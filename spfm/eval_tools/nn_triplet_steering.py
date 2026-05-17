#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from accelerate.state import PartialState
from accelerate.utils import set_seed
from datasets import load_dataset
from torchvision import transforms as TVT
from torchvision.utils import make_grid, save_image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint
import run_experiment
import train
from preprocessing.encoders import load_invae
from trainer.db import build_primary_db
from trainer.model_factory import build_model
from utils.train_helpers import (
    _apply_label_filter,
    _apply_label_split,
    decode_latents,
    ensure_dir,
    get_vae_scaling,
    infer_vae_latent_spec,
    nearest_neighbors,
    vae_tag_from_name,
)

LOGGER = logging.getLogger("spfm_nn_triplet_eval")
NN_SEARCH_CHUNK = 1024


def _parse_bool(value):
    return eval_checkpoint._parse_bool(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate 8 images and latent nearest neighbours for DB variants plus full-DB steering variants."
    )
    ap.add_argument("--config", default="experiments/spfm.yaml")
    ap.add_argument("--ckpt", default="out/spfm_afhq_cat_dog/model_step10000.pt")
    ap.add_argument(
        "--results_dir", default="out/evals/spfm_afhq_cat_dog_step10000/nn_triplet_steer"
    )
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--sample_steps", type=int, default=None)
    ap.add_argument("--decode_batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use_ema", type=_parse_bool, default=False)
    ap.add_argument("--steer_strength", type=float, default=1.0)
    ap.add_argument(
        "--beta_schedule",
        choices=["constant", "bell", "quadratic-decay"],
        default="quadratic-decay",
    )
    ap.add_argument("--steer_start_frac", type=float, default=0.15)
    ap.add_argument("--steer_end_frac", type=float, default=0.95)
    ap.add_argument("--steer_topk", type=int, default=0, help="0 means use the full steering bank.")
    ap.add_argument(
        "--class_subset_size",
        type=int,
        default=0,
        help="Use this many selected examples for cats_only/dogs_only and steering banks. 0 uses the full class DBs.",
    )
    ap.add_argument(
        "--class_subset_mode", choices=["random", "white_background"], default="white_background"
    )
    ap.add_argument("--patch_rows", type=int, default=None)
    ap.add_argument("--patch_cols", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _setup_logging(results_dir: str) -> None:
    ensure_dir(results_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "eval.log")),
        ],
        force=True,
    )


def _load_args(
    config_path: str, ckpt: str | None, sample_steps: int | None, decode_batch: int | None
):
    ckpt_args_path = None
    if ckpt is not None:
        ckpt_path = Path(ckpt).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (REPO_ROOT / ckpt_path).resolve()
        ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        candidate = ckpt_dir / "args.json"
        if candidate.exists():
            ckpt_args_path = candidate

    if ckpt_args_path is not None:
        raw_args = json.loads(ckpt_args_path.read_text(encoding="utf-8"))
        cli_args = run_experiment._to_cli_args(raw_args)
        args = train.parse_args(cli_args)
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        train_cfg = cfg.get("train", cfg)
        flat_cfg = run_experiment._merge_train_cfg(train_cfg)
        cli_args = run_experiment._to_cli_args(flat_cfg)
        args = train.parse_args(cli_args)

    if ckpt is not None:
        args.ckpt = ckpt
    if sample_steps is not None:
        args.sample_steps = int(sample_steps)
    if decode_batch is not None:
        args.decode_batch = int(decode_batch)
    args.report_to = None
    args.wandb_project = None
    args.wandb_entity = None
    args.wandb_name = None
    args.wandb_resume = None
    args.wandb_run_id = None
    args.exp_name = None
    args.self_mask_db = False
    args.alt_db = "none"
    return args


def _resolve_ckpt_path(ckpt: str, train_out_dir: str) -> str:
    ckpt_path = Path(ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    return eval_checkpoint._resolve_model_weights_path(str(ckpt_path), train_out_dir)


def _build_db_artifacts(
    base_args, *, db_spec: str, vae, device: torch.device, return_indices: bool = False
):
    variant_args = argparse.Namespace(**vars(base_args))
    variant_args.db = db_spec
    variant_args.alt_db = "none"
    variant_args.self_mask_db = bool(return_indices)

    class _Accel:
        is_main_process = True

        @staticmethod
        def wait_for_everyone() -> None:
            return None

    primary = build_primary_db(variant_args, vae=vae, device=device, accelerator=_Accel())
    if return_indices and primary.db_indices is None:
        raise ValueError(f"DB indices were not returned for {db_spec}")
    return primary


def _build_db_variant(base_args, *, db_spec: str, vae, device: torch.device):
    primary = _build_db_artifacts(
        base_args, db_spec=db_spec, vae=vae, device=device, return_indices=False
    )
    return primary.Xdb


def _save_image_set(imgs: torch.Tensor, out_dir: str, prefix: str) -> list[str]:
    ensure_dir(out_dir)
    paths = []
    for idx in range(imgs.shape[0]):
        path = os.path.join(out_dir, f"{prefix}_{idx:02d}.png")
        save_image(imgs[idx].detach().cpu(), path)
        paths.append(path)
    return paths


def _label_value_from_db_spec(db_spec: str) -> str:
    labels = []
    for entry in [e.strip() for e in db_spec.split(",") if e.strip()]:
        if ":" in entry:
            labels.append(entry.split(":", 1)[0].strip())
        elif "=" in entry:
            labels.append(entry.split("=", 1)[0].strip())
        else:
            labels.append(entry)
    return ",".join(label for label in labels if label)


def _white_background_score(img: torch.Tensor) -> float:
    # img is [C,H,W] in [0,1]. AFHQ faces are already centered, so favor clean white borders.
    border = torch.cat(
        [
            img[:, :32, :].permute(1, 2, 0).reshape(-1, 3),
            img[:, -32:, :].permute(1, 2, 0).reshape(-1, 3),
            img[:, :, :32].permute(1, 2, 0).reshape(-1, 3),
            img[:, :, -32:].permute(1, 2, 0).reshape(-1, 3),
        ],
        dim=0,
    )
    center = img[:, 64:192, 64:192].permute(1, 2, 0).reshape(-1, 3)
    border_white = ((border > 0.86).all(dim=1)).float().mean()
    border_sat = (border.max(dim=1).values - border.min(dim=1).values).mean()
    center_white = ((center > 0.86).all(dim=1)).float().mean()
    return float(border_white.item() - 0.35 * border_sat.item() - 0.10 * center_white.item())


def _select_class_subset(
    *,
    args,
    db_spec: str,
    db: torch.Tensor,
    db_indices: torch.Tensor,
    subset_size: int,
    mode: str,
    seed: int,
    out_dir: str,
    tag: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    if subset_size <= 0:
        return db, {"subset_size": int(db.shape[0]), "subset_mode": "full"}
    if int(db.shape[0]) < subset_size:
        raise ValueError(
            f"{tag}: requested {subset_size} examples but DB has only {int(db.shape[0])}"
        )

    label_value = _label_value_from_db_spec(db_spec)
    ds = load_dataset(args.dataset, split=args.split, streaming=False)
    ds = _apply_label_filter(ds, args.label_field, label_value)
    ds = _apply_label_split(
        ds,
        label_field=args.label_field,
        label_split_spec=db_spec,
        split_seed=int(args.seed),
        use_complement=False,
    )

    idx_to_row = {int(idx): row for row, idx in enumerate(db_indices.detach().cpu().tolist())}
    candidate_indices = sorted(idx_to_row.keys())
    if len(candidate_indices) < subset_size:
        raise ValueError(f"{tag}: only {len(candidate_indices)} indexed examples available")

    if mode == "random":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        perm = torch.randperm(len(candidate_indices), generator=gen)[:subset_size].tolist()
        selected_ds_indices = [candidate_indices[i] for i in perm]
        scores = [None for _ in selected_ds_indices]
        selected_imgs = []
    else:
        tf = TVT.Compose(
            [TVT.Resize(args.image_size), TVT.CenterCrop(args.image_size), TVT.ToTensor()]
        )
        scored: list[tuple[float, int, torch.Tensor]] = []
        for ds_idx in candidate_indices:
            ex = ds[int(ds_idx)]
            im = ex.get("img", None) or ex.get("image", None)
            if im is None:
                raise KeyError("Example missing 'img' or 'image'")
            img = tf(im.convert("RGB"))
            scored.append((_white_background_score(img), int(ds_idx), img))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[:subset_size]
        selected_ds_indices = [idx for _score, idx, _img in chosen]
        scores = [score for score, _idx, _img in chosen]
        selected_imgs = [img for _score, _idx, img in chosen]

    selected_rows = [idx_to_row[int(idx)] for idx in selected_ds_indices]
    row_tensor = torch.tensor(selected_rows, device=db.device, dtype=torch.long)
    subset = db.index_select(0, row_tensor)

    ensure_dir(out_dir)
    if mode == "white_background" and selected_imgs:
        ref_grid = make_grid(torch.stack(selected_imgs, dim=0), nrow=subset_size)
        save_image(ref_grid, os.path.join(out_dir, f"{tag}_selected_examples_grid.png"))

    with open(os.path.join(out_dir, f"{tag}_selected_examples.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "db_spec": db_spec,
                "index_scope": "dataset after label filter and label split",
                "subset_size": int(subset_size),
                "subset_mode": mode,
                "dataset_indices": [int(x) for x in selected_ds_indices],
                "db_rows": [int(x) for x in selected_rows],
                "white_background_scores": scores,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    return subset, {
        "subset_size": int(subset_size),
        "subset_mode": mode,
        "selected_dataset_indices": [int(x) for x in selected_ds_indices],
        "selected_db_rows": [int(x) for x in selected_rows],
    }


@torch.no_grad()
def _sample_from_initial_latents(
    *,
    Xdb: torch.Tensor,
    model,
    vae,
    initial_latents: torch.Tensor,
    steps: int,
    t_eps: float,
    decode_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = Xdb.device
    x = initial_latents.to(device=device).clone()
    num_gen = x.shape[0]
    ts = torch.linspace(t_eps, 1.0 - t_eps, steps + 1, device=device)
    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        t = t_curr.expand(num_gen)
        out = model(x, t, Xdb)
        mu_step = out[0] if isinstance(out, (tuple, list)) else out
        dt = t_next - t_curr
        denom = (1.0 - t_curr).clamp(min=t_eps)
        x = x + ((mu_step - x) / denom) * dt
    imgs = decode_latents(vae, x, decode_batch=decode_batch)
    return imgs, x


def _split_spatial_patches(x: torch.Tensor, patch_rows: int, patch_cols: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("Expected latent tensor with shape [B, C, H, W].")
    batch, channels, height, width = x.shape
    if height % patch_rows != 0 or width % patch_cols != 0:
        raise ValueError(
            f"Latent shape {(height, width)} not divisible by patch grid {(patch_rows, patch_cols)}."
        )
    patch_h = height // patch_rows
    patch_w = width // patch_cols
    x = x.view(batch, channels, patch_rows, patch_h, patch_cols, patch_w)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(batch, patch_rows * patch_cols, channels, patch_h, patch_w)


def _merge_spatial_patches(patches: torch.Tensor, patch_rows: int, patch_cols: int) -> torch.Tensor:
    batch, num_patches, channels, patch_h, patch_w = patches.shape
    if num_patches != patch_rows * patch_cols:
        raise ValueError("Patch count does not match patch_rows * patch_cols.")
    x = patches.view(batch, patch_rows, patch_cols, channels, patch_h, patch_w)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(batch, channels, patch_rows * patch_h, patch_cols * patch_w)


def _schedule_beta(steer_strength: float, beta_schedule: str, t: float) -> float:
    if beta_schedule == "bell":
        return 4.0 * steer_strength * t * (1.0 - t)
    if beta_schedule == "quadratic-decay":
        return steer_strength * (1.0 - t) ** 2
    return steer_strength


def _in_steering_window(
    step_index: int, total_steps: int, start_frac: float, end_frac: float
) -> bool:
    frac = float(step_index) / float(max(total_steps - 1, 1))
    return start_frac <= frac <= end_frac


@torch.no_grad()
def _patchwise_retrieval_update(
    *,
    current_latents: torch.Tensor,
    bank_latents: torch.Tensor,
    t: float,
    patch_rows: int,
    patch_cols: int,
    topk: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    current_patches = _split_spatial_patches(current_latents.float(), patch_rows, patch_cols)
    bank_patches = _split_spatial_patches(bank_latents.float(), patch_rows, patch_cols)

    batch, num_patches, channels, patch_h, patch_w = current_patches.shape
    bank_size = bank_patches.shape[0]
    patch_dim = channels * patch_h * patch_w

    current_flat = current_patches.view(batch, num_patches, patch_dim)
    bank_flat = bank_patches.view(bank_size, num_patches, patch_dim)

    t = float(max(min(t, 1.0 - 1e-4), 1e-4))
    projected_bank = t * bank_flat
    centered = current_flat[:, None, :, :] - projected_bank[None, :, :, :]
    dist2 = centered.pow(2).mean(dim=-1).permute(0, 2, 1).contiguous()
    logits = -dist2 / (2.0 * ((1.0 - t) ** 2 + 1e-8))

    if topk > 0 and topk < logits.shape[-1]:
        vals, idx = torch.topk(logits, k=topk, dim=-1)
        weights = torch.softmax(vals, dim=-1)
        bank_by_patch = bank_flat.permute(1, 0, 2).unsqueeze(0).expand(batch, -1, -1, -1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, patch_dim)
        selected_bank = torch.gather(bank_by_patch, 2, gather_idx)
        mu_patch = (weights[..., None] * selected_bank).sum(dim=2)
    else:
        weights = torch.softmax(logits, dim=-1)
        mu_patch = torch.einsum("bpm,mpd->bpd", weights, bank_flat)

    mu_patch = mu_patch.view(batch, num_patches, channels, patch_h, patch_w)
    mu_latents = _merge_spatial_patches(mu_patch, patch_rows, patch_cols).to(current_latents.dtype)
    v_star = ((mu_latents.float() - current_latents.float()) / max(1.0 - t, 1e-4)).to(
        current_latents.dtype
    )

    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
    stats = {
        "mu_star_norm": float(mu_latents.float().reshape(batch, -1).norm(dim=-1).mean().item()),
        "current_norm": float(
            current_latents.float().reshape(batch, -1).norm(dim=-1).mean().item()
        ),
        "v_star_norm": float(v_star.float().reshape(batch, -1).norm(dim=-1).mean().item()),
        "posterior_entropy": float(entropy.mean().item()),
        "top1_weight": float(weights.max(dim=-1).values.mean().item()),
    }
    return v_star, stats


@torch.no_grad()
def _sample_from_initial_latents_with_steering(
    *,
    model_db: torch.Tensor,
    steer_db: torch.Tensor,
    model,
    vae,
    initial_latents: torch.Tensor,
    steps: int,
    t_eps: float,
    decode_batch: int,
    patch_rows: int,
    patch_cols: int,
    steer_strength: float,
    beta_schedule: str,
    steer_start_frac: float,
    steer_end_frac: float,
    steer_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = model_db.device
    x = initial_latents.to(device=device).clone()
    num_gen = x.shape[0]
    ts = torch.linspace(t_eps, 1.0 - t_eps, steps + 1, device=device)
    history: list[dict[str, float]] = []

    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        t = t_curr.expand(num_gen)
        out = model(x, t, model_db)
        mu_step = out[0] if isinstance(out, (tuple, list)) else out
        dt = t_next - t_curr
        denom = (1.0 - t_curr).clamp(min=t_eps)
        v_total = (mu_step - x) / denom

        if _in_steering_window(i, steps, steer_start_frac, steer_end_frac):
            beta_t = _schedule_beta(steer_strength, beta_schedule, float(t_curr.item()))
            v_star, stats = _patchwise_retrieval_update(
                current_latents=x,
                bank_latents=steer_db,
                t=float(t_curr.item()),
                patch_rows=patch_rows,
                patch_cols=patch_cols,
                topk=int(steer_topk),
            )
            v_total = v_total + beta_t * v_star
            stats["step"] = float(i)
            stats["t"] = float(t_curr.item())
            stats["dt"] = float(dt.item())
            stats["beta_t"] = float(beta_t)
            history.append(stats)

        x = x + v_total * dt

    imgs = decode_latents(vae, x, decode_batch=decode_batch)
    return imgs, x, history


def _make_initial_latents(
    args, *, device: torch.device, base_seed: int
) -> tuple[torch.Tensor, list[int]]:
    seeds = [int(base_seed) + i for i in range(int(args.num_gen))]
    latents = []
    for seed in seeds:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        latents.append(
            float(args.noise_scale)
            * torch.randn(
                1,
                int(args.latent_c),
                int(args.latent_h),
                int(args.latent_w),
                device=device,
                generator=gen,
            )
        )
    return torch.cat(latents, dim=0), seeds


def _run_variant(
    *,
    name: str,
    db_spec: str,
    model,
    vae,
    args,
    device: torch.device,
    results_dir: str,
    initial_latents: torch.Tensor,
    sample_seeds: list[int],
) -> dict[str, object]:
    variant_dir = os.path.join(results_dir, name)
    ensure_dir(variant_dir)
    db = _build_db_variant(args, db_spec=db_spec, vae=vae, device=device)
    LOGGER.info("[%s] DB shape=%s spec=%s", name, tuple(db.shape), db_spec)

    imgs, latents = _sample_from_initial_latents(
        Xdb=db,
        model=model,
        vae=vae,
        initial_latents=initial_latents,
        steps=int(args.sample_steps),
        t_eps=float(args.t_eps),
        decode_batch=int(args.decode_batch),
    )

    return _write_variant_outputs(
        name=name,
        db_spec=db_spec,
        db=db,
        imgs=imgs,
        latents=latents,
        vae=vae,
        args=args,
        results_dir=results_dir,
        sample_seeds=sample_seeds,
    )


def _run_variant_with_db(
    *,
    name: str,
    db_spec: str,
    db: torch.Tensor,
    model,
    vae,
    args,
    results_dir: str,
    initial_latents: torch.Tensor,
    sample_seeds: list[int],
    extra_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    LOGGER.info("[%s] DB shape=%s spec=%s", name, tuple(db.shape), db_spec)
    imgs, latents = _sample_from_initial_latents(
        Xdb=db,
        model=model,
        vae=vae,
        initial_latents=initial_latents,
        steps=int(args.sample_steps),
        t_eps=float(args.t_eps),
        decode_batch=int(args.decode_batch),
    )
    return _write_variant_outputs(
        name=name,
        db_spec=db_spec,
        db=db,
        imgs=imgs,
        latents=latents,
        vae=vae,
        args=args,
        results_dir=results_dir,
        sample_seeds=sample_seeds,
        extra_metrics=extra_metrics,
    )


def _write_variant_outputs(
    *,
    name: str,
    db_spec: str,
    db: torch.Tensor,
    imgs: torch.Tensor,
    latents: torch.Tensor,
    vae,
    args,
    results_dir: str,
    sample_seeds: list[int],
    extra_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    variant_dir = os.path.join(results_dir, name)
    ensure_dir(variant_dir)

    nn_idx, nn_dist = nearest_neighbors(latents, db, chunk=NN_SEARCH_CHUNK)
    nn_latents = db.index_select(0, nn_idx.to(device=db.device, dtype=torch.long))
    nn_imgs = decode_latents(vae, nn_latents, decode_batch=int(args.decode_batch))

    sample_dir = os.path.join(variant_dir, "generated")
    nn_dir = os.path.join(variant_dir, "nearest_neighbors")
    sample_paths = _save_image_set(imgs, sample_dir, "img")
    nn_paths = _save_image_set(nn_imgs, nn_dir, "nn")

    sample_grid = make_grid(imgs.detach().cpu(), nrow=int(args.num_gen))
    nn_grid = make_grid(nn_imgs.detach().cpu(), nrow=int(args.num_gen))
    save_image(sample_grid, os.path.join(variant_dir, "generated_grid.png"))
    save_image(nn_grid, os.path.join(variant_dir, "nearest_neighbors_grid.png"))
    paired_grid = make_grid(
        torch.cat([imgs.detach().cpu(), nn_imgs.detach().cpu()], dim=0), nrow=int(args.num_gen)
    )
    save_image(paired_grid, os.path.join(variant_dir, "generated_vs_nn_grid.png"))

    metrics = {
        "variant": name,
        "db_spec": db_spec,
        "db_size": int(db.shape[0]),
        "num_gen": int(args.num_gen),
        "sample_seeds": sample_seeds,
        "sample_steps": int(args.sample_steps),
        "nn_mean_l2_squared": float(nn_dist.mean().item()),
        "nn_max_l2_squared": float(nn_dist.max().item()),
        "nn_indices": [int(x) for x in nn_idx.detach().cpu().tolist()],
        "generated_paths": sample_paths,
        "nearest_neighbor_paths": nn_paths,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    with open(os.path.join(variant_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def _run_steered_variant(
    *,
    name: str,
    model_db: torch.Tensor,
    model_db_spec: str,
    steer_db: torch.Tensor,
    steer_db_spec: str,
    model,
    vae,
    args,
    results_dir: str,
    initial_latents: torch.Tensor,
    sample_seeds: list[int],
    patch_rows: int,
    patch_cols: int,
    steer_strength: float,
    beta_schedule: str,
    steer_start_frac: float,
    steer_end_frac: float,
    steer_topk: int,
    extra_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    LOGGER.info(
        "[%s] model DB shape=%s spec=%s steer DB shape=%s spec=%s",
        name,
        tuple(model_db.shape),
        model_db_spec,
        tuple(steer_db.shape),
        steer_db_spec,
    )
    imgs, latents, history = _sample_from_initial_latents_with_steering(
        model_db=model_db,
        steer_db=steer_db,
        model=model,
        vae=vae,
        initial_latents=initial_latents,
        steps=int(args.sample_steps),
        t_eps=float(args.t_eps),
        decode_batch=int(args.decode_batch),
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        steer_strength=steer_strength,
        beta_schedule=beta_schedule,
        steer_start_frac=steer_start_frac,
        steer_end_frac=steer_end_frac,
        steer_topk=steer_topk,
    )
    variant_dir = os.path.join(results_dir, name)
    ensure_dir(variant_dir)
    with open(os.path.join(variant_dir, "steer_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics = {
        "model_db_spec": model_db_spec,
        "model_db_size": int(model_db.shape[0]),
        "steer_db_spec": steer_db_spec,
        "steer_db_size": int(steer_db.shape[0]),
        "steer_strength": float(steer_strength),
        "beta_schedule": str(beta_schedule),
        "steer_start_frac": float(steer_start_frac),
        "steer_end_frac": float(steer_end_frac),
        "steer_topk": int(steer_topk),
        "patch_rows": int(patch_rows),
        "patch_cols": int(patch_cols),
        "steer_history_path": os.path.join(variant_dir, "steer_history.json"),
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    return _write_variant_outputs(
        name=name,
        db_spec=model_db_spec,
        db=model_db,
        imgs=imgs,
        latents=latents,
        vae=vae,
        args=args,
        results_dir=results_dir,
        sample_seeds=sample_seeds,
        extra_metrics=metrics,
    )


def main() -> int:
    ns = parse_args()
    results_dir = str(Path(ns.results_dir).expanduser().resolve())
    _setup_logging(results_dir)
    PartialState()
    set_seed(int(ns.seed))

    device = torch.device(ns.device)
    args = _load_args(ns.config, ns.ckpt, ns.sample_steps, ns.decode_batch)
    args.num_gen = int(ns.num_gen)

    vae = load_invae(args.vae_name, device=device)
    vae.eval().requires_grad_(False)
    latent_c, latent_h, latent_w, latent_downsample = infer_vae_latent_spec(
        vae, args.image_size, device
    )
    args.latent_c = latent_c
    args.latent_h = latent_h
    args.latent_w = latent_w
    args.latent_downsample = latent_downsample
    args.vae_scaling = get_vae_scaling(vae, args.vae_name)
    args.vae_tag = vae_tag_from_name(args.vae_name)
    vae._pairflow_scaling = args.vae_scaling
    eval_checkpoint._maybe_override_embed_dim(args)

    model = build_model(args, device)
    ckpt_path = _resolve_ckpt_path(args.ckpt, str(Path(args.out_dir).expanduser().resolve()))
    eval_checkpoint._load_model_weights(model, ckpt_path, use_ema=bool(ns.use_ema))
    raw_model = train._unwrap_model_for_runtime(model)
    raw_model.eval()
    initial_latents, sample_seeds = _make_initial_latents(
        args, device=device, base_seed=int(ns.seed)
    )
    torch.save(
        {"sample_seeds": sample_seeds, "initial_latents": initial_latents.detach().cpu()},
        os.path.join(results_dir, "initial_latents.pt"),
    )

    patch_rows = int(ns.patch_rows or (args.latent_h // args.cross_patch_size))
    patch_cols = int(ns.patch_cols or (args.latent_w // args.cross_patch_size))

    variants = [
        ("full_db", "dog:1.0,cat:1.0"),
    ]
    all_metrics = []
    for name, db_spec in variants:
        all_metrics.append(
            _run_variant(
                name=name,
                db_spec=db_spec,
                model=raw_model,
                vae=vae,
                args=args,
                device=device,
                results_dir=results_dir,
                initial_latents=initial_latents,
                sample_seeds=sample_seeds,
            )
        )

    full_db_spec = "dog:1.0,cat:1.0"
    cats_db_spec = "cat:1.0,dog:0.0"
    dogs_db_spec = "cat:0.0,dog:1.0"
    full_db = _build_db_variant(args, db_spec=full_db_spec, vae=vae, device=device)
    need_class_indices = int(ns.class_subset_size) > 0
    cats_artifacts = _build_db_artifacts(
        args,
        db_spec=cats_db_spec,
        vae=vae,
        device=device,
        return_indices=need_class_indices,
    )
    dogs_artifacts = _build_db_artifacts(
        args,
        db_spec=dogs_db_spec,
        vae=vae,
        device=device,
        return_indices=need_class_indices,
    )
    cats_db = cats_artifacts.Xdb
    dogs_db = dogs_artifacts.Xdb
    cats_subset_meta: dict[str, object] = {
        "subset_size": int(cats_db.shape[0]),
        "subset_mode": "full",
    }
    dogs_subset_meta: dict[str, object] = {
        "subset_size": int(dogs_db.shape[0]),
        "subset_mode": "full",
    }
    if need_class_indices:
        cats_db, cats_subset_meta = _select_class_subset(
            args=args,
            db_spec=cats_db_spec,
            db=cats_db,
            db_indices=cats_artifacts.db_indices,
            subset_size=int(ns.class_subset_size),
            mode=str(ns.class_subset_mode),
            seed=int(ns.seed) + 17,
            out_dir=results_dir,
            tag="cats",
        )
        dogs_db, dogs_subset_meta = _select_class_subset(
            args=args,
            db_spec=dogs_db_spec,
            db=dogs_db,
            db_indices=dogs_artifacts.db_indices,
            subset_size=int(ns.class_subset_size),
            mode=str(ns.class_subset_mode),
            seed=int(ns.seed) + 23,
            out_dir=results_dir,
            tag="dogs",
        )

    all_metrics.append(
        _run_variant_with_db(
            name="cats_only",
            db_spec=cats_db_spec,
            db=cats_db,
            model=raw_model,
            vae=vae,
            args=args,
            results_dir=results_dir,
            initial_latents=initial_latents,
            sample_seeds=sample_seeds,
            extra_metrics=cats_subset_meta,
        )
    )
    all_metrics.append(
        _run_variant_with_db(
            name="dogs_only",
            db_spec=dogs_db_spec,
            db=dogs_db,
            model=raw_model,
            vae=vae,
            args=args,
            results_dir=results_dir,
            initial_latents=initial_latents,
            sample_seeds=sample_seeds,
            extra_metrics=dogs_subset_meta,
        )
    )
    all_metrics.append(
        _run_steered_variant(
            name="full_db_steer_dogs",
            model_db=full_db,
            model_db_spec=full_db_spec,
            steer_db=dogs_db,
            steer_db_spec=dogs_db_spec,
            model=raw_model,
            vae=vae,
            args=args,
            results_dir=results_dir,
            initial_latents=initial_latents,
            sample_seeds=sample_seeds,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            steer_strength=float(ns.steer_strength),
            beta_schedule=str(ns.beta_schedule),
            steer_start_frac=float(ns.steer_start_frac),
            steer_end_frac=float(ns.steer_end_frac),
            steer_topk=int(ns.steer_topk),
            extra_metrics={"steer_subset": dogs_subset_meta},
        )
    )
    all_metrics.append(
        _run_steered_variant(
            name="full_db_steer_cats",
            model_db=full_db,
            model_db_spec=full_db_spec,
            steer_db=cats_db,
            steer_db_spec=cats_db_spec,
            model=raw_model,
            vae=vae,
            args=args,
            results_dir=results_dir,
            initial_latents=initial_latents,
            sample_seeds=sample_seeds,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            steer_strength=float(ns.steer_strength),
            beta_schedule=str(ns.beta_schedule),
            steer_start_frac=float(ns.steer_start_frac),
            steer_end_frac=float(ns.steer_end_frac),
            steer_topk=int(ns.steer_topk),
            extra_metrics={"steer_subset": cats_subset_meta},
        )
    )

    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "config": str(Path(ns.config).expanduser().resolve()),
                "num_gen": int(ns.num_gen),
                "sample_seeds": sample_seeds,
                "sample_steps": int(args.sample_steps),
                "steering": {
                    "steer_strength": float(ns.steer_strength),
                    "beta_schedule": str(ns.beta_schedule),
                    "steer_start_frac": float(ns.steer_start_frac),
                    "steer_end_frac": float(ns.steer_end_frac),
                    "steer_topk": int(ns.steer_topk),
                    "patch_rows": int(patch_rows),
                    "patch_cols": int(patch_cols),
                },
                "class_subset": {
                    "size": int(ns.class_subset_size),
                    "mode": str(ns.class_subset_mode) if int(ns.class_subset_size) > 0 else "full",
                    "cats": cats_subset_meta,
                    "dogs": dogs_subset_meta,
                },
                "variants": all_metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    LOGGER.info("[done] wrote %s", os.path.join(results_dir, "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
