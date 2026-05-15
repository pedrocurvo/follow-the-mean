#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate.state import PartialState
from accelerate.utils import set_seed
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
from utils.train_helpers import decode_latents, ensure_dir, get_vae_scaling, infer_vae_latent_spec, vae_tag_from_name


LOGGER = logging.getLogger("afhq_fixed_seed_comparison")


@dataclass
class Runtime:
    name: str
    args: object
    ckpt_path: str
    model: torch.nn.Module
    vae: torch.nn.Module


class _SingleProcessAccel:
    is_main_process = True
    process_index = 0

    @staticmethod
    def wait_for_everyone() -> None:
        return None


def _parse_bool(value):
    return eval_checkpoint._parse_bool(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate a fixed-seed AFHQ comparison across baseline, retrieval DB selection, and retrieval steering."
    )
    ap.add_argument("--baseline-ckpt", default="out/model_baseline_dit_afhq")
    ap.add_argument("--retrieval-ckpt", default="out/model_spf_fullattention_afhq")
    ap.add_argument("--config", default="experiments/model.yaml")
    ap.add_argument("--output-root", default="out/afhq_fixed_seed_comparison")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--decode-batch", type=int, default=None)
    ap.add_argument("--use-ema", type=_parse_bool, default=False)
    ap.add_argument("--cats", type=int, default=50)
    ap.add_argument("--dogs", type=int, default=50)
    ap.add_argument("--steer-strength", type=float, default=1.0)
    ap.add_argument("--beta-schedule", choices=["constant", "bell", "quadratic-decay"], default="quadratic-decay")
    ap.add_argument("--steer-start-frac", type=float, default=0.15)
    ap.add_argument("--steer-end-frac", type=float, default=0.95)
    ap.add_argument("--steer-topk", type=int, default=0, help="0 means use the full bank.")
    ap.add_argument("--patch-rows", type=int, default=None)
    ap.add_argument("--patch-cols", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _setup_logging(output_root: str) -> None:
    ensure_dir(output_root)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_root, "run.log")),
        ],
        force=True,
    )


def _load_training_args(
    config_path: str,
    ckpt_override: str | None,
    sample_steps: int | None,
    decode_batch: int | None,
):
    ckpt_args_path = None
    if ckpt_override is not None:
        ckpt_path = Path(ckpt_override).expanduser()
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
        args = eval_checkpoint._load_config_args(config_path)

    args._train_out_dir = str(args.out_dir)
    if ckpt_override is not None:
        args.ckpt = ckpt_override
    if sample_steps is not None:
        args.sample_steps = int(sample_steps)
    if decode_batch is not None:
        args.decode_batch = int(decode_batch)
    args.out_dir = "out/tmp_unused_fixed_seed"
    args.exp_name = None
    args.report_to = None
    args.wandb_project = None
    args.wandb_entity = None
    args.wandb_name = None
    args.wandb_resume = None
    args.wandb_run_id = None
    args.fid = False
    args.kid = False
    args.alt_db = "none"
    return args


def _resolve_ckpt_path(ckpt: str, train_out_dir: str) -> str:
    ckpt_path = Path(ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if ckpt_path.is_file():
        return str(ckpt_path.resolve())
    if ckpt_path.is_dir():
        direct_last = ckpt_path / "model_last.pt"
        if direct_last.exists():
            return str(direct_last.resolve())
        step_candidates = sorted(ckpt_path.glob("model_step*.pt"))
        if step_candidates:
            return str(step_candidates[-1].resolve())
        parent = ckpt_path.parent
        step_tag = ckpt_path.name.replace("checkpoint_", "")
        candidate = parent / f"model_{step_tag}.pt"
        if candidate.exists():
            return str(candidate.resolve())
        last_candidate = parent / "model_last.pt"
        if last_candidate.exists():
            return str(last_candidate.resolve())
        parent_step_candidates = sorted(parent.glob("model_step*.pt"))
        if parent_step_candidates:
            return str(parent_step_candidates[-1].resolve())
        raise FileNotFoundError(f"Could not resolve model weights from checkpoint dir: {ckpt_path}")
    return eval_checkpoint._resolve_model_weights_path(str(ckpt_path), train_out_dir)


def _load_runtime(
    *,
    name: str,
    config_path: str,
    ckpt: str,
    sample_steps: int | None,
    decode_batch: int | None,
    device: torch.device,
    use_ema: bool,
) -> Runtime:
    args = _load_training_args(config_path, ckpt, sample_steps, decode_batch)
    is_raw_vae = str(args.vae_name).strip().lower() == "raw"
    if is_raw_vae:
        raise ValueError("Raw VAE mode is not supported.")

    vae = load_invae(args.vae_name, device=device)
    vae.eval().requires_grad_(False)

    latent_c, latent_h, latent_w, latent_downsample = infer_vae_latent_spec(vae, args.image_size, device)
    args.latent_c = latent_c
    args.latent_h = latent_h
    args.latent_w = latent_w
    args.latent_downsample = latent_downsample
    args.vae_scaling = get_vae_scaling(vae, args.vae_name)
    args.vae_tag = vae_tag_from_name(args.vae_name)
    vae._pairflow_scaling = args.vae_scaling
    eval_checkpoint._maybe_override_embed_dim(args)

    model = build_model(args, device)
    ckpt_path = _resolve_ckpt_path(args.ckpt, str(args._train_out_dir))
    eval_checkpoint._load_model_weights(model, ckpt_path, use_ema=use_ema)
    raw_model = train._unwrap_model_for_runtime(model)
    raw_model.eval()

    return Runtime(name=name, args=args, ckpt_path=ckpt_path, model=raw_model, vae=vae)


def _make_initial_latent(args, *, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return float(args.noise_scale) * torch.randn(
        1,
        int(args.latent_c),
        int(args.latent_h),
        int(args.latent_w),
        device=device,
        generator=gen,
    )


def _db_cache_dir(output_root: str, db_spec: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", db_spec).strip("_")
    return os.path.join(output_root, "db_cache", tag or "db")


def _build_db(
    runtime: Runtime,
    db_spec: str,
    device: torch.device,
    *,
    output_root: str,
) -> tuple[torch.Tensor, list[int] | None]:
    args = argparse.Namespace(**vars(runtime.args))
    args.db = db_spec
    args.db_dir = _db_cache_dir(output_root, db_spec)
    args.alt_db = "none"
    args.self_mask_db = True
    primary = build_primary_db(args, vae=runtime.vae, device=device, accelerator=_SingleProcessAccel())
    indices = None
    if primary.db_indices is not None:
        indices = [int(x) for x in primary.db_indices.detach().cpu().tolist()]
    return primary.Xdb, indices


@torch.no_grad()
def _sample_from_initial_latent(
    *,
    model,
    Xdb: torch.Tensor,
    vae,
    initial_latent: torch.Tensor,
    steps: int,
    t_eps: float,
    decode_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = initial_latent.clone()
    ts = torch.linspace(t_eps, 1.0 - t_eps, steps + 1, device=x.device)
    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        t = t_curr.expand(x.shape[0])
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
        raise ValueError(f"Latent shape {(height, width)} not divisible by patch grid {(patch_rows, patch_cols)}.")
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


def _in_steering_window(step_index: int, total_steps: int, start_frac: float, end_frac: float) -> bool:
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
    v_star = ((mu_latents.float() - current_latents.float()) / max(1.0 - t, 1e-4)).to(current_latents.dtype)

    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1)
    stats = {
        "mu_star_norm": float(mu_latents.float().reshape(batch, -1).norm(dim=-1).mean().item()),
        "current_norm": float(current_latents.float().reshape(batch, -1).norm(dim=-1).mean().item()),
        "v_star_norm": float(v_star.float().reshape(batch, -1).norm(dim=-1).mean().item()),
        "posterior_entropy": float(entropy.mean().item()),
        "top1_weight": float(weights.max(dim=-1).values.mean().item()),
    }
    return v_star, stats


@torch.no_grad()
def _sample_baseline_with_retrieval_steering(
    *,
    baseline_runtime: Runtime,
    bank_latents: torch.Tensor,
    initial_latent: torch.Tensor,
    patch_rows: int,
    patch_cols: int,
    steer_strength: float,
    beta_schedule: str,
    steer_start_frac: float,
    steer_end_frac: float,
    steer_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    args = baseline_runtime.args
    model = baseline_runtime.model
    x = initial_latent.clone()
    dummy_db = bank_latents[:1]
    ts = torch.linspace(float(args.t_eps), 1.0 - float(args.t_eps), int(args.sample_steps) + 1, device=x.device)
    history: list[dict[str, float]] = []

    for step_idx in range(int(args.sample_steps)):
        t_curr = ts[step_idx]
        t_next = ts[step_idx + 1]
        t = t_curr.expand(x.shape[0])
        out = model(x, t, dummy_db)
        mu_step = out[0] if isinstance(out, (tuple, list)) else out
        dt = t_next - t_curr
        denom = (1.0 - t_curr).clamp(min=float(args.t_eps))
        v_model = (mu_step - x) / denom
        v_total = v_model

        if _in_steering_window(step_idx, int(args.sample_steps), steer_start_frac, steer_end_frac):
            beta_t = _schedule_beta(steer_strength, beta_schedule, float(t_curr.item()))
            v_star, stats = _patchwise_retrieval_update(
                current_latents=x,
                bank_latents=bank_latents,
                t=float(t_curr.item()),
                patch_rows=patch_rows,
                patch_cols=patch_cols,
                topk=int(steer_topk),
            )
            v_total = v_total + beta_t * v_star
            stats["step"] = float(step_idx)
            stats["t"] = float(t_curr.item())
            stats["dt"] = float(dt.item())
            stats["beta_t"] = float(beta_t)
            history.append(stats)

        x = x + v_total * dt

    imgs = decode_latents(baseline_runtime.vae, x, decode_batch=int(args.decode_batch))
    return imgs, x, history


@torch.no_grad()
def _sample_model_with_retrieval_steering(
    *,
    runtime: Runtime,
    model_db: torch.Tensor,
    steer_bank_latents: torch.Tensor,
    initial_latent: torch.Tensor,
    patch_rows: int,
    patch_cols: int,
    steer_strength: float,
    beta_schedule: str,
    steer_start_frac: float,
    steer_end_frac: float,
    steer_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    args = runtime.args
    model = runtime.model
    x = initial_latent.clone()
    ts = torch.linspace(float(args.t_eps), 1.0 - float(args.t_eps), int(args.sample_steps) + 1, device=x.device)
    history: list[dict[str, float]] = []

    for step_idx in range(int(args.sample_steps)):
        t_curr = ts[step_idx]
        t_next = ts[step_idx + 1]
        t = t_curr.expand(x.shape[0])
        out = model(x, t, model_db)
        mu_step = out[0] if isinstance(out, (tuple, list)) else out
        dt = t_next - t_curr
        denom = (1.0 - t_curr).clamp(min=float(args.t_eps))
        v_model = (mu_step - x) / denom
        v_total = v_model

        if _in_steering_window(step_idx, int(args.sample_steps), steer_start_frac, steer_end_frac):
            beta_t = _schedule_beta(steer_strength, beta_schedule, float(t_curr.item()))
            v_star, stats = _patchwise_retrieval_update(
                current_latents=x,
                bank_latents=steer_bank_latents,
                t=float(t_curr.item()),
                patch_rows=patch_rows,
                patch_cols=patch_cols,
                topk=int(steer_topk),
            )
            v_total = v_total + beta_t * v_star
            stats["step"] = float(step_idx)
            stats["t"] = float(t_curr.item())
            stats["dt"] = float(dt.item())
            stats["beta_t"] = float(beta_t)
            history.append(stats)

        x = x + v_total * dt

    imgs = decode_latents(runtime.vae, x, decode_batch=int(args.decode_batch))
    return imgs, x, history


def _save_single_image(imgs: torch.Tensor, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    save_image(imgs[0].detach().cpu(), path)


def _save_bank_preview(bank_latents: torch.Tensor, vae, decode_batch: int, out_path: str, limit: int = 16) -> None:
    preview = bank_latents[: min(limit, int(bank_latents.shape[0]))]
    if preview.numel() == 0:
        return
    imgs = decode_latents(vae, preview, decode_batch=decode_batch)
    grid = make_grid(imgs.detach().cpu(), nrow=int(math.sqrt(max(1, imgs.shape[0]))))
    save_image(grid, out_path)


def main() -> int:
    ns = parse_args()
    _setup_logging(ns.output_root)
    PartialState()
    set_seed(int(ns.seed))
    device = torch.device(ns.device)
    LOGGER.info("device=%s", device)
    LOGGER.info("seed=%d", int(ns.seed))

    baseline_runtime = _load_runtime(
        name="baseline",
        config_path=ns.config,
        ckpt=ns.baseline_ckpt,
        sample_steps=ns.sample_steps,
        decode_batch=ns.decode_batch,
        device=device,
        use_ema=bool(ns.use_ema),
    )
    retrieval_runtime = _load_runtime(
        name="retrieval",
        config_path=ns.config,
        ckpt=ns.retrieval_ckpt,
        sample_steps=ns.sample_steps,
        decode_batch=ns.decode_batch,
        device=device,
        use_ema=bool(ns.use_ema),
    )

    if (
        baseline_runtime.args.latent_c != retrieval_runtime.args.latent_c
        or baseline_runtime.args.latent_h != retrieval_runtime.args.latent_h
        or baseline_runtime.args.latent_w != retrieval_runtime.args.latent_w
    ):
        raise ValueError("Baseline and retrieval runtimes do not share the same latent shape.")

    patch_rows = int(ns.patch_rows or (baseline_runtime.args.latent_h // baseline_runtime.args.cross_patch_size))
    patch_cols = int(ns.patch_cols or (baseline_runtime.args.latent_w // baseline_runtime.args.cross_patch_size))

    cat_major = int(round(0.75 * int(ns.cats)))
    cat_minor = int(ns.cats) - cat_major
    dog_major = int(round(0.75 * int(ns.dogs)))
    dog_minor = int(ns.dogs) - dog_major
    mixed_cats = cat_major + dog_minor
    mixed_dogs = dog_major + cat_minor

    cat_spec = f"cat={cat_major},dog={cat_minor}"
    dog_spec = f"dog={dog_major},cat={dog_minor}"
    mixed_spec = f"dog={mixed_dogs},cat={mixed_cats}"
    cat_bank, cat_indices = _build_db(retrieval_runtime, cat_spec, device, output_root=ns.output_root)
    dog_bank, dog_indices = _build_db(retrieval_runtime, dog_spec, device, output_root=ns.output_root)
    mixed_bank, mixed_indices = _build_db(retrieval_runtime, mixed_spec, device, output_root=ns.output_root)
    LOGGER.info("cat bank shape=%s spec=%s", tuple(cat_bank.shape), cat_spec)
    LOGGER.info("dog bank shape=%s spec=%s", tuple(dog_bank.shape), dog_spec)
    LOGGER.info("mixed bank shape=%s spec=%s", tuple(mixed_bank.shape), mixed_spec)

    initial_latent = _make_initial_latent(baseline_runtime.args, device=device, seed=int(ns.seed))
    baseline_db = torch.zeros_like(cat_bank[:1])

    baseline_img, baseline_latent = _sample_from_initial_latent(
        model=baseline_runtime.model,
        Xdb=baseline_db,
        vae=baseline_runtime.vae,
        initial_latent=initial_latent,
        steps=int(baseline_runtime.args.sample_steps),
        t_eps=float(baseline_runtime.args.t_eps),
        decode_batch=int(baseline_runtime.args.decode_batch),
    )
    retrieval_cat_img, retrieval_cat_latent = _sample_from_initial_latent(
        model=retrieval_runtime.model,
        Xdb=cat_bank,
        vae=retrieval_runtime.vae,
        initial_latent=initial_latent,
        steps=int(retrieval_runtime.args.sample_steps),
        t_eps=float(retrieval_runtime.args.t_eps),
        decode_batch=int(retrieval_runtime.args.decode_batch),
    )
    retrieval_dog_img, retrieval_dog_latent = _sample_from_initial_latent(
        model=retrieval_runtime.model,
        Xdb=dog_bank,
        vae=retrieval_runtime.vae,
        initial_latent=initial_latent,
        steps=int(retrieval_runtime.args.sample_steps),
        t_eps=float(retrieval_runtime.args.t_eps),
        decode_batch=int(retrieval_runtime.args.decode_batch),
    )
    retrieval_mixed_img, retrieval_mixed_latent = _sample_from_initial_latent(
        model=retrieval_runtime.model,
        Xdb=mixed_bank,
        vae=retrieval_runtime.vae,
        initial_latent=initial_latent,
        steps=int(retrieval_runtime.args.sample_steps),
        t_eps=float(retrieval_runtime.args.t_eps),
        decode_batch=int(retrieval_runtime.args.decode_batch),
    )
    steer_cat_img, steer_cat_latent, steer_cat_history = _sample_baseline_with_retrieval_steering(
        baseline_runtime=baseline_runtime,
        bank_latents=cat_bank,
        initial_latent=initial_latent,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        steer_strength=float(ns.steer_strength),
        beta_schedule=str(ns.beta_schedule),
        steer_start_frac=float(ns.steer_start_frac),
        steer_end_frac=float(ns.steer_end_frac),
        steer_topk=int(ns.steer_topk),
    )
    steer_dog_img, steer_dog_latent, steer_dog_history = _sample_baseline_with_retrieval_steering(
        baseline_runtime=baseline_runtime,
        bank_latents=dog_bank,
        initial_latent=initial_latent,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        steer_strength=float(ns.steer_strength),
        beta_schedule=str(ns.beta_schedule),
        steer_start_frac=float(ns.steer_start_frac),
        steer_end_frac=float(ns.steer_end_frac),
        steer_topk=int(ns.steer_topk),
    )
    mixed_plus_steer_dog_img, mixed_plus_steer_dog_latent, mixed_plus_steer_dog_history = (
        _sample_model_with_retrieval_steering(
            runtime=retrieval_runtime,
            model_db=mixed_bank,
            steer_bank_latents=dog_bank,
            initial_latent=initial_latent,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            steer_strength=float(ns.steer_strength),
            beta_schedule=str(ns.beta_schedule),
            steer_start_frac=float(ns.steer_start_frac),
            steer_end_frac=float(ns.steer_end_frac),
            steer_topk=int(ns.steer_topk),
        )
    )
    mixed_plus_steer_cat_img, mixed_plus_steer_cat_latent, mixed_plus_steer_cat_history = (
        _sample_model_with_retrieval_steering(
            runtime=retrieval_runtime,
            model_db=mixed_bank,
            steer_bank_latents=cat_bank,
            initial_latent=initial_latent,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            steer_strength=float(ns.steer_strength),
            beta_schedule=str(ns.beta_schedule),
            steer_start_frac=float(ns.steer_start_frac),
            steer_end_frac=float(ns.steer_end_frac),
            steer_topk=int(ns.steer_topk),
        )
    )

    cat_tag = f"{cat_major}cats_{cat_minor}dogs"
    dog_tag = f"{dog_major}dogs_{dog_minor}cats"
    mixed_tag = f"{mixed_dogs}dogs_{mixed_cats}cats"
    cases = [
        ("baseline", "01_baseline_seed.png", baseline_img),
        (f"retrieval_{cat_tag}", f"02_retrieval_{cat_tag}_seed.png", retrieval_cat_img),
        (f"retrieval_{dog_tag}", f"03_retrieval_{dog_tag}_seed.png", retrieval_dog_img),
        (f"baseline_plus_steer_{cat_tag}", f"04_baseline_plus_steer_{cat_tag}_seed.png", steer_cat_img),
        (f"baseline_plus_steer_{dog_tag}", f"05_baseline_plus_steer_{dog_tag}_seed.png", steer_dog_img),
        (f"retrieval_{mixed_tag}", f"06_retrieval_{mixed_tag}_seed.png", retrieval_mixed_img),
        (
            f"retrieval_{mixed_tag}_plus_steer_{dog_tag}",
            f"07_retrieval_{mixed_tag}_plus_steer_{dog_tag}_seed.png",
            mixed_plus_steer_dog_img,
        ),
        (
            f"retrieval_{mixed_tag}_plus_steer_{cat_tag}",
            f"08_retrieval_{mixed_tag}_plus_steer_{cat_tag}_seed.png",
            mixed_plus_steer_cat_img,
        ),
    ]
    for _, filename, imgs in cases:
        _save_single_image(imgs, os.path.join(ns.output_root, filename))

    _save_bank_preview(
        cat_bank,
        retrieval_runtime.vae,
        decode_batch=int(retrieval_runtime.args.decode_batch),
        out_path=os.path.join(ns.output_root, "cat_bank_preview.png"),
    )
    _save_bank_preview(
        dog_bank,
        retrieval_runtime.vae,
        decode_batch=int(retrieval_runtime.args.decode_batch),
        out_path=os.path.join(ns.output_root, "dog_bank_preview.png"),
    )
    _save_bank_preview(
        mixed_bank,
        retrieval_runtime.vae,
        decode_batch=int(retrieval_runtime.args.decode_batch),
        out_path=os.path.join(ns.output_root, "mixed_bank_preview.png"),
    )

    outputs = {key: filename for key, filename, _ in cases}

    summary = {
        "seed": int(ns.seed),
        "device": str(device),
        "baseline_ckpt": baseline_runtime.ckpt_path,
        "retrieval_ckpt": retrieval_runtime.ckpt_path,
        "sample_steps": int(baseline_runtime.args.sample_steps),
        "t_eps": float(baseline_runtime.args.t_eps),
        "noise_scale": float(baseline_runtime.args.noise_scale),
        "cat_db_spec": cat_spec,
        "dog_db_spec": dog_spec,
        "mixed_db_spec": mixed_spec,
        "cat_db_major_count": cat_major,
        "cat_db_minor_dog_count": cat_minor,
        "dog_db_major_count": dog_major,
        "dog_db_minor_cat_count": dog_minor,
        "cat_db_size": int(cat_bank.shape[0]),
        "dog_db_size": int(dog_bank.shape[0]),
        "mixed_db_size": int(mixed_bank.shape[0]),
        "cat_db_indices": cat_indices,
        "dog_db_indices": dog_indices,
        "mixed_db_indices": mixed_indices,
        "patch_rows": patch_rows,
        "patch_cols": patch_cols,
        "steer_strength": float(ns.steer_strength),
        "beta_schedule": str(ns.beta_schedule),
        "steer_start_frac": float(ns.steer_start_frac),
        "steer_end_frac": float(ns.steer_end_frac),
        "steer_topk": int(ns.steer_topk),
        "outputs": outputs,
        "latent_norms": {
            "baseline": float(baseline_latent.float().reshape(1, -1).norm(dim=-1).item()),
            f"retrieval_{cat_tag}": float(retrieval_cat_latent.float().reshape(1, -1).norm(dim=-1).item()),
            f"retrieval_{dog_tag}": float(retrieval_dog_latent.float().reshape(1, -1).norm(dim=-1).item()),
            f"baseline_plus_steer_{cat_tag}": float(steer_cat_latent.float().reshape(1, -1).norm(dim=-1).item()),
            f"baseline_plus_steer_{dog_tag}": float(steer_dog_latent.float().reshape(1, -1).norm(dim=-1).item()),
            f"retrieval_{mixed_tag}": float(retrieval_mixed_latent.float().reshape(1, -1).norm(dim=-1).item()),
            f"retrieval_{mixed_tag}_plus_steer_{dog_tag}": float(
                mixed_plus_steer_dog_latent.float().reshape(1, -1).norm(dim=-1).item()
            ),
            f"retrieval_{mixed_tag}_plus_steer_{cat_tag}": float(
                mixed_plus_steer_cat_latent.float().reshape(1, -1).norm(dim=-1).item()
            ),
        },
    }
    with open(os.path.join(ns.output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(os.path.join(ns.output_root, "steer_cat_history.json"), "w", encoding="utf-8") as f:
        json.dump(steer_cat_history, f, indent=2)
    with open(os.path.join(ns.output_root, "steer_dog_history.json"), "w", encoding="utf-8") as f:
        json.dump(steer_dog_history, f, indent=2)
    with open(os.path.join(ns.output_root, "mixed_plus_steer_dog_history.json"), "w", encoding="utf-8") as f:
        json.dump(mixed_plus_steer_dog_history, f, indent=2)
    with open(os.path.join(ns.output_root, "mixed_plus_steer_cat_history.json"), "w", encoding="utf-8") as f:
        json.dump(mixed_plus_steer_cat_history, f, indent=2)

    LOGGER.info("saved outputs under %s", Path(ns.output_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
