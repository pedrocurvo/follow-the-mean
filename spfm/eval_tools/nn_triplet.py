#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import yaml
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
from utils.train_helpers import (
    decode_latents,
    ensure_dir,
    get_vae_scaling,
    infer_vae_latent_spec,
    nearest_neighbors,
    vae_tag_from_name,
)


LOGGER = logging.getLogger("full_attention_nn_triplet_eval")
NN_SEARCH_CHUNK = 1024


def _parse_bool(value):
    return eval_checkpoint._parse_bool(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate 8 images and latent nearest neighbours for three DB variants.")
    ap.add_argument("--config", default="experiments/model.yaml")
    ap.add_argument("--ckpt", default="out/model_spf_fullattention_afhq_cat_dog/model_step10000.pt")
    ap.add_argument("--results_dir", default="out/evals/model_spf_fullattention_afhq_cat_dog_step10000/nn_triplet")
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--sample_steps", type=int, default=None)
    ap.add_argument("--decode_batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use_ema", type=_parse_bool, default=False)
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


def _load_args(config_path: str, ckpt: str | None, sample_steps: int | None, decode_batch: int | None):
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


def _build_db_variant(base_args, *, db_spec: str, vae, device: torch.device):
    variant_args = argparse.Namespace(**vars(base_args))
    variant_args.db = db_spec
    variant_args.alt_db = "none"
    variant_args.self_mask_db = False

    class _Accel:
        is_main_process = True

        @staticmethod
        def wait_for_everyone() -> None:
            return None

    primary = build_primary_db(variant_args, vae=vae, device=device, accelerator=_Accel())
    return primary.Xdb


def _save_image_set(imgs: torch.Tensor, out_dir: str, prefix: str) -> list[str]:
    ensure_dir(out_dir)
    paths = []
    for idx in range(imgs.shape[0]):
        path = os.path.join(out_dir, f"{prefix}_{idx:02d}.png")
        save_image(imgs[idx].detach().cpu(), path)
        paths.append(path)
    return paths


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


def _make_initial_latents(args, *, device: torch.device, base_seed: int) -> tuple[torch.Tensor, list[int]]:
    seeds = [int(base_seed) + i for i in range(int(args.num_gen))]
    latents = []
    for seed in seeds:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        latents.append(
            float(args.noise_scale) * torch.randn(
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
    paired_grid = make_grid(torch.cat([imgs.detach().cpu(), nn_imgs.detach().cpu()], dim=0), nrow=int(args.num_gen))
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
    with open(os.path.join(variant_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


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
    ckpt_path = _resolve_ckpt_path(args.ckpt, str(Path(args.out_dir).expanduser().resolve()))
    eval_checkpoint._load_model_weights(model, ckpt_path, use_ema=bool(ns.use_ema))
    raw_model = train._unwrap_model_for_runtime(model)
    raw_model.eval()
    initial_latents, sample_seeds = _make_initial_latents(args, device=device, base_seed=int(ns.seed))
    torch.save(
        {"sample_seeds": sample_seeds, "initial_latents": initial_latents.detach().cpu()},
        os.path.join(results_dir, "initial_latents.pt"),
    )

    variants = [
        ("full_db", "dog:1.0,cat:1.0"),
        ("cats_only", "cat:1.0,dog:0.0"),
        ("dogs_only", "cat:0.0,dog:1.0"),
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

    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "config": str(Path(ns.config).expanduser().resolve()),
                "num_gen": int(ns.num_gen),
                "sample_seeds": sample_seeds,
                "sample_steps": int(args.sample_steps),
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
