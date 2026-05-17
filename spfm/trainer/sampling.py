from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from utils.train_helpers import decode_latents, ensure_dir, nearest_neighbors, sample_closedform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NN_SEARCH_CHUNK = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_nrow(num_gen: int) -> int:
    return max(1, int(math.sqrt(max(1, int(num_gen)))))


# ---------------------------------------------------------------------------
# Artifact Containers
# ---------------------------------------------------------------------------


@dataclass
class QuickSampleArtifacts:
    main_grid: torch.Tensor
    main_nn_grid: torch.Tensor
    main_grid_path: str
    main_nn_grid_path: str
    alt_grid: torch.Tensor | None = None
    alt_nn_grid: torch.Tensor | None = None
    alt_grid_path: str | None = None
    alt_nn_grid_path: str | None = None


# ---------------------------------------------------------------------------
# Quick Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_quick_sample_nn(
    *,
    raw_model: torch.nn.Module,
    step_now: int,
    args,
    vae,
    save_dir: str,
    Xdb: torch.Tensor,
    alt_db: torch.Tensor | None,
    alt_db_label: str | None,
    logger,
) -> QuickSampleArtifacts:
    imgs, x = sample_closedform(
        Xdb=Xdb,
        model=raw_model,
        vae=vae,
        out_dir=save_dir,
        num_gen=args.num_gen,
        steps=args.sample_steps,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        decode_batch=args.decode_batch,
        latent_c=args.latent_c,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        return_latents=True,
    )
    grid_path = os.path.join(save_dir, f"train_grid_step{step_now}.png")
    grid = make_grid(imgs, nrow=_grid_nrow(args.num_gen))
    save_image(grid, grid_path)
    logger.info("[train] saved %s", grid_path)
    nn_idx, nn_dist = nearest_neighbors(x, Xdb, chunk=NN_SEARCH_CHUNK)
    nn_latents = Xdb[nn_idx]
    nn_imgs = decode_latents(vae, nn_latents, decode_batch=args.decode_batch)
    nn_grid_path = os.path.join(save_dir, f"train_grid_step{step_now}_nn.png")
    nn_grid = make_grid(nn_imgs, nrow=_grid_nrow(args.num_gen))
    save_image(nn_grid, nn_grid_path)
    logger.info("[train] saved %s", nn_grid_path)
    logger.info("[train] nn mean l2^2 %.6f", float(nn_dist.mean().item()))
    out = QuickSampleArtifacts(
        main_grid=grid,
        main_nn_grid=nn_grid,
        main_grid_path=grid_path,
        main_nn_grid_path=nn_grid_path,
    )

    if alt_db is None:
        return out

    alt_out_dir = os.path.join(save_dir, f"alt_db_{alt_db_label}")
    ensure_dir(alt_out_dir)
    alt_imgs, alt_x = sample_closedform(
        Xdb=alt_db,
        model=raw_model,
        vae=vae,
        out_dir=alt_out_dir,
        num_gen=args.num_gen,
        steps=args.sample_steps,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        decode_batch=args.decode_batch,
        latent_c=args.latent_c,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        return_latents=True,
    )
    alt_grid_path = os.path.join(alt_out_dir, f"train_grid_step{step_now}.png")
    alt_grid = make_grid(alt_imgs, nrow=_grid_nrow(args.num_gen))
    save_image(alt_grid, alt_grid_path)
    logger.info("[train] saved %s", alt_grid_path)
    alt_nn_idx, alt_nn_dist = nearest_neighbors(alt_x, alt_db, chunk=NN_SEARCH_CHUNK)
    alt_nn_latents = alt_db[alt_nn_idx]
    alt_nn_imgs = decode_latents(vae, alt_nn_latents, decode_batch=args.decode_batch)
    alt_nn_grid_path = os.path.join(alt_out_dir, f"train_grid_step{step_now}_nn.png")
    alt_nn_grid = make_grid(alt_nn_imgs, nrow=_grid_nrow(args.num_gen))
    save_image(alt_nn_grid, alt_nn_grid_path)
    logger.info("[train] saved %s", alt_nn_grid_path)
    logger.info("[train] alt nn mean l2^2 %.6f", float(alt_nn_dist.mean().item()))
    out.alt_grid = alt_grid
    out.alt_nn_grid = alt_nn_grid
    out.alt_grid_path = alt_grid_path
    out.alt_nn_grid_path = alt_nn_grid_path
    return out


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_quick_sample(
    *,
    accelerator: Accelerator,
    step_now: int,
    sample_artifacts: QuickSampleArtifacts,
    report_to: str | None,
    alt_db_label: str | None,
    wandb_module: Any | None,
) -> None:
    if not report_to:
        return
    if wandb_module is not None and "wandb" in str(report_to):
        log_payload = {
            "train/sample_grid": wandb_module.Image(sample_artifacts.main_grid.clamp(0, 1).cpu()),
            "train/sample_nn_grid": wandb_module.Image(
                sample_artifacts.main_nn_grid.clamp(0, 1).cpu()
            ),
            "train/step": step_now,
        }
        if sample_artifacts.alt_grid is not None and sample_artifacts.alt_nn_grid is not None:
            alt_tag = str(alt_db_label) if alt_db_label is not None else "alt"
            log_payload[f"train/alt_sample_grid_{alt_tag}"] = wandb_module.Image(
                sample_artifacts.alt_grid.clamp(0, 1).cpu()
            )
            log_payload[f"train/alt_sample_nn_grid_{alt_tag}"] = wandb_module.Image(
                sample_artifacts.alt_nn_grid.clamp(0, 1).cpu()
            )
        accelerator.log(log_payload, step=step_now)
        return
    log_payload = {
        "train/sample_grid_path": sample_artifacts.main_grid_path,
        "train/sample_nn_grid_path": sample_artifacts.main_nn_grid_path,
        "train/step": step_now,
    }
    if sample_artifacts.alt_grid_path is not None and sample_artifacts.alt_nn_grid_path is not None:
        alt_tag = str(alt_db_label) if alt_db_label is not None else "alt"
        log_payload[f"train/alt_sample_grid_path_{alt_tag}"] = sample_artifacts.alt_grid_path
        log_payload[f"train/alt_sample_nn_grid_path_{alt_tag}"] = sample_artifacts.alt_nn_grid_path
    accelerator.log(log_payload, step=step_now)
