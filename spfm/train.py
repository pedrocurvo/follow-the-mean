# train.py
#
# PairFlow-style closed-form generation in VAE (InVAE) LATENT SPACE
# with single-token retrieval in a flattened embedding space.
#
# Dataset points: images resized to --image_size, encoded to InVAE latents
# (C x H x W, with H=W=image_size/latent_downsample).
#
# Posterior mean (single-token attention):
#   q,k,v from flattened latents (single-token retrieval)
#   AdaLN time conditioning on query tokens
#   mu_theta assembled from attention over database tokens
#
# Training loss:
#   L = ||mu_theta(x_t,t) - x_data||^2
#
# Run:
#   python train.py --N_img 200 --train_steps 1000
#
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import warnings
from copy import deepcopy
from pathlib import Path

import torch
from cleanfid import fid as cleanfid_fid
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

try:
    import wandb
except Exception:
    wandb = None

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from preprocessing.encoders import load_invae
from trainer.checkpoint import (
    is_full_training_checkpoint,
    load_full_training_checkpoint,
    save_full_training_checkpoint,
)
from trainer.clip_eval import clip_eval as run_clip_eval
from trainer.clip_eval import load_clip_runtime, prompt_metric_suffix
from trainer.db import build_alt_db, build_primary_db, build_train_latent_loader
from trainer.loss import (
    compute_drifting_loss,
    compute_main_loss,
    compute_refiner_loss,
    drifting_penalty_schedule,
)
from trainer.model_factory import build_model
from trainer.optim import build_optimizer, compute_grad_norm
from trainer.sampling import log_quick_sample, run_quick_sample_nn
from utils.io import load_encoders
from utils.train_helpers import (
    DEFAULT_IMAGE_SIZE,
    RawVAE,
    create_logger,
    decode_latents,
    ensure_dir,
    generate_images_to_dir,
    get_vae_scaling,
    infer_vae_latent_spec,
    update_ema,
    vae_tag_from_name,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI Parsing Helpers
# ---------------------------------------------------------------------------


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_optional_n_img(value):
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "N_img must be > 0, or use 'none'/'all' for full dataset."
            )
        return value
    text = str(value).strip().lower()
    if text in {"none", "null", "all"}:
        return None
    try:
        n_img = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid N_img value: {value!r} (expected positive int or one of: none, null, all)"
        ) from exc
    if n_img <= 0:
        raise argparse.ArgumentTypeError("N_img must be > 0, or use 'none'/'all' for full dataset.")
    return n_img


# ---------------------------------------------------------------------------
# Runtime Helpers
# ---------------------------------------------------------------------------


def _unwrap_model_for_runtime(model: torch.nn.Module) -> torch.nn.Module:
    raw = model
    while hasattr(raw, "module"):
        raw = getattr(raw, "module")
    # Handle torch.compile wrappers without relying on Accelerate internals.
    while isinstance(getattr(raw, "__dict__", None), dict) and "_orig_mod" in raw.__dict__:
        raw = raw.__dict__["_orig_mod"]
    return raw


@torch.no_grad()
def _write_latents_to_dir(
    vae,
    latents: torch.Tensor,
    out_dir: str,
    decode_batch: int,
    prefix: str = "img_",
):
    ensure_dir(out_dir)
    count = 0
    for s in range(0, latents.shape[0], decode_batch):
        imgs = decode_latents(vae, latents[s : s + decode_batch], decode_batch=decode_batch)
        for i in range(imgs.shape[0]):
            save_image(imgs[i], os.path.join(out_dir, f"{prefix}{count:06d}.png"))
            count += 1


def _wait_for_path(path: str, timeout_s: float = 7200.0, poll_s: float = 2.0) -> None:
    deadline = time.time() + float(timeout_s)
    while not os.path.exists(path):
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_s)


def _transient_root(args) -> str:
    root = args.transient_root
    if root is None:
        root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    root = os.path.expanduser(str(root))
    path = os.path.join(root, "scalable-fm", Path(args.out_dir).name)
    ensure_dir(path)
    return path


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_loop(
    args,
    Xdb: torch.Tensor,
    db_indices: torch.Tensor | None,
    db_group_ids: torch.Tensor | None,
    alt_db: torch.Tensor | None,
    alt_db_group_ids: torch.Tensor | None,
    alt_db_label: str | None,
    loader: DataLoader,
    vae,
    device: torch.device,
    accelerator: Accelerator,
    save_dir: str,
):
    model = build_model(args, device)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("[train] model params: %d", num_params)
        logger.info("[train] trainable params: %d", num_trainable)
        logger.info("[train] model arch:\n%s", model)
        if wandb is not None and "wandb" in str(args.report_to):
            accelerator.log(
                {
                    "model/params_total": num_params,
                    "model/params_trainable": num_trainable,
                },
                step=0,
            )
            if wandb.run is not None:
                wandb.run.summary["model/arch"] = str(model)

    opt = build_optimizer(
        model=model,
        args=args,
        logger=logger,
        is_main_process=accelerator.is_main_process,
    )

    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    accelerator.register_for_checkpointing(ema)

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "torch.compile is not available in this PyTorch build; refiner compilation is required."
        )
    try:
        import torch._dynamo as dynamo  # type: ignore[attr-defined]

        dynamo.config.capture_scalar_outputs = True
    except Exception:
        pass
    refiner = getattr(model, "refiner", None)
    if isinstance(refiner, torch.nn.Module):
        model.refiner = torch.compile(refiner)
        if accelerator.is_main_process:
            logger.info(
                "[train] torch.compile enabled for refiner only (capture_scalar_outputs=True)"
            )
    elif accelerator.is_main_process:
        logger.warning("[train] model has no refiner module; skipping refiner compile.")

    model, opt, loader = accelerator.prepare(model, opt, loader)

    start_step = 0
    if args.ckpt is not None and os.path.exists(args.ckpt):
        if is_full_training_checkpoint(args.ckpt):
            start_step = load_full_training_checkpoint(accelerator, args.ckpt)
            if accelerator.is_main_process:
                logger.info(
                    "[train] resumed full training state from %s at step %d", args.ckpt, start_step
                )
        else:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            _unwrap_model_for_runtime(model).load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            ema.load_state_dict(ckpt["ema"])
            start_step = int(ckpt.get("step", 0))
            if accelerator.is_main_process:
                logger.info(
                    "[train] resumed weights/optimizer from %s at step %d", args.ckpt, start_step
                )

    model.train()
    step = start_step
    t0 = time.time()

    it = iter(loader)
    pbar = tqdm(
        total=args.train_steps,
        initial=step,
        desc="train",
        disable=not accelerator.is_local_main_process,
    )
    clip_runtime = None
    clip_prompts = args.clip_eval_prompts or ["a photo of a dog", "a photo of a cat"]
    clip_prompts = [p.strip() for p in clip_prompts if p and p.strip()]
    if not clip_prompts:
        clip_prompts = ["a photo of a dog", "a photo of a cat"]

    try:
        while step < args.train_steps:
            for _ in range(args.gradient_accumulation_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)

                if isinstance(batch, (tuple, list)):
                    row_idx = batch[0]
                    batch_idx = batch[1] if len(batch) > 1 else None
                else:
                    row_idx = batch
                    batch_idx = None
                row_idx = row_idx.to(device=device, dtype=torch.long)
                if Xdb.device.type == "cpu":
                    z1 = Xdb.index_select(0, row_idx.to(device="cpu", dtype=torch.long))
                    z1 = z1.to(device=device, non_blocking=True)
                else:
                    z1 = Xdb[row_idx]
                db_mask = None
                if args.self_mask_db and db_indices is not None and batch_idx is not None:
                    batch_idx = batch_idx.to(device=device, dtype=torch.long)
                    db_mask = db_indices[None, :] == batch_idx[:, None]

                B = z1.shape[0]
                # Training loop
                t = torch.rand(B, device=device) * (1.0 - 2.0 * args.t_eps) + args.t_eps
                x_data = z1
                x_noise = args.noise_scale * torch.randn_like(z1)
                x_t = (1.0 - t[:, None, None, None]) * x_noise + t[:, None, None, None] * x_data
                Xdb_step = Xdb
                db_mask_step = db_mask
                if db_mask_step is not None and db_mask_step.all(dim=1).any():
                    db_mask_step = db_mask_step.clone()
                    all_masked = db_mask_step.all(dim=1)
                    db_mask_step[all_masked] = False

                model_kwargs = dict(
                    db_mask=db_mask_step,
                    return_delta=True,
                    return_mu_ret=True,
                )
                mu, delta, mu_ret = model(
                    x_t,
                    t,
                    Xdb_step,
                    **model_kwargs,
                )
                main_loss = compute_main_loss(
                    mu=mu, x_data=x_data, t=t, loss_weight=args.loss_weight
                )

                # One-time diagnostic: measure pairwise distances for drifting_tau calibration
                if step == 0 and accelerator.is_main_process and mu_ret is not None:
                    with torch.no_grad():
                        _z = mu_ret.reshape(mu_ret.shape[0], -1).float()
                        _dists = torch.cdist(_z, _z)
                        _mask = ~torch.eye(_z.shape[0], dtype=torch.bool, device=_z.device)
                        _mean_d = float(_dists[_mask].mean().item())
                        _std_d = float(_dists[_mask].std().item())
                        logger.info(
                            "[drift-diag] mu_ret pairwise dist: mean=%.4f std=%.4f "
                            "(suggested drifting_tau=%.4f..%.4f)",
                            _mean_d,
                            _std_d,
                            0.3 * _mean_d,
                            0.5 * _mean_d,
                        )

                refiner_loss = None
                total_loss = main_loss
                if args.refiner_penalty > 0 and delta is not None:
                    # Reconstruct same base proposal used in model forward.
                    raw_model = _unwrap_model_for_runtime(model)
                    g = raw_model.compute_g(t, mu.dtype)
                    alpha = raw_model.compute_alpha(t, mu.dtype)
                    # Match model's residual scaling schedule.
                    refiner_loss = compute_refiner_loss(
                        delta=delta,
                        x_data=x_data,
                        x_t=x_t,
                        mu_ret=mu_ret,
                        alpha=alpha,
                        g=g,
                    )
                    total_loss = total_loss + args.refiner_penalty * refiner_loss

                drifting_loss = None
                drift_v_mag = None
                if args.drifting_penalty > 0 and mu_ret is not None:
                    drift_w = drifting_penalty_schedule(
                        step,
                        args.drifting_penalty,
                        args.drifting_warmup,
                    )
                    if drift_w > 0:
                        drifting_loss, drift_v_mag = compute_drifting_loss(
                            mu=mu,
                            mu_ret=mu_ret,
                            tau=args.drifting_tau,
                        )
                        total_loss = total_loss + drift_w * drifting_loss

                if not torch.isfinite(total_loss):
                    opt.zero_grad(set_to_none=True)
                    if accelerator.is_main_process:
                        logger.warning(
                            "[train] non-finite loss at step=%d (main=%s total=%s); skipping update",
                            step,
                            float(main_loss.detach().cpu()),
                            float(total_loss.detach().cpu()),
                        )
                    continue
                accelerator.backward(total_loss)

            if args.max_grad_norm > 0:
                grad_norm = float(
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()
                )
            else:
                grad_norm = compute_grad_norm(model.parameters())
            if not math.isfinite(grad_norm):
                opt.zero_grad(set_to_none=True)
                if accelerator.is_main_process:
                    logger.warning(
                        "[train] non-finite grad norm at step=%d (main=%s total=%s); skipping optimizer step",
                        step,
                        float(main_loss.detach().cpu()),
                        float(total_loss.detach().cpu()),
                    )
                continue

            opt.step()
            opt.zero_grad(set_to_none=True)

            update_ema(ema, _unwrap_model_for_runtime(model), decay=args.ema_decay)

            step += 1
            pbar.update(1)
            did_quick_sample_this_step = False

            if accelerator.is_main_process and step % args.log_every == 0:
                dt = max(1e-6, time.time() - t0)
                it_s = args.log_every / dt
                msg = (
                    f"[train] step {step}/{args.train_steps} "
                    f"main={main_loss.item():.6f} total={total_loss.item():.6f} "
                    f"grad_norm={grad_norm:.4f} it/s={it_s:.2f}"
                )
                if refiner_loss is not None:
                    msg += f" refiner={refiner_loss.item():.6f}"
                if drifting_loss is not None:
                    msg += f" drift={drifting_loss.item():.6f}"
                    if drift_v_mag is not None:
                        msg += f" |V|={drift_v_mag:.4f}"
                logger.info(msg)
                pbar.set_postfix(
                    main=f"{main_loss.item():.4f}",
                    total=f"{total_loss.item():.4f}",
                    gnorm=f"{grad_norm:.3f}",
                    it_s=f"{it_s:.2f}",
                )
                if args.report_to:
                    log_data = {
                        "train/main_loss": main_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/grad_norm": grad_norm,
                        "train/step": step,
                    }
                    if refiner_loss is not None:
                        log_data["train/refiner_loss"] = refiner_loss.item()
                    if drifting_loss is not None:
                        log_data["train/drifting_loss"] = drifting_loss.item()
                        if drift_v_mag is not None:
                            log_data["train/drift_v_magnitude"] = drift_v_mag
                        log_data["train/drift_penalty_eff"] = drift_w
                    accelerator.log(log_data, step=step)
                t0 = time.time()

            if args.save_every > 0 and step % args.save_every == 0:
                full_ckpt_dir = os.path.join(save_dir, f"checkpoint_step{step}")
                ensure_dir(full_ckpt_dir)
                save_full_training_checkpoint(accelerator, full_ckpt_dir, step)
                if accelerator.is_main_process:
                    logger.info("[train] saved full state %s", full_ckpt_dir)
                    save_path = os.path.join(save_dir, f"model_step{step}.pt")
                    torch.save(
                        {
                            "model": _unwrap_model_for_runtime(model).state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "step": step,
                        },
                        save_path,
                    )
                    logger.info("[train] saved %s", save_path)

                raw_model = _unwrap_model_for_runtime(model)
                was_training = raw_model.training
                raw_model.eval()
                eval_metrics = {}
                eval_root = os.path.join(_transient_root(args), f"eval_step{step}")
                eval_done = os.path.join(eval_root, "_done")
                if accelerator.is_main_process and os.path.exists(eval_root):
                    shutil.rmtree(eval_root, ignore_errors=True)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ensure_dir(eval_root)
                accelerator.wait_for_everyone()
                db_real_dir = os.path.join(eval_root, "db_real")
                db_gen_dir = os.path.join(eval_root, "db_gen")
                alt_real_dir = None
                alt_gen_dir = None
                try:
                    if accelerator.is_main_process:
                        _write_latents_to_dir(
                            vae=vae,
                            latents=Xdb,
                            out_dir=db_real_dir,
                            decode_batch=args.decode_batch,
                        )
                    generate_images_to_dir(
                        Xdb=Xdb,
                        model=raw_model,
                        vae=vae,
                        out_dir=db_gen_dir,
                        total_gen=int(Xdb.shape[0]),
                        gen_batch=args.batch_size,
                        steps=args.sample_steps,
                        noise_scale=args.noise_scale,
                        t_eps=args.t_eps,
                        decode_batch=args.decode_batch,
                        prefix="img_",
                        latent_c=args.latent_c,
                        latent_h=args.latent_h,
                        latent_w=args.latent_w,
                        db_group_ids=db_group_ids,
                        process_index=accelerator.process_index,
                        num_processes=accelerator.num_processes,
                    )
                    if alt_db is not None:
                        alt_real_dir = os.path.join(eval_root, "alt_real")
                        alt_gen_dir = os.path.join(eval_root, "alt_gen")
                        if accelerator.is_main_process:
                            _write_latents_to_dir(
                                vae=vae,
                                latents=alt_db,
                                out_dir=alt_real_dir,
                                decode_batch=args.decode_batch,
                            )
                        generate_images_to_dir(
                            Xdb=alt_db,
                            model=raw_model,
                            vae=vae,
                            out_dir=alt_gen_dir,
                            total_gen=int(alt_db.shape[0]),
                            gen_batch=args.batch_size,
                            steps=args.sample_steps,
                            noise_scale=args.noise_scale,
                            t_eps=args.t_eps,
                            decode_batch=args.decode_batch,
                            prefix="img_",
                            latent_c=args.latent_c,
                            latent_h=args.latent_h,
                            latent_w=args.latent_w,
                            db_group_ids=alt_db_group_ids,
                            process_index=accelerator.process_index,
                            num_processes=accelerator.num_processes,
                        )

                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        if args.fid:
                            eval_metrics["eval/db_fid"] = float(
                                cleanfid_fid.compute_fid(db_real_dir, db_gen_dir)
                            )
                            if alt_real_dir is not None and alt_gen_dir is not None:
                                eval_metrics["eval/alt_fid"] = float(
                                    cleanfid_fid.compute_fid(alt_real_dir, alt_gen_dir)
                                )
                        if args.kid:
                            eval_metrics["eval/db_kid"] = float(
                                cleanfid_fid.compute_kid(db_real_dir, db_gen_dir)
                            )
                            if alt_real_dir is not None and alt_gen_dir is not None:
                                eval_metrics["eval/alt_kid"] = float(
                                    cleanfid_fid.compute_kid(alt_real_dir, alt_gen_dir)
                                )

                        if clip_runtime is None:
                            try:
                                clip_runtime = load_clip_runtime(
                                    model_name=args.clip_model,
                                    device=device,
                                    local_files_only=args.nn_clip_local_files_only,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "[eval] CLIP unavailable, skipping prompt percentages (%s)", exc
                                )
                                clip_runtime = False
                        if clip_runtime and clip_runtime is not False:
                            db_clip_scores = run_clip_eval(
                                directory_with_images=db_gen_dir,
                                prompts=clip_prompts,
                                runtime=clip_runtime,
                                batch_size=args.clip_eval_batch_size or args.batch_size,
                            )
                            for prompt, pct in db_clip_scores.items():
                                suffix = prompt_metric_suffix(prompt)
                                eval_metrics[f"eval/db_gen_clip_pct_{suffix}"] = float(pct)
                            if alt_gen_dir is not None:
                                alt_clip_scores = run_clip_eval(
                                    directory_with_images=alt_gen_dir,
                                    prompts=clip_prompts,
                                    runtime=clip_runtime,
                                    batch_size=args.clip_eval_batch_size or args.batch_size,
                                )
                                for prompt, pct in alt_clip_scores.items():
                                    suffix = prompt_metric_suffix(prompt)
                                    eval_metrics[f"eval/alt_gen_clip_pct_{suffix}"] = float(pct)

                        sample_artifacts = run_quick_sample_nn(
                            raw_model=raw_model,
                            step_now=step,
                            args=args,
                            vae=vae,
                            save_dir=save_dir,
                            Xdb=Xdb,
                            alt_db=alt_db,
                            alt_db_label=alt_db_label,
                            logger=logger,
                        )
                        log_quick_sample(
                            accelerator=accelerator,
                            step_now=step,
                            sample_artifacts=sample_artifacts,
                            report_to=args.report_to,
                            alt_db_label=alt_db_label,
                            wandb_module=wandb,
                        )
                        did_quick_sample_this_step = True
                        if eval_metrics:
                            logger.info("[eval] step %d metrics: %s", step, eval_metrics)
                            if args.report_to:
                                accelerator.log(eval_metrics, step=step)
                        Path(eval_done).touch()
                    else:
                        _wait_for_path(eval_done)
                    accelerator.wait_for_everyone()
                finally:
                    if accelerator.is_main_process:
                        shutil.rmtree(eval_root, ignore_errors=True)
                    if was_training:
                        raw_model.train()

            if (
                args.sample_every > 0
                and step % args.sample_every == 0
                and not did_quick_sample_this_step
            ):
                sample_sync_dir = os.path.join(_transient_root(args), f"sample_step{step}")
                sample_done = os.path.join(sample_sync_dir, "_done")
                if accelerator.is_main_process and os.path.exists(sample_sync_dir):
                    shutil.rmtree(sample_sync_dir, ignore_errors=True)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ensure_dir(sample_sync_dir)
                raw_model = _unwrap_model_for_runtime(model)
                was_training = raw_model.training
                raw_model.eval()
                try:
                    if accelerator.is_main_process:
                        sample_artifacts = run_quick_sample_nn(
                            raw_model=raw_model,
                            step_now=step,
                            args=args,
                            vae=vae,
                            save_dir=save_dir,
                            Xdb=Xdb,
                            alt_db=alt_db,
                            alt_db_label=alt_db_label,
                            logger=logger,
                        )
                        log_quick_sample(
                            accelerator=accelerator,
                            step_now=step,
                            sample_artifacts=sample_artifacts,
                            report_to=args.report_to,
                            alt_db_label=alt_db_label,
                            wandb_module=wandb,
                        )
                        Path(sample_done).touch()
                    else:
                        _wait_for_path(sample_done)
                    accelerator.wait_for_everyone()
                finally:
                    if accelerator.is_main_process:
                        shutil.rmtree(sample_sync_dir, ignore_errors=True)
                    if was_training:
                        raw_model.train()
    finally:
        pbar.close()

    full_ckpt_dir = os.path.join(save_dir, "checkpoint_last")
    ensure_dir(full_ckpt_dir)
    save_full_training_checkpoint(accelerator, full_ckpt_dir, step)
    if accelerator.is_main_process:
        logger.info("[train] saved full state %s", full_ckpt_dir)

    if accelerator.is_main_process:
        save_path = os.path.join(save_dir, "model_last.pt")
        torch.save(
            {
                "model": _unwrap_model_for_runtime(model).state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
            },
            save_path,
        )
        logger.info("[train] saved %s", save_path)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    global logger
    if args.report_to is not None and str(args.report_to).lower() == "none":
        args.report_to = None

    ddp_find_unused = args.ddp_find_unused_parameters
    if ddp_find_unused is None:
        # Auto mode: spfm defines modules that may stay unused in forward,
        # which trips strict DDP bucket rebuild checks on multi-GPU runs.
        ddp_find_unused = args.model == "spfm"
    ddp_find_unused = bool(ddp_find_unused)

    logging_dir = Path(args.out_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.out_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)],
    )
    if accelerator.is_main_process:
        logger.info(
            "[dist] world_size=%d process_index=%d local_process_index=%d device=%s",
            accelerator.num_processes,
            accelerator.process_index,
            accelerator.local_process_index,
            accelerator.device,
        )
        logger.info("[dist] ddp_find_unused_parameters=%s", ddp_find_unused)

    save_dir = os.path.join(args.out_dir, args.exp_name) if args.exp_name else args.out_dir

    if accelerator.is_main_process:
        ensure_dir(save_dir)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, "w") as f:
            json.dump(args_dict, f, indent=4)
        logger = create_logger(save_dir)
        logger.info("Experiment directory created at %s", save_dir)
        if args.report_to:
            wandb_init = {
                "entity": args.wandb_entity,
                "name": args.wandb_name,
            }
            if args.wandb_run_id is not None:
                wandb_init["id"] = args.wandb_run_id
            if args.wandb_resume is not None:
                wandb_init["resume"] = args.wandb_resume
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=args_dict,
                init_kwargs={"wandb": wandb_init},
            )

    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    if args.enc_type is not None and str(args.enc_type).lower() != "none":
        if accelerator.is_main_process:
            _ = load_encoders(args.enc_type, device, args.image_size)
        accelerator.wait_for_everyone()
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.image_size
        )
        encoder_embed_dims = [enc.embed_dim for enc in encoders]
        if accelerator.is_main_process:
            logger.info("Encoders: %s", list(zip(encoder_types, architectures, encoder_embed_dims)))

    is_raw_vae = str(args.vae_name).strip().lower() == "raw"
    if is_raw_vae:
        vae = RawVAE().to(device)
        if accelerator.is_main_process:
            logger.info("[vae] using raw identity backend")
    else:
        if accelerator.is_main_process:
            _ = load_invae(args.vae_name, device=device)
        accelerator.wait_for_everyone()
        vae = load_invae(args.vae_name, device=device)
    vae.eval().requires_grad_(False)
    latent_c, latent_h, latent_w, latent_downsample = infer_vae_latent_spec(
        vae, args.image_size, device
    )
    args.latent_c = latent_c
    args.latent_h = latent_h
    args.latent_w = latent_w
    args.latent_downsample = latent_downsample
    latent_dim = latent_c * latent_h * latent_w
    expected_embed_dim = args.embed_dim
    model_requires_fixed_embed_dim = args.model != "dit"
    cross_patchwise_effective = bool(args.cross_patchwise)
    if model_requires_fixed_embed_dim:
        if not cross_patchwise_effective:
            expected_embed_dim = latent_dim
        elif not args.cross_decouple_embed:
            expected_embed_dim = latent_c * args.cross_patch_size * args.cross_patch_size
    if args.embed_dim != expected_embed_dim:
        if accelerator.is_main_process:
            logger.info(
                "[model] overriding embed_dim to expected=%d (was %d)",
                expected_embed_dim,
                args.embed_dim,
            )
        args.embed_dim = expected_embed_dim
    args.vae_scaling = get_vae_scaling(vae, args.vae_name)
    args.vae_tag = vae_tag_from_name(args.vae_name)
    vae._pairflow_scaling = args.vae_scaling
    primary_db = build_primary_db(
        args=args,
        vae=vae,
        device=device,
        accelerator=accelerator,
    )
    Xdb = primary_db.Xdb
    db_indices = primary_db.db_indices
    db_group_ids = primary_db.db_group_ids
    loader = build_train_latent_loader(
        Xdb=Xdb,
        db_indices=db_indices,
        batch_size=args.batch_size,
        include_db_indices=bool(args.self_mask_db),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    if accelerator.is_main_process:
        logger.info("[ok] DB ready: %s", tuple(Xdb.shape))

    alt_db_artifacts = build_alt_db(
        args=args,
        vae=vae,
        device=device,
        accelerator=accelerator,
    )

    train_loop(
        args,
        Xdb,
        db_indices,
        db_group_ids,
        alt_db_artifacts.alt_db,
        alt_db_artifacts.alt_db_group_ids,
        alt_db_artifacts.alt_db_label,
        loader,
        vae,
        device,
        accelerator,
        save_dir,
    )
    return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(input_args=None):
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", default="out/out_learned_vae")
    ap.add_argument("--exp_name", default=None)
    ap.add_argument("--logging_dir", default="logs")
    ap.add_argument(
        "--transient_root",
        default=None,
        help="Root for temporary eval/sample artifacts. Defaults to $TMPDIR or the system temp dir.",
    )
    ap.add_argument("--db_dir", default="data/pairflow_vae_db")
    ap.add_argument("--hf_limit", type=int, default=0)
    ap.add_argument("--hf_streaming", action="store_true")
    ap.add_argument("--hf_streaming_buffer", type=int, default=10_000)
    ap.add_argument(
        "--N_img",
        type=_parse_optional_n_img,
        default=20000,
        help="Number of DB images to encode; use 'none' or 'all' to encode the full dataset once.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--db_batch_size",
        type=int,
        default=None,
        help="Batch size for one-time VAE encoding when building/loading the latent DB. Defaults to --batch_size.",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=_parse_bool, default=True)
    ap.add_argument("--persistent_workers", type=_parse_bool, default=True)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--num_gen", type=int, default=16)
    ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--t_eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode_batch", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)

    # VAE / encoders
    ap.add_argument("--vae_name", default="REPA-E/e2e-invae")
    ap.add_argument("--enc_type", type=str, default="None")

    # Patchwise retrieval model
    ap.add_argument("--embed_dim", type=int, default=768)
    ap.add_argument("--time_embed_dim", type=int, default=256)
    ap.add_argument("--cross_use_entmax", action="store_true")
    ap.add_argument("--cross_entmax_alpha", type=float, default=1.5)
    ap.add_argument("--cross_patchwise", action="store_true")
    ap.add_argument("--cross_patch_size", type=int, default=4)
    ap.add_argument(
        "--cross_attn_chunk_size",
        type=int,
        default=0,
        help=("Chunk size over DB tokens for SPFM softmax retrieval; 0 disables chunking."),
    )
    ap.add_argument(
        "--cross_db_dropout",
        type=float,
        default=0.0,
        help="Train-only probability of masking each DB image for spfm retrieval.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="spfm",
        choices=["dit", "spfm"],
        help="Model family to train.",
    )
    ap.add_argument(
        "--cross_kv_freeze",
        action="store_true",
        help="Freeze k_proj and optional v_proj inside cross-attention blocks.",
    )
    ap.add_argument(
        "--cross_decouple_embed",
        action="store_true",
        help="Keep embed_dim independent from cross_patch_size by projecting patch vectors in/out of embed space.",
    )
    ap.add_argument("--value_adaln", action="store_true")
    ap.add_argument("--qk_dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--refiner_depth", type=int, default=1)
    ap.add_argument("--refiner_patch_size", type=int, default=4)
    ap.add_argument("--refiner_embed_dim", type=int, default=384)
    ap.add_argument("--refiner_num_heads", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--refiner_mlp_ratio", type=float, default=None)
    ap.add_argument(
        "--learned_g",
        action="store_true",
        help="Use an MLP gate g(t) instead of fixed g(t)=4t(1-t).",
    )
    ap.add_argument(
        "--learned_alpha",
        action="store_true",
        help="Use an MLP gate alpha(t) instead of fixed alpha(t)=alpha_max*t.",
    )
    ap.add_argument(
        "--learned_g_init",
        type=float,
        default=0.5,
        help="Initial output value for learned g(t) in (0,1).",
    )
    ap.add_argument(
        "--learned_alpha_init",
        type=float,
        default=0.5,
        help="Initial output value for learned alpha(t) in (0,1).",
    )

    # Training
    ap.add_argument("--train_steps", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lion", "muon"])
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--adam_epsilon", type=float, default=1e-8)
    ap.add_argument("--weight_decay", type=float, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=50000)
    ap.add_argument(
        "--fid", type=_parse_bool, default=True, help="Enable FID evaluation on save_every."
    )
    ap.add_argument(
        "--kid", type=_parse_bool, default=True, help="Enable KID evaluation on save_every."
    )
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Path to .pt weights or full-state checkpoint directory for training resume.",
    )
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument(
        "--db",
        type=str,
        default="none",
        help="Main DB selection: 'none' (entire dataset) or class split spec, e.g. 'dog:0.9,cat:0.1'.",
    )
    ap.add_argument(
        "--alt_db",
        type=str,
        default="none",
        help=(
            "Alt DB selection: 'none', 'complementary' (relative to --db split), "
            "or an explicit class split spec like 'cat:1.0,dog:0.0'."
        ),
    )
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument(
        "--nn_clip_model", dest="clip_model", type=str, default=None, help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--nn_clip_local_files_only",
        action="store_true",
        help="Load CLIP model from local cache only (no downloads).",
    )
    ap.add_argument(
        "--clip_eval_prompts",
        action="append",
        default=None,
        help="Prompt used by CLIP eval. Repeat this flag to pass multiple prompts.",
    )
    ap.add_argument(
        "--clip_eval_batch_size",
        type=int,
        default=0,
        help="Batch size for CLIP prompt evaluation during save_every eval (0 uses --batch_size).",
    )
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--refiner_penalty", type=float, default=1e-3)
    ap.add_argument("--drifting_penalty", type=float, default=0.0)
    ap.add_argument("--drifting_tau", type=float, default=0.1)
    ap.add_argument("--drifting_warmup", type=int, default=5000)
    ap.add_argument("--loss_weight", choices=["none", "cos", "inv_1_minus_t2"], default="none")
    ap.add_argument(
        "--self_mask_db", action="store_true", help="Mask DB self-match during training."
    )

    # Accelerate
    ap.add_argument("--allow_tf32", action="store_true")
    ap.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument(
        "--ddp_find_unused_parameters",
        type=_parse_bool,
        default=None,
        help=(
            "Override DDP find_unused_parameters. "
            "If unset, auto-enables for model=spfm and disables otherwise."
        ),
    )
    ap.add_argument("--report_to", type=str, default="wandb")
    ap.add_argument("--wandb_project", default="pairflow-learned-vae")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_name", default=None)
    ap.add_argument(
        "--wandb_run_id",
        default=None,
        help="Explicit Weights & Biases run id for resume/continuation.",
    )
    ap.add_argument(
        "--wandb_resume",
        type=str,
        default=None,
        choices=["allow", "must", "never", "auto"],
        help="Weights & Biases resume policy; set with --wandb_run_id to continue the same run.",
    )
    ap.add_argument("--label_field", type=str, default=None)

    args = ap.parse_args(input_args)
    raw_argv = input_args if input_args is not None else sys.argv[1:]
    if any(tok == "--nn_clip_model" or tok.startswith("--nn_clip_model=") for tok in raw_argv):
        warnings.warn(
            "--nn_clip_model is deprecated; use --clip_model instead.",
            UserWarning,
            stacklevel=2,
        )
    args.model = str(args.model).replace("-", "_")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
