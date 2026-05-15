#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from cleanfid import fid as cleanfid_fid
from torchvision.utils import make_grid, save_image
from torchvision.models import Inception_V3_Weights, inception_v3

import run_experiment
import train
from preprocessing.encoders import load_invae
from trainer.clip_eval import clip_eval as run_clip_eval
from trainer.clip_eval import load_clip_runtime, prompt_metric_suffix
from trainer.db import build_alt_db, build_primary_db
from trainer.model_factory import build_model
from utils.train_helpers import (
    RawVAE,
    decode_latents,
    ensure_dir,
    generate_images_to_dir,
    get_vae_scaling,
    infer_vae_latent_spec,
    vae_tag_from_name,
)


LOGGER = logging.getLogger("eval_checkpoint")


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "all"}:
        return None
    return int(text)


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


def _load_config_args(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")
    train_cfg = cfg.get("train", cfg)
    if not isinstance(train_cfg, dict):
        raise ValueError("`train` section must be a mapping.")
    flat_cfg = run_experiment._merge_train_cfg(train_cfg)
    flat_cfg, launch_cfg = run_experiment._extract_launch_cfg(flat_cfg)
    del launch_cfg
    cli_args = run_experiment._to_cli_args(flat_cfg)
    return train.parse_args(cli_args)


def _apply_overrides(args, ns) -> None:
    args.out_dir = ns.results_dir
    args.exp_name = None
    args.report_to = None
    args.wandb_entity = None
    args.wandb_name = None
    args.wandb_project = None
    args.wandb_resume = None
    args.wandb_run_id = None
    if ns.ckpt is not None:
        args.ckpt = ns.ckpt
    if ns.n_img_override is not None:
        args.N_img = ns.n_img_override
    if ns.db_override is not None:
        args.db = ns.db_override
    if ns.alt_db_override is not None:
        args.alt_db = ns.alt_db_override
    if ns.sample_steps is not None:
        args.sample_steps = int(ns.sample_steps)
    if ns.decode_batch is not None:
        args.decode_batch = int(ns.decode_batch)
    if ns.gen_batch is not None:
        args.batch_size = int(ns.gen_batch)
    if ns.clip_eval_batch_size is not None:
        args.clip_eval_batch_size = int(ns.clip_eval_batch_size)
    if ns.clip_eval_prompts:
        args.clip_eval_prompts = list(ns.clip_eval_prompts)
    args.fid = bool(ns.fid)
    args.kid = bool(ns.kid)


def _resolve_model_weights_path(requested_ckpt: str | None, train_out_dir: str) -> str:
    if requested_ckpt is not None:
        path = Path(requested_ckpt).expanduser()
        if path.is_file():
            return str(path.resolve())
        if path.is_dir():
            parent = path.parent
            step_tag = path.name.replace("checkpoint_", "")
            candidate = parent / f"model_{step_tag}.pt"
            if candidate.exists():
                return str(candidate.resolve())
            last_candidate = parent / "model_last.pt"
            if last_candidate.exists():
                return str(last_candidate.resolve())
            step_candidates = sorted(path.glob("model_step*.pt"))
            if step_candidates:
                return str(step_candidates[-1].resolve())
            parent_step_candidates = sorted(parent.glob("model_step*.pt"))
            if parent_step_candidates:
                return str(parent_step_candidates[-1].resolve())
            raise FileNotFoundError(f"Could not resolve model weights from checkpoint dir: {path}")
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    root = Path(train_out_dir).expanduser().resolve()
    direct_last = root / "model_last.pt"
    if direct_last.exists():
        return str(direct_last)
    step_candidates = sorted(root.glob("model_step*.pt"))
    if step_candidates:
        return str(step_candidates[-1].resolve())
    raise FileNotFoundError(f"No model_last.pt or model_step*.pt found under {root}")


def _prepare_reference_args(args):
    ref_args = deepcopy(args)
    ref_args.db = "none"
    ref_args.alt_db = "none"
    ref_args.N_img = None
    ref_args.self_mask_db = False
    return ref_args


def _maybe_override_embed_dim(args) -> None:
    latent_dim = args.latent_c * args.latent_h * args.latent_w
    expected_embed_dim = args.embed_dim
    model_requires_fixed_embed_dim = args.model != "dit"
    cross_patchwise_effective = bool(args.cross_patchwise)
    if model_requires_fixed_embed_dim:
        if not cross_patchwise_effective:
            expected_embed_dim = latent_dim
        elif not args.cross_decouple_embed:
            expected_embed_dim = args.latent_c * args.cross_patch_size * args.cross_patch_size
    if args.embed_dim != expected_embed_dim:
        LOGGER.info("[model] overriding embed_dim to expected=%d (was %d)", expected_embed_dim, args.embed_dim)
        args.embed_dim = expected_embed_dim


def _write_latents_to_dir_if_needed(vae, latents: torch.Tensor, out_dir: str, decode_batch: int) -> None:
    existing = sorted(Path(out_dir).glob("*.png"))
    if existing:
        LOGGER.info("[real] reusing %d existing reference images at %s", len(existing), out_dir)
        return
    ensure_dir(out_dir)
    count = 0
    for s in range(0, latents.shape[0], decode_batch):
        imgs = decode_latents(vae, latents[s:s + decode_batch], decode_batch=decode_batch)
        for i in range(imgs.shape[0]):
            from torchvision.utils import save_image
            save_image(imgs[i], os.path.join(out_dir, f"img_{count:06d}.png"))
            count += 1
    LOGGER.info("[real] wrote %d reference images to %s", count, out_dir)


def _compute_inception_score(directory_with_images: str, device: torch.device, batch_size: int, splits: int = 10):
    image_paths = sorted(Path(directory_with_images).glob("*.png"))
    if not image_paths:
        return None
    weights = Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()
    model = inception_v3(weights=weights, transform_input=False).to(device)
    model.eval()
    probs_parts: list[torch.Tensor] = []
    eff_batch = max(1, int(batch_size))
    with torch.no_grad():
        for s in range(0, len(image_paths), eff_batch):
            batch_paths = image_paths[s:s + eff_batch]
            batch = []
            for path in batch_paths:
                with Image.open(path) as im:
                    batch.append(preprocess(im.convert("RGB")))
            pix = torch.stack(batch, dim=0).to(device)
            logits = model(pix)
            probs_parts.append(F.softmax(logits, dim=1).cpu())
    probs = torch.cat(probs_parts, dim=0)
    num_images = int(probs.shape[0])
    eff_splits = max(1, min(int(splits), num_images))
    split_size = max(1, num_images // eff_splits)
    scores = []
    for split_idx in range(eff_splits):
        start = split_idx * split_size
        end = num_images if split_idx == eff_splits - 1 else min(num_images, (split_idx + 1) * split_size)
        if end <= start:
            continue
        part = probs[start:end]
        py = part.mean(dim=0, keepdim=True)
        kl = part * (torch.log(part.clamp_min(1e-12)) - torch.log(py.clamp_min(1e-12)))
        scores.append(torch.exp(kl.sum(dim=1).mean()).item())
    if not scores:
        return None
    mean = float(sum(scores) / len(scores))
    var = float(sum((x - mean) ** 2 for x in scores) / len(scores))
    return {
        "inception_score_mean": mean,
        "inception_score_std": math.sqrt(var),
        "inception_score_splits": eff_splits,
    }


def _save_preview_grid(
    directory_with_images: str,
    out_path: str,
    grid_size: int = 10,
) -> int:
    image_paths = sorted(Path(directory_with_images).glob("*.png"))
    max_images = int(grid_size * grid_size)
    selected = image_paths[:max_images]
    if not selected:
        return 0
    imgs = []
    for path in selected:
        with Image.open(path) as im:
            arr = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1).float() / 255.0
            imgs.append(arr)
    grid = make_grid(torch.stack(imgs, dim=0), nrow=grid_size)
    save_image(grid, out_path)
    return len(selected)


def _load_model_weights(model: torch.nn.Module, ckpt_path: str, use_ema: bool) -> None:
    def _normalize_state_dict_keys(state_dict):
        normalized = {}
        for name, value in state_dict.items():
            key = str(name)
            for prefix in ("module.", "_orig_mod."):
                while key.startswith(prefix):
                    key = key[len(prefix):]
                key = key.replace(f".{prefix}", ".")
            normalized[key] = value
        return normalized

    state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")
    if use_ema and "ema" in state:
        state_dict = state["ema"]
        chosen = "ema"
    elif "model" in state:
        state_dict = state["model"]
        chosen = "model"
    else:
        raise KeyError(f"Checkpoint {ckpt_path} missing 'model'/'ema' keys")
    state_dict = _normalize_state_dict_keys(state_dict)
    train._unwrap_model_for_runtime(model).load_state_dict(state_dict)
    LOGGER.info("[ckpt] loaded %s weights from %s", chosen, ckpt_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate images and evaluate a trained checkpoint.")
    ap.add_argument("--config", required=True, help="Path to experiments YAML config.")
    ap.add_argument("--results_dir", required=True, help="Directory where generated images and metrics are saved.")
    ap.add_argument("--ckpt", default=None, help="Optional explicit .pt checkpoint path.")
    ap.add_argument("--total_gen", type=int, required=True, help="Number of samples to generate.")
    ap.add_argument("--gen_batch", type=int, default=None, help="Generation batch size override.")
    ap.add_argument("--sample_steps", type=int, default=None, help="Sampling steps override.")
    ap.add_argument("--decode_batch", type=int, default=None, help="Decode batch size override.")
    ap.add_argument("--n_img_override", type=_parse_optional_int, default=None, help="Override DB size; use 'none' for full DB.")
    ap.add_argument("--db_override", default=None, help="Override --db selection spec.")
    ap.add_argument("--alt_db_override", default="none", help="Override --alt_db selection spec.")
    ap.add_argument("--reference_mode", choices=["db", "dataset", "none"], default="db")
    ap.add_argument("--reference_dir", default=None, help="Optional shared directory for decoded reference images.")
    ap.add_argument("--generated_dir", default=None, help="Optional directory for generated PNG samples.")
    ap.add_argument("--fid", type=_parse_bool, default=True)
    ap.add_argument("--kid", type=_parse_bool, default=True)
    ap.add_argument("--inception_score", type=_parse_bool, default=True)
    ap.add_argument("--clip_eval", type=_parse_bool, default=True)
    ap.add_argument("--clip_eval_prompts", action="append", default=None)
    ap.add_argument("--clip_eval_batch_size", type=int, default=None)
    ap.add_argument("--use_ema", type=_parse_bool, default=True)
    ns = ap.parse_args()

    config_path = str(Path(ns.config).expanduser().resolve())
    training_args = _load_config_args(config_path)
    train_out_dir = str(training_args.out_dir)
    _apply_overrides(training_args, ns)
    _setup_logging(training_args.out_dir)

    LOGGER.info("[config] using config=%s", config_path)
    LOGGER.info("[results] saving artifacts to %s", training_args.out_dir)

    ddp_find_unused = training_args.ddp_find_unused_parameters
    if ddp_find_unused is None:
        ddp_find_unused = training_args.model == "spfm"
    ddp_find_unused = bool(ddp_find_unused)

    accelerator_project_config = ProjectConfiguration(
        project_dir=training_args.out_dir,
        logging_dir=Path(training_args.out_dir, training_args.logging_dir),
    )
    accelerator = Accelerator(
        mixed_precision=training_args.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)],
    )

    if accelerator.is_main_process:
        LOGGER.info(
            "[dist] world_size=%d process_index=%d local_process_index=%d device=%s",
            accelerator.num_processes,
            accelerator.process_index,
            accelerator.local_process_index,
            accelerator.device,
        )
    if training_args.seed is not None:
        set_seed(training_args.seed + accelerator.process_index)
    if training_args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = accelerator.device
    is_raw_vae = str(training_args.vae_name).strip().lower() == "raw"
    if is_raw_vae:
        vae = RawVAE().to(device)
    else:
        if accelerator.is_main_process:
            _ = load_invae(training_args.vae_name, device=device)
        accelerator.wait_for_everyone()
        vae = load_invae(training_args.vae_name, device=device)
    vae.eval().requires_grad_(False)

    latent_c, latent_h, latent_w, latent_downsample = infer_vae_latent_spec(vae, training_args.image_size, device)
    training_args.latent_c = latent_c
    training_args.latent_h = latent_h
    training_args.latent_w = latent_w
    training_args.latent_downsample = latent_downsample
    training_args.vae_scaling = get_vae_scaling(vae, training_args.vae_name)
    training_args.vae_tag = vae_tag_from_name(training_args.vae_name)
    vae._pairflow_scaling = training_args.vae_scaling
    _maybe_override_embed_dim(training_args)

    primary_db = build_primary_db(training_args, vae=vae, device=device, accelerator=accelerator)
    Xdb = primary_db.Xdb
    db_group_ids = primary_db.db_group_ids
    LOGGER.info("[db] primary DB shape=%s", tuple(Xdb.shape))

    alt_db_artifacts = build_alt_db(training_args, vae=vae, device=device, accelerator=accelerator)
    if alt_db_artifacts.alt_db is not None:
        LOGGER.info("[db] alt DB shape=%s label=%s", tuple(alt_db_artifacts.alt_db.shape), alt_db_artifacts.alt_db_label)

    model = build_model(training_args, device)
    model = accelerator.prepare(model)
    ckpt_path = _resolve_model_weights_path(ns.ckpt, train_out_dir)
    _load_model_weights(model, ckpt_path, use_ema=bool(ns.use_ema))
    raw_model = train._unwrap_model_for_runtime(model)
    raw_model.eval()

    results_dir = training_args.out_dir
    generated_dir = ns.generated_dir or os.path.join(results_dir, "generated")
    reference_dir = ns.reference_dir or os.path.join(results_dir, "reference_real")
    if accelerator.is_main_process and os.path.exists(generated_dir):
        shutil.rmtree(generated_dir, ignore_errors=True)
    accelerator.wait_for_everyone()

    gen_start_time = time.time()
    generate_images_to_dir(
        Xdb=Xdb,
        model=raw_model,
        vae=vae,
        out_dir=generated_dir,
        total_gen=int(ns.total_gen),
        gen_batch=int(ns.gen_batch or training_args.batch_size),
        steps=int(training_args.sample_steps),
        noise_scale=float(training_args.noise_scale),
        t_eps=float(training_args.t_eps),
        decode_batch=int(training_args.decode_batch),
        prefix="img_",
        latent_c=training_args.latent_c,
        latent_h=training_args.latent_h,
        latent_w=training_args.latent_w,
        db_group_ids=db_group_ids,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        LOGGER.info("[generate] finished total_gen=%d elapsed_sec=%.1f", int(ns.total_gen), time.time() - gen_start_time)

    metrics: dict[str, object] = {
        "config": config_path,
        "results_dir": results_dir,
        "training_out_dir": train_out_dir,
        "checkpoint": ckpt_path,
        "model": training_args.model,
        "db": str(training_args.db),
        "alt_db": str(training_args.alt_db),
        "N_img": training_args.N_img,
        "total_gen": int(ns.total_gen),
        "sample_steps": int(training_args.sample_steps),
        "use_ema": bool(ns.use_ema),
        "reference_mode": ns.reference_mode,
        "reference_dir": reference_dir,
        "generated_dir": generated_dir,
    }

    reference_latents = None
    if ns.reference_mode == "db":
        reference_latents = Xdb
    elif ns.reference_mode == "dataset":
        ref_args = _prepare_reference_args(training_args)
        ref_db = build_primary_db(ref_args, vae=vae, device=device, accelerator=accelerator)
        reference_latents = ref_db.Xdb
        metrics["reference_N_img"] = ref_args.N_img
        metrics["reference_db"] = ref_args.db

    if accelerator.is_main_process:
        preview_grid_path = os.path.join(results_dir, "generated_grid_10x10.png")
        preview_count = _save_preview_grid(generated_dir, preview_grid_path, grid_size=10)
        if preview_count > 0:
            metrics["preview_grid"] = preview_grid_path
            metrics["preview_grid_num_images"] = preview_count

        if reference_latents is not None and (training_args.fid or training_args.kid):
            _write_latents_to_dir_if_needed(
                vae=vae,
                latents=reference_latents,
                out_dir=reference_dir,
                decode_batch=int(training_args.decode_batch),
            )
            if training_args.fid:
                metrics["fid"] = float(cleanfid_fid.compute_fid(reference_dir, generated_dir))
            if training_args.kid:
                metrics["kid"] = float(cleanfid_fid.compute_kid(reference_dir, generated_dir))

        if ns.inception_score:
            is_metrics = _compute_inception_score(
                directory_with_images=generated_dir,
                device=device,
                batch_size=int(training_args.clip_eval_batch_size or training_args.batch_size),
            )
            if is_metrics is not None:
                metrics.update(is_metrics)

        clip_prompts = training_args.clip_eval_prompts or ["a photo of a dog", "a photo of a cat"]
        clip_prompts = [p.strip() for p in clip_prompts if p and p.strip()]
        if ns.clip_eval and clip_prompts:
            clip_runtime = load_clip_runtime(
                model_name=training_args.clip_model,
                device=device,
                local_files_only=bool(training_args.nn_clip_local_files_only),
            )
            clip_scores = run_clip_eval(
                directory_with_images=generated_dir,
                prompts=clip_prompts,
                runtime=clip_runtime,
                batch_size=int(training_args.clip_eval_batch_size or training_args.batch_size),
            )
            metrics["clip_eval"] = clip_scores
            for prompt, pct in clip_scores.items():
                metrics[f"clip_pct_{prompt_metric_suffix(prompt)}"] = float(pct)

        with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        with open(os.path.join(results_dir, "resolved_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(training_args), f, indent=2, sort_keys=True, default=str)
        LOGGER.info("[done] metrics=%s", json.dumps(metrics, indent=2, sort_keys=True))

    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
