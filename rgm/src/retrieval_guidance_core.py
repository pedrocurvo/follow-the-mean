#!/usr/bin/env python3
"""Core primitives for Reference-Mean Guidance (RMG) experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    model_id: str
    prompt: str
    reference_prompt: str
    negative_prompt: str
    seed: int
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    reference_size: int
    guidance_strength: float
    guidance_start_frac: float
    guidance_end_frac: float
    topk: int
    out_dir: str
    reuse_reference: bool
    reference_cache_path: str
    callback_verbose: bool
    log_callback_keys: bool
    debug_intervention: str
    debug_intervention_step: int


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def save_pil(image: Image.Image, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"saved: {path}")


def flatten_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents.flatten(start_dim=1)


# ---------------------------------------------------------------------------
# Latent layout helpers
# ---------------------------------------------------------------------------


def _packed_to_spatial_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError(
            f"Expected packed latents with shape [B, T, C], got {tuple(latents.shape)}"
        )

    batch_size, tokens, channels = latents.shape
    side = int(math.isqrt(tokens))
    if side * side != tokens:
        raise ValueError(f"Packed latent token count must be a square, got {tokens}")

    return latents.permute(0, 2, 1).reshape(batch_size, channels, side, side).contiguous()


def _spatial_to_packed_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(
            f"Expected spatial latents with shape [B, C, H, W], got {tuple(latents.shape)}"
        )

    batch_size, channels, height, width = latents.shape
    return latents.reshape(batch_size, channels, height * width).permute(0, 2, 1).contiguous()


def _resize_packed_reference_latents_to_match(
    reference_latents: torch.Tensor,
    target_latents: torch.Tensor,
) -> torch.Tensor:
    if reference_latents.shape[1:] == target_latents.shape[1:]:
        return reference_latents

    reference_spatial = _packed_to_spatial_latents(reference_latents)
    target_spatial = _packed_to_spatial_latents(target_latents)
    target_h, target_w = target_spatial.shape[-2:]

    resized = F.interpolate(
        reference_spatial.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(reference_latents.dtype)
    return _spatial_to_packed_latents(resized)


# ---------------------------------------------------------------------------
# Step / window utilities
# ---------------------------------------------------------------------------


def step_fraction(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return step_index / (total_steps - 1)


def in_guidance_window(
    step_index: int,
    total_steps: int,
    start_frac: float,
    end_frac: float,
) -> bool:
    frac = step_fraction(step_index, total_steps)
    return start_frac <= frac <= end_frac


# ---------------------------------------------------------------------------
# Reference paths
# ---------------------------------------------------------------------------


def resolve_reference_cache_path(cfg: RuntimeConfig) -> Path:
    if cfg.reference_cache_path:
        return Path(cfg.reference_cache_path)
    cache_path = Path(cfg.out_dir) / "reference_cache.pt"
    legacy_path = Path(cfg.out_dir) / "bank_cache.pt"
    if not cache_path.exists() and legacy_path.exists():
        return legacy_path
    return cache_path


def resolve_reference_images_dir(cfg: RuntimeConfig) -> Path:
    return Path(cfg.out_dir) / "reference_images"


# ---------------------------------------------------------------------------
# VAE encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_images_to_latents(
    images: List[Image.Image],
    pipe,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    images = [pil_to_rgb(im).resize((width, height)) for im in images]
    arr = np.stack([np.array(im).astype(np.float32) / 255.0 for im in images], axis=0)

    vae = pipe.vae
    vae_dtype = getattr(vae, "dtype", torch.float32)
    if not isinstance(vae_dtype, torch.dtype) or not vae_dtype.is_floating_point:
        vae_dtype = torch.float32

    x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device=device, dtype=vae_dtype)
    x = x * 2.0 - 1.0
    enc = vae.encode(x)
    if hasattr(enc, "latent_dist"):
        if hasattr(enc.latent_dist, "mode"):
            latents = enc.latent_dist.mode()
        else:
            latents = enc.latent_dist.sample()
    else:
        latents = enc.latents

    if hasattr(pipe, "_patchify_latents") and hasattr(pipe, "_pack_latents") and hasattr(vae, "bn"):
        latents = pipe._patchify_latents(latents)
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_bn_mean) / latents_bn_std
        latents = pipe._pack_latents(latents)
    else:
        scaling_factor = getattr(getattr(vae, "config", object()), "scaling_factor", 1.0)
        latents = latents * scaling_factor

    return latents.detach()


# ---------------------------------------------------------------------------
# RMG update
# ---------------------------------------------------------------------------


@torch.no_grad()
def rmg_velocity_update(
    current_latents: torch.Tensor,
    reference_latents: torch.Tensor,
    t: float,
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute:
        logits_i = - ||x_t - t d_i||^2 / (2 (1-t)^2 + eps)
        mu_ref(x,t) = sum_i softmax(logits_i) d_i
        v_ref(x,t) = (mu_ref(x,t) - x_t) / (1-t)

    The caller converts this reference-only velocity into the paper's RMG
    residual correction, beta_t * (mu_ref - mu_theta) / (1-t).
    """
    reference_latents = _resize_packed_reference_latents_to_match(
        reference_latents, current_latents
    )
    x = flatten_latents(current_latents.float())
    d = flatten_latents(reference_latents.float())

    t = float(np.clip(t, 1e-4, 1.0 - 1e-4))
    projected_reference = t * d
    centered = x[:, None, :] - projected_reference[None, :, :]
    dist2 = centered.pow(2).mean(dim=-1)
    bandwidth = (1.0 - t) ** 2 + 1e-8
    logits = -dist2 / (2.0 * bandwidth)

    full_w = torch.softmax(logits, dim=-1)

    if topk is not None and topk > 0 and topk < logits.shape[-1]:
        vals, idx = torch.topk(logits, k=topk, dim=-1)
        w = torch.softmax(vals, dim=-1)
        selected_d = d[idx]
        mu_ref = (w[..., None] * selected_d).sum(dim=1)
    else:
        w = full_w
        mu_ref = w @ d
        idx = None

    v_guided = ((mu_ref - x) / (1.0 - t)).view_as(current_latents).to(current_latents.dtype)
    entropy = -(full_w * torch.log(full_w.clamp_min(1e-12))).sum(dim=-1)

    stats = {
        "mu_ref_norm": float(mu_ref.norm(dim=-1).mean().item()),
        "current_norm": float(x.norm(dim=-1).mean().item()),
        "v_guided_norm": float(flatten_latents(v_guided.float()).norm(dim=-1).mean().item()),
        "posterior_entropy": float(entropy.mean().item()),
        "top1_weight": float(full_w.max(dim=-1).values.mean().item()),
        "topk_indices": None if idx is None else idx.detach().cpu(),
    }
    return v_guided, stats


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipe(model_id: str, dtype: torch.dtype, device: str):
    pipe, pipe_kind = _load_pipe(model_id, dtype)
    _configure_pipe(pipe, device)
    print("Loaded pipeline kind:", pipe_kind)
    return pipe


def _load_pipe(model_id: str, dtype: torch.dtype):
    pipe = None
    pipe_kind = None

    if "klein" in model_id.lower():
        try:
            from diffusers import Flux2KleinPipeline
        except Exception as e:
            raise RuntimeError(
                "This checkpoint requires diffusers support for Flux2KleinPipeline, "
                "but the installed diffusers package does not provide it. "
                f"Model: {model_id}. "
                "Upgrade diffusers to a version that includes Flux2KleinPipeline."
            ) from e

        try:
            pipe = Flux2KleinPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            pipe_kind = "flux2_klein"
        except Exception as e:
            raise RuntimeError(
                "Flux2KleinPipeline is available, but loading the checkpoint failed. "
                "Check model access, Hugging Face cache location/quota, and any partial downloads. "
                f"Model: {model_id}."
            ) from e

    if pipe is None:
        try:
            from diffusers import Flux2Pipeline

            pipe = Flux2Pipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            pipe_kind = "flux2"
        except Exception as e:
            raise RuntimeError(
                "Could not load either Flux2KleinPipeline or Flux2Pipeline. "
                "Check diffusers version, model access, and hardware."
            ) from e

    return pipe, pipe_kind


def _configure_pipe(pipe, device: str) -> None:
    if device == "cuda":
        # Do not eagerly move the full pipeline to GPU before offload is enabled.
        # Prefer sequential offload because it uses less VRAM than model offload.
        offload_enabled = False
        for fn_name in [
            "enable_sequential_cpu_offload",
            "enable_model_cpu_offload",
        ]:
            if hasattr(pipe, fn_name):
                try:
                    getattr(pipe, fn_name)()
                    print("Enabled:", fn_name)
                    offload_enabled = True
                    break
                except Exception:
                    pass

        if not offload_enabled:
            pipe = pipe.to(device)

    else:
        pipe.to(device)

    for fn_name in [
        "enable_attention_slicing",
        "vae_enable_slicing",
        "vae_enable_tiling",
    ]:
        if hasattr(pipe, fn_name):
            try:
                getattr(pipe, fn_name)()
                print("Enabled:", fn_name)
            except Exception:
                pass


def _base_pipe_kwargs(prompt: str, cfg: RuntimeConfig) -> dict:
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": cfg.num_inference_steps,
        "height": cfg.height,
        "width": cfg.width,
        "guidance_scale": cfg.guidance_scale,
    }
    if cfg.negative_prompt:
        kwargs["negative_prompt"] = cfg.negative_prompt

    return kwargs


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_single_image(
    pipe,
    prompt: str,
    seed: int,
    cfg: RuntimeConfig,
    device: str,
) -> Image.Image:
    g = torch.Generator(device=device).manual_seed(seed)
    kwargs = _base_pipe_kwargs(prompt, cfg)
    kwargs["generator"] = g

    out = pipe(**kwargs)
    return out.images[0]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    if len(images) == 0:
        raise ValueError("Need at least one image for a grid.")
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(pil_to_rgb(img), (x, y))
    return grid


def make_text_tile(width: int, height: int, lines: List[str]) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 16
    for line in lines:
        draw.text((16, y), line, fill=(20, 20, 20), font=font)
        y += 18
    return image


def add_labels_and_title(
    images: List[Image.Image],
    labels: List[str],
    title: str,
) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("Need one label per image.")

    font = ImageFont.load_default()
    panel_width, panel_height = images[0].size
    title_height = 40
    label_height = 28
    canvas = Image.new(
        "RGB",
        (panel_width * len(images), title_height + panel_height + label_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), title, font=font)
    title_x = max(12, (canvas.width - (bbox[2] - bbox[0])) // 2)
    draw.text((title_x, 12), title, fill=(0, 0, 0), font=font)

    for idx, (image, label) in enumerate(zip(images, labels)):
        x = idx * panel_width
        canvas.paste(pil_to_rgb(image), (x, title_height))
        draw.rectangle(
            [
                (x, title_height + panel_height),
                (x + panel_width, title_height + panel_height + label_height),
            ],
            fill=(240, 240, 240),
        )
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_x = x + max(8, (panel_width - (label_bbox[2] - label_bbox[0])) // 2)
        draw.text((label_x, title_height + panel_height + 7), label, fill=(0, 0, 0), font=font)

    return canvas


# ---------------------------------------------------------------------------
# Reference persistence
# ---------------------------------------------------------------------------


def save_reference_cache(
    cache_path: Path,
    reference_latents: torch.Tensor,
    meta: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_latents": reference_latents.detach().cpu(),
        "meta": meta,
    }
    torch.save(payload, cache_path)
    print(f"saved: {cache_path}")


def load_reference_cache(cache_path: Path) -> Tuple[torch.Tensor, dict]:
    payload = torch.load(cache_path, map_location="cpu")
    reference_latents = payload.get("reference_latents", payload.get("bank_latents"))
    if reference_latents is None:
        raise KeyError(f"Cache missing reference_latents: {cache_path}")
    meta = payload.get("meta", {})
    return reference_latents, meta


def save_reference_images(images: List[Image.Image], cfg: RuntimeConfig) -> None:
    reference_images_dir = resolve_reference_images_dir(cfg)
    reference_images_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        save_pil(image, str(reference_images_dir / f"reference_{idx:04d}.png"))


def load_reference_image(cfg: RuntimeConfig, index: int) -> Optional[Image.Image]:
    image_path = resolve_reference_images_dir(cfg) / f"reference_{index:04d}.png"
    if not image_path.exists():
        image_path = Path(cfg.out_dir) / "bank_images" / f"bank_{index:04d}.png"
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return pil_to_rgb(image.copy())


# ---------------------------------------------------------------------------
# Nearest-neighbour retrieval
# ---------------------------------------------------------------------------


@torch.no_grad()
def find_nearest_reference(
    query_image: Image.Image,
    reference_latents: torch.Tensor,
    pipe,
    cfg: RuntimeConfig,
    device: str,
) -> Tuple[int, torch.Tensor]:
    query_latents = encode_images_to_latents([query_image], pipe, cfg.height, cfg.width, device)
    reference_latents = _resize_packed_reference_latents_to_match(reference_latents, query_latents)
    reference_device = reference_latents.device
    query = flatten_latents(query_latents.float().to(reference_device))
    reference = flatten_latents(reference_latents.float().to(reference_device))
    dist2 = ((reference - query[0:1]) ** 2).mean(dim=-1)
    index = int(dist2.argmin().item())
    return index, dist2.detach().cpu()
