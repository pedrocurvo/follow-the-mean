#!/usr/bin/env python3
"""Reusable runtime helpers for RGM experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, List, Sequence

import retrieval_guidance_core as poc
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# General Utilities
# ---------------------------------------------------------------------------


def slugify(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {" ", "-", "_"}:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "case"


# ---------------------------------------------------------------------------
# RMG Callback
# ---------------------------------------------------------------------------


def beta_schedule_value(guidance_strength: float, beta_schedule: str, t: float) -> float:
    if beta_schedule == "bell":
        return 4.0 * guidance_strength * t * (1.0 - t)
    if beta_schedule == "quadratic-decay":
        return guidance_strength * (1.0 - t) ** 2
    return guidance_strength


class RMGCallback:
    def __init__(
        self,
        reference_latents: torch.Tensor,
        total_steps: int,
        guidance_strength: float,
        beta_schedule: str,
        guidance_start_frac: float,
        guidance_end_frac: float,
        topk: int,
        verbose: bool = False,
    ):
        self.reference_latents = reference_latents
        self.total_steps = total_steps
        self.guidance_strength = guidance_strength
        self.beta_schedule = beta_schedule
        self.guidance_start_frac = guidance_start_frac
        self.guidance_end_frac = guidance_end_frac
        self.topk = topk
        self.verbose = verbose
        self.history = []
        self.prev_latents = None

    def __call__(self, pipeline, step_index: int, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        if latents is None:
            return callback_kwargs

        if not poc.in_guidance_window(
            step_index=step_index,
            total_steps=self.total_steps,
            start_frac=self.guidance_start_frac,
            end_frac=self.guidance_end_frac,
        ):
            return callback_kwargs

        t = poc.step_fraction(step_index, self.total_steps)
        t_next = poc.step_fraction(min(step_index + 1, self.total_steps - 1), self.total_steps)
        if step_index + 1 >= self.total_steps:
            t_next = 1.0
        dt = t_next - t
        beta_t = beta_schedule_value(self.guidance_strength, self.beta_schedule, t)
        v_guided, stats = poc.rmg_velocity_update(
            current_latents=latents,
            reference_latents=self.reference_latents,
            t=t,
            topk=self.topk,
        )

        # Compute the RMG velocity correction from Eq. (4) of the paper:
        #   u_t^pi = u_t^theta + beta_t * (mu_hat_t^rho - mu_t^theta) / (1 - t)
        #
        # v_guided = (mu_hat_t^rho - x_t) / (1 - t), which expands via the
        # flow matching identity mu_t^theta = x_t + (1 - t) * u_t^theta to:
        #   v_guided = (mu_hat_t^rho - mu_t^theta) / (1 - t) + u_t^theta
        #
        # So the net latent update is:
        #   delta_x = beta_t * (v_guided - u_t^theta) * dt
        #
        # We reconstruct u_t^theta from the rectified flow finite difference:
        #   u_t^theta ~= (x_t - x_{t-1}) / dt
        #
        # NOTE: using beta_t * v_guided * dt without the u_t^theta correction
        # is a valid approximation for all steps. It applies a slightly larger
        # update but avoids needing prev_latents entirely. Use it if pipeline
        # internals change.
        if self.prev_latents is not None and dt > 0:
            u_theta = (
                latents - self.prev_latents.to(device=latents.device, dtype=latents.dtype)
            ) / dt
            correction = beta_t * (v_guided - u_theta) * dt
            stats["u_theta_norm"] = float(
                poc.flatten_latents(u_theta.float()).norm(dim=-1).mean().item()
            )
        else:
            # First step in the guidance window: prev_latents is not yet
            # available. Fall back to the approximate update; the error is
            # negligible when beta_t is near zero at the window boundary.
            correction = beta_t * v_guided * dt
            stats["u_theta_norm"] = None

        self.prev_latents = latents.detach().clone()
        callback_kwargs["latents"] = latents + correction.to(latents.dtype)
        stats["dt"] = float(dt)
        stats["beta_t"] = float(beta_t)
        stats["correction_norm"] = float(
            poc.flatten_latents(correction.float()).norm(dim=-1).mean().item()
        )
        self.history.append((step_index, t, stats))

        if self.verbose:
            print(
                f"step={step_index:02d} timestep={timestep} t={t:.3f} "
                f"dt={dt:.4f} beta_t={beta_t:.4f} v_guided={stats['v_guided_norm']:.4f}"
            )
        return callback_kwargs


# ---------------------------------------------------------------------------
# File Utilities
# ---------------------------------------------------------------------------


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"saved: {path}")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def normalize_yes_no(text: str) -> str:
    lowered = text.strip().lower()
    has_yes = "yes" in lowered
    has_no = "no" in lowered
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "unknown"


def generate_vlm_answer(
    model, processor, image: Image.Image, question: str, device: str, max_new_tokens: int
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = []
    for input_ids, output_ids in zip(inputs["input_ids"], generated_ids):
        trimmed.append(output_ids[len(input_ids) :])
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0].strip()


# ---------------------------------------------------------------------------
# Reference IDs And Paths
# ---------------------------------------------------------------------------


def shared_references_dir(root_dir: Path) -> Path:
    return root_dir / "references"


def make_reference_label(reference_prompts: Sequence[str]) -> str:
    prompt_counts = Counter(reference_prompts)
    return "-".join(f"{slugify(prompt)}x{count}" for prompt, count in sorted(prompt_counts.items()))


def make_reference_id(args: argparse.Namespace, reference_prompts: Sequence[str]) -> str:
    # Keep the hash schema compatible with the original scalable-fm scripts so
    # renamed references resolve to the same cached images/latents.
    payload = {
        "model_id": args.model_id,
        "height": args.height,
        "width": args.width,
        "bank_size": len(reference_prompts),
        "bank_seed": args.reference_seed,
        "bank_prompts": list(reference_prompts),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    label = make_reference_label(reference_prompts)
    return f"{label}-{digest}"[:180]


def has_reference_cache(reference_dir: Path) -> bool:
    return (reference_dir / "reference_cache.pt").exists() or (
        reference_dir / "bank_cache.pt"
    ).exists()


def resolve_staged_reference_dir(
    root_dir: Path, reference_id: str, reference_prompts: Sequence[str]
) -> Path:
    reference_root = shared_references_dir(root_dir)
    canonical_dir = reference_root / reference_id
    if has_reference_cache(canonical_dir):
        return canonical_dir

    label = make_reference_label(reference_prompts)
    candidates = sorted(
        path for path in reference_root.glob(f"{label}-*") if has_reference_cache(path)
    )
    if candidates:
        print(f"Reusing staged reference with legacy id: {candidates[0]}")
        return candidates[0]

    return canonical_dir


def list_image_paths(image_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in exts
    )


def make_image_reference_id(args: argparse.Namespace, image_paths: Sequence[Path]) -> str:
    payload = {
        "image_preprocess": "aspect_preserving_resize_then_pad",
        "model_id": args.model_id,
        "height": args.height,
        "width": args.width,
        "image_paths": [str(path.resolve()) for path in image_paths],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"image-reference-{digest}"


# ---------------------------------------------------------------------------
# Runtime Configuration
# ---------------------------------------------------------------------------


def make_runtime_cfg(
    args: argparse.Namespace,
    prompt: str,
    reference_prompt: str,
    out_dir: Path,
    seed: int,
) -> poc.RuntimeConfig:
    return poc.RuntimeConfig(
        model_id=args.model_id,
        prompt=prompt,
        reference_prompt=reference_prompt,
        negative_prompt=getattr(args, "negative_prompt", ""),
        seed=seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        reference_size=args.reference_size,
        guidance_strength=args.guidance_strength,
        guidance_start_frac=args.guidance_start_frac,
        guidance_end_frac=args.guidance_end_frac,
        topk=args.topk,
        out_dir=str(out_dir),
        reuse_reference=args.reuse_reference,
        reference_cache_path="",
        callback_verbose=args.callback_verbose,
        log_callback_keys=False,
        debug_intervention="none",
        debug_intervention_step=5,
    )


def override_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(updates)
    return argparse.Namespace(**payload)


def make_reference_cfg(
    args: argparse.Namespace,
    root_dir: Path,
    reference_label: str,
    reference_prompts: Sequence[str],
) -> tuple[poc.RuntimeConfig, str]:
    reference_id = make_reference_id(args, reference_prompts)
    reference_dir = resolve_staged_reference_dir(root_dir, reference_id, reference_prompts)
    cfg = make_runtime_cfg(
        args,
        prompt=reference_prompts[0] if reference_prompts else reference_label,
        reference_prompt=reference_label,
        out_dir=reference_dir,
        seed=args.reference_seed,
    )
    return cfg, reference_id


# ---------------------------------------------------------------------------
# Reference Construction
# ---------------------------------------------------------------------------


def build_reference_from_prompts(
    pipe, cfg: poc.RuntimeConfig, reference_prompts: Sequence[str], device: str
):
    reference_images: List[Image.Image] = []
    for i, prompt in enumerate(reference_prompts):
        print(f"[reference {i + 1}/{len(reference_prompts)}] {prompt}")
        image = poc.generate_single_image(pipe, prompt, cfg.seed + 1000 + i, cfg, device)
        reference_images.append(image)

    rows = max(1, math.ceil(len(reference_images) / 4))
    cols = min(4, len(reference_images))
    poc.save_pil(
        poc.image_grid(reference_images, rows=rows, cols=cols),
        str(Path(cfg.out_dir) / "reference_grid.png"),
    )
    poc.save_reference_images(reference_images, cfg)

    reference_latents = poc.encode_images_to_latents(
        reference_images, pipe, cfg.height, cfg.width, device
    )
    reference_meta = {
        "reference_prompt": cfg.reference_prompt,
        "reference_size": len(reference_prompts),
        "height": cfg.height,
        "width": cfg.width,
        "seed": cfg.seed,
        "reference_dir": str(Path(cfg.out_dir)),
        "reference_images_dir": str(poc.resolve_reference_images_dir(cfg)),
        "reference_prompt_list": list(reference_prompts),
        "reference_prompt_counts": dict(Counter(reference_prompts)),
    }
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"reference_latents": reference_latents.detach().cpu(), "meta": reference_meta},
        poc.resolve_reference_cache_path(cfg),
    )
    print(f"saved: {poc.resolve_reference_cache_path(cfg)}")
    save_json(reference_meta, Path(cfg.out_dir) / "reference_metadata.json")
    return reference_latents, reference_meta


def build_reference_from_images(
    image_paths: Sequence[Path],
    pipe,
    cfg: poc.RuntimeConfig,
    image_reference_dir: Path,
    device: str,
):
    reference_images: List[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            rgb = poc.pil_to_rgb(image.copy())
            fitted = ImageOps.contain(rgb, (cfg.width, cfg.height), method=Image.Resampling.LANCZOS)
            processed = Image.new("RGB", (cfg.width, cfg.height), (0, 0, 0))
            offset_x = (cfg.width - fitted.width) // 2
            offset_y = (cfg.height - fitted.height) // 2
            processed.paste(fitted, (offset_x, offset_y))
            reference_images.append(processed)

    rows = max(1, math.ceil(len(reference_images) / 4))
    cols = min(4, len(reference_images))
    poc.save_pil(
        poc.image_grid(reference_images, rows=rows, cols=cols),
        str(image_reference_dir / "reference_grid.png"),
    )

    reference_images_dir = image_reference_dir / "reference_images"
    reference_images_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(reference_images):
        poc.save_pil(image, str(reference_images_dir / f"reference_{idx:04d}.png"))

    reference_latents = poc.encode_images_to_latents(
        reference_images, pipe, cfg.height, cfg.width, device
    )
    reference_meta = {
        "reference_prompt": cfg.reference_prompt,
        "reference_size": len(reference_images),
        "height": cfg.height,
        "width": cfg.width,
        "seed": cfg.seed,
        "reference_dir": str(image_reference_dir),
        "reference_images_dir": str(reference_images_dir),
        "image_paths": [str(path.resolve()) for path in image_paths],
        "preprocess": "aspect_preserving_resize_then_pad",
        "reference_source": "images",
    }
    image_reference_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"reference_latents": reference_latents.detach().cpu(), "meta": reference_meta},
        image_reference_dir / "reference_cache.pt",
    )
    print(f"saved: {image_reference_dir / 'reference_cache.pt'}")
    save_json(reference_meta, image_reference_dir / "reference_metadata.json")
    return reference_latents, reference_meta


def load_or_build_image_reference(
    image_paths: Sequence[Path],
    pipe,
    cfg: poc.RuntimeConfig,
    image_reference_dir: Path,
    device: str,
):
    cache_path = image_reference_dir / "reference_cache.pt"
    legacy_cache_path = image_reference_dir / "bank_cache.pt"
    if not cache_path.exists() and legacy_cache_path.exists():
        cache_path = legacy_cache_path
    if cfg.reuse_reference and cache_path.exists():
        print(f"Reusing cached image reference: {cache_path}")
        reference_latents, reference_meta = poc.load_reference_cache(cache_path)
        save_json(reference_meta, image_reference_dir / "reference_metadata.json")
        return reference_latents, reference_meta
    return build_reference_from_images(image_paths, pipe, cfg, image_reference_dir, device)


def load_or_build_reference(
    pipe, cfg: poc.RuntimeConfig, reference_prompts: Sequence[str], device: str
):
    cache_path = poc.resolve_reference_cache_path(cfg)
    if cfg.reuse_reference and cache_path.exists():
        print(f"Reusing cached reference: {cache_path}")
        reference_latents, reference_meta = poc.load_reference_cache(cache_path)
        save_json(reference_meta, Path(cfg.out_dir) / "reference_metadata.json")
        return reference_latents, reference_meta
    return build_reference_from_prompts(pipe, cfg, reference_prompts, device)


# ---------------------------------------------------------------------------
# Guided Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_with_rmg(
    pipe,
    cfg: poc.RuntimeConfig,
    device: str,
    reference_latents: torch.Tensor,
    beta_schedule: str,
):
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 7)
    callback = RMGCallback(
        reference_latents=reference_latents,
        total_steps=cfg.num_inference_steps,
        guidance_strength=cfg.guidance_strength,
        beta_schedule=beta_schedule,
        guidance_start_frac=cfg.guidance_start_frac,
        guidance_end_frac=cfg.guidance_end_frac,
        topk=cfg.topk,
        verbose=cfg.callback_verbose,
    )

    kwargs = dict(
        prompt=cfg.prompt,
        num_inference_steps=cfg.num_inference_steps,
        height=cfg.height,
        width=cfg.width,
        guidance_scale=cfg.guidance_scale,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    kwargs["generator"] = generator

    result = pipe(**kwargs)
    return result.images[0], callback


def save_callback_history(history, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for step_index, t_frac, stats in history:
            parts = [
                f"step={step_index}",
                f"t_frac={t_frac:.4f}",
                f"dt={stats.get('dt', 0.0):.6f}",
                f"beta_t={stats.get('beta_t', 0.0):.6f}",
                f"mu_ref_norm={stats.get('mu_ref_norm', 0.0):.6f}",
                f"v_guided_norm={stats.get('v_guided_norm', 0.0):.6f}",
                f"current_norm={stats.get('current_norm', 0.0):.6f}",
                f"top1_weight={stats.get('top1_weight', 0.0):.6f}",
                f"posterior_entropy={stats.get('posterior_entropy', 0.0):.6f}",
            ]
            f.write("\t".join(parts) + "\n")
    print(f"saved: {path}")


# ---------------------------------------------------------------------------
# Case Runners
# ---------------------------------------------------------------------------


def run_case(
    pipe,
    args: argparse.Namespace,
    root_dir: Path,
    case_dir: Path,
    prompt: str,
    reference_label: str,
    reference_prompts: Sequence[str],
    seed: int,
    device: str,
    dtype: torch.dtype,
    save_baseline: bool = True,
):
    cfg = make_runtime_cfg(
        args, prompt=prompt, reference_prompt=reference_label, out_dir=case_dir, seed=seed
    )
    reference_cfg, reference_id = make_reference_cfg(
        args=args,
        root_dir=root_dir,
        reference_label=reference_label,
        reference_prompts=reference_prompts,
    )
    case_dir.mkdir(parents=True, exist_ok=True)

    baseline = None
    if save_baseline:
        baseline = poc.generate_single_image(pipe, cfg.prompt, cfg.seed + 7, cfg, device)
        poc.save_pil(baseline, str(case_dir / "baseline.png"))

    reference_latents, reference_meta = load_or_build_reference(
        pipe, reference_cfg, reference_prompts, device
    )
    guided, callback = generate_with_rmg(
        pipe=pipe,
        cfg=cfg,
        device=device,
        reference_latents=reference_latents.to(device=device, dtype=dtype),
        beta_schedule=args.beta_schedule,
    )
    poc.save_pil(guided, str(case_dir / "rmg_guided.png"))

    nn_index, nn_dist2 = poc.find_nearest_reference(guided, reference_latents, pipe, cfg, device)
    nearest = poc.load_reference_image(reference_cfg, nn_index)
    if nearest is None:
        nearest = poc.make_text_tile(
            guided.width, guided.height, [f"Nearest reference image missing for index {nn_index}"]
        )
    poc.save_pil(nearest, str(case_dir / "nearest_reference_neighbor.png"))

    if baseline is not None:
        comparison = poc.add_labels_and_title(
            [baseline, guided, nearest],
            ["Baseline", "RMG-guided", f"Nearest reference neighbour #{nn_index}"],
            title=f"Prompt: {prompt} | Reference: {reference_label}",
        )
        poc.save_pil(comparison, str(case_dir / "comparison.png"))
    save_callback_history(callback.history, case_dir / "callback_history.txt")

    metadata = {
        "prompt": prompt,
        "reference_label": reference_label,
        "reference_prompts": list(reference_prompts),
        "seed": seed,
        "beta_schedule": args.beta_schedule,
        "guidance_strength": args.guidance_strength,
        "nearest_reference_index": nn_index,
        "nearest_reference_latent_mse": float(nn_dist2[nn_index].item()),
        "shared_reference_id": reference_id,
        "shared_reference_dir": str(Path(reference_cfg.out_dir)),
        "runtime_cfg": asdict(cfg),
        "reference_meta": reference_meta,
    }
    save_json(metadata, case_dir / "run_metadata.json")
    result = {"case_dir": str(case_dir), "guided": guided, "nearest": nearest, "metadata": metadata}
    if baseline is not None:
        result["baseline"] = baseline
    return result


# ---------------------------------------------------------------------------
# Contact Sheets
# ---------------------------------------------------------------------------


def labeled_contact_sheet(
    rows: List[List[Image.Image]],
    row_labels: List[str],
    col_labels: List[str],
    title: str,
) -> Image.Image:
    if not rows or not rows[0]:
        raise ValueError("Need at least one image")
    font = ImageFont.load_default()
    cell_w, cell_h = rows[0][0].size
    left_w = 160
    top_h = 56
    label_h = 26
    canvas = Image.new(
        "RGB",
        (left_w + len(col_labels) * cell_w, top_h + len(row_labels) * (cell_h + label_h)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 16), title, fill=(0, 0, 0), font=font)

    for col_idx, label in enumerate(col_labels):
        x = left_w + col_idx * cell_w + 10
        draw.text((x, top_h - 28), label, fill=(0, 0, 0), font=font)

    for row_idx, row_label in enumerate(row_labels):
        y = top_h + row_idx * (cell_h + label_h)
        draw.text((16, y + cell_h // 2), row_label, fill=(0, 0, 0), font=font)
        for col_idx, image in enumerate(rows[row_idx]):
            x = left_w + col_idx * cell_w
            canvas.paste(poc.pil_to_rgb(image), (x, y))
    return canvas
