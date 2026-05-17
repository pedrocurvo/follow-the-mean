#!/usr/bin/env python3
"""
Reference-set size ablation for training-free control in FLUX.

This experiment fixes a single target-aligned reference, builds a large reusable
reference set once, then varies only the subset size M used at inference time.

For each M and random subset repeat, it reports:
- CLIP target-vs-base directional score
- VLM target success rate
- LPIPS diversity (if available)

It also aggregates mean/std across subset repeats for each M.
"""

from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path
from statistics import mean
from typing import List, Sequence

import torch
from PIL import Image

import experiment_runtime as exp
import parser as cli_parser
import retrieval_guidance_core as poc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    return cli_parser.config_args("Reference-set size ablation for FLUX", "ablate_reference_size.py")

# ---------------------------------------------------------------------------
# LPIPS Evaluation
# ---------------------------------------------------------------------------

def maybe_compute_lpips(image_paths: Sequence[Path], device: str):
    try:
        import lpips  # type: ignore
        import numpy as np
    except Exception:
        return None, {"lpips_available": False}

    if len(image_paths) < 2:
        return 0.0, {"lpips_available": True, "num_pairs": 0}

    model = lpips.LPIPS(net="alex").to(device)
    model.eval()

    tensors = []
    for path in image_paths:
        with Image.open(path) as image:
            arr = torch.from_numpy(np.array(image.convert("RGB")).astype("float32") / 255.0)
        tensor = arr.permute(2, 0, 1).unsqueeze(0).to(device)
        tensor = tensor * 2.0 - 1.0
        tensors.append(tensor)

    distances = []
    with torch.no_grad():
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                distances.append(float(model(tensors[i], tensors[j]).item()))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return (float(mean(distances)) if distances else 0.0), {
        "lpips_available": True,
        "num_pairs": len(distances),
    }

# ---------------------------------------------------------------------------
# Reference Construction
# ---------------------------------------------------------------------------

def build_or_load_full_reference(pipe, args: argparse.Namespace, out_dir: Path, device: str):
    reference_prompts = [args.target_prompt] * args.reference_size
    reference_dir = out_dir / "references" / "full_target_reference"
    reference_label = f"{args.target_label} reference"
    reference_args = exp.override_args(args, reference_size=args.reference_size)
    cfg = exp.make_runtime_cfg(
        reference_args,
        prompt=args.target_prompt,
        reference_prompt=reference_label,
        out_dir=reference_dir,
        seed=args.reference_seed,
    )
    latents, meta = exp.load_or_build_reference(pipe, cfg, reference_prompts, device)
    return latents, meta, reference_dir

# ---------------------------------------------------------------------------
# Subset Sampling
# ---------------------------------------------------------------------------

def effective_repeat_count(size: int, full_reference_size: int, num_subsets: int) -> int:
    return 1 if size >= full_reference_size else num_subsets


def choose_subset_indices(full_reference_size: int, size: int, subset_seed: int, repeat_index: int) -> List[int]:
    if size > full_reference_size:
        raise ValueError(f"subset size {size} exceeds full reference size {full_reference_size}")
    if size == full_reference_size:
        return list(range(full_reference_size))
    rng = random.Random(subset_seed + size * 1009 + repeat_index * 9176)
    indices = rng.sample(range(full_reference_size), size)
    indices.sort()
    return indices


# ---------------------------------------------------------------------------
# Generation Phase
# ---------------------------------------------------------------------------

def run_generation_phase(
    pipe,
    args: argparse.Namespace,
    out_dir: Path,
    device: str,
    dtype: torch.dtype,
    full_reference_latents: torch.Tensor,
    full_reference_meta: dict,
    reference_dir: Path,
    seeds: List[int],
) -> dict:
    overall = {
        "prompt": args.prompt,
        "target_prompt": args.target_prompt,
        "target_label": args.target_label,
        "subset_sizes": args.subset_sizes,
        "num_subsets_requested": args.num_subsets,
        "seeds": seeds,
        "full_reference_dir": str(reference_dir),
        "full_reference_meta": full_reference_meta,
        "generation": {
            "model_id": args.model_id,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "guidance_strength": args.guidance_strength,
            "beta_schedule": args.beta_schedule,
            "guidance_start_frac": args.guidance_start_frac,
            "guidance_end_frac": args.guidance_end_frac,
            "topk": args.topk,
        },
        "sizes": {},
    }

    for size in args.subset_sizes:
        size_slug = f"M_{size:03d}"
        size_dir = out_dir / size_slug
        repeat_count = effective_repeat_count(size, args.reference_size, args.num_subsets)
        size_summary = {
            "subset_size": size,
            "num_subsets": repeat_count,
            "repeats": {},
        }

        for repeat_index in range(repeat_count):
            repeat_slug = f"repeat_{repeat_index:02d}"
            repeat_dir = size_dir / repeat_slug
            seeds_dir = repeat_dir / "seeds"
            histories_dir = repeat_dir / "seed_histories"
            seeds_dir.mkdir(parents=True, exist_ok=True)
            histories_dir.mkdir(parents=True, exist_ok=True)

            subset_indices = choose_subset_indices(args.reference_size, size, args.subset_seed, repeat_index)
            subset_latents = full_reference_latents[subset_indices].clone()
            subset_meta = {
                "subset_size": size,
                "repeat_index": repeat_index,
                "subset_seed": args.subset_seed,
                "subset_indices": subset_indices,
                "source_reference_dir": str(reference_dir),
                "source_reference_size": args.reference_size,
                "reference_prompt": full_reference_meta.get("reference_prompt"),
            }
            exp.save_json(subset_meta, repeat_dir / "subset_metadata.json")

            generated = []
            run_args = exp.override_args(args, reference_size=size)
            subset_latents_device = subset_latents.to(device=device, dtype=dtype)
            for sample_seed in seeds:
                cfg = exp.make_runtime_cfg(
                    run_args,
                    prompt=args.prompt,
                    reference_prompt=f"{args.target_label} reference subset",
                    out_dir=repeat_dir,
                    seed=sample_seed,
                )
                image, callback = exp.generate_with_rmg(
                    pipe=pipe,
                    cfg=cfg,
                    device=device,
                    reference_latents=subset_latents_device,
                    beta_schedule=args.beta_schedule,
                )
                image_path = seeds_dir / f"{sample_seed:04d}_generated.png"
                poc.save_pil(image, str(image_path))
                exp.save_callback_history(callback.history, histories_dir / f"{sample_seed:04d}_callback_history.txt")
                generated.append({"seed": sample_seed, "image_path": str(image_path)})

            size_summary["repeats"][repeat_slug] = {
                "subset_indices": subset_indices,
                "images": generated,
            }

        overall["sizes"][size_slug] = size_summary

    exp.save_json(overall, out_dir / "generation_manifest.json")
    return overall


# ---------------------------------------------------------------------------
# Evaluation Phase
# ---------------------------------------------------------------------------

def evaluate_image(
    image_path: Path,
    args: argparse.Namespace,
    device: str,
    clip_processor,
    clip_model,
    text_features: torch.Tensor,
    vlm_processor,
    vlm_model,
) -> dict:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        image_inputs = clip_processor(images=image_rgb, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze(0)
        positive_sim = float(sims[0].item())
        negative_sim = float(sims[1].item())

        raw_answer = ""
        prediction = "unknown"
        if vlm_model is not None and vlm_processor is not None:
            raw_answer = exp.generate_vlm_answer(
                vlm_model,
                vlm_processor,
                image_rgb,
                args.target_question,
                device,
                args.max_vlm_new_tokens,
            )
            prediction = exp.normalize_yes_no(raw_answer)

    target_score = positive_sim - negative_sim
    return {
        "clip_positive_sim": positive_sim,
        "clip_negative_sim": negative_sim,
        "target_score": target_score,
        "vlm_raw_answer": raw_answer,
        "target": 1 if prediction == "yes" else 0,
        "not_target": 1 if prediction == "no" else 0,
        "unknown": 1 if prediction == "unknown" else 0,
    }


def run_evaluation_phase(manifest_path: Path, args: argparse.Namespace, out_dir: Path, device: str, dtype: torch.dtype) -> dict:
    overall = exp.load_json(manifest_path)

    # Imports are deferred until after the generation pipeline is deleted so GPU
    # memory can be reclaimed before loading CLIP and the VLM.
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, Qwen2VLForConditionalGeneration

    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    clip_model.eval()
    text_inputs = clip_processor(text=[args.positive_text, args.negative_text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    vlm_model = None
    vlm_processor = None
    if not args.skip_vlm:
        vlm_processor = AutoProcessor.from_pretrained(args.vlm_model_id)
        vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vlm_model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            vlm_model = vlm_model.to(device)
        vlm_model.eval()

    for size_slug, size_summary in overall["sizes"].items():
        for repeat_slug, repeat_summary in size_summary["repeats"].items():
            image_paths = []
            target_scores = []
            yes_count = 0
            no_count = 0
            unknown_count = 0

            for info in repeat_summary["images"]:
                image_path = Path(info["image_path"])
                image_paths.append(image_path)
                metrics = evaluate_image(
                    image_path=image_path,
                    args=args,
                    device=device,
                    clip_processor=clip_processor,
                    clip_model=clip_model,
                    text_features=text_features,
                    vlm_processor=vlm_processor,
                    vlm_model=vlm_model,
                )
                info.update(metrics)
                target_scores.append(metrics["target_score"])
                yes_count += metrics["target"]
                no_count += metrics["not_target"]
                unknown_count += metrics["unknown"]

            target_mean, target_std = exp.summarize(target_scores)
            if args.skip_lpips:
                lpips_value = None
                lpips_meta = {"lpips_available": False, "skipped": True}
            else:
                lpips_value, lpips_meta = maybe_compute_lpips(image_paths, device)

            repeat_summary.update(
                {
                    "positive_text": args.positive_text,
                    "negative_text": args.negative_text,
                    "target_question": args.target_question,
                    "average_target_score": target_mean,
                    "std_target_score": target_std,
                    "target_success_rate": yes_count / len(image_paths) if image_paths else 0.0,
                    "number_of_target_classified": yes_count,
                    "number_of_not_target_classified": no_count,
                    "number_of_unknown": unknown_count,
                    "lpips_diversity": lpips_value,
                    "lpips_meta": lpips_meta,
                }
            )

            exp.save_json(repeat_summary, out_dir / size_slug / repeat_slug / "summary.json")

    return overall


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(overall: dict) -> None:
    for size_summary in overall["sizes"].values():
        repeat_success_rates = []
        repeat_target_scores = []
        repeat_lpips_values = []

        for repeat_summary in size_summary["repeats"].values():
            repeat_success_rates.append(repeat_summary["target_success_rate"])
            repeat_target_scores.append(repeat_summary["average_target_score"])
            lpips_value = repeat_summary["lpips_diversity"]
            if lpips_value is not None:
                repeat_lpips_values.append(lpips_value)

        success_mean, success_std = exp.summarize(repeat_success_rates)
        score_mean, score_std = exp.summarize(repeat_target_scores)
        lpips_mean, lpips_std = exp.summarize(repeat_lpips_values)

        size_summary["aggregate"] = {
            "target_success_rate_mean": success_mean,
            "target_success_rate_std": success_std,
            "average_target_score_mean": score_mean,
            "average_target_score_std": score_std,
            "lpips_diversity_mean": lpips_mean,
            "lpips_diversity_std": lpips_std,
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.num_subsets <= 0:
        raise ValueError("--num-subsets must be positive")
    if any(size > args.reference_size for size in args.subset_sizes):
        raise ValueError("subset_sizes cannot exceed reference_size")

    poc.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("device:", device)
    print("dtype:", dtype)
    print("output dir:", out_dir)

    pipe = poc.build_pipe(args.model_id, dtype=dtype, device=device)
    full_reference_latents, full_reference_meta, reference_dir = build_or_load_full_reference(pipe, args, out_dir, device)
    seeds = [args.seed + i * args.seed_stride for i in range(args.num_samples)]
    run_generation_phase(
        pipe=pipe,
        args=args,
        out_dir=out_dir,
        device=device,
        dtype=dtype,
        full_reference_latents=full_reference_latents,
        full_reference_meta=full_reference_meta,
        reference_dir=reference_dir,
        seeds=seeds,
    )

    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    overall = run_evaluation_phase(out_dir / "generation_manifest.json", args, out_dir, device, dtype)
    aggregate_results(overall)
    overall.update(
        {
            "clip_model_id": args.clip_model_id,
            "vlm_model_id": None if args.skip_vlm else args.vlm_model_id,
            "question": None if args.skip_vlm else args.target_question,
        }
    )
    exp.save_json(overall, out_dir / "ablate_reference_size_summary.json")


if __name__ == "__main__":
    main()
