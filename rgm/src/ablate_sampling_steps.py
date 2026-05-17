#!/usr/bin/env python3
"""
NFE ablation for control-anatomy with a fixed image reference.

This experiment keeps the prompt, reference, seed, and guidance setup fixed while
varying only the number of inference steps (NFE). For each NFE it reports:
- ring-leap success (VLM yes/no)
- artifact presence (VLM yes/no)
- optional pose quality score (1-5)
- runtime
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Optional

import experiment_runtime as exp
import parser as cli_parser
import retrieval_guidance_core as poc
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    return cli_parser.config_args(
        "NFE ablation for FLUX control-anatomy", "ablate_sampling_steps.py"
    )


# ---------------------------------------------------------------------------
# VLM Evaluation
# ---------------------------------------------------------------------------


def parse_pose_score(text: str) -> Optional[int]:
    stripped = text.strip()
    digits = [ch for ch in stripped if ch.isdigit()]
    if not digits:
        return None
    value = int(digits[0])
    if 1 <= value <= 5:
        return value
    return None


# ---------------------------------------------------------------------------
# Generation Phase
# ---------------------------------------------------------------------------


def run_generation_phase(
    pipe,
    args: argparse.Namespace,
    out_dir: Path,
    image_dir: Path,
    image_paths: list[Path],
    device: str,
    dtype: torch.dtype,
) -> dict:
    reference_label = f"image reference: {image_dir.name}"
    reference_id = exp.make_image_reference_id(args, image_paths)
    image_reference_dir = out_dir / "references" / reference_id
    reference_cfg = exp.make_runtime_cfg(
        args,
        prompt=args.prompt,
        reference_prompt=reference_label,
        out_dir=image_reference_dir,
        seed=args.reference_seed,
    )
    reference_latents, reference_meta = exp.load_or_build_image_reference(
        image_paths=image_paths,
        pipe=pipe,
        cfg=reference_cfg,
        image_reference_dir=image_reference_dir,
        device=device,
    )
    reference_latents_device = reference_latents.to(device=device, dtype=dtype)

    overall = {
        "prompt": args.prompt,
        "seed": args.seed,
        "case_slug": args.case_slug,
        "control_anatomy_image_dir": str(image_dir),
        "reference_meta": reference_meta,
        "nfe_values": args.nfe_values,
        "guidance_strength_values": args.guidance_strength_values,
        "generation": {
            "model_id": args.model_id,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
            "beta_schedule": args.beta_schedule,
            "guidance_start_frac": args.guidance_start_frac,
            "guidance_end_frac": args.guidance_end_frac,
            "topk": args.topk,
        },
        "strengths": {},
    }

    for guidance_strength in args.guidance_strength_values:
        strength_slug = f"b0_{str(guidance_strength).replace('.', 'p')}"
        strength_dir = out_dir / strength_slug
        strength_summary = {
            "guidance_strength": guidance_strength,
            "runs": {},
        }

        for nfe in args.nfe_values:
            run_slug = f"nfe_{nfe:03d}"
            run_dir = strength_dir / run_slug
            run_dir.mkdir(parents=True, exist_ok=True)

            run_args = exp.override_args(
                args, num_inference_steps=nfe, guidance_strength=guidance_strength
            )
            cfg = exp.make_runtime_cfg(
                run_args,
                prompt=args.prompt,
                reference_prompt=reference_label,
                out_dir=run_dir,
                seed=args.seed,
            )

            baseline_start = time.perf_counter()
            baseline = poc.generate_single_image(pipe, cfg.prompt, cfg.seed + 7, cfg, device)
            baseline_runtime_seconds = time.perf_counter() - baseline_start
            baseline_image_path = run_dir / "baseline.png"
            poc.save_pil(baseline, str(baseline_image_path))

            start = time.perf_counter()
            image, callback = exp.generate_with_rmg(
                pipe=pipe,
                cfg=cfg,
                device=device,
                reference_latents=reference_latents_device,
                beta_schedule=args.beta_schedule,
            )
            runtime_seconds = time.perf_counter() - start

            image_path = run_dir / "rmg_guided.png"
            poc.save_pil(image, str(image_path))
            exp.save_callback_history(callback.history, run_dir / "callback_history.txt")

            strength_summary["runs"][run_slug] = {
                "nfe": nfe,
                "guidance_strength": guidance_strength,
                "baseline_runtime_seconds": baseline_runtime_seconds,
                "runtime_seconds": runtime_seconds,
                "baseline_image_path": str(baseline_image_path),
                "image_path": str(image_path),
                "runtime_cfg": vars(cfg),
            }

        overall["strengths"][strength_slug] = strength_summary

    exp.save_json(overall, out_dir / "generation_manifest.json")
    return overall


# ---------------------------------------------------------------------------
# Evaluation Phase
# ---------------------------------------------------------------------------


def evaluate_run_images(
    run_summary: dict, args: argparse.Namespace, device: str, vlm_model, vlm_processor
) -> None:
    baseline_success_raw = ""
    baseline_artifact_raw = ""
    baseline_pose_raw = ""
    guided_success_raw = ""
    guided_artifact_raw = ""
    guided_pose_raw = ""
    baseline_success_pred = "unknown"
    baseline_artifact_pred = "unknown"
    guided_success_pred = "unknown"
    guided_artifact_pred = "unknown"
    baseline_pose_score = None
    guided_pose_score = None

    if vlm_model is not None and vlm_processor is not None:
        with Image.open(run_summary["baseline_image_path"]) as image:
            image_rgb = image.convert("RGB")
            baseline_success_raw = exp.generate_vlm_answer(
                vlm_model, vlm_processor, image_rgb, args.success_question, device, 8
            )
            baseline_artifact_raw = exp.generate_vlm_answer(
                vlm_model, vlm_processor, image_rgb, args.artifact_question, device, 8
            )
            baseline_success_pred = exp.normalize_yes_no(baseline_success_raw)
            baseline_artifact_pred = exp.normalize_yes_no(baseline_artifact_raw)
            if not args.skip_pose_score:
                baseline_pose_raw = exp.generate_vlm_answer(
                    vlm_model, vlm_processor, image_rgb, args.pose_score_question, device, 8
                )
                baseline_pose_score = parse_pose_score(baseline_pose_raw)

        with Image.open(run_summary["image_path"]) as image:
            image_rgb = image.convert("RGB")
            guided_success_raw = exp.generate_vlm_answer(
                vlm_model, vlm_processor, image_rgb, args.success_question, device, 8
            )
            guided_artifact_raw = exp.generate_vlm_answer(
                vlm_model, vlm_processor, image_rgb, args.artifact_question, device, 8
            )
            guided_success_pred = exp.normalize_yes_no(guided_success_raw)
            guided_artifact_pred = exp.normalize_yes_no(guided_artifact_raw)
            if not args.skip_pose_score:
                guided_pose_raw = exp.generate_vlm_answer(
                    vlm_model, vlm_processor, image_rgb, args.pose_score_question, device, 8
                )
                guided_pose_score = parse_pose_score(guided_pose_raw)

    run_summary.update(
        {
            "success_question": args.success_question,
            "artifact_question": args.artifact_question,
            "pose_score_question": None if args.skip_pose_score else args.pose_score_question,
            "baseline_success_vlm_raw_answer": baseline_success_raw,
            "baseline_artifact_vlm_raw_answer": baseline_artifact_raw,
            "baseline_pose_score_vlm_raw_answer": baseline_pose_raw,
            "guided_success_vlm_raw_answer": guided_success_raw,
            "guided_artifact_vlm_raw_answer": guided_artifact_raw,
            "guided_pose_score_vlm_raw_answer": guided_pose_raw,
            "baseline_ring_leap_success": 1 if baseline_success_pred == "yes" else 0,
            "baseline_ring_leap_failure": 1 if baseline_success_pred == "no" else 0,
            "baseline_ring_leap_unknown": 1 if baseline_success_pred == "unknown" else 0,
            "guided_ring_leap_success": 1 if guided_success_pred == "yes" else 0,
            "guided_ring_leap_failure": 1 if guided_success_pred == "no" else 0,
            "guided_ring_leap_unknown": 1 if guided_success_pred == "unknown" else 0,
            "baseline_artifact_present": 1 if baseline_artifact_pred == "yes" else 0,
            "baseline_artifact_absent": 1 if baseline_artifact_pred == "no" else 0,
            "baseline_artifact_unknown": 1 if baseline_artifact_pred == "unknown" else 0,
            "guided_artifact_present": 1 if guided_artifact_pred == "yes" else 0,
            "guided_artifact_absent": 1 if guided_artifact_pred == "no" else 0,
            "guided_artifact_unknown": 1 if guided_artifact_pred == "unknown" else 0,
            "baseline_pose_score": baseline_pose_score,
            "guided_pose_score": guided_pose_score,
            "delta_ring_leap_success": (1 if guided_success_pred == "yes" else 0)
            - (1 if baseline_success_pred == "yes" else 0),
            "delta_artifact_present": (1 if guided_artifact_pred == "yes" else 0)
            - (1 if baseline_artifact_pred == "yes" else 0),
            "delta_pose_score": None
            if baseline_pose_score is None or guided_pose_score is None
            else guided_pose_score - baseline_pose_score,
        }
    )


def run_evaluation_phase(
    manifest_path: Path, args: argparse.Namespace, out_dir: Path, device: str, dtype: torch.dtype
) -> dict:
    overall = exp.load_json(manifest_path)

    vlm_model = None
    vlm_processor = None
    if not args.skip_vlm:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        vlm_processor = AutoProcessor.from_pretrained(args.vlm_model_id)
        vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vlm_model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            vlm_model = vlm_model.to(device)
        vlm_model.eval()

    for strength_slug, strength_summary in overall["strengths"].items():
        baseline_guided_runtime = None
        baseline_baseline_runtime = None
        for run_slug, run_summary in strength_summary["runs"].items():
            if baseline_guided_runtime is None:
                baseline_guided_runtime = run_summary["runtime_seconds"]
            if baseline_baseline_runtime is None:
                baseline_baseline_runtime = run_summary["baseline_runtime_seconds"]

            evaluate_run_images(run_summary, args, device, vlm_model, vlm_processor)
            run_summary["relative_guided_runtime"] = (
                (run_summary["runtime_seconds"] / baseline_guided_runtime)
                if baseline_guided_runtime
                else None
            )
            run_summary["relative_baseline_runtime"] = (
                (run_summary["baseline_runtime_seconds"] / baseline_baseline_runtime)
                if baseline_baseline_runtime
                else None
            )
            exp.save_json(run_summary, out_dir / strength_slug / run_slug / "summary.json")

    return overall


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(overall: dict) -> None:
    for strength_summary in overall["strengths"].values():
        guided_runtime_values = []
        baseline_runtime_values = []
        guided_success_values = []
        baseline_success_values = []
        guided_artifact_values = []
        baseline_artifact_values = []
        guided_pose_values = []
        baseline_pose_values = []

        for run_summary in strength_summary["runs"].values():
            guided_runtime_values.append(run_summary["runtime_seconds"])
            baseline_runtime_values.append(run_summary["baseline_runtime_seconds"])
            guided_success_values.append(run_summary["guided_ring_leap_success"])
            baseline_success_values.append(run_summary["baseline_ring_leap_success"])
            guided_artifact_values.append(run_summary["guided_artifact_present"])
            baseline_artifact_values.append(run_summary["baseline_artifact_present"])
            if run_summary["guided_pose_score"] is not None:
                guided_pose_values.append(float(run_summary["guided_pose_score"]))
            if run_summary["baseline_pose_score"] is not None:
                baseline_pose_values.append(float(run_summary["baseline_pose_score"]))

        guided_runtime_mean, guided_runtime_std = exp.summarize(guided_runtime_values)
        baseline_runtime_mean, baseline_runtime_std = exp.summarize(baseline_runtime_values)
        guided_success_mean, guided_success_std = exp.summarize(guided_success_values)
        baseline_success_mean, baseline_success_std = exp.summarize(baseline_success_values)
        guided_artifact_mean, guided_artifact_std = exp.summarize(guided_artifact_values)
        baseline_artifact_mean, baseline_artifact_std = exp.summarize(baseline_artifact_values)
        guided_pose_mean, guided_pose_std = exp.summarize(guided_pose_values)
        baseline_pose_mean, baseline_pose_std = exp.summarize(baseline_pose_values)

        strength_summary["aggregate"] = {
            "guided_runtime_seconds_mean": guided_runtime_mean,
            "guided_runtime_seconds_std": guided_runtime_std,
            "baseline_runtime_seconds_mean": baseline_runtime_mean,
            "baseline_runtime_seconds_std": baseline_runtime_std,
            "guided_ring_leap_success_mean": guided_success_mean,
            "guided_ring_leap_success_std": guided_success_std,
            "baseline_ring_leap_success_mean": baseline_success_mean,
            "baseline_ring_leap_success_std": baseline_success_std,
            "guided_artifact_present_mean": guided_artifact_mean,
            "guided_artifact_present_std": guided_artifact_std,
            "baseline_artifact_present_mean": baseline_artifact_mean,
            "baseline_artifact_present_std": baseline_artifact_std,
            "guided_pose_score_mean": guided_pose_mean,
            "guided_pose_score_std": guided_pose_std,
            "baseline_pose_score_mean": baseline_pose_mean,
            "baseline_pose_score_std": baseline_pose_std,
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    poc.set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.control_anatomy_image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Control-anatomy image directory does not exist: {image_dir}")
    image_paths = exp.list_image_paths(image_dir)
    if not image_paths:
        raise ValueError(
            f"No supported images found in control-anatomy image directory: {image_dir}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("device:", device)
    print("dtype:", dtype)
    print("output dir:", out_dir)

    pipe = poc.build_pipe(args.model_id, dtype=dtype, device=device)
    run_generation_phase(pipe, args, out_dir, image_dir, image_paths, device, dtype)

    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    overall = run_evaluation_phase(
        out_dir / "generation_manifest.json", args, out_dir, device, dtype
    )
    aggregate_results(overall)
    overall["vlm_model_id"] = None if args.skip_vlm else args.vlm_model_id
    exp.save_json(overall, out_dir / "ablate_sampling_steps_summary.json")


if __name__ == "__main__":
    main()
