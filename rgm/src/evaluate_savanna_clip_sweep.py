#!/usr/bin/env python3
"""
Generate multiple seeds for each savanna_object controllability reference mix and
score each output with CLIP for zebra and giraffe similarity.

This reuses the existing mixed references from:
    outputs_flux2_training_free_control/controllability/savanna_object/mix_*/

and writes a nested JSON summary with per-image and aggregate statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import experiment_runtime as exp
import parser as cli_parser
import retrieval_guidance_core as poc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = cli_parser.argument_parser("Savanna-object CLIP sweep over reference mixes")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="runs/ablation_reference_composition/controllability/savanna_object",
        help="Directory containing mix_*/run_metadata.json from the savanna_object controllability experiment",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/savanna_object_clip_sweep",
        help="Directory to save generated images and JSON summary",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--guidance-strength", type=float, default=0.2)
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="bell",
        choices=["constant", "bell", "quadratic-decay"],
    )
    parser.add_argument("--guidance-start-frac", type=float, default=0.15)
    parser.add_argument("--guidance-end-frac", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--zebra-text",
        type=str,
        default="a zebra in a savanna",
    )
    parser.add_argument(
        "--giraffe-text",
        type=str,
        default="a giraffe in a savanna",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Input Discovery
# ---------------------------------------------------------------------------

def load_mix_metadata(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("mix_*/run_metadata.json")):
        rows.append(exp.load_json(path))
    if not rows:
        raise FileNotFoundError(f"No mix_*/run_metadata.json files found under: {input_dir}")
    return rows


def load_shared_reference(reference_dir: Path):
    cache_path = reference_dir / "reference_cache.pt"
    if not cache_path.exists():
        cache_path = reference_dir / "bank_cache.pt"
    reference_latents, reference_meta = poc.load_reference_cache(cache_path)
    return reference_latents, reference_meta

# ---------------------------------------------------------------------------
# CLIP Evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    poc.set_seed(args.seed)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("device:", device)
    print("dtype:", dtype)
    print("input dir:", input_dir)
    print("output dir:", out_dir)

    mix_rows = load_mix_metadata(input_dir)
    pipe = poc.build_pipe(args.model_id, dtype=dtype, device=device)

    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    clip_model.eval()

    text_inputs = processor(
        text=[args.zebra_text, args.giraffe_text],
        return_tensors="pt",
        padding=True,
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    experiment_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "prompt": "an animal in a savanna",
        "clip_model_id": args.clip_model_id,
        "zebra_text": args.zebra_text,
        "giraffe_text": args.giraffe_text,
        "num_samples": args.num_samples,
        "beta_schedule": args.beta_schedule,
        "guidance_strength": args.guidance_strength,
        "topk": args.topk,
        "mixes": {},
    }

    for mix_meta in mix_rows:
        mix_slug = Path(mix_meta["runtime_cfg"]["out_dir"]).name
        mix_dir = out_dir / mix_slug
        mix_dir.mkdir(parents=True, exist_ok=True)
        shared_reference_dir = Path(mix_meta["shared_reference_dir"])
        reference_latents, reference_meta = load_shared_reference(shared_reference_dir)

        prompt = mix_meta["prompt"]
        reference_label = mix_meta["reference_label"]
        seeds_dir = mix_dir / "seeds"
        histories_dir = mix_dir / "seed_histories"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        histories_dir.mkdir(parents=True, exist_ok=True)
        runtime_args = exp.override_args(
            args,
            reference_size=len(mix_meta.get("reference_prompts", [])),
            callback_verbose=False,
            reuse_reference=True,
            reference_cache_path="",
        )

        per_image = []
        zebra_scores = []
        giraffe_scores = []
        zebra_count = 0
        giraffe_count = 0

        for sample_idx in range(args.num_samples):
            sample_seed = args.seed + sample_idx * args.seed_stride
            sample_cfg = exp.make_runtime_cfg(
                runtime_args,
                prompt=prompt,
                reference_prompt=reference_label,
                out_dir=mix_dir,
                seed=sample_seed,
            )

            guided, callback = exp.generate_with_rmg(
                pipe=pipe,
                cfg=sample_cfg,
                device=device,
                reference_latents=reference_latents.to(device=device, dtype=dtype),
                beta_schedule=args.beta_schedule,
            )
            image_path = seeds_dir / f"{sample_seed:04d}_rmg_guided.png"
            poc.save_pil(guided, str(image_path))
            exp.save_callback_history(callback.history, histories_dir / f"{sample_seed:04d}_callback_history.txt")

            with Image.open(image_path) as image:
                image_inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            sims = (image_features @ text_features.T).squeeze(0)
            zebra_score = float(sims[0].item())
            giraffe_score = float(sims[1].item())
            zebra_cls = 1 if zebra_score > giraffe_score else 0
            giraffe_cls = 1 if giraffe_score > zebra_score else 0

            zebra_scores.append(zebra_score)
            giraffe_scores.append(giraffe_score)
            zebra_count += zebra_cls
            giraffe_count += giraffe_cls

            per_image.append({
                "seed": sample_seed,
                "image_path": str(image_path),
                "clip_score_zebra": zebra_score,
                "clip_score_giraffe": giraffe_score,
                "zebra": zebra_cls,
                "giraffe": giraffe_cls,
            })

        zebra_mean, zebra_std = exp.summarize(zebra_scores)
        giraffe_mean, giraffe_std = exp.summarize(giraffe_scores)
        mix_summary = {
            "images": per_image,
            "prompt": prompt,
            "reference_label": reference_label,
            "shared_reference_dir": str(shared_reference_dir),
            "reference_meta": reference_meta,
            "average_clip_score_zebra": zebra_mean,
            "average_clip_score_giraffe": giraffe_mean,
            "std_clip_score_zebra": zebra_std,
            "std_clip_score_giraffe": giraffe_std,
            "number_of_zebra_classified": zebra_count,
            "number_of_giraffe_classified": giraffe_count,
            "num_ties": args.num_samples - zebra_count - giraffe_count,
        }
        experiment_summary["mixes"][mix_slug] = mix_summary
        exp.save_json(mix_summary, mix_dir / "summary.json")

    exp.save_json(experiment_summary, out_dir / "savanna_object_clip_sweep.json")


if __name__ == "__main__":
    main()
