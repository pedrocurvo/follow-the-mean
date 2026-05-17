#!/usr/bin/env python3
"""
Classify existing savanna_object sweep images with a VLM as zebra vs giraffe.

For each image, asks:
    Which animal does the main animal in this image most resemble: zebra or giraffe?
    Answer with exactly one word: zebra or giraffe.

Writes per-mix JSON summaries with per-image predictions and aggregate rates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import experiment_runtime as exp
import parser as cli_parser
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

QUESTION = (
    "Which animal does the main animal in this image most resemble: zebra or giraffe? "
    "Answer with exactly one word: zebra or giraffe."
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = cli_parser.argument_parser("VLM zebra/giraffe classification for savanna-object sweep")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="runs/savanna_object_clip_sweep",
        help="Directory containing mix_*/seeds/*_rmg_guided.png",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional explicit output JSON path",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="VLM model id used for classification",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# VLM Evaluation
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    has_zebra = "zebra" in lowered
    has_giraffe = "giraffe" in lowered
    if has_zebra and not has_giraffe:
        return "zebra"
    if has_giraffe and not has_zebra:
        return "giraffe"
    return "unknown"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("device:", device)
    print("dtype:", dtype)
    print("input dir:", input_dir)
    print("model id:", args.model_id)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    experiment_summary = {
        "input_dir": str(input_dir),
        "model_id": args.model_id,
        "question": QUESTION,
        "mixes": {},
    }

    for mix_dir in sorted(input_dir.glob("mix_*")):
        seeds_dir = mix_dir / "seeds"
        if not seeds_dir.exists():
            continue

        rows = []
        zebra_count = 0
        giraffe_count = 0
        unknown_count = 0

        image_paths = sorted(seeds_dir.glob("*_rmg_guided.png"))
        for image_path in image_paths:
            seed_str = image_path.name.split("_")[0]
            seed = int(seed_str)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                raw_answer = exp.generate_vlm_answer(
                    model, processor, image, QUESTION, device, args.max_new_tokens
                )

            answer = normalize_answer(raw_answer)
            zebra = 1 if answer == "zebra" else 0
            giraffe = 1 if answer == "giraffe" else 0
            if answer == "zebra":
                zebra_count += 1
            elif answer == "giraffe":
                giraffe_count += 1
            else:
                unknown_count += 1

            rows.append(
                {
                    "seed": seed,
                    "image_path": str(image_path),
                    "raw_answer": raw_answer,
                    "prediction": answer,
                    "zebra": zebra,
                    "giraffe": giraffe,
                }
            )

        total = len(image_paths)
        mix_summary = {
            "images": rows,
            "ZebraRate": zebra_count / total if total else 0.0,
            "GiraffeRate": giraffe_count / total if total else 0.0,
            "UnknownRate": unknown_count / total if total else 0.0,
            "number_of_zebra_classified": zebra_count,
            "number_of_giraffe_classified": giraffe_count,
            "number_of_unknown": unknown_count,
            "num_images": total,
        }
        experiment_summary["mixes"][mix_dir.name] = mix_summary
        exp.save_json(mix_summary, mix_dir / "vlm_summary.json")

    out_json = (
        Path(args.out_json) if args.out_json else input_dir / "savanna_object_vlm_summary.json"
    )
    exp.save_json(experiment_summary, out_json)


if __name__ == "__main__":
    main()
