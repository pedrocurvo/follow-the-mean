#!/usr/bin/env python3
"""
Score prompt-reference interaction RMG-guided images with CLIP.

Default metric:
    score(image) = sim(image, positive_text) - sim(image, negative_text)

This is intended as a simple continuous measure of an injected attribute,
for example:
    positive_text = "a pink elephant in a jungle"
    negative_text = "an elephant in a jungle"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import experiment_runtime as exp
import parser as cli_parser
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = cli_parser.argument_parser("CLIP scoring for prompt-reference interaction outputs")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="runs/ablation_prompt_reference_pink_elephant/prompt_reference_interaction",
        help="Directory containing prompt_reference_interaction case folders",
    )
    parser.add_argument(
        "--clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model id",
    )
    parser.add_argument(
        "--positive-text",
        type=str,
        default="a pink elephant in a jungle",
        help="Positive text prompt for CLIP similarity",
    )
    parser.add_argument(
        "--negative-text",
        type=str,
        default="a gray elephant in a jungle",
        help="Negative text prompt for CLIP similarity",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional explicit output JSON path",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional explicit output CSV path",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Case Discovery
# ---------------------------------------------------------------------------


def load_cases(input_dir: Path) -> List[dict]:
    cases = []
    seen_case_dirs = set()
    for image_path in sorted(input_dir.glob("prompt_*_reference_*/rmg_guided.png")):
        metadata_path = image_path.parent / "run_metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = exp.load_json(metadata_path)
        cases.append(
            {
                "image_path": str(image_path),
                "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                "prompt": metadata.get("prompt", ""),
                "reference_label": metadata.get("reference_label", ""),
                "case_dir": str(image_path.parent),
            }
        )
        seen_case_dirs.add(str(image_path.parent))

    # Add one baseline per prompt row from the corresponding reference case directory.
    for baseline_path in sorted(input_dir.glob("prompt_*_reference_*/baseline.png")):
        case_dir = str(baseline_path.parent)
        if case_dir not in seen_case_dirs:
            continue
        metadata_path = baseline_path.parent / "run_metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = exp.load_json(metadata_path)
        cases.append(
            {
                "image_path": str(baseline_path),
                "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                "prompt": metadata.get("prompt", ""),
                "reference_label": "no reference set",
                "case_dir": str(baseline_path.parent),
            }
        )
    return cases


# ---------------------------------------------------------------------------
# CLIP Evaluation
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    cases = load_cases(input_dir)
    if not cases:
        raise FileNotFoundError(f"No rmg_guided.png files found under: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    model.eval()

    text_inputs = processor(
        text=[args.positive_text, args.negative_text],
        return_tensors="pt",
        padding=True,
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = []
    for case in cases:
        with Image.open(case["image_path"]) as image:
            image_inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ text_features.T).squeeze(0)
        positive_sim = float(sims[0].item())
        negative_sim = float(sims[1].item())
        score = positive_sim - negative_sim

        rows.append(
            {
                **case,
                "positive_text": args.positive_text,
                "negative_text": args.negative_text,
                "positive_sim": positive_sim,
                "negative_sim": negative_sim,
                "score_diff": score,
            }
        )

    rows.sort(key=lambda row: row["score_diff"], reverse=True)

    out_json = Path(args.out_json) if args.out_json else input_dir / "clip_scores.json"
    out_csv = Path(args.out_csv) if args.out_csv else input_dir / "clip_scores.csv"

    exp.save_json(
        {
            "clip_model_id": args.clip_model_id,
            "positive_text": args.positive_text,
            "negative_text": args.negative_text,
            "rows": rows,
        },
        out_json,
    )

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_dir",
                "prompt",
                "reference_label",
                "image_path",
                "metadata_path",
                "positive_text",
                "negative_text",
                "positive_sim",
                "negative_sim",
                "score_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
