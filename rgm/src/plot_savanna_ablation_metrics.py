#!/usr/bin/env python3
"""
Plot savanna-object controllability metrics from CLIP and VLM summaries.

Produces:
1. giraffe generation rate vs giraffe mix percentage
2. average CLIP zebra/giraffe score vs giraffe mix percentage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import parser as cli_parser

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = cli_parser.argument_parser("Plot savanna-object controllability metrics")
    parser.add_argument(
        "--clip-json",
        type=str,
        default="runs/savanna_object_clip_sweep/savanna_object_clip_sweep.json",
        help="Path to CLIP sweep summary JSON",
    )
    parser.add_argument(
        "--vlm-json",
        type=str,
        default="runs/savanna_object_clip_sweep/savanna_object_vlm_summary.json",
        help="Path to VLM summary JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/savanna_object_clip_sweep/plots",
        help="Directory to save the plots",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mix_to_fraction(mix_slug: str) -> float:
    return int(mix_slug.split("_")[1]) / 100.0

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    clip_data = load_json(Path(args.clip_json))
    vlm_data = load_json(Path(args.vlm_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixes = sorted(clip_data["mixes"].keys(), key=mix_to_fraction)
    x = [100.0 * mix_to_fraction(mix) for mix in mixes]

    clip_giraffe_rate = []
    vlm_giraffe_rate = []
    clip_zebra_score = []
    clip_giraffe_score = []

    for mix in mixes:
        clip_mix = clip_data["mixes"][mix]
        vlm_mix = vlm_data["mixes"][mix]

        total_clip = (
            clip_mix["number_of_zebra_classified"]
            + clip_mix["number_of_giraffe_classified"]
            + clip_mix["num_ties"]
        )
        clip_giraffe_rate.append(
            100.0 * clip_mix["number_of_giraffe_classified"] / total_clip if total_clip else 0.0
        )
        vlm_giraffe_rate.append(100.0 * vlm_mix["GiraffeRate"])
        clip_zebra_score.append(clip_mix["average_clip_score_zebra"])
        clip_giraffe_score.append(clip_mix["average_clip_score_giraffe"])

    colors = {
        "vlm": "#3A94C5",
        "clip_cls": "#D14D41",
        "zebra": "#879A39",
        "giraffe": "#CE5D97",
    }

    plt.figure(figsize=(9.4, 6.4))
    plt.plot(x, vlm_giraffe_rate, marker="o", linewidth=3.8, markersize=16, color=colors["vlm"], label="VLM")
    plt.plot(x, clip_giraffe_rate, marker="s", linewidth=3.8, markersize=16, color=colors["clip_cls"], label="CLIP")
    plt.xlabel("Reference composition\n(% target attribute)", fontsize=30)
    plt.ylabel("Output composition\n(% target attribute)", fontsize=30)
    plt.xticks(x, [f"{int(v)}" for v in x], fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(
        fontsize=22,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True,
        edgecolor="#B7B5AC",
        facecolor="white",
        framealpha=1.0,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(out_dir / "savanna_object_giraffe_rate.pdf")
    print(f"saved: {out_dir / 'savanna_object_giraffe_rate.pdf'}")
    plt.close()

    plt.figure(figsize=(9.4, 6.4))
    plt.plot(x, clip_zebra_score, marker="o", linewidth=3.8, markersize=16, color=colors["zebra"], label="Zebra")
    plt.plot(x, clip_giraffe_score, marker="s", linewidth=3.8, markersize=16, color=colors["giraffe"], label="Giraffe")
    plt.xlabel("Reference composition\n(% target attribute)", fontsize=30)
    plt.ylabel("Mean CLIP similarity\n(target attribute)", fontsize=30)
    plt.xticks(x, [f"{int(v)}" for v in x], fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(True, alpha=0.3)
    plt.legend(
        fontsize=22,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True,
        edgecolor="#B7B5AC",
        facecolor="white",
        framealpha=1.0,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(out_dir / "savanna_object_clip_scores.pdf")
    print(f"saved: {out_dir / 'savanna_object_clip_scores.pdf'}")
    plt.close()


if __name__ == "__main__":
    main()
