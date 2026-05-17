#!/usr/bin/env python3
"""
Plot dataset-size ablation metrics from an aggregated summary JSON.

Produces three figures:
1. average target score vs dataset size
2. LPIPS diversity vs dataset size
3. target success rate vs dataset size
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

import parser as cli_parser

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = cli_parser.argument_parser("Plot dataset-size ablation metrics")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="runs/ablation_dataset_size/ablate_reference_size_summary.json",
        help="Path to the dataset-size ablation summary JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/ablation_dataset_size/plots",
        help="Directory where the plots will be saved",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def size_key(size_slug: str) -> int:
    return int(size_slug.split("_")[1])


def extract_series(summary: dict) -> tuple[list[int], list[str], dict[str, list[float]]]:
    sizes = sorted(summary["sizes"].keys(), key=size_key)
    x = [size_key(size) for size in sizes]
    labels = [str(size_key(size)) for size in sizes]

    metrics = {
        "average_target_score_mean": [],
        "average_target_score_std": [],
        "lpips_diversity_mean": [],
        "lpips_diversity_std": [],
        "target_success_rate_mean": [],
        "target_success_rate_std": [],
    }

    for size in sizes:
        aggregate = summary["sizes"][size]["aggregate"]
        for key in metrics:
            metrics[key].append(aggregate[key])

    return x, labels, metrics

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric(
    x: list[int],
    labels: list[str],
    y: list[float],
    yerr: list[float],
    *,
    xlabel: str,
    ylabel: str,
    color: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9.2, 6.2))
    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=3.0,
        markersize=14,
        capsize=5,
        color=color,
        ecolor=color,
    )
    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.xscale("log", base=2)
    plt.xticks(x, labels, fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved: {out_path}")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    summary = load_json(Path(args.summary_json))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, labels, metrics = extract_series(summary)

    plot_metric(
        x,
        labels,
        metrics["average_target_score_mean"],
        metrics["average_target_score_std"],
        xlabel="Reference Set Size",
        ylabel="Average target score",
        color="#D14D41",
        out_path=out_dir / "average_target_score_vs_dataset_size.pdf",
    )
    plot_metric(
        x,
        labels,
        metrics["lpips_diversity_mean"],
        metrics["lpips_diversity_std"],
        xlabel="Reference Set Size",
        ylabel="LPIPS",
        color="#3A94C5",
        out_path=out_dir / "lpips_diversity_vs_dataset_size.pdf",
    )
    plot_metric(
        x,
        labels,
        metrics["target_success_rate_mean"],
        metrics["target_success_rate_std"],
        xlabel="Reference Set Size",
        ylabel="Target success rate",
        color="#879A39",
        out_path=out_dir / "target_success_rate_vs_dataset_size.pdf",
    )


if __name__ == "__main__":
    main()
