#!/usr/bin/env python3
"""
Plot reference beta schedules for RMG experiments.

Uses beta0 = 1 by default and saves a figure comparing:
- constant
- bell
- quadratic-decay
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parser as cli_parser

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = cli_parser.argument_parser("Plot beta schedules as a function of time")
    parser.add_argument("--beta0", type=float, default=1.0, help="Reference beta0 value")
    parser.add_argument(
        "--num-points",
        type=int,
        default=500,
        help="Number of time samples between 0 and 1",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="outputs_flux2_training_free_control/beta_schedule_reference.pdf",
        help="Path to save the figure",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def constant_schedule(beta0: float, t: np.ndarray) -> np.ndarray:
    return beta0 * np.ones_like(t)


def bell_schedule(beta0: float, t: np.ndarray) -> np.ndarray:
    return 4.0 * beta0 * t * (1.0 - t)


def quadratic_decay_schedule(beta0: float, t: np.ndarray) -> np.ndarray:
    return beta0 * (1.0 - t) ** 2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    t = np.linspace(0.0, 1.0, args.num_points)

    schedules = {
        "constant": constant_schedule(args.beta0, t),
        "bell": bell_schedule(args.beta0, t),
        "quadratic-decay": quadratic_decay_schedule(args.beta0, t),
    }
    colors = {
        "constant": "#D14D41",
        "bell": "#3A94C5",
        "quadratic-decay": "#879A39",
    }
    labels = {
        "constant": r"$\beta_0$",
        "bell": r"$4\beta_0\, t(1-t)$",
        "quadratic-decay": r"$\beta_0(1-t)^2$",
    }

    plt.figure(figsize=(9.2, 6.2))
    for label, values in schedules.items():
        plt.plot(t, values, linewidth=3.0, label=labels[label], color=colors[label])

    plt.xlabel(r"$t$", fontsize=28)
    plt.ylabel(r"$\beta_t$", fontsize=28)
    plt.xlim(0.0, 1.0)
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(
        fontsize=22,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
        edgecolor="#B7B5AC",
        facecolor="white",
        framealpha=1.0,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
