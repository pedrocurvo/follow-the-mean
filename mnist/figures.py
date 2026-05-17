#!/usr/bin/env python3
"""Generate closed-form MNIST steering figures.

This experiment restricts MNIST to digits 0 and 1. Sparse labels are used to
estimate class posteriors under the flow kernel, and generated samples are
steered by those posteriors. No neural network is trained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Plot Style
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.8,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
    }
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_PURPLE = "#7a1fa2"
C_RED = "#d1495b"
C_GREEN = "#0b6e4f"
C_BLUE = "#1d4e89"
C_ORANGE = "#c06c00"
C_TEAL = "#1f7a8c"
C_LIME = "#2f9e44"
C_GRAY = "#6c757d"

# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------


def _load_steerability_cache(
    cache_path: Path,
    m_values: list[int],
    n_trials: int,
    n_eval: int,
    gen_steps: int,
    target_class: int,
) -> dict[str, np.ndarray] | None:
    if not cache_path.exists():
        return None

    cache = np.load(cache_path)
    expected = {
        "m_values",
        "n_trials",
        "n_eval",
        "gen_steps",
        "target_class",
        "steer_means",
        "steer_stds",
        "uncond_accs",
        "hard_accs",
    }
    if not expected.issubset(cache.files):
        return None

    if not np.array_equal(cache["m_values"], np.array(m_values, dtype=int)):
        return None
    if int(cache["n_trials"]) != n_trials:
        return None
    if int(cache["n_eval"]) != n_eval:
        return None
    if int(cache["gen_steps"]) != gen_steps:
        return None
    if int(cache["target_class"]) != target_class:
        return None

    print(f"[mnist/figures] loading cached steerability data from {cache_path}")
    return {key: cache[key] for key in expected}


def _render_steerability_plot(
    m_values: list[int] | np.ndarray,
    steer_means: list[float] | np.ndarray,
    steer_stds: list[float] | np.ndarray,
    uncond_accs: list[float] | np.ndarray,
    hard_accs: list[float] | np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.plot(
        m_values,
        steer_means,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=C_PURPLE,
        label="Semi-Supervised",
    )
    ax.fill_between(
        m_values,
        np.array(steer_means) - np.array(steer_stds),
        np.array(steer_means) + np.array(steer_stds),
        color=C_PURPLE,
        alpha=0.12,
    )
    ax.axhline(np.mean(uncond_accs), linestyle=":", linewidth=1.5, color=C_GRAY, label="Unconditional")
    ax.axhline(np.mean(hard_accs), linestyle="--", linewidth=1.5, color=C_RED, label="Hard Filter")
    ax.set_xscale("log")
    ax.set_xlabel(r"$M$", fontsize=28)
    ax.set_ylabel("Target-Digit\nGeneration Rate (%)", fontsize=28)
    ax.set_xticks(m_values, [str(v) for v in m_values], fontsize=24)
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=15,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MNIST paper figures for closed-form semi-supervised flow matching."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/mnist"),
        help="Directory where MNIST will be stored/read.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--n-mnist",
        type=int,
        default=1000,
        help="Number of MNIST digits (0/1 only) to use.",
    )
    parser.add_argument(
        "--m-labeled",
        type=int,
        default=50,
        help="Balanced number of labeled examples used in the sample-grid experiment.",
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=16,
        help="Number of generated samples per class for the sample grids.",
    )
    parser.add_argument(
        "--gen-steps",
        type=int,
        default=30,
        help="Euler steps for generation.",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 50],
        help="Balanced labeled counts per experiment.",
    )
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Timesteps for the label-accuracy-vs-t figure.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of labeled-set trials for each point estimate.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=64,
        help="Number of generated samples for steerability evaluation.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device selection.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["all", "accuracy-t", "accuracy-m", "samples", "steerability"],
        default=["all"],
        help="Subset of figures to generate.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Runtime Utilities
# ---------------------------------------------------------------------------

def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Flow Kernels And Posteriors
# ---------------------------------------------------------------------------

def sigma_t(t: torch.Tensor | float, sigma_min: float = 1e-4) -> torch.Tensor | float:
    return (1 - t) + sigma_min


def flow_kernel(z: torch.Tensor, X: torch.Tensor, t: torch.Tensor | float, sigma_min: float = 1e-4) -> torch.Tensor:
    sig = sigma_t(t, sigma_min)
    means = t * X
    if z.dim() == 1:
        diff = z.unsqueeze(0) - means
        return -0.5 * (diff**2).sum(-1) / sig**2
    diff = z.unsqueeze(1) - means.unsqueeze(0)
    return -0.5 * (diff**2).sum(-1) / sig**2


def time_kernel(
    X_unlabeled: torch.Tensor, X_labeled: torch.Tensor, t: torch.Tensor | float, sigma_min: float = 1e-4
) -> torch.Tensor:
    sig = sigma_t(t, sigma_min)
    bw = 2 * sig**2 / (t**2 + 1e-8)
    diff = X_unlabeled.unsqueeze(1) - X_labeled.unsqueeze(0)
    return -0.5 * (diff**2).sum(-1) / bw


def infer_label_posteriors(
    X_unlabeled: torch.Tensor,
    X_labeled: torch.Tensor,
    y_labeled: torch.Tensor,
    K: int,
    t: torch.Tensor | float,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    log_k = time_kernel(X_unlabeled, X_labeled, t, sigma_min)
    log_k_norm = log_k - torch.logsumexp(log_k, dim=1, keepdim=True)
    k_norm = log_k_norm.exp()
    posteriors = torch.zeros(len(X_unlabeled), K, device=X_unlabeled.device)
    for cls in range(K):
        mask = (y_labeled == cls).float()
        posteriors[:, cls] = (k_norm * mask.unsqueeze(0)).sum(1)
    return posteriors


def conditional_mean(
    z: torch.Tensor,
    X_unlabeled: torch.Tensor,
    X_labeled: torch.Tensor,
    y_labeled: torch.Tensor,
    k_class: int,
    t: torch.Tensor | float,
    posteriors: torch.Tensor | None = None,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    K = int(y_labeled.max().item()) + 1
    log_w_u = flow_kernel(z, X_unlabeled, t, sigma_min)
    log_w_l = flow_kernel(z, X_labeled, t, sigma_min)

    if posteriors is None:
        posteriors = infer_label_posteriors(X_unlabeled, X_labeled, y_labeled, K, t, sigma_min)

    p_u = posteriors[:, k_class]
    w_u = log_w_u + torch.log(p_u.clamp(min=1e-10).unsqueeze(0))

    w_l = log_w_l.clone()
    w_l[:, y_labeled != k_class] = float("-inf")

    all_log_w = torch.cat([w_u, w_l], dim=1)
    all_X = torch.cat([X_unlabeled, X_labeled], dim=0)
    log_Z = torch.logsumexp(all_log_w, dim=1, keepdim=True)
    weights = (all_log_w - log_Z).exp()
    return weights @ all_X


def velocity_field(
    z: torch.Tensor,
    X_unlabeled: torch.Tensor,
    X_labeled: torch.Tensor,
    y_labeled: torch.Tensor,
    k_class: int,
    t: torch.Tensor | float,
    posteriors: torch.Tensor | None = None,
    sigma_min: float = 1e-4,
    t_eps: float = 1e-3,
) -> torch.Tensor:
    mu = conditional_mean(z, X_unlabeled, X_labeled, y_labeled, k_class, t, posteriors, sigma_min)
    return (mu - z) / (1 - t + t_eps)


def unconditional_mean(z: torch.Tensor, X_db: torch.Tensor, t: torch.Tensor | float, sigma_min: float = 1e-4) -> torch.Tensor:
    log_w = flow_kernel(z, X_db, t, sigma_min)
    log_Z = torch.logsumexp(log_w, dim=1, keepdim=True)
    weights = (log_w - log_Z).exp()
    return weights @ X_db


def unconditional_velocity(
    z: torch.Tensor, X_db: torch.Tensor, t: torch.Tensor | float, sigma_min: float = 1e-4, t_eps: float = 1e-3
) -> torch.Tensor:
    mu = unconditional_mean(z, X_db, t, sigma_min)
    return (mu - z) / (1 - t + t_eps)


# ---------------------------------------------------------------------------
# Data Selection
# ---------------------------------------------------------------------------

def select_labeled(
    X: torch.Tensor, y: torch.Tensor, M: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    K = int(y.max().item()) + 1
    labeled_idx: list[int] = []
    per_class = M // K
    if per_class <= 0:
        raise ValueError(f"M={M} is too small for {K} classes.")
    for cls in range(K):
        cls_idx = (y == cls).nonzero(as_tuple=True)[0].cpu().numpy()
        chosen = rng.choice(cls_idx, size=per_class, replace=False)
        labeled_idx.extend(chosen.tolist())
    labeled_idx_t = torch.tensor(labeled_idx, device=X.device, dtype=torch.long)
    unlabeled_mask = torch.ones(len(X), dtype=torch.bool, device=X.device)
    unlabeled_mask[labeled_idx_t] = False
    return X[labeled_idx_t], y[labeled_idx_t], X[unlabeled_mask], y[unlabeled_mask]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_unconditional(
    X_db: torch.Tensor,
    device: torch.device,
    n_samples: int = 16,
    n_steps: int = 30,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    d = X_db.shape[1]
    ts = torch.linspace(0.01, 0.99, n_steps, device=device)
    z = torch.randn(n_samples, d, device=device)
    trajectories = [z.clone()]

    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        v = unconditional_velocity(z, X_db, t, sigma_min)
        z = z + dt * v
        trajectories.append(z.clone())

    return torch.stack(trajectories)


@torch.no_grad()
def generate_samples(
    X_unlabeled: torch.Tensor,
    X_labeled: torch.Tensor,
    y_labeled: torch.Tensor,
    k_class: int,
    device: torch.device,
    n_samples: int = 16,
    n_steps: int = 30,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    d = X_unlabeled.shape[1]
    ts = torch.linspace(0.01, 0.99, n_steps, device=device)
    z = torch.randn(n_samples, d, device=device)
    trajectories = [z.clone()]
    K = int(y_labeled.max().item()) + 1

    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        post = infer_label_posteriors(X_unlabeled, X_labeled, y_labeled, K, t, sigma_min)
        v = velocity_field(z, X_unlabeled, X_labeled, y_labeled, k_class, t, post, sigma_min)
        z = z + dt * v
        trajectories.append(z.clone())

    return torch.stack(trajectories)


# ---------------------------------------------------------------------------
# Data Loading And Plot Helpers
# ---------------------------------------------------------------------------

def save_mnist_grid(
    samples: torch.Tensor,
    title: str,
    output_path: Path,
    mean_vec: torch.Tensor,
    std_vec: torch.Tensor,
    cmap: str = "gray",
) -> None:
    n = len(samples)
    n_cols = min(8, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    fig.suptitle(title, fontsize=11)

    for i, ax in enumerate(axes_arr):
        if i < n:
            img = (samples[i].detach().cpu() * std_vec.cpu() + mean_vec.cpu()).reshape(28, 28).numpy()
            ax.imshow(np.clip(img, 0.0, 1.0), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_mnist_binary_subset(
    data_root: Path, n_mnist: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    mnist = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    idx_01 = [i for i, (_, label) in enumerate(mnist) if label in [0, 1]]
    np.random.shuffle(idx_01)
    idx_01 = idx_01[:n_mnist]

    X_list = []
    y_list = []
    for i in idx_01:
        img, label = mnist[i]
        X_list.append(img)
        y_list.append(label)

    X = torch.stack(X_list).to(device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)
    mean_vec = X.mean(0)
    std_vec = X.std(0) + 1e-6
    X = (X - mean_vec) / std_vec
    return X, y, mean_vec, std_vec


# ---------------------------------------------------------------------------
# Figure Builders
# ---------------------------------------------------------------------------

def plot_accuracy_vs_t(
    X: torch.Tensor,
    y: torch.Tensor,
    m_labeled: int,
    t_values: list[float],
    n_trials: int,
    output_path: Path,
) -> None:
    means = []
    stds = []
    for t_val in t_values:
        trial_accs = []
        for seed in range(n_trials):
            X_l, y_l, X_u, y_u = select_labeled(X, y, m_labeled, seed=seed)
            post = infer_label_posteriors(X_u, X_l, y_l, K=2, t=torch.tensor(t_val, device=X.device))
            acc = (post.argmax(1) == y_u).float().mean().item()
            trial_accs.append(acc)
        means.append(float(np.mean(trial_accs)))
        stds.append(float(np.std(trial_accs)))

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.plot(
        t_values,
        means,
        marker="o",
        linewidth=2.5,
        markersize=7,
        color=C_GREEN,
        label=f"M={m_labeled}",
    )
    ax.fill_between(t_values, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=C_GREEN, alpha=0.12)
    ax.axhline(0.5, linestyle="--", linewidth=1.5, color=C_GRAY, label="chance")
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel("Label inference accuracy", fontsize=14)
    ax.set_ylim(0.45, 1.02)
    ax.set_xticks(t_values)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_m(
    X: torch.Tensor,
    y: torch.Tensor,
    m_values: list[int],
    n_trials: int,
    output_path: Path,
    t_fixed: float = 0.5,
) -> None:
    means = []
    stds = []
    for m_val in m_values:
        trial_accs = []
        for seed in range(n_trials):
            X_l, y_l, X_u, y_u = select_labeled(X, y, m_val, seed=seed)
            post = infer_label_posteriors(X_u, X_l, y_l, K=2, t=torch.tensor(t_fixed, device=X.device))
            acc = (post.argmax(1) == y_u).float().mean().item()
            trial_accs.append(acc)
        means.append(float(np.mean(trial_accs)))
        stds.append(float(np.std(trial_accs)))

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.plot(
        m_values,
        means,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=C_BLUE,
        label=f"t={t_fixed}",
    )
    ax.fill_between(m_values, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=C_BLUE, alpha=0.12)
    ax.axhline(0.5, linestyle="--", linewidth=1.5, color=C_GRAY, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("Labeled points M", fontsize=14)
    ax.set_ylabel("Label inference accuracy", fontsize=14)
    ax.set_xticks(m_values, [str(v) for v in m_values], fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_steerability_vs_m(
    X: torch.Tensor,
    y: torch.Tensor,
    m_values: list[int],
    n_trials: int,
    n_eval: int,
    gen_steps: int,
    output_path: Path,
) -> None:
    target_class = 1
    cache_path = output_path.with_suffix(".npz")
    cached = _load_steerability_cache(cache_path, m_values, n_trials, n_eval, gen_steps, target_class)

    if cached is None:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X.detach().cpu().numpy(), y.detach().cpu().numpy())

        steer_means = []
        steer_stds = []
        uncond_accs = []
        hard_accs = []

        for seed in range(n_trials):
            _, _, X_u_ref, _ = select_labeled(X, y, m_values[-1], seed=100 + seed)
            traj_uncond = generate_unconditional(X_u_ref, device=X.device, n_samples=n_eval, n_steps=gen_steps)
            preds_uncond = knn.predict(traj_uncond[-1].detach().cpu().numpy())
            uncond_accs.append(float((preds_uncond == target_class).mean()))

            X_hard = X[y == target_class]
            y_hard = y[y == target_class]
            traj_hard = generate_samples(
                X_hard,
                X_hard,
                y_hard,
                k_class=target_class,
                device=X.device,
                n_samples=n_eval,
                n_steps=gen_steps,
            )
            preds_hard = knn.predict(traj_hard[-1].detach().cpu().numpy())
            hard_accs.append(float((preds_hard == target_class).mean()))

        for m_val in m_values:
            trial_accs = []
            for seed in range(n_trials):
                X_l, y_l, X_u, _ = select_labeled(X, y, m_val, seed=seed)
                traj = generate_samples(
                    X_u,
                    X_l,
                    y_l,
                    k_class=target_class,
                    device=X.device,
                    n_samples=n_eval,
                    n_steps=gen_steps,
                )
                preds = knn.predict(traj[-1].detach().cpu().numpy())
                trial_accs.append(float((preds == target_class).mean()))
            steer_means.append(float(np.mean(trial_accs)))
            steer_stds.append(float(np.std(trial_accs)))

        np.savez(
            cache_path,
            m_values=np.array(m_values, dtype=int),
            n_trials=n_trials,
            n_eval=n_eval,
            gen_steps=gen_steps,
            target_class=target_class,
            steer_means=np.array(steer_means),
            steer_stds=np.array(steer_stds),
            uncond_accs=np.array(uncond_accs),
            hard_accs=np.array(hard_accs),
        )
        print(f"[mnist/figures] wrote cached steerability data to {cache_path}")
    else:
        steer_means = cached["steer_means"]
        steer_stds = cached["steer_stds"]
        uncond_accs = cached["uncond_accs"]
        hard_accs = cached["hard_accs"]

    _render_steerability_plot(m_values, steer_means, steer_stds, uncond_accs, hard_accs, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    requested = set(args.figures)
    generate_all = "all" in requested

    if not generate_all and requested == {"steerability"}:
        steerability_path = args.output_dir / "mnist_steerability_vs_m.pdf"
        cached = _load_steerability_cache(
            steerability_path.with_suffix(".npz"),
            args.m_values,
            min(args.n_trials, 3),
            args.n_eval,
            args.gen_steps,
            target_class=1,
        )
        if cached is not None:
            _render_steerability_plot(
                cached["m_values"],
                cached["steer_means"],
                cached["steer_stds"],
                cached["uncond_accs"],
                cached["hard_accs"],
                steerability_path,
            )
            print(f"[mnist/figures] wrote figures to {args.output_dir}")
            return 0

    print(f"[mnist/figures] device={device}")
    print(f"[mnist/figures] loading MNIST 0/1 subset with N={args.n_mnist}")
    X, y, mean_vec, std_vec = load_mnist_binary_subset(args.data_root, args.n_mnist, device)
    print(
        f"[mnist/figures] loaded {len(X)} samples: zeros={(y == 0).sum().item()} ones={(y == 1).sum().item()}"
    )

    if generate_all or "accuracy-t" in requested:
        print("[mnist/figures] plotting label accuracy vs t")
        plot_accuracy_vs_t(
            X=X,
            y=y,
            m_labeled=args.m_labeled,
            t_values=args.t_values,
            n_trials=args.n_trials,
            output_path=args.output_dir / "mnist_accuracy_vs_t.pdf",
        )

    if generate_all or "accuracy-m" in requested:
        print("[mnist/figures] plotting label accuracy vs M")
        plot_accuracy_vs_m(
            X=X,
            y=y,
            m_values=args.m_values,
            n_trials=args.n_trials,
            output_path=args.output_dir / "mnist_accuracy_vs_m.pdf",
        )

    if generate_all or "samples" in requested:
        print(f"[mnist/figures] generating sample grids with M={args.m_labeled}")
        X_l, y_l, X_u, _ = select_labeled(X, y, args.m_labeled, seed=args.seed)
        traj_0 = generate_samples(
            X_u,
            X_l,
            y_l,
            k_class=0,
            device=device,
            n_samples=args.n_gen,
            n_steps=args.gen_steps,
        )
        traj_1 = generate_samples(
            X_u,
            X_l,
            y_l,
            k_class=1,
            device=device,
            n_samples=args.n_gen,
            n_steps=args.gen_steps,
        )
        save_mnist_grid(
            traj_0[-1],
            f"Generated zeros (M={args.m_labeled})",
            args.output_dir / "mnist_generated_zeros.pdf",
            mean_vec,
            std_vec,
        )
        save_mnist_grid(
            traj_1[-1],
            f"Generated ones (M={args.m_labeled})",
            args.output_dir / "mnist_generated_ones.pdf",
            mean_vec,
            std_vec,
        )

    if generate_all or "steerability" in requested:
        print("[mnist/figures] plotting steerability vs M")
        plot_steerability_vs_m(
            X=X,
            y=y,
            m_values=args.m_values,
            n_trials=min(args.n_trials, 3),
            n_eval=args.n_eval,
            gen_steps=args.gen_steps,
            output_path=args.output_dir / "mnist_steerability_vs_m.pdf",
        )

    print(f"[mnist/figures] wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
