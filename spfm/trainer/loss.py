from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Main Objective
# ---------------------------------------------------------------------------

def compute_main_loss(
    mu: torch.Tensor,
    x_data: torch.Tensor,
    t: torch.Tensor,
    loss_weight: str = "none",
) -> torch.Tensor:
    if loss_weight == "cos":
        weight = 1.0 - torch.cos(0.5 * math.pi * t)
    elif loss_weight == "inv_1_minus_t2":
        # Avoid singularities as t approaches 1.
        weight = 1.0 / torch.clamp(1.0 - t.pow(2), min=1e-4)
    else:
        weight = torch.ones_like(t)
    per_sample = (mu - x_data).pow(2).mean(dim=(1, 2, 3))
    return (per_sample * weight).mean()


def compute_refiner_loss(
    delta: torch.Tensor,
    x_data: torch.Tensor,
    x_t: torch.Tensor,
    mu_ret: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    mu_base = (1.0 - g) * x_t + g * mu_ret
    target = x_data - mu_base.detach()
    return ((alpha * delta) - target).pow(2).mean()


# ---------------------------------------------------------------------------
# Drifting Penalty
# ---------------------------------------------------------------------------

def compute_drift_field(
    gen: torch.Tensor,
    pos: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Anti-symmetric drifting field V = V_pos - V_neg.

    Attracts ``gen`` toward ``pos`` (retrieved neighbors) and repels ``gen``
    from other generated samples in the batch.  Joint normalization over
    positive and negative distances enforces the anti-symmetry property
    V(p, q) = -V(q, p), guaranteeing V = 0 at equilibrium.
    """
    B = gen.shape[0]
    gen_flat = gen.reshape(B, -1)
    pos_flat = pos.reshape(B, -1)

    # V_pos: attraction toward positives
    diff_pos = gen_flat.unsqueeze(1) - pos_flat.unsqueeze(0)  # (B, B, D)
    dist_pos = diff_pos.norm(dim=-1)                          # (B, B)

    # V_neg: repulsion from other generated samples
    diff_neg = gen_flat.unsqueeze(1) - gen_flat.unsqueeze(0)  # (B, B, D)
    dist_neg = diff_neg.norm(dim=-1)                          # (B, B)
    mask = torch.eye(B, device=gen.device, dtype=torch.bool)
    dist_neg = dist_neg.masked_fill(mask, float('inf'))

    # Joint softmax over pos + neg for anti-symmetry
    all_dists = torch.cat([dist_pos, dist_neg], dim=1)        # (B, 2B)
    all_w = torch.softmax(-all_dists / tau, dim=1)            # (B, 2B)
    w_pos = all_w[:, :B]
    w_neg = all_w[:, B:]

    V_pos = torch.einsum('ij,ijd->id', w_pos, -diff_pos)
    V_neg = torch.einsum('ij,ijd->id', w_neg, -diff_neg)
    V = V_pos - V_neg

    # Unit-normalize drift direction (paper appendix A.6)
    V = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return V.reshape_as(gen)


def compute_drifting_loss(
    mu: torch.Tensor,
    mu_ret: torch.Tensor,
    tau: float = 0.1,
    drift_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Drifting loss: MSE(mu, stopgrad(mu + V)).

    Returns (loss, v_magnitude) where v_magnitude is the mean norm of V
    before unit-normalization — useful as a diagnostic.
    """
    with torch.no_grad():
        V_raw = compute_drift_field(mu, mu_ret, tau=tau)
        # V_raw is already unit-normalized; recover pre-norm magnitude
        # by recomputing just the norms before normalization.
        B = mu.shape[0]
        gen_flat = mu.reshape(B, -1)
        pos_flat = mu_ret.reshape(B, -1)
        diff_pos = gen_flat.unsqueeze(1) - pos_flat.unsqueeze(0)
        dist_pos = diff_pos.norm(dim=-1)
        diff_neg = gen_flat.unsqueeze(1) - gen_flat.unsqueeze(0)
        dist_neg = diff_neg.norm(dim=-1)
        mask = torch.eye(B, device=mu.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(mask, float('inf'))
        all_dists = torch.cat([dist_pos, dist_neg], dim=1)
        all_w = torch.softmax(-all_dists / tau, dim=1)
        w_pos, w_neg = all_w[:, :B], all_w[:, B:]
        V_pos = torch.einsum('ij,ijd->id', w_pos, -diff_pos)
        V_neg = torch.einsum('ij,ijd->id', w_neg, -diff_neg)
        V_unnorm = V_pos - V_neg
        v_mag = float(V_unnorm.norm(dim=-1).mean().item())
        target = mu + V_raw
    loss = drift_weight * (mu - target).pow(2).mean()
    return loss, v_mag


def drifting_penalty_schedule(
    step: int,
    max_penalty: float,
    warmup_steps: int = 5000,
) -> float:
    """Quadratic warmup ramp for drifting penalty."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return max_penalty
    return max_penalty * (step / warmup_steps) ** 2


# ---------------------------------------------------------------------------
# Perturbation Losses
# ---------------------------------------------------------------------------

def compute_spatial_dev_loss(
    delta: torch.Tensor,
    mu_ret: torch.Tensor,
    mu_ret_perturbed: torch.Tensor,
    eps: torch.Tensor,
) -> torch.Tensor:
    target_spatial_dev = ((mu_ret_perturbed - mu_ret) / eps).detach()
    return (delta - target_spatial_dev).pow(2).mean()


def compute_time_dev_loss(
    delta: torch.Tensor,
    mu_ret: torch.Tensor,
    mu_ret_t_perturbed: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    target_time_dev = ((mu_ret_t_perturbed - mu_ret) / dt).detach()
    return (delta - target_time_dev).pow(2).mean()
