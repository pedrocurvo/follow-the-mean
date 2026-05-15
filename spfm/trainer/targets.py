from __future__ import annotations

import torch


def build_spatial_perturbation(
    x_t: torch.Tensor,
    spatial_dev_eps: float,
    norm_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.randn_like(x_t)
    v = v / (v.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + norm_eps)
    rms = x_t.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
    eps = spatial_dev_eps * rms
    x_t_perturbed = x_t + eps * v
    return x_t_perturbed, eps


def build_time_perturbation(
    t: torch.Tensor,
    time_dev_eps: float,
    min_dt: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_perturbed = (t + time_dev_eps).clamp(0.0, 1.0)
    dt = (t_perturbed - t).view(-1, 1, 1, 1).clamp_min(min_dt)
    return t_perturbed, dt
