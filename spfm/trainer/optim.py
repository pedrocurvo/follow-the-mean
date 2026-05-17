from __future__ import annotations

from typing import Iterable

import torch
from utils.optim import MuonAdamW


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        return float(max(g.detach().abs().max().item() for g in grads))
    total = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        total = total + g.detach().float().norm(norm_type).pow(norm_type)
    return float(total.pow(1.0 / norm_type).item())


def partition_muon_parameters(
    model: torch.nn.Module,
) -> tuple[dict[tuple[int, ...], list[torch.nn.Parameter]], list[torch.nn.Parameter]]:
    muon_params_by_shape: dict[tuple[int, ...], list[torch.nn.Parameter]] = {}
    adamw_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim == 2 and "embed" not in lname and "lm_head" not in lname:
            muon_params_by_shape.setdefault(tuple(param.shape), []).append(param)
        else:
            adamw_params.append(param)
    return muon_params_by_shape, adamw_params


def build_optimizer(
    model: torch.nn.Module,
    args,
    logger=None,
    is_main_process: bool = False,
):
    optimizer_name = args.optimizer
    if optimizer_name == "lion":
        try:
            from lion_pytorch import Lion
        except ImportError as exc:
            raise ImportError(
                "Lion optimizer requested but lion_pytorch is not installed."
            ) from exc
        return Lion(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    if optimizer_name == "muon":
        muon_params_by_shape, adamw_params = partition_muon_parameters(model)
        param_groups: list[dict] = []
        if adamw_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=adamw_params,
                    lr=args.lr,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_epsilon,
                    weight_decay=args.weight_decay,
                )
            )
        for shape in sorted(muon_params_by_shape):
            param_groups.append(
                dict(
                    kind="muon",
                    params=muon_params_by_shape[shape],
                    lr=args.lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=args.weight_decay,
                )
            )
        if not param_groups:
            raise ValueError("No trainable parameters found for Muon optimizer.")
        opt = MuonAdamW(param_groups)
        if logger is not None and is_main_process:
            muon_group_count = len(muon_params_by_shape)
            muon_param_count = sum(p.numel() for ps in muon_params_by_shape.values() for p in ps)
            adamw_param_count = sum(p.numel() for p in adamw_params)
            logger.info(
                "[train] optimizer=muon (muon_groups=%d, muon_params=%d, adamw_params=%d)",
                muon_group_count,
                muon_param_count,
                adamw_param_count,
            )
        return opt
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
