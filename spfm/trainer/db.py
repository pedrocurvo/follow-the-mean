from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

from utils.train_helpers import (
    _apply_label_filter,
    _apply_label_split,
    build_or_load_db,
    get_filtered_label_counts,
    make_loader,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Artifact Containers
# ---------------------------------------------------------------------------

@dataclass
class PrimaryDBArtifacts:
    loader: object
    Xdb: torch.Tensor
    db_indices: torch.Tensor | None
    db_group_ids: torch.Tensor | None
    need_group_ids: bool
    use_db_indices: bool


@dataclass
class AltDBArtifacts:
    alt_db: torch.Tensor | None
    alt_db_group_ids: torch.Tensor | None
    alt_db_label: str | None


# ---------------------------------------------------------------------------
# Training Loader
# ---------------------------------------------------------------------------

def build_train_latent_loader(
    *,
    Xdb: torch.Tensor,
    db_indices: torch.Tensor | None,
    batch_size: int,
    drop_last: bool = True,
    shuffle: bool = True,
    include_db_indices: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    n = int(Xdb.shape[0])
    row_idx = torch.arange(n, dtype=torch.long)
    if include_db_indices and db_indices is not None:
        db_idx_cpu = db_indices.detach().to(device="cpu", dtype=torch.long)
        dataset = TensorDataset(row_idx, db_idx_cpu)
    else:
        dataset = TensorDataset(row_idx)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and int(prefetch_factor) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


# ---------------------------------------------------------------------------
# DB Spec Helpers
# ---------------------------------------------------------------------------

def _normalize_db_spec(spec: str | None) -> str | None:
    if spec is None:
        return None
    cleaned = str(spec).strip()
    if cleaned == "" or cleaned.lower() == "none":
        return None
    return cleaned


def _parse_labels_from_db_spec(db_spec: str) -> str:
    labels: list[str] = []
    for entry in [e.strip() for e in db_spec.split(",") if e.strip()]:
        if ":" in entry:
            label = entry.split(":", 1)[0].strip()
        elif "=" in entry:
            label = entry.split("=", 1)[0].strip()
        else:
            raise ValueError(f"Invalid db spec entry '{entry}', expected label:fraction or label=count")
        if not label:
            raise ValueError(f"Invalid db spec entry '{entry}', empty label")
        labels.append(label)
    if not labels:
        raise ValueError("db spec must contain at least one label:fraction pair")
    return ",".join(labels)


def _split_seed(args) -> int:
    return int(getattr(args, "seed", 0))


def _loader_perf_kwargs(args) -> dict[str, object]:
    return {
        "num_workers": int(getattr(args, "num_workers", 0)),
        "pin_memory": bool(getattr(args, "pin_memory", True)),
        "persistent_workers": bool(getattr(args, "persistent_workers", True)),
        "prefetch_factor": int(getattr(args, "prefetch_factor", 2)),
    }


def build_db_group_ids_from_indices(
    dataset_name: str,
    split: str,
    label_field: str | None,
    label_value: str | None,
    label_split_spec: str | None,
    split_seed: int,
    label_split_complement: bool,
    db_indices: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if db_indices is None:
        return None
    if label_field is None:
        return torch.zeros(int(db_indices.numel()), device=device, dtype=torch.long)
    ds = load_dataset(dataset_name, split=split, streaming=False)
    ds = _apply_label_filter(ds, label_field, label_value)
    ds = _apply_label_split(
        ds,
        label_field=label_field,
        label_split_spec=label_split_spec,
        split_seed=split_seed,
        use_complement=label_split_complement,
    )
    if label_field not in ds.features:
        raise KeyError(f"label field '{label_field}' not found in dataset features")
    raw_labels = ds[label_field]
    idx_all = db_indices.detach().cpu().tolist()
    mapping: dict[object, int] = {}
    group_ids: list[int] = []
    for idx in idx_all:
        raw = raw_labels[int(idx)]
        key = int(raw) if isinstance(raw, (int, np.integer)) else str(raw)
        if key not in mapping:
            mapping[key] = len(mapping)
        group_ids.append(mapping[key])
    return torch.tensor(group_ids, device=device, dtype=torch.long)


# ---------------------------------------------------------------------------
# Primary Database
# ---------------------------------------------------------------------------

def _compute_primary_db_flags(args) -> tuple[bool, bool]:
    need_group_ids = False
    use_db_indices = bool(args.self_mask_db)
    return need_group_ids, use_db_indices


def _resolve_primary_label_args(args) -> tuple[str | None, str | None, bool]:
    db_spec = _normalize_db_spec(getattr(args, "db", None))
    if db_spec is None:
        return None, None, False
    if args.label_field is None:
        raise ValueError("--label_field is required when --db is not 'none'")
    return _parse_labels_from_db_spec(db_spec), db_spec, True


def build_primary_db(
    args,
    vae,
    device: torch.device,
    accelerator: Accelerator,
) -> PrimaryDBArtifacts:
    db_batch_size = int(getattr(args, "db_batch_size", 0) or args.batch_size)
    need_group_ids, use_db_indices = _compute_primary_db_flags(args)

    label_value, label_split_spec, use_full_filtered = _resolve_primary_label_args(args)
    label_split_complement = False

    if use_full_filtered:
        if args.hf_streaming:
            raise ValueError("--db with class split requires non-streaming dataset")
        args.hf_limit = 0
        args.N_img, db_counts = get_filtered_label_counts(
            args.dataset,
            args.split,
            args.label_field,
            label_value,
            label_split_spec=label_split_spec,
            label_split_seed=_split_seed(args),
            label_split_complement=label_split_complement,
        )
        if accelerator.is_main_process:
            counts_msg = ", ".join(f"{k}:{v}" for k, v in sorted(db_counts.items())) if db_counts else "n/a"
            logger.info("[db] using full filtered dataset: N_img=%d labels={%s}", args.N_img, counts_msg)

    loader = make_loader(
        args.dataset,
        args.split,
        db_batch_size,
        args.image_size,
        hf_limit=args.hf_limit,
        hf_streaming=args.hf_streaming,
        hf_streaming_buffer=args.hf_streaming_buffer,
        label_field=args.label_field,
        label_value=label_value,
        label_split_spec=label_split_spec,
        label_split_seed=_split_seed(args),
        label_split_complement=label_split_complement,
        return_indices=use_db_indices,
        **_loader_perf_kwargs(args),
    )
    db_loader = loader
    if args.N_img is None:
        # For "full dataset" DB builds, include the final partial batch and do a single pass.
        db_loader = make_loader(
            args.dataset,
            args.split,
            db_batch_size,
            args.image_size,
            drop_last=False,
            hf_limit=args.hf_limit,
            hf_streaming=args.hf_streaming,
            hf_streaming_buffer=args.hf_streaming_buffer,
            label_field=args.label_field,
            label_value=label_value,
            label_split_spec=label_split_spec,
            label_split_seed=_split_seed(args),
            label_split_complement=label_split_complement,
            return_indices=use_db_indices,
            **_loader_perf_kwargs(args),
        )
    db_out = build_or_load_db(
        db_loader,
        vae,
        args.db_dir,
        args.N_img,
        device,
        accelerator,
        args.image_size,
        vae_tag=args.vae_tag,
        return_indices=use_db_indices,
    )
    if use_db_indices:
        Xdb, db_indices = db_out
        db_indices = db_indices.to(device=device, dtype=torch.long)
    else:
        Xdb = db_out
        db_indices = None
    db_group_ids = None
    if need_group_ids and db_indices is not None:
        db_group_ids = build_db_group_ids_from_indices(
            args.dataset,
            args.split,
            args.label_field,
            label_value,
            label_split_spec,
            _split_seed(args),
            label_split_complement,
            db_indices,
            device=device,
        )
    return PrimaryDBArtifacts(
        loader=loader,
        Xdb=Xdb,
        db_indices=db_indices,
        db_group_ids=db_group_ids,
        need_group_ids=need_group_ids,
        use_db_indices=use_db_indices,
    )


# ---------------------------------------------------------------------------
# Alternate Database
# ---------------------------------------------------------------------------

def build_alt_db(
    args,
    vae,
    device: torch.device,
    accelerator: Accelerator,
) -> AltDBArtifacts:
    db_batch_size = int(getattr(args, "db_batch_size", 0) or args.batch_size)
    alt_db_raw = str(getattr(args, "alt_db", "none")).strip()
    alt_mode = alt_db_raw.lower()
    if alt_mode in {"", "none"}:
        return AltDBArtifacts(alt_db=None, alt_db_group_ids=None, alt_db_label=None)
    if args.hf_streaming:
        raise ValueError("--alt_db requires non-streaming dataset")
    if args.label_field is None:
        raise ValueError("--label_field is required when --alt_db is not 'none'")

    if alt_mode == "complementary":
        db_spec = _normalize_db_spec(getattr(args, "db", None))
        if db_spec is None:
            raise ValueError("--alt_db complementary requires --db with a class split spec")
        label_split_spec = db_spec
        label_split_complement = True
        alt_db_label = "complementary"
        log_mode = "complementary split"
    else:
        alt_spec = _normalize_db_spec(alt_db_raw)
        if alt_spec is None:
            raise ValueError("invalid --alt_db value")
        label_split_spec = alt_spec
        label_split_complement = False
        alt_db_label = f"split_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', alt_spec)}"
        log_mode = f"explicit split '{alt_spec}'"

    label_value = _parse_labels_from_db_spec(label_split_spec)
    dataset = args.dataset
    split = args.split
    dataset_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dataset))
    alt_db_dir = os.path.join(args.db_dir, f"alt_{dataset_tag}_{alt_db_label}")

    alt_n_img, alt_counts = get_filtered_label_counts(
        dataset,
        split,
        args.label_field,
        label_value,
        label_split_spec=label_split_spec,
        label_split_seed=_split_seed(args),
        label_split_complement=label_split_complement,
    )
    if accelerator.is_main_process:
        counts_msg = ", ".join(f"{k}:{v}" for k, v in sorted(alt_counts.items())) if alt_counts else "n/a"
        logger.info("[alt_db] using %s: N_img=%d labels={%s}", log_mode, alt_n_img, counts_msg)

    alt_need_group_ids = False
    alt_loader = make_loader(
        dataset,
        split,
        db_batch_size,
        args.image_size,
        hf_limit=0,
        hf_streaming=False,
        hf_streaming_buffer=args.hf_streaming_buffer,
        label_field=args.label_field,
        label_value=label_value,
        label_split_spec=label_split_spec,
        label_split_seed=_split_seed(args),
        label_split_complement=label_split_complement,
        return_indices=alt_need_group_ids,
        **_loader_perf_kwargs(args),
    )
    alt_db_out = build_or_load_db(
        alt_loader,
        vae,
        alt_db_dir,
        alt_n_img,
        device,
        accelerator,
        args.image_size,
        vae_tag=args.vae_tag,
        return_indices=alt_need_group_ids,
    )
    if alt_need_group_ids:
        alt_db, alt_db_indices = alt_db_out
        alt_db_indices = alt_db_indices.to(device=device, dtype=torch.long)
        alt_db_group_ids = build_db_group_ids_from_indices(
            dataset,
            split,
            args.label_field,
            label_value,
            label_split_spec,
            _split_seed(args),
            label_split_complement,
            alt_db_indices,
            device=device,
        )
    else:
        alt_db = alt_db_out
        alt_db_group_ids = None
    return AltDBArtifacts(
        alt_db=alt_db,
        alt_db_group_ids=alt_db_group_ids,
        alt_db_label=alt_db_label,
    )
