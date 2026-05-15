#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


BOOL_VALUE_KEYS = {
    "pin_memory",
    "persistent_workers",
    "fid",
    "kid",
    "ddp_find_unused_parameters",
}

NONE_VALUE_KEYS = {
    # Preserve explicit YAML null for keys where train.py supports a semantic "none"/"all" mode.
    "N_img",
}

LAUNCH_ONLY_KEYS = {
    "accelerate_num_processes",
    "accelerate_num_machines",
    "accelerate_machine_rank",
    "accelerate_main_process_port",
    "accelerate_multi_gpu",
}


def _to_cli_args(cfg: dict) -> list[str]:
    args: list[str] = []
    for key, value in cfg.items():
        if value is None:
            if key in NONE_VALUE_KEYS:
                args.extend([f"--{key}", "none"])
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if key in BOOL_VALUE_KEYS:
                args.extend([flag, "true" if value else "false"])
            elif value:
                args.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def _flatten_mapping(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_mapping(v, prefix=f"{key}_"))
        else:
            out[key] = v
    return out


def _merge_train_cfg(train_cfg: dict) -> dict:
    # Sections that only organize config and should not appear as CLI prefixes.
    unwrap_sections = {
        "dataset",
        "model",
        "training",
        "optimizer",
        "sampling",
        "logging",
        "runtime",
        "additional",
    }
    merged: dict = {}
    for k, v in train_cfg.items():
        if isinstance(v, dict) and k in unwrap_sections:
            flat = _flatten_mapping(v)
        elif isinstance(v, dict):
            flat = _flatten_mapping(v, prefix=f"{k}_")
        else:
            flat = {k: v}
        overlap = set(merged).intersection(flat)
        if overlap:
            dup = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate config keys after flattening: {dup}")
        merged.update(flat)
    # Semantic aliases: keep structured names while mapping to train.py CLI keys.
    aliases = {
        "cross_depth": "depth",
        "cross_mlp_ratio": "mlp_ratio",
        "cross_num_heads": "num_heads",
        "cross_qk_dim": "qk_dim",
        "cross_chunk_size": "cross_attn_chunk_size",
        "cross_learned_g": "learned_g",
        "cross_learned_alpha": "learned_alpha",
        "metrics_fid": "fid",
        "metrics_kid": "kid",
    }
    for src, dst in aliases.items():
        if src not in merged:
            continue
        if dst in merged:
            raise ValueError(f"Both '{src}' and '{dst}' are set; keep only one.")
        merged[dst] = merged.pop(src)
    for key in ("fid", "kid"):
        if key in merged and isinstance(merged[key], bool):
            merged[key] = "true" if merged[key] else "false"
    return merged


def _parse_bool_like(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _extract_launch_cfg(flat_cfg: dict) -> tuple[dict, dict]:
    cfg = dict(flat_cfg)
    launch = {k: cfg.pop(k) for k in LAUNCH_ONLY_KEYS if k in cfg}
    num_processes = int(launch.get("accelerate_num_processes", 1))
    multi_gpu = _parse_bool_like(launch.get("accelerate_multi_gpu", False)) or num_processes > 1
    launch_cfg = {
        "enabled": multi_gpu,
        "num_processes": num_processes,
        "num_machines": int(launch["accelerate_num_machines"]) if "accelerate_num_machines" in launch else None,
        "machine_rank": int(launch["accelerate_machine_rank"]) if "accelerate_machine_rank" in launch else None,
        "main_process_port": (
            int(launch["accelerate_main_process_port"])
            if "accelerate_main_process_port" in launch
            else None
        ),
    }
    return cfg, launch_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Run train.py from an experiments YAML config.")
    ap.add_argument("--config", required=True, help="Path to experiments YAML file.")
    ap.add_argument("--dry-run", action="store_true", help="Print command and exit.")
    ns = ap.parse_args()

    cfg_path = Path(ns.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")

    train_cfg = cfg.get("train", cfg)
    if not isinstance(train_cfg, dict):
        raise ValueError("`train` section must be a mapping.")
    flat_cfg = _merge_train_cfg(train_cfg)
    flat_cfg, launch_cfg = _extract_launch_cfg(flat_cfg)

    root = Path(__file__).resolve().parent
    train_py = root / "train.py"
    if launch_cfg["enabled"]:
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            str(launch_cfg["num_processes"]),
        ]
        if launch_cfg["num_machines"] is not None:
            cmd.extend(["--num_machines", str(launch_cfg["num_machines"])])
        if launch_cfg["machine_rank"] is not None:
            cmd.extend(["--machine_rank", str(launch_cfg["machine_rank"])])
        if launch_cfg["main_process_port"] is not None:
            cmd.extend(["--main_process_port", str(launch_cfg["main_process_port"])])
        cmd.extend([str(train_py), *_to_cli_args(flat_cfg)])
    else:
        cmd = [sys.executable, str(train_py), *_to_cli_args(flat_cfg)]
    print("[run_experiment] cmd:", shlex.join(cmd))
    if ns.dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
