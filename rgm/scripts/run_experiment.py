#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import experiment_config as cfg_loader

# ---------------------------------------------------------------------------
# Command Construction
# ---------------------------------------------------------------------------


def build_command(config: cfg_loader.ExperimentConfig) -> list[str]:
    python = config.python or sys.executable
    script = SRC / config.script
    return [python, str(script), "--config", str(config.path)]


# ---------------------------------------------------------------------------
# Reference Staging
# ---------------------------------------------------------------------------


def stage_references(config: cfg_loader.ExperimentConfig) -> None:
    ns = cfg_loader.namespace_from_config(config.path, script=config.script)
    out_dir = Path(ns.out_dir)
    reference_root = out_dir / "references"
    reference_root.mkdir(parents=True, exist_ok=True)
    for item in config.stage_references:
        source = ROOT / "references" / item["source"]
        target = reference_root / item.get("target", item["source"])
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(source, target_is_directory=True)
        except OSError:
            shutil.copytree(source, target)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an RGM FLUX experiment YAML")
    parser.add_argument("config", type=Path, help="Path to experiments/*.yaml")
    parser.add_argument("--dry-run", action="store_true")
    ns = parser.parse_args()

    config_path = ns.config if ns.config.is_absolute() else ROOT / ns.config
    config = cfg_loader.load_experiment_config(config_path)

    command = build_command(config)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.update({str(k): str(v) for k, v in config.env.items()})

    print(" ".join(command))
    if ns.dry_run:
        return 0
    stage_references(config)
    return subprocess.call(command, cwd=SRC, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
