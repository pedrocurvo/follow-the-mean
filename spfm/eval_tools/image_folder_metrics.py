#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from cleanfid import fid as cleanfid_fid

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_checkpoint


LOGGER = logging.getLogger("compute_image_folder_metrics")


def _parse_bool(value):
    return eval_checkpoint._parse_bool(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute FID/KID/Inception Score for an existing generated image folder.")
    ap.add_argument("--generated_dir", required=True)
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--config", default="experiments/model.yaml")
    ap.add_argument("--fid", type=_parse_bool, default=True)
    ap.add_argument("--kid", type=_parse_bool, default=True)
    ap.add_argument("--inception_score", type=_parse_bool, default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _setup_logging(results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "metrics_only.log")),
        ],
        force=True,
    )


def main() -> int:
    ns = parse_args()
    generated_dir = str(Path(ns.generated_dir).expanduser().resolve())
    reference_dir = str(Path(ns.reference_dir).expanduser().resolve())
    results_dir = str(Path(ns.results_dir).expanduser().resolve())
    config_path = str(Path(ns.config).expanduser().resolve())
    _setup_logging(results_dir)

    if not os.path.isdir(generated_dir):
        raise FileNotFoundError(f"generated_dir does not exist: {generated_dir}")
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"reference_dir does not exist: {reference_dir}")

    metrics: dict[str, object] = {
        "config": config_path,
        "generated_dir": generated_dir,
        "reference_dir": reference_dir,
    }

    if ns.fid:
        LOGGER.info("[fid] computing FID")
        metrics["fid"] = float(cleanfid_fid.compute_fid(reference_dir, generated_dir))
    if ns.kid:
        LOGGER.info("[kid] computing KID")
        metrics["kid"] = float(cleanfid_fid.compute_kid(reference_dir, generated_dir))
    if ns.inception_score:
        LOGGER.info("[is] computing Inception Score")
        training_args = train.parse_args(["--config", config_path]) if False else None
        batch_size = 64
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    import yaml

                    cfg = yaml.safe_load(f) or {}
                train_cfg = cfg.get("train", cfg)
                sampling_cfg = train_cfg.get("sampling", {})
                dataset_cfg = train_cfg.get("dataset", {})
                batch_size = int(sampling_cfg.get("clip_eval_batch_size") or dataset_cfg.get("batch_size") or 64)
            except Exception:
                batch_size = 64
        is_metrics = eval_checkpoint._compute_inception_score(
            directory_with_images=generated_dir,
            device=torch.device(ns.device),
            batch_size=batch_size,
        )
        if is_metrics is not None:
            metrics.update(is_metrics)

    out_path = os.path.join(results_dir, "metrics_only.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    LOGGER.info("[done] wrote %s", out_path)
    LOGGER.info("[done] metrics=%s", json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
