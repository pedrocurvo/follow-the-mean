#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

LOGGER = logging.getLogger("lpips_metrics")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute LPIPS diversity over a folder of generated images."
    )
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Maximum number of image pairs to evaluate. Use 0 or a negative value for exact all-pairs.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _setup_logging(output_json: str) -> None:
    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(out_dir or ".", "lpips.log")),
        ],
        force=True,
    )


def _load_images(image_dir: str) -> tuple[torch.Tensor, list[str]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = sorted(str(p.resolve()) for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in {image_dir}, found {len(paths)}")
    images = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = pil_to_tensor(img).float().div_(255.0)
            images.append(tensor)
    return torch.stack(images, dim=0), paths


def _sample_pairs(num_images: int, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    total_pairs = num_images * (num_images - 1) // 2
    if max_pairs <= 0 or total_pairs <= max_pairs:
        return [(i, j) for i in range(num_images) for j in range(i + 1, num_images)]

    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < max_pairs:
        i = rng.randrange(num_images)
        j = rng.randrange(num_images - 1)
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        seen.add((i, j))
    return sorted(seen)


def main() -> int:
    ns = parse_args()
    output_json = str(Path(ns.output_json).expanduser().resolve())
    image_dir = str(Path(ns.image_dir).expanduser().resolve())
    _setup_logging(output_json)

    images, image_paths = _load_images(image_dir)
    num_images = int(images.shape[0])
    pairs = _sample_pairs(num_images, int(ns.max_pairs), int(ns.seed))
    LOGGER.info(
        "[lpips] image_dir=%s num_images=%d num_pairs=%d", image_dir, num_images, len(pairs)
    )

    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as exc:
        metrics = {
            "image_dir": image_dir,
            "num_images": num_images,
            "lpips_mean_pairwise": None,
            "lpips_num_pairs": 0,
            "lpips_error": f"import_failed: {exc}",
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        LOGGER.info("[done] wrote %s", output_json)
        return 0

    device = torch.device(ns.device)
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    metric.eval()

    imgs = images * 2.0 - 1.0
    score_sum = 0.0
    score_count = 0
    batch_size = max(1, int(ns.batch_size))
    with torch.no_grad():
        for s in range(0, len(pairs), batch_size):
            batch_pairs = pairs[s : s + batch_size]
            idx_i = [i for i, _ in batch_pairs]
            idx_j = [j for _, j in batch_pairs]
            batch_i = imgs[idx_i].to(device=device, non_blocking=True)
            batch_j = imgs[idx_j].to(device=device, non_blocking=True)
            batch_scores = metric(batch_i, batch_j)
            if batch_scores.ndim == 0:
                # TorchMetrics LPIPS may return a mean over the batch.
                batch_mean = float(batch_scores.item())
                score_sum += batch_mean * len(batch_pairs)
                score_count += len(batch_pairs)
            else:
                vals = [float(x) for x in batch_scores.detach().cpu().flatten().tolist()]
                score_sum += float(sum(vals))
                score_count += len(vals)

    mean_lpips = (score_sum / score_count) if score_count > 0 else None
    total_pairs = num_images * (num_images - 1) // 2
    metrics = {
        "image_dir": image_dir,
        "num_images": num_images,
        "lpips_mean_pairwise": mean_lpips,
        "lpips_num_pairs": score_count,
        "lpips_error": None,
        "max_pairs": int(ns.max_pairs),
        "batch_size": batch_size,
        "seed": int(ns.seed),
        "exact_all_pairs": bool(ns.max_pairs <= 0 or score_count == total_pairs),
        "total_possible_pairs": total_pairs,
        "sampled_pair_fraction": (score_count / total_pairs) if total_pairs > 0 else None,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    LOGGER.info("[done] wrote %s", output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
