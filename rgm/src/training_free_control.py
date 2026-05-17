#!/usr/bin/env python3
"""Training-free control experiment orchestration for FLUX."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

import experiment_runtime as exp
import parser as cli_parser
import retrieval_guidance_core as poc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    return cli_parser.config_args("Training-free control experiments for FLUX", "training_free_control.py")

# ---------------------------------------------------------------------------
# Prompt Reference Experiment
# ---------------------------------------------------------------------------

def prompt_reference_spec(args: argparse.Namespace) -> tuple[list[tuple[str, str]], list[tuple[str, list[str]]], str]:
    data = args.prompt_reference_data
    prompts = [(item["label"], item["prompt"]) for item in data.get("prompts", [])]
    references = []
    for item in data.get("references", []):
        if "prompts" in item:
            reference_prompts = list(item["prompts"])
        else:
            reference_prompts = [item["prompt"]] * args.reference_size
        references.append((item["label"], reference_prompts))
    return prompts, references, data["title"]


def run_prompt_reference_interaction(pipe, args: argparse.Namespace, root_dir: Path, device: str, dtype: torch.dtype):
    section_dir = root_dir / "prompt_reference_interaction"
    prompts, references, title = prompt_reference_spec(args)
    rows = []
    row_labels = []
    manifest = []

    for prompt_idx, (prompt_label, prompt_text) in enumerate(prompts):
        row_labels.append(prompt_label)
        row_images: List[Image.Image] = []
        for reference_idx, (reference_label, reference_prompts) in enumerate(references):
            case_dir = section_dir / f"prompt_{prompt_idx:02d}_reference_{reference_idx:02d}"
            result = exp.run_case(
                pipe=pipe,
                args=args,
                root_dir=root_dir,
                case_dir=case_dir,
                prompt=prompt_text,
                reference_label=reference_label,
                reference_prompts=reference_prompts,
                seed=args.seed,
                device=device,
                dtype=dtype,
            )
            row_images.append(result["guided"])
            manifest.append(result["metadata"])
        rows.append(row_images)

    grid = exp.labeled_contact_sheet(rows=rows, row_labels=row_labels, col_labels=[label for label, _ in references], title=title)
    poc.save_pil(grid, str(section_dir / "prompt_reference_interaction_grid.png"))
    exp.save_json({"cases": manifest}, section_dir / "manifest.json")

# ---------------------------------------------------------------------------
# Controllability Experiment
# ---------------------------------------------------------------------------

def mixed_reference_prompts(reference_size: int, source_prompt: str, target_prompt: str, target_fraction: float) -> List[str]:
    target_count = int(round(reference_size * target_fraction))
    neutral_count = reference_size - target_count
    return [target_prompt] * target_count + [source_prompt] * neutral_count


def controllability_specs(args: argparse.Namespace) -> list[dict]:
    return list(args.controllability_data["specs"])


def controllability_fractions(args: argparse.Namespace) -> list[float]:
    return [float(value) for value in args.controllability_data["fractions"]]


def run_single_controllability_spec(
    pipe,
    args: argparse.Namespace,
    root_dir: Path,
    device: str,
    dtype: torch.dtype,
    spec: Dict[str, str],
):
    section_dir = root_dir / "controllability" / spec["slug"]
    fractions = controllability_fractions(args)
    images = []
    manifest = []

    baseline_cfg = exp.make_runtime_cfg(
        args,
        prompt=spec["prompt"],
        reference_prompt="baseline",
        out_dir=section_dir / "baseline_only",
        seed=args.seed,
    )
    baseline = poc.generate_single_image(pipe, baseline_cfg.prompt, baseline_cfg.seed + 7, baseline_cfg, device)
    poc.save_pil(baseline, str(Path(baseline_cfg.out_dir) / "baseline.png"))
    images.append(baseline)

    for fraction in fractions:
        pct = int(round(100 * fraction))
        reference_prompts = mixed_reference_prompts(args.reference_size, spec["source_prompt"], spec["target_prompt"], fraction)
        case_dir = section_dir / f"mix_{pct:03d}"
        result = exp.run_case(
            pipe=pipe,
            args=args,
            root_dir=root_dir,
            case_dir=case_dir,
            prompt=spec["prompt"],
            reference_label=f"{exp.slugify(spec['target_prompt'])}-{pct:03d}-percent",
            reference_prompts=reference_prompts,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        images.append(result["guided"])
        manifest.append(result["metadata"])

    grid = exp.labeled_contact_sheet(
        rows=[images],
        row_labels=["Generated"],
        col_labels=["Baseline"] + [f"{int(round(value * 100))}%" for value in fractions],
        title=spec["title"],
    )
    poc.save_pil(grid, str(section_dir / "controllability_grid.png"))
    exp.save_json({"cases": manifest, "fractions": fractions, "spec": spec}, section_dir / "manifest.json")


def run_controllability(pipe, args: argparse.Namespace, root_dir: Path, device: str, dtype: torch.dtype):
    selected_specs = None
    if getattr(args, "controllability_specs", "").strip():
        selected_specs = {token.strip() for token in args.controllability_specs.split(",") if token.strip()}

    for spec in controllability_specs(args):
        if selected_specs is not None and spec["slug"] not in selected_specs:
            continue
        run_single_controllability_spec(pipe, args, root_dir, device, dtype, spec)

# ---------------------------------------------------------------------------
# Beta Ablation Experiment
# ---------------------------------------------------------------------------

def ablation_beta_specs(args: argparse.Namespace) -> list[dict]:
    specs = []
    for item in args.ablation_beta_data["specs"]:
        spec = dict(item)
        if "reference_prompts" not in spec and "reference_prompt" in spec:
            spec["reference_prompts"] = [spec["reference_prompt"]] * args.reference_size
        specs.append(spec)
    return specs


def strength_slug(strength: float) -> str:
    return f"{strength:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def run_single_ablation_beta_spec(
    pipe,
    args: argparse.Namespace,
    root_dir: Path,
    device: str,
    dtype: torch.dtype,
    spec: Dict,
):
    section_dir = root_dir / "ablation_beta" / spec["slug"]
    schedules = list(args.ablation_beta_data["schedules"])
    strengths = [float(value) for value in args.ablation_beta_data["strengths"]]
    manifest = []

    baseline_cfg = exp.make_runtime_cfg(
        args,
        prompt=spec["prompt"],
        reference_prompt="baseline",
        out_dir=section_dir / "baseline_only",
        seed=args.seed,
    )
    baseline = poc.generate_single_image(pipe, baseline_cfg.prompt, baseline_cfg.seed + 7, baseline_cfg, device)
    poc.save_pil(baseline, str(Path(baseline_cfg.out_dir) / "baseline.png"))

    for schedule in schedules:
        row_images = [baseline]
        for strength in strengths:
            case_dir = section_dir / schedule / f"lambda_{strength_slug(strength)}"
            case_args = exp.override_args(args, beta_schedule=schedule, guidance_strength=strength)
            result = exp.run_case(
                pipe=pipe,
                args=case_args,
                root_dir=root_dir,
                case_dir=case_dir,
                prompt=spec["prompt"],
                reference_label=spec["reference_label"],
                reference_prompts=spec["reference_prompts"],
                seed=args.seed,
                device=device,
                dtype=dtype,
                save_baseline=False,
            )
            row_images.append(result["guided"])
            manifest.append(result["metadata"])

        grid = exp.labeled_contact_sheet(
            rows=[row_images],
            row_labels=[str(schedule)],
            col_labels=["Baseline"] + [str(value) for value in strengths],
            title=f"{spec['title']} | {schedule}",
        )
        poc.save_pil(grid, str(section_dir / f"{schedule}_grid.png"))

    exp.save_json({"spec": spec, "schedules": schedules, "strengths": strengths, "cases": manifest}, section_dir / "manifest.json")


def run_ablation_beta(pipe, args: argparse.Namespace, root_dir: Path, device: str, dtype: torch.dtype):
    for spec in ablation_beta_specs(args):
        run_single_ablation_beta_spec(pipe, args, root_dir, device, dtype, spec)

# ---------------------------------------------------------------------------
# Single Case Execution
# ---------------------------------------------------------------------------

def run_image_reference_case(
    pipe,
    args: argparse.Namespace,
    root_dir: Path,
    case_dir: Path,
    slug: str,
    prompt: str,
    reference: dict,
    device: str,
    dtype: torch.dtype,
):
    image_dir = Path(reference["image_dir"])
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Reference image directory does not exist: {image_dir}")

    image_paths = exp.list_image_paths(image_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in reference image directory: {image_dir}")

    case_dir.mkdir(parents=True, exist_ok=True)
    reference_label = reference.get("label") or f"image reference: {image_dir.name}"
    runtime_cfg = exp.make_runtime_cfg(args, prompt=prompt, reference_prompt=reference_label, out_dir=case_dir, seed=args.seed)

    reference_id = exp.make_image_reference_id(args, image_paths)
    image_reference_dir = exp.shared_references_dir(root_dir) / reference_id
    reference_cfg = exp.make_runtime_cfg(args, prompt=prompt, reference_prompt=reference_label, out_dir=image_reference_dir, seed=args.reference_seed)

    baseline = poc.generate_single_image(pipe, runtime_cfg.prompt, runtime_cfg.seed + 7, runtime_cfg, device)
    poc.save_pil(baseline, str(case_dir / "baseline.png"))

    reference_latents, reference_meta = exp.load_or_build_image_reference(
        image_paths=image_paths,
        pipe=pipe,
        cfg=reference_cfg,
        image_reference_dir=image_reference_dir,
        device=device,
    )
    guided, callback = exp.generate_with_rmg(
        pipe=pipe,
        cfg=runtime_cfg,
        device=device,
        reference_latents=reference_latents.to(device=device, dtype=dtype),
        beta_schedule=args.beta_schedule,
    )
    poc.save_pil(guided, str(case_dir / "rmg_guided.png"))

    nn_index, nn_dist2 = poc.find_nearest_reference(guided, reference_latents, pipe, runtime_cfg, device)
    nearest = poc.load_reference_image(reference_cfg, nn_index)
    if nearest is None:
        nearest = poc.make_text_tile(baseline.width, baseline.height, [f"Nearest reference image missing for index {nn_index}"])
    poc.save_pil(nearest, str(case_dir / "nearest_reference_neighbor.png"))

    comparison = poc.add_labels_and_title(
        [baseline, guided, nearest],
        ["Baseline", "RMG-guided", f"Nearest reference neighbour #{nn_index}"],
        title=f"Prompt: {prompt} | Reference: {image_dir.name}",
    )
    poc.save_pil(comparison, str(case_dir / "comparison.png"))
    exp.save_callback_history(callback.history, case_dir / "callback_history.txt")

    exp.save_json(
        {
            "case_slug": slug,
            "prompt": prompt,
            "reference_label": reference_label,
            "reference_set": {"type": "images", "image_dir": str(image_dir)},
            "seed": args.seed,
            "beta_schedule": args.beta_schedule,
            "guidance_strength": args.guidance_strength,
            "nearest_reference_index": nn_index,
            "nearest_reference_latent_mse": float(nn_dist2[nn_index].item()),
            "shared_reference_id": reference_id,
            "shared_reference_dir": str(image_reference_dir),
            "runtime_cfg": runtime_cfg.__dict__,
            "reference_meta": reference_meta,
        },
        case_dir / "run_metadata.json",
    )


def run_single_case(pipe, args: argparse.Namespace, root_dir: Path, device: str, dtype: torch.dtype):
    case = args.case_data
    prompt = case["prompt"]
    slug = case.get("slug") or exp.slugify(prompt)
    reference = case["reference_set"]
    case_dir = root_dir / "case" / slug

    if reference["type"] == "prompts":
        reference_prompts = list(reference["prompts"])
        reference_label = reference.get("label") or reference.get("prompt") or "prompt reference set"
        exp.run_case(
            pipe=pipe,
            args=args,
            root_dir=root_dir,
            case_dir=case_dir,
            prompt=prompt,
            reference_label=reference_label,
            reference_prompts=reference_prompts,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        return

    if reference["type"] == "images":
        run_image_reference_case(
            pipe=pipe,
            args=args,
            root_dir=root_dir,
            case_dir=case_dir,
            slug=slug,
            prompt=prompt,
            reference=reference,
            device=device,
            dtype=dtype,
        )
        return

    raise ValueError(f"Unsupported reference set type: {reference['type']}")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    poc.set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp.shared_references_dir(out_dir).mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("dtype:", dtype)
    print("output dir:", out_dir)
    print("beta schedule:", args.beta_schedule)
    print("guidance strength:", args.guidance_strength)

    pipe = poc.build_pipe(args.model_id, dtype=dtype, device=device)
    sections = {section.strip() for section in args.sections.split(",") if section.strip()}
    if "case" in sections:
        run_single_case(pipe, args, out_dir, device, dtype)
    if "prompt-reference" in sections:
        run_prompt_reference_interaction(pipe, args, out_dir, device, dtype)
    if "controllability" in sections:
        run_controllability(pipe, args, out_dir, device, dtype)
    if "ablation-beta" in sections:
        run_ablation_beta(pipe, args, out_dir, device, dtype)

    exp.save_json({"args": vars(args), "sections": sorted(sections), "device": device, "dtype": str(dtype)}, out_dir / "experiment_manifest.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
