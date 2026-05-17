#!/usr/bin/env python3
"""Typed YAML loading for RGM experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
VALID_TRAINING_FREE_SECTIONS = {
    "case",
    "prompt-reference",
    "controllability",
    "ablation-beta",
}
PATH_KEYS = {
    "out_dir",
    "control_anatomy_image_dir",
    "reference_set_image_dir",
    "input_dir",
    "output_json",
    "output_csv",
}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class CommonConfig:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    seed: int = 123
    reference_seed: int = 123
    num_inference_steps: int = 50
    guidance_scale: float = 3.5
    height: int = 768
    width: int = 768
    out_dir: str = "outputs_flux2_training_free_control"


@dataclass
class RetrievalConfig:
    reference_size: int = 20
    guidance_strength: float = 0.2
    beta_schedule: str = "bell"
    guidance_start_frac: float = 0.15
    guidance_end_frac: float = 0.95
    topk: int = 4
    reuse_reference: bool = False
    callback_verbose: bool = False


@dataclass
class ExperimentConfig:
    name: str
    script: str
    common: CommonConfig = field(default_factory=CommonConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    args: dict[str, Any] = field(default_factory=dict)
    sections: list[str] = field(default_factory=list)
    case: dict[str, Any] = field(default_factory=dict)
    prompt_reference: dict[str, Any] = field(default_factory=dict)
    controllability: dict[str, Any] = field(default_factory=dict)
    ablation_beta: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    stage_references: list[dict[str, Any]] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    python: str | None = None
    path: Path | None = None

# ---------------------------------------------------------------------------
# Key And Path Normalization
# ---------------------------------------------------------------------------

def normalize_key(key: str) -> str:
    return key.replace("-", "_")


def normalize_mapping(values: dict[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    return {normalize_key(str(key)): value for key, value in values.items()}


def resolve_repo_path(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)

# ---------------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    config_path = path if path.is_absolute() else ROOT / path
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")
    raw["_config_path"] = config_path
    return raw


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = load_yaml(path)
    args = normalize_mapping(raw.get("args"))
    common_values = normalize_mapping(raw.get("common"))
    retrieval_values = normalize_mapping(raw.get("retrieval"))

    common = CommonConfig(**{**common_values})
    retrieval = RetrievalConfig(**{**retrieval_values})
    config = ExperimentConfig(
        name=str(raw.get("name") or Path(raw["_config_path"]).stem),
        script=str(raw.get("script") or args.get("script") or "training_free_control.py"),
        common=common,
        retrieval=retrieval,
        args=args,
        sections=[str(item) for item in raw.get("sections", [])],
        case=raw.get("case") or {},
        prompt_reference=raw.get("prompt_reference") or {},
        controllability=raw.get("controllability") or {},
        ablation_beta=raw.get("ablation_beta") or {},
        evaluation=raw.get("evaluation") or {},
        stage_references=list(raw.get("stage_references") or []),
        env=dict(raw.get("env") or {}),
        python=raw.get("python"),
        path=raw["_config_path"],
    )
    validate_config(config)
    return config

# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

def validate_config(config: ExperimentConfig) -> None:
    if not config.script:
        raise ValueError("Experiment config requires a script")
    validate_retrieval(config)
    for item in config.stage_references:
        if not isinstance(item, dict) or not item.get("source"):
            raise ValueError("Each stage_references entry must define source")
    if config.script == "training_free_control.py":
        sections = effective_sections(config)
        unknown = sorted(set(sections) - VALID_TRAINING_FREE_SECTIONS)
        if unknown:
            raise ValueError(f"Unknown training_free_control section(s): {', '.join(unknown)}")
        if "case" in sections:
            validate_case(config.case)
        if "prompt-reference" in sections:
            validate_prompt_reference(config.prompt_reference)
        if "controllability" in sections:
            validate_controllability(config.controllability)
        if "ablation-beta" in sections:
            validate_ablation_beta(config.ablation_beta)
    if config.script == "ablate_reference_size.py":
        require_args(
            config,
            [
                "prompt",
                "target_prompt",
                "target_label",
                "target_question",
                "positive_text",
                "negative_text",
                "clip_model_id",
                "vlm_model_id",
            ],
        )
    if config.script == "ablate_sampling_steps.py":
        require_args(
            config,
            [
                "prompt",
                "control_anatomy_image_dir",
                "case_slug",
                "nfe_values",
                "guidance_strength_values",
                "vlm_model_id",
                "success_question",
                "artifact_question",
                "pose_score_question",
            ],
        )


def require_args(config: ExperimentConfig, keys: list[str]) -> None:
    missing = [key for key in keys if key not in config.args]
    if missing:
        raise ValueError(f"{config.script} requires args: {', '.join(missing)}")


def validate_retrieval(config: ExperimentConfig) -> None:
    raw = load_yaml(config.path) if config.path else {}
    retrieval = raw.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("retrieval section is required")
    required = [
        "reuse_reference",
        "reference_size",
        "guidance_strength",
        "beta_schedule",
        "guidance_start_frac",
        "guidance_end_frac",
        "topk",
        "callback_verbose",
    ]
    missing = [key for key in required if key not in retrieval]
    if missing:
        raise ValueError(f"retrieval missing: {', '.join(missing)}")


def validate_case(case: dict[str, Any]) -> None:
    if not case:
        raise ValueError("case section requires a case mapping")
    if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
        raise ValueError("case.prompt is required")
    reference = case.get("reference_set")
    if not isinstance(reference, dict):
        raise ValueError("case.reference_set is required")
    reference_type = reference.get("type")
    if reference_type == "prompts":
        if "prompts" not in reference and "prompt" not in reference:
            raise ValueError("prompt reference_set requires prompt or prompts")
    elif reference_type == "images":
        if not isinstance(reference.get("image_dir"), str) or not reference["image_dir"].strip():
            raise ValueError("image reference_set requires image_dir")
    else:
        raise ValueError("case.reference_set.type must be prompts or images")


def validate_prompt_reference(prompt_reference: dict[str, Any]) -> None:
    prompts = prompt_reference.get("prompts")
    references = prompt_reference.get("references")
    if not isinstance(prompt_reference.get("title"), str) or not prompt_reference["title"].strip():
        raise ValueError("prompt_reference.title is required")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt_reference.prompts is required")
    if not isinstance(references, list) or not references:
        raise ValueError("prompt_reference.references is required")


def validate_controllability(controllability: dict[str, Any]) -> None:
    fractions = controllability.get("fractions")
    specs = controllability.get("specs")
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("controllability.fractions is required")
    if not isinstance(specs, list) or not specs:
        raise ValueError("controllability.specs is required")
    required = {"slug", "title", "prompt", "source_prompt", "target_prompt"}
    for index, spec in enumerate(specs):
        missing = sorted(required - set(spec))
        if missing:
            raise ValueError(f"controllability.specs[{index}] missing: {', '.join(missing)}")


def validate_ablation_beta(ablation_beta: dict[str, Any]) -> None:
    schedules = ablation_beta.get("schedules")
    strengths = ablation_beta.get("strengths")
    specs = ablation_beta.get("specs")
    if not isinstance(schedules, list) or not schedules:
        raise ValueError("ablation_beta.schedules is required")
    if not isinstance(strengths, list) or not strengths:
        raise ValueError("ablation_beta.strengths is required")
    if not isinstance(specs, list) or not specs:
        raise ValueError("ablation_beta.specs is required")
    required = {"slug", "title", "prompt", "reference_label"}
    for index, spec in enumerate(specs):
        missing = sorted(required - set(spec))
        if missing:
            raise ValueError(f"ablation_beta.specs[{index}] missing: {', '.join(missing)}")
        if "reference_prompt" not in spec and "reference_prompts" not in spec:
            raise ValueError(f"ablation_beta.specs[{index}] requires reference_prompt or reference_prompts")

# ---------------------------------------------------------------------------
# Experiment Normalization
# ---------------------------------------------------------------------------

def effective_sections(config: ExperimentConfig) -> list[str]:
    if config.sections:
        return config.sections
    sections = []
    if config.case:
        sections.append("case")
    if config.prompt_reference:
        sections.append("prompt-reference")
    if config.controllability:
        sections.append("controllability")
    if config.ablation_beta:
        sections.append("ablation-beta")
    return sections


def repeated_prompts(prompt: str, count: int) -> list[str]:
    return [prompt] * count


def normalize_case(case: dict[str, Any], reference_size: int) -> dict[str, Any]:
    item = dict(case)
    reference = dict(item.get("reference_set") or {})
    if reference.get("type") == "prompts":
        if "prompts" not in reference and "prompt" in reference:
            reference["prompts"] = repeated_prompts(str(reference["prompt"]), reference_size)
    item["reference_set"] = reference
    return item


def list_or_csv(value: Any, cast=str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [cast(item) for item in value]
    if isinstance(value, tuple):
        return [cast(item) for item in value]
    if isinstance(value, str):
        return [cast(item.strip()) for item in value.split(",") if item.strip()]
    return [cast(value)]


def defaults_for_script(script: str) -> dict[str, Any]:
    base = {
        **CommonConfig().__dict__,
        **RetrievalConfig().__dict__,
        "sections": "",
        "controllability_specs": "",
    }
    if script == "ablate_reference_size.py":
        base.update(
            {
                "subset_seed": 123,
                "num_samples": 50,
                "seed_stride": 1,
                "num_subsets": 5,
                "subset_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
                "reference_size": 128,
                "guidance_strength": 1.0,
                "beta_schedule": "quadratic-decay",
                "guidance_start_frac": 0.0,
                "guidance_end_frac": 0.85,
                "topk": 20,
                "max_vlm_new_tokens": 8,
                "skip_vlm": False,
                "skip_lpips": False,
            }
        )
    if script == "ablate_sampling_steps.py":
        base.update(
            {
                "nfe_values": [10, 20, 30, 50, 100, 200],
                "num_inference_steps": 20,
                "reference_size": 27,
                "guidance_strength": 0.2,
                "guidance_strength_values": [0.1, 0.2, 0.4, 0.5, 1.0],
                "beta_schedule": "quadratic-decay",
                "guidance_start_frac": 0.0,
                "guidance_end_frac": 0.85,
                "topk": 10,
                "skip_vlm": False,
                "skip_pose_score": False,
            }
        )
    return base

# ---------------------------------------------------------------------------
# Namespace Export
# ---------------------------------------------------------------------------

def namespace_from_config(config_path: Path, script: str | None = None) -> argparse.Namespace:
    config = load_experiment_config(config_path)
    script_name = script or config.script
    values = defaults_for_script(script_name)
    values.update(config.common.__dict__)
    values.update(config.retrieval.__dict__)
    values.update(config.args)
    values["name"] = config.name
    values["script"] = config.script
    values["config"] = str(config.path)
    values["sections"] = ",".join(effective_sections(config))

    if config.case:
        case = normalize_case(config.case, int(values["reference_size"]))
        reference = dict(case.get("reference_set") or {})
        if reference.get("type") == "images" and reference.get("image_dir"):
            reference["image_dir"] = resolve_repo_path(reference["image_dir"])
        case["reference_set"] = reference
        values["case_data"] = case
    if config.prompt_reference:
        values["prompt_reference_data"] = config.prompt_reference
    if config.controllability:
        values["controllability_data"] = config.controllability
        selected = config.controllability.get("selected")
        if selected:
            values["controllability_specs"] = ",".join(str(item) for item in selected)
    if config.ablation_beta:
        values["ablation_beta_data"] = config.ablation_beta
    if config.evaluation:
        values.update(normalize_mapping(config.evaluation))

    for key in ("subset_sizes", "nfe_values"):
        if key in values:
            values[key] = list_or_csv(values[key], int)
    if "guidance_strength_values" in values:
        values["guidance_strength_values"] = list_or_csv(values["guidance_strength_values"], float)

    for key in PATH_KEYS:
        if values.get(key):
            values[key] = resolve_repo_path(values[key])
    return argparse.Namespace(**values)
