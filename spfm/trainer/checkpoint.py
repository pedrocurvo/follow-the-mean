from __future__ import annotations

import json
import os

from accelerate import Accelerator


def is_full_training_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "step.json"))


def save_full_training_checkpoint(accelerator: Accelerator, ckpt_dir: str, step: int) -> None:
    accelerator.wait_for_everyone()
    accelerator.save_state(ckpt_dir)
    if accelerator.is_main_process:
        with open(os.path.join(ckpt_dir, "step.json"), "w") as f:
            json.dump({"step": int(step)}, f, indent=2)


def load_full_training_checkpoint(accelerator: Accelerator, ckpt_dir: str) -> int:
    accelerator.load_state(ckpt_dir)
    step = 0
    step_path = os.path.join(ckpt_dir, "step.json")
    if os.path.exists(step_path):
        with open(step_path, "r") as f:
            step = int(json.load(f).get("step", 0))
    return step
