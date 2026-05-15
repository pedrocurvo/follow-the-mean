from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TVT


@dataclass
class ClipEvalRuntime:
    model: object
    tokenizer: object
    device: torch.device
    text_features_cache: dict[tuple[str, ...], torch.Tensor] = field(default_factory=dict)


def load_clip_runtime(
    model_name: str,
    device: torch.device,
    local_files_only: bool = False,
) -> ClipEvalRuntime:
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    return ClipEvalRuntime(model=model, tokenizer=tokenizer, device=device)


def prompt_metric_suffix(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return slug or "prompt"


def _normalize_prompts(prompts: list[str]) -> list[str]:
    out = [str(p).strip() for p in prompts if str(p).strip()]
    if not out:
        raise ValueError("prompts must contain at least one non-empty prompt")
    return out


def _get_text_features(runtime: ClipEvalRuntime, prompts: list[str]) -> torch.Tensor:
    key = tuple(prompts)
    cached = runtime.text_features_cache.get(key)
    if cached is not None:
        return cached
    txt = runtime.tokenizer(prompts, padding=True, return_tensors="pt").to(runtime.device)
    with torch.no_grad():
        text_features = F.normalize(runtime.model.get_text_features(**txt), dim=-1)
    runtime.text_features_cache[key] = text_features
    return text_features


@torch.no_grad()
def clip_eval(
    directory_with_images: str,
    prompts: list[str],
    *,
    runtime: ClipEvalRuntime | None = None,
    model_name: str = "openai/clip-vit-base-patch32",
    local_files_only: bool = False,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> dict[str, float]:
    if runtime is None:
        runtime = load_clip_runtime(
            model_name=model_name,
            device=device or torch.device("cpu"),
            local_files_only=local_files_only,
        )
    prompts = _normalize_prompts(prompts)
    image_paths = sorted(str(p) for p in Path(directory_with_images).glob("*.png"))
    if not image_paths:
        return {prompt: 0.0 for prompt in prompts}
    text_features = _get_text_features(runtime, prompts)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=runtime.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=runtime.device).view(1, 3, 1, 1)
    counts = torch.zeros((len(prompts),), dtype=torch.long, device=runtime.device)

    eff_batch = int(max(1, batch_size))
    for s in range(0, len(image_paths), eff_batch):
        batch_paths = image_paths[s:s + eff_batch]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                imgs.append(TVT.ToTensor()(im.convert("RGB")))
        pix = torch.stack(imgs, dim=0).to(runtime.device)
        pix = F.interpolate(pix, size=(224, 224), mode="bicubic", align_corners=False)
        pix = ((pix - mean) / std).to(dtype=torch.float32)
        image_features = F.normalize(runtime.model.get_image_features(pixel_values=pix), dim=-1)
        pred = (image_features @ text_features.T).argmax(dim=1)
        counts += torch.bincount(pred, minlength=len(prompts))

    total = int(max(1, counts.sum().item()))
    return {prompt: float(100.0 * counts[i].item() / total) for i, prompt in enumerate(prompts)}
