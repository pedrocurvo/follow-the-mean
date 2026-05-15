from __future__ import annotations

import logging
import math
import os
import re
import zlib
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms as TVT
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

DEFAULT_IMAGE_SIZE = 256
DEFAULT_INVAE_LATENT_C = 32
DEFAULT_INVAE_LATENT_DOWNSAMPLE = 16
DEFAULT_INVAE_SCALING = 0.3099
DEFAULT_SD_SCALING = 0.18215

logger = get_logger(__name__)
class _RawPosterior:
    def __init__(self, sample: torch.Tensor):
        self._sample = sample

    def sample(self) -> torch.Tensor:
        return self._sample


class _RawDecoderOut:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class RawVAE(torch.nn.Module):
    """Identity VAE adapter for raw-image experiments."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0)
        self._pairflow_scaling = 1.0

    def encode(self, x: torch.Tensor):
        return _RawPosterior(x)

    def decode(self, z: torch.Tensor):
        return _RawDecoderOut(z)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def shard_range(total: int, process_index: int, num_processes: int) -> tuple[int, int]:
    total = int(max(0, total))
    num_processes = int(max(1, num_processes))
    process_index = int(min(max(0, process_index), num_processes - 1))
    start = (total * process_index) // num_processes
    end = (total * (process_index + 1)) // num_processes
    return start, end


def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    def _normalize_name(name: str) -> str:
        # DDP / torch.compile wrappers may prepend these prefixes.
        for prefix in ("module.", "_orig_mod."):
            while name.startswith(prefix):
                name = name[len(prefix):]
        return name

    ema_params = OrderedDict((_normalize_name(n), p) for n, p in ema_model.named_parameters())
    model_params = OrderedDict((_normalize_name(n), p) for n, p in model.named_parameters())

    for name, param in model_params.items():
        if name not in ema_params:
            continue
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def _extract_latent_sample(posterior):
    if hasattr(posterior, "latent_dist"):
        return posterior.latent_dist.sample()
    sample = getattr(posterior, "sample", None)
    if callable(sample):
        return sample()
    if sample is not None:
        return sample
    raise TypeError(f"Unsupported VAE encoder output type: {type(posterior)}")


def _extract_decode_sample(decoded):
    sample = getattr(decoded, "sample", None)
    if callable(sample):
        return sample()
    if sample is not None:
        return sample
    return decoded


def get_vae_scaling(vae, vae_name: str | None = None) -> float:
    if hasattr(vae, "_pairflow_scaling"):
        return float(vae._pairflow_scaling)
    cfg = getattr(vae, "config", None)
    if cfg is not None and getattr(cfg, "scaling_factor", None) is not None:
        return float(cfg.scaling_factor)
    if vae_name is not None and ("sd-vae" in vae_name or vae_name.startswith("stabilityai/")):
        return DEFAULT_SD_SCALING
    return DEFAULT_INVAE_SCALING


@torch.no_grad()
def infer_vae_latent_spec(vae, image_size: int, device: torch.device):
    x = torch.zeros(1, 3, image_size, image_size, device=device, dtype=torch.float32)
    z = _extract_latent_sample(vae.encode(x))
    if z.ndim != 4:
        raise RuntimeError(f"Unexpected latent shape from VAE encode: {tuple(z.shape)}")
    _, c, h, w = z.shape
    if image_size % h != 0:
        raise ValueError(f"VAE latent height {h} does not divide image_size {image_size}")
    downsample = image_size // h
    return int(c), int(h), int(w), int(downsample)


def vae_tag_from_name(vae_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", vae_name).strip("._-") or "vae"


@torch.no_grad()
def vae_encode(vae, x_01: torch.Tensor) -> torch.Tensor:
    x = x_01 * 2.0 - 1.0
    posterior = vae.encode(x)
    z = _extract_latent_sample(posterior)
    scaling = get_vae_scaling(vae)
    return z * scaling


@torch.no_grad()
def vae_decode(vae, z: torch.Tensor) -> torch.Tensor:
    scaling = get_vae_scaling(vae)
    z = z / scaling
    x = _extract_decode_sample(vae.decode(z))
    return ((x + 1.0) * 0.5).clamp(0, 1)


# ----------------------------
# Data loader (CIFAR-10 -> image_size)
# ----------------------------

def _apply_label_filter(ds, label_field: str | None, label_value: str | None):
    if label_field is None or label_value is None:
        return ds
    if label_field not in ds.features:
        raise KeyError(f"label field '{label_field}' not found in dataset features")
    raw_values = [v.strip() for v in str(label_value).split(",") if v.strip()]
    if len(raw_values) == 0:
        return ds
    feature = ds.features[label_field]
    target_values: list[int | str] = []
    if hasattr(feature, "names"):
        for raw in raw_values:
            if raw.isdigit():
                target_values.append(int(raw))
            else:
                if raw not in feature.names:
                    raise ValueError(f"label '{raw}' not in names: {feature.names}")
                target_values.append(int(feature.names.index(raw)))
    else:
        for raw in raw_values:
            if raw.isdigit():
                target_values.append(int(raw))
            else:
                target_values.append(raw)
    allowed = set(target_values)
    return ds.filter(lambda ex: ex.get(label_field, None) in allowed)


def _parse_label_split_spec(spec: str | None) -> dict[str, tuple[str, float | int]]:
    if spec is None:
        return {}
    parsed: dict[str, tuple[str, float | int]] = {}
    entries = [item.strip() for item in str(spec).split(",") if item.strip()]
    for entry in entries:
        if "=" in entry:
            raw_label, raw_count = entry.split("=", 1)
            label = raw_label.strip()
            count = int(raw_count.strip())
            if not label:
                raise ValueError("Empty label in --label_split_spec")
            if count < 0:
                raise ValueError(f"Invalid count for label '{label}': {count} (must be >= 0)")
            parsed[label] = ("count", count)
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid --label_split_spec entry '{entry}', expected label:fraction or label=count"
            )
        raw_label, raw_frac = entry.split(":", 1)
        label = raw_label.strip()
        frac = float(raw_frac.strip())
        if not label:
            raise ValueError("Empty label in --label_split_spec")
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"Invalid fraction for label '{label}': {frac} (must be in [0,1])")
        parsed[label] = ("fraction", frac)
    return parsed


def _resolve_split_label_to_target(feature, raw_label: str):
    if hasattr(feature, "names"):
        if raw_label.isdigit():
            target = int(raw_label)
            if target < 0 or target >= len(feature.names):
                raise ValueError(f"Label id '{raw_label}' out of range for feature names")
            return target
        if raw_label not in feature.names:
            raise ValueError(f"Label '{raw_label}' not in names: {feature.names}")
        return int(feature.names.index(raw_label))
    if raw_label.isdigit():
        return int(raw_label)
    return raw_label


def _apply_label_split(
    ds,
    label_field: str | None,
    label_split_spec: str | None,
    split_seed: int,
    use_complement: bool = False,
):
    if not label_split_spec:
        return ds
    if label_field is None:
        raise ValueError("--label_split_spec requires --label_field")
    if label_field not in ds.features:
        raise KeyError(f"label field '{label_field}' not found in dataset features")

    split_map = _parse_label_split_spec(label_split_spec)
    if len(split_map) == 0:
        return ds

    feature = ds.features[label_field]
    label_column = ds[label_field]
    keep_indices: list[int] = []
    for raw_label, spec in split_map.items():
        target = _resolve_split_label_to_target(feature, raw_label)
        cls_indices = [i for i, lbl in enumerate(label_column) if lbl == target]
        n_cls = len(cls_indices)
        if n_cls == 0:
            logger.warning("[label_split] no examples found for label '%s'", raw_label)
            continue
        mode, value = spec
        if mode == "count":
            n_take = int(value)
        else:
            n_take = int(round(float(value) * n_cls))
        n_take = min(max(n_take, 0), n_cls)
        seed_offset = zlib.crc32(str(target).encode("utf-8")) % (2**16)
        rng = np.random.default_rng(split_seed + seed_offset)
        order = rng.permutation(n_cls)
        if use_complement:
            selected = [cls_indices[j] for j in order[n_take:]]
        else:
            selected = [cls_indices[j] for j in order[:n_take]]
        keep_indices.extend(selected)

    keep_indices = sorted(set(keep_indices))
    return ds.select(keep_indices)


def _unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[0]):
        return batch[0], batch[1]
    return batch, None


def make_loader(
    dataset_name: str,
    split: str,
    batch_size: int,
    image_size: int,
    drop_last: bool = True,
    hf_limit: int = 0,
    hf_streaming: bool = False,
    hf_streaming_buffer: int = 10_000,
    label_field: str | None = None,
    label_value: str | None = None,
    label_split_spec: str | None = None,
    label_split_seed: int = 0,
    label_split_complement: bool = False,
    return_indices: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
):
    tf = TVT.Compose([
        TVT.Resize(image_size),
        TVT.CenterCrop(image_size),
        TVT.ToTensor(),
    ])
    ds = load_dataset(dataset_name, split=split, streaming=hf_streaming)
    ds = _apply_label_filter(ds, label_field, label_value)
    if label_split_spec:
        if hf_streaming:
            raise ValueError("label_split_spec requires non-streaming dataset")
        ds = _apply_label_split(
            ds,
            label_field=label_field,
            label_split_spec=label_split_spec,
            split_seed=label_split_seed,
            use_complement=label_split_complement,
        )

    if hf_streaming:
        if return_indices:
            raise ValueError("return_indices requires non-streaming dataset")
        ds = ds.shuffle(seed=0, buffer_size=hf_streaming_buffer)
    else:
        if hf_limit > 0:
            ds = ds.select(range(min(hf_limit, len(ds))))
        if return_indices:
            ds = ds.add_column("__db_index__", list(range(len(ds))))

    def collate(batch):
        imgs = []
        idxs = []
        for ex in batch:
            im = ex.get("img", None) or ex.get("image", None)
            if im is None:
                raise KeyError("Example missing 'img' or 'image'")
            imgs.append(tf(im.convert("RGB")))
            if return_indices:
                if "__db_index__" not in ex:
                    raise KeyError("Example missing '__db_index__' for return_indices")
                idxs.append(int(ex["__db_index__"]))
        imgs = torch.stack(imgs, dim=0)
        if return_indices:
            return imgs, torch.tensor(idxs, dtype=torch.long)
        return imgs

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=collate,
        num_workers=max(0, int(num_workers)),
        pin_memory=bool(pin_memory),
    )
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and int(prefetch_factor) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(ds, **loader_kwargs)


def get_filtered_length(
    dataset_name: str,
    split: str,
    label_field: str | None,
    label_value: str | None,
    label_split_spec: str | None = None,
    label_split_seed: int = 0,
    label_split_complement: bool = False,
) -> int:
    ds = load_dataset(dataset_name, split=split, streaming=False)
    ds = _apply_label_filter(ds, label_field, label_value)
    ds = _apply_label_split(
        ds,
        label_field=label_field,
        label_split_spec=label_split_spec,
        split_seed=label_split_seed,
        use_complement=label_split_complement,
    )
    return len(ds)


def get_filtered_label_counts(
    dataset_name: str,
    split: str,
    label_field: str | None,
    label_value: str | None,
    label_split_spec: str | None = None,
    label_split_seed: int = 0,
    label_split_complement: bool = False,
) -> tuple[int, dict[str, int]]:
    ds = load_dataset(dataset_name, split=split, streaming=False)
    ds = _apply_label_filter(ds, label_field, label_value)
    ds = _apply_label_split(
        ds,
        label_field=label_field,
        label_split_spec=label_split_spec,
        split_seed=label_split_seed,
        use_complement=label_split_complement,
    )
    total = len(ds)
    if label_field is None or label_field not in ds.features:
        return total, {}
    counts: dict[str, int] = {}
    feature = ds.features[label_field]
    for raw in ds[label_field]:
        if hasattr(feature, "names"):
            key = str(feature.names[int(raw)])
        else:
            key = str(raw)
        counts[key] = counts.get(key, 0) + 1
    return total, counts


# ----------------------------
# Build / load latent DB
# ----------------------------

def build_or_load_db(
    loader: DataLoader,
    vae,
    db_root: str,
    N_img: int | None,
    device: torch.device,
    accelerator: Accelerator,
    image_size: int,
    vae_tag: str = "invae",
    return_indices: bool = False,
):
    use_cuda_transfer = (device.type == "cuda")

    def _to_device_fast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x_cpu = x.to(dtype=dtype)
        if use_cuda_transfer:
            x_cpu = x_cpu.pin_memory()
            return x_cpu.to(device=device, non_blocking=True)
        return x_cpu.to(device=device)

    db_dir = os.path.join(db_root, f"db_vae_latents_{vae_tag}")
    if accelerator.is_main_process:
        ensure_dir(db_dir)
    n_tag = "all" if N_img is None else str(int(N_img))
    lat_path = os.path.join(db_dir, f"X_latent_{vae_tag}_N{n_tag}_img{image_size}.npy")
    idx_path = os.path.join(db_dir, f"X_idx_{vae_tag}_N{n_tag}_img{image_size}.npy")

    if os.path.exists(lat_path):
        X = np.load(lat_path)
        X = _to_device_fast(torch.from_numpy(np.asarray(X)), dtype=torch.float32)
        if accelerator.is_main_process:
            logger.info("[db] loaded latents %s", tuple(X.shape))
        if not return_indices:
            return X
        if not os.path.exists(idx_path):
            raise ValueError(f"DB indices missing at {idx_path}; rebuild with return_indices=True")
        idxs = np.load(idx_path)
        idxs = _to_device_fast(torch.from_numpy(np.asarray(idxs)), dtype=torch.long)
        return X, idxs

    if accelerator.is_main_process:
        if N_img is None:
            logger.info("[db] building latent DB from all available loader images")
        else:
            logger.info("[db] building latent DB from %d images", int(N_img))
        chunks = []
        idx_chunks = [] if return_indices else None
        seen = 0
        if N_img is None:
            for batch in loader:
                imgs, idxs = _unpack_batch(batch)
                if return_indices and idxs is None:
                    raise ValueError("return_indices=True but loader did not return indices")
                if use_cuda_transfer:
                    imgs = imgs.pin_memory().to(device=device, non_blocking=True)
                else:
                    imgs = imgs.to(device)
                z = vae_encode(vae, imgs)
                chunks.append(z.cpu())
                if return_indices:
                    idx_chunks.append(idxs.cpu())
                seen += int(imgs.shape[0])
                if seen % 500 == 0:
                    logger.info("[db] %d/all", seen)
        else:
            it = iter(loader)
            target_n = int(N_img)
            while seen < target_n:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                imgs, idxs = _unpack_batch(batch)
                if return_indices and idxs is None:
                    raise ValueError("return_indices=True but loader did not return indices")
                B = imgs.shape[0]
                take = min(B, target_n - seen)
                imgs = imgs[:take]
                if use_cuda_transfer:
                    imgs = imgs.pin_memory().to(device=device, non_blocking=True)
                else:
                    imgs = imgs.to(device)
                z = vae_encode(vae, imgs)
                chunks.append(z.cpu())
                if return_indices:
                    idx_chunks.append(idxs[:take].cpu())
                seen += take
                if seen % 500 == 0 or seen == target_n:
                    logger.info("[db] %d/%d", seen, target_n)

        if seen == 0:
            raise ValueError("No images were loaded for DB construction.")

        X = torch.cat(chunks, dim=0).to(torch.float32)
        np.save(lat_path, X.numpy().astype(np.float32))
        if return_indices:
            idx_all = torch.cat(idx_chunks, dim=0).to(torch.long)
            np.save(idx_path, idx_all.numpy().astype(np.int64))
        logger.info("[db] saved %s shape %s", lat_path, tuple(X.shape))

    accelerator.wait_for_everyone()
    X = np.load(lat_path)
    X = _to_device_fast(torch.from_numpy(np.asarray(X)), dtype=torch.float32)
    if not return_indices:
        return X
    if not os.path.exists(idx_path):
        raise ValueError(f"DB indices missing at {idx_path}; rebuild with return_indices=True")
    idxs = np.load(idx_path)
    idxs = _to_device_fast(torch.from_numpy(np.asarray(idxs)), dtype=torch.long)
    return X, idxs


# ----------------------------
# Sampling
# ----------------------------

@torch.no_grad()

def sample_closedform(
    Xdb: torch.Tensor,
    model: LearnedPosteriorMean,
    vae,
    out_dir: str,
    num_gen: int,
    steps: int,
    noise_scale: float,
    t_eps: float,
    decode_batch: int,
    latent_c: int,
    latent_h: int,
    latent_w: int,
    return_latents: bool = False,
    return_entropies: bool = False,
    save_grid: bool = True,
    time_schedule: str = "uniform",
    db_group_ids: torch.Tensor | None = None,
    model_extra_kwargs: dict[str, object] | None = None,
    db_retriever = None,
    model_kwargs_retriever = None,
):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = Xdb.device
    x = noise_scale * torch.randn(num_gen, latent_c, latent_h, latent_w, device=device)

    entropies = []
    # Integrate from noise (t=t_eps) to data (t=1-t_eps) using the flow-matching ODE.
    if time_schedule == "exp":
        # Exponential spacing over [t_eps, 1-t_eps].
        s = torch.linspace(0.0, 1.0, steps + 1, device=device)
        ts = t_eps + (1.0 - 2.0 * t_eps) * (torch.exp(s) - 1.0) / (math.e - 1.0)
    else:
        ts = torch.linspace(t_eps, 1.0 - t_eps, steps + 1, device=device)
    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        t = t_curr.expand(num_gen)
        model_db_kwargs = {}
        if db_group_ids is not None:
            model_db_kwargs["db_group_ids"] = db_group_ids
        if model_extra_kwargs:
            model_db_kwargs.update(model_extra_kwargs)
        if model_kwargs_retriever is not None:
            model_db_kwargs.update(model_kwargs_retriever(x, t))
        db_step = Xdb
        if db_retriever is not None:
            db_step, db_mask_step = db_retriever(x)
            model_db_kwargs["db_mask"] = db_mask_step
        if return_entropies:
            out = model(
                x,
                t,
                db_step,
                return_entropy=True,
                **model_db_kwargs,
            )
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                mu_step, ent = out[0], out[1]
            else:
                raise RuntimeError("Model did not return (mu, entropy) but return_entropies=True.")
        else:
            out = model(
                x,
                t,
                db_step,
                **model_db_kwargs,
            )
            mu_step = out[0] if isinstance(out, (tuple, list)) else out
        dt = t_next - t_curr  # positive
        denom = (1.0 - t_curr).clamp(min=t_eps)
        # dx/dt = (mu - x) / (1 - t)  -> Euler step towards data
        x = x + ((mu_step - x) / denom) * dt
        if return_entropies:
            entropies.append(ent.detach().cpu())

    imgs = decode_latents(vae, x, decode_batch=decode_batch)
    if save_grid:
        ensure_dir(out_dir)
        grid = make_grid(imgs, nrow=int(math.sqrt(num_gen)))
        out_path = os.path.join(out_dir, f"fullsum_learned_vae_N{Xdb.shape[0]}_S{steps}.png")
        save_image(grid, out_path)

    if return_entropies:
        entropies = torch.stack(entropies, dim=0)

    if return_latents and return_entropies:
        return imgs, x, entropies
    if return_latents:
        return imgs, x
    if return_entropies:
        return imgs, entropies
    return imgs


@torch.no_grad()

def generate_images_to_dir(
    Xdb: torch.Tensor,
    model: LearnedPosteriorMean,
    vae,
    out_dir: str,
    total_gen: int,
    gen_batch: int,
    steps: int,
    noise_scale: float,
    t_eps: float,
    decode_batch: int,
    prefix: str,
    latent_c: int,
    latent_h: int,
    latent_w: int,
    db_group_ids: torch.Tensor | None = None,
    model_extra_kwargs: dict[str, object] | None = None,
    db_retriever = None,
    model_kwargs_retriever = None,
    process_index: int = 0,
    num_processes: int = 1,
):
    ensure_dir(out_dir)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = Xdb.device
    model.eval()
    start_idx, end_idx = shard_range(total_gen, process_index, num_processes)
    count = start_idx
    progress = tqdm(
        total=end_idx - start_idx,
        desc="generate",
        dynamic_ncols=True,
        disable=process_index != 0,
    )
    while count < end_idx:
        cur = min(gen_batch, end_idx - count)
        imgs, _ = sample_closedform(
            Xdb=Xdb,
            model=model,
            vae=vae,
            out_dir=out_dir,
            num_gen=cur,
            steps=steps,
            noise_scale=noise_scale,
            t_eps=t_eps,
            decode_batch=decode_batch,
            latent_c=latent_c,
            latent_h=latent_h,
            latent_w=latent_w,
            return_latents=True,
            save_grid=False,
            db_group_ids=db_group_ids,
            model_extra_kwargs=model_extra_kwargs,
            db_retriever=db_retriever,
            model_kwargs_retriever=model_kwargs_retriever,
        )
        for i in range(cur):
            out_path = os.path.join(out_dir, f"{prefix}{count + i:06d}.png")
            save_image(imgs[i], out_path)
        count += cur
        progress.update(cur)
    progress.close()


@torch.no_grad()

def decode_latents(vae, z: torch.Tensor, decode_batch: int) -> torch.Tensor:
    imgs = []
    for s in range(0, z.shape[0], decode_batch):
        imgs.append(vae_decode(vae, z[s:s + decode_batch]))
    return torch.cat(imgs, dim=0)


@torch.no_grad()

def nearest_neighbors(queries: torch.Tensor, db: torch.Tensor, chunk: int):
    device = queries.device
    if queries.ndim > 2:
        queries = queries.reshape(queries.shape[0], -1)
    if db.ndim > 2:
        db = db.reshape(db.shape[0], -1)
    queries = queries.to(device=device)
    B = queries.shape[0]
    best_dist = torch.full((B,), float("inf"), device=device)
    best_idx = torch.zeros((B,), device=device, dtype=torch.long)
    q_norm2 = (queries * queries).sum(dim=1)

    for s in range(0, db.shape[0], chunk):
        e = min(db.shape[0], s + chunk)
        d = db[s:e].to(device=device, non_blocking=(db.device.type == "cpu"))
        d_norm2 = (d * d).sum(dim=1)
        dist = q_norm2[:, None] + d_norm2[None, :] - 2.0 * (queries @ d.t())
        min_dist, min_idx = dist.min(dim=1)
        better = min_dist < best_dist
        best_dist[better] = min_dist[better]
        best_idx[better] = min_idx[better] + s

    return best_idx, best_dist


@torch.no_grad()
def project_mu_cross_to_close_plane(
    model: LearnedPosteriorMean,
    x_latent: torch.Tensor,
    db: torch.Tensor,
    t_eps: float,
    topk: int,
    vae=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vae is None:
        raise ValueError("vae is required for proj_close")
    bsz = x_latent.shape[0]
    t = torch.full((bsz,), 1.0 - float(t_eps), device=x_latent.device, dtype=x_latent.dtype)
    mu_step, mu_cross = model(
        x_latent,
        t,
        db,
        return_mu_ret=True,
    )
    _ = mu_step
    db_weights = None
    if topk <= 0:
        raise ValueError("topk must be > 0")
    mu_flat = mu_cross.reshape(mu_cross.shape[0], -1).to(dtype=torch.float32)
    db_flat = db.reshape(db.shape[0], -1).to(dtype=torch.float32)
    k = min(int(topk), int(db_flat.shape[0]))
    if db_weights is not None:
        if db_weights.ndim != 2 or db_weights.shape[1] != db_flat.shape[0]:
            raise ValueError("db_weights must have shape (B, M) matching DB size")
        top_idx = db_weights.to(dtype=torch.float32).topk(k=k, dim=1).indices
    else:
        scores = mu_flat @ db_flat.t()
        top_idx = scores.topk(k=k, dim=1).indices
    proj_flat = torch.empty_like(mu_flat)
    for b in range(mu_flat.shape[0]):
        neigh = db_flat.index_select(0, top_idx[b])
        center = neigh.mean(dim=0, keepdim=True)
        basis = neigh - center
        vec = (mu_flat[b:b + 1] - center).squeeze(0)
        gram = basis @ basis.t()
        coeff = torch.linalg.pinv(gram) @ (basis @ vec)
        proj_flat[b] = center.squeeze(0) + basis.t() @ coeff
    mu_proj = proj_flat.reshape_as(mu_cross).to(dtype=mu_cross.dtype)
    return mu_cross, mu_proj


@torch.no_grad()
def sample_proj_close_rollout_grid(
    Xdb: torch.Tensor,
    model: LearnedPosteriorMean,
    vae,
    steps: int,
    noise_scale: float,
    t_eps: float,
    decode_batch: int,
    latent_c: int,
    latent_h: int,
    latent_w: int,
    topk: int = 50,
) -> torch.Tensor:
    device = Xdb.device
    x = noise_scale * torch.randn(1, latent_c, latent_h, latent_w, device=device)
    ts = torch.linspace(t_eps, 1.0 - t_eps, steps + 1, device=device)
    mu_proj_steps = []
    mu_steps = []

    for i in range(steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        t = t_curr.expand(1)
        mu_step, mu_cross = model(
            x,
            t,
            Xdb,
            return_mu_ret=True,
        )
        db_weights = None
        if topk <= 0:
            raise ValueError("topk must be > 0")
        mu_flat = mu_cross.reshape(mu_cross.shape[0], -1).to(dtype=torch.float32)
        db_flat = Xdb.reshape(Xdb.shape[0], -1).to(dtype=torch.float32)
        k = min(int(topk), int(db_flat.shape[0]))
        if db_weights is not None:
            if db_weights.ndim != 2 or db_weights.shape[1] != db_flat.shape[0]:
                raise ValueError("db_weights must have shape (B, M) matching DB size")
            top_idx = db_weights.to(dtype=torch.float32).topk(k=k, dim=1).indices
        else:
            scores = mu_flat @ db_flat.t()
            top_idx = scores.topk(k=k, dim=1).indices
        proj_flat = torch.empty_like(mu_flat)
        for b in range(mu_flat.shape[0]):
            neigh = db_flat.index_select(0, top_idx[b])
            center = neigh.mean(dim=0, keepdim=True)
            basis = neigh - center
            vec = (mu_flat[b:b + 1] - center).squeeze(0)
            gram = basis @ basis.t()
            coeff = torch.linalg.pinv(gram) @ (basis @ vec)
            proj_flat[b] = center.squeeze(0) + basis.t() @ coeff
        mu_proj = proj_flat.reshape_as(mu_cross).to(dtype=mu_cross.dtype)
        mu_proj_steps.append(mu_proj.detach())
        mu_steps.append(mu_step.detach())
        dt = t_next - t_curr
        denom = (1.0 - t_curr).clamp(min=t_eps)
        x = x + ((mu_step - x) / denom) * dt

    mu_proj_latents = torch.cat(mu_proj_steps, dim=0)
    mu_latents = torch.cat(mu_steps, dim=0)
    mu_proj_imgs = decode_latents(vae, mu_proj_latents, decode_batch=decode_batch)
    mu_imgs = decode_latents(vae, mu_latents, decode_batch=decode_batch)
    two_row = torch.cat([mu_proj_imgs, mu_imgs], dim=0)
    return make_grid(two_row, nrow=steps)


@torch.no_grad()
def nearest_neighbors_chunked_features(
    query_feats: torch.Tensor,
    db_size: int,
    chunk: int,
    build_db_feats,
):
    device = query_feats.device
    q = query_feats.reshape(query_feats.shape[0], -1)
    q_norm2 = (q * q).sum(dim=1)
    best_dist = torch.full((q.shape[0],), float("inf"), device=device)
    best_idx = torch.zeros((q.shape[0],), device=device, dtype=torch.long)

    for s in range(0, db_size, chunk):
        e = min(db_size, s + chunk)
        d = build_db_feats(s, e).reshape(e - s, -1).to(device=device, dtype=q.dtype)
        d_norm2 = (d * d).sum(dim=1)
        dist = q_norm2[:, None] + d_norm2[None, :] - 2.0 * (q @ d.t())
        min_dist, min_idx = dist.min(dim=1)
        better = min_dist < best_dist
        best_dist[better] = min_dist[better]
        best_idx[better] = min_idx[better] + s
    return best_idx, best_dist


class ClipImageEncoder:
    def __init__(self, model_name: str, device: torch.device, local_files_only: bool = False):
        from transformers import CLIPModel

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, imgs_01: torch.Tensor) -> torch.Tensor:
        imgs = F.interpolate(imgs_01.to(self.device), size=(224, 224), mode="bicubic", align_corners=False)
        imgs = ((imgs - self.mean) / self.std).to(dtype=torch.float32)
        feats = self.model.get_image_features(pixel_values=imgs)
        return F.normalize(feats, dim=-1)


class DinoImageEncoder:
    def __init__(self, model_name: str, device: torch.device, local_repo: str | None = None):
        self.device = device
        self.model_name = model_name
        if local_repo:
            self.model = torch.hub.load(local_repo, model_name, source="local")
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(device)
        self.model.eval().requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.embed_dim = int(getattr(self.model, "embed_dim", 768))

    @torch.no_grad()
    def encode_tokens(self, imgs_01: torch.Tensor, out_grid: tuple[int, int]) -> torch.Tensor:
        imgs = F.interpolate(imgs_01.to(self.device), size=(224, 224), mode="bicubic", align_corners=False)
        imgs = ((imgs - self.mean) / self.std).to(dtype=torch.float32)
        feats = self.model.forward_features(imgs)
        patch_tokens = None
        if isinstance(feats, dict):
            patch_tokens = feats.get("x_norm_patchtokens", None)
            if patch_tokens is None:
                maybe = feats.get("x_prenorm", None)
                if isinstance(maybe, torch.Tensor) and maybe.ndim == 3 and maybe.shape[1] > 1:
                    patch_tokens = maybe[:, 1:, :]
        elif isinstance(feats, torch.Tensor):
            if feats.ndim == 3:
                patch_tokens = feats
            elif feats.ndim == 2:
                patch_tokens = feats[:, None, :]
        if patch_tokens is None:
            raise RuntimeError("Could not extract DINO patch tokens from forward_features output")

        bsz, npatch, dim = patch_tokens.shape
        src = int(math.isqrt(npatch))
        if src * src != npatch:
            raise ValueError(f"DINO patch token count must be square, got {npatch}")
        gh, gw = out_grid
        token_map = patch_tokens.transpose(1, 2).reshape(bsz, dim, src, src)
        token_map = F.interpolate(token_map, size=(gh, gw), mode="bilinear", align_corners=False)
        tokens = token_map.flatten(2).transpose(1, 2).contiguous()
        return F.normalize(tokens, dim=-1)


def _dino_target_grid(cross_patchwise: bool, latent_h: int, latent_w: int, cross_patch_size: int) -> tuple[int, int]:
    if cross_patchwise:
        return latent_h // cross_patch_size, latent_w // cross_patch_size
    return 1, 1


def build_or_load_dino_k_from_indices(
    dataset_name: str,
    split: str,
    image_size: int,
    label_field: str | None,
    label_value: str | None,
    label_split_spec: str | None,
    label_split_seed: int,
    label_split_complement: bool,
    db_indices: torch.Tensor,
    db_root: str,
    N_img: int,
    batch_size: int,
    dino_encoder: DinoImageEncoder,
    accelerator: Accelerator,
    vae_tag: str,
    out_grid: tuple[int, int],
):
    db_dir = os.path.join(db_root, f"db_vae_latents_{vae_tag}")
    if accelerator.is_main_process:
        ensure_dir(db_dir)
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", dino_encoder.model_name)
    gh, gw = out_grid
    dino_path = os.path.join(
        db_dir,
        f"K_dino_{safe_model}_N{N_img}_img{image_size}_gh{gh}_gw{gw}_d{dino_encoder.embed_dim}.npy",
    )
    if os.path.exists(dino_path):
        K = np.load(dino_path)
        K = torch.from_numpy(np.asarray(K)).to(device=accelerator.device, dtype=torch.float32)
        if accelerator.is_main_process:
            logger.info("[db] loaded dino K %s", tuple(K.shape))
        return K

    if accelerator.is_main_process:
        logger.info("[db] building dino K from cached DB indices (%d rows)", int(db_indices.numel()))
        tf = TVT.Compose([
            TVT.Resize(image_size),
            TVT.CenterCrop(image_size),
            TVT.ToTensor(),
        ])
        ds = load_dataset(dataset_name, split=split, streaming=False)
        ds = _apply_label_filter(ds, label_field, label_value)
        ds = _apply_label_split(
            ds,
            label_field=label_field,
            label_split_spec=label_split_spec,
            split_seed=label_split_seed,
            use_complement=label_split_complement,
        )
        idx_all = db_indices.detach().cpu().tolist()
        chunks = []
        for s in range(0, len(idx_all), batch_size):
            e = min(len(idx_all), s + batch_size)
            idx_batch = idx_all[s:e]
            ex_batch = ds.select(idx_batch)
            imgs = []
            for ex in ex_batch:
                im = ex.get("img", None) or ex.get("image", None)
                if im is None:
                    raise KeyError("Example missing 'img' or 'image'")
                imgs.append(tf(im.convert("RGB")))
            imgs = torch.stack(imgs, dim=0).to(accelerator.device)
            tok = dino_encoder.encode_tokens(imgs, out_grid=out_grid).cpu()
            chunks.append(tok)
            if e % 500 == 0 or e == len(idx_all):
                logger.info("[db] dino K %d/%d", e, len(idx_all))
        K = torch.cat(chunks, dim=0).to(torch.float32)
        np.save(dino_path, K.numpy().astype(np.float32))
        logger.info("[db] saved %s shape %s", dino_path, tuple(K.shape))

    accelerator.wait_for_everyone()
    K = np.load(dino_path)
    K = torch.from_numpy(np.asarray(K)).to(device=accelerator.device, dtype=torch.float32)
    return K
