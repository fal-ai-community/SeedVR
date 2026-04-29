from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from omegaconf import ListConfig
from einops import rearrange
from torch.utils.data import DataLoader

from seedvr.common.config import load_config
from seedvr.common.diffusion.types import PredictionType
from seedvr.models.dit import na
from seedvr.models.embeds import PrecomputedEmbeddings
from seedvr.models.video_vae_v3.modules.types import MemoryState
from seedvr.projects.training_hdr.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    write_result_manifest,
)
from seedvr.projects.training_hdr.config import ExtraValidationConfig, TrainingConfig
from seedvr.projects.training_hdr.dataset import (
    SeedVRHDRImageDataset,
    SeedVRHDRVideoDataset,
)
from seedvr.projects.training_hdr.losses import (
    denoise_loss,
    dwt_high_frequency_loss,
    fft_high_frequency_loss,
    image_reconstruction_loss,
    total_variation_map,
)
from seedvr.projects.training_hdr.validation import (
    compute_hdr_metrics,
    linear_hdr_from_target_tensor,
    save_dataset_sample_preview,
    save_triptych,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional at runtime
    wandb = None

try:
    from torchao.float8 import (  # type: ignore
        Float8LinearConfig,
        convert_to_float8_training,
    )
except ImportError:  # pragma: no cover - optional dependency
    convert_to_float8_training = None  # type: ignore[assignment]
    Float8LinearConfig = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SeedVR on image-first HDR pairs."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the JSON training config."
    )
    parser.add_argument(
        "--result-manifest",
        required=True,
        help="Path to write the final training result manifest.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SeedVR HDR training.")
    return torch.device("cuda", 0)


@contextlib.contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def download_checkpoints(
    config: TrainingConfig, repo_root: Path
) -> tuple[Path, Path, Path]:
    spec = config.resolve_checkpoint_spec()
    cache_root = Path(config.base_checkpoint_cache_root)
    if str(cache_root).startswith("/data"):
        ckpt_root = cache_root
    else:
        ckpt_root = repo_root / "ckpts"
    ckpt_dir = ckpt_root / spec.repo_id.replace("/", "__")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    required_paths = [ckpt_dir / spec.dit_filename, ckpt_dir / spec.vae_filename]
    if not all(path.exists() and path.stat().st_size > 0 for path in required_paths):
        lock_path = cache_root / "locks" / f"{spec.repo_id.replace('/', '__')}.lock"
        with file_lock(lock_path):
            if not all(
                path.exists() and path.stat().st_size > 0 for path in required_paths
            ):
                print(
                    "[seedvr-hdr] downloading base checkpoints "
                    f"repo={spec.repo_id} cache_dir={ckpt_dir}"
                )
                snapshot_download(
                    repo_id=spec.repo_id,
                    local_dir=str(ckpt_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    allow_patterns=[
                        spec.dit_filename,
                        spec.vae_filename,
                    ],
                )
    else:
        print(f"[seedvr-hdr] reusing base checkpoint cache dir={ckpt_dir}")
    config_path = repo_root / spec.config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Missing SeedVR config path: {config_path}")
    return ckpt_dir / spec.dit_filename, ckpt_dir / spec.vae_filename, config_path


def sample_training_timesteps(
    batch_size: int,
    schedule_T: float,
    device: torch.device,
) -> torch.Tensor:
    # Matches the logitnormal config in upstream YAML: sigmoid(N(0, 1)) * T
    timesteps = torch.sigmoid(torch.randn(batch_size, device=device)) * schedule_T
    return timesteps.clamp(1.0e-3, schedule_T - 1.0e-3)


def expand_timesteps_to_latents(
    timesteps: torch.Tensor,
    latent_shapes: torch.Tensor,
) -> torch.Tensor:
    """Repeat one batch timestep for each flattened latent token."""
    lengths = latent_shapes.to(device=timesteps.device).prod(dim=1).long()
    return torch.repeat_interleave(timesteps, lengths, dim=0)


def ensure_text_condition_batch(
    txt: torch.Tensor,
    txt_shape: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat shared text conditioning to match a batched latent forward."""
    text_batch_size = int(txt_shape.shape[0])
    if text_batch_size == batch_size:
        return txt, txt_shape
    if text_batch_size != 1:
        raise ValueError(
            "Text conditioning batch size must be 1 or match latent batch size: "
            f"text_batch_size={text_batch_size}, latent_batch_size={batch_size}"
        )
    if batch_size < 1:
        raise ValueError(f"Latent batch size must be positive, got {batch_size}")
    text_sample = na.unflatten(txt, txt_shape)[0]
    return na.flatten([text_sample] * batch_size)


def _enable_safe_attention_fallback() -> None:
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def run_dit_forward(
    *,
    runner: VideoDiffusionInfer,
    config: TrainingConfig,
    vid: torch.Tensor,
    txt: torch.Tensor,
    vid_shape: torch.Tensor,
    txt_shape: torch.Tensor,
    timestep: torch.Tensor,
    stage: str,
    step: int | None = None,
):
    try:
        txt, txt_shape = ensure_text_condition_batch(
            txt=txt,
            txt_shape=txt_shape,
            batch_size=int(vid_shape.shape[0]),
        )
        return runner.dit(
            vid=vid,
            txt=txt,
            vid_shape=vid_shape,
            txt_shape=txt_shape,
            timestep=timestep,
        )
    except Exception as exc:
        context = {
            "stage": stage,
            "step": step,
            "vid_shape": list(vid.shape),
            "txt_shape": list(txt.shape),
            "latent_shapes": vid_shape.detach().cpu().tolist(),
            "text_shapes": txt_shape.detach().cpu().tolist(),
            "timestep_shape": list(timestep.shape),
            "train_width": config.train_width,
            "train_height": config.train_height,
            "data_mode": config.data_mode,
            "phase_jitter": config.phase_jitter,
            "size_jitter_steps": config.size_jitter_steps,
            "use_fa3": config.use_fa3,
            "allow_attention_fallback": config.allow_attention_fallback,
        }
        print(
            "[seedvr-hdr][stage=dit_forward] failed "
            f"context={json.dumps(context, default=str)} error={exc}"
        )
        if not config.allow_attention_fallback:
            raise
        _enable_safe_attention_fallback()
        try:
            return runner.dit(
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=timestep,
            )
        except Exception as retry_exc:
            context["fallback_retry_error"] = str(retry_exc)
            print(
                "[seedvr-hdr][stage=dit_forward] fallback_failed "
                f"context={json.dumps(context, default=str)}"
            )
            raise


def add_condition_noise(
    runner: VideoDiffusionInfer,
    latent: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    aug_noise = torch.randn_like(latent)
    t = torch.tensor(
        [runner.schedule.T * noise_scale],
        device=latent.device,
        dtype=latent.dtype,
    )
    shape = torch.tensor(latent.shape[:3], device=latent.device).unsqueeze(0)
    t = runner.timestep_transform(t, shape).to(latent.dtype)
    return runner.schedule.forward(latent, aug_noise, t)


def configure_runtime_optimizations(
    config: TrainingConfig, device: torch.device
) -> dict[str, Any]:
    runtime_info: dict[str, Any] = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    capability = torch.cuda.get_device_capability(device)
    runtime_info["compute_capability"] = f"{capability[0]}.{capability[1]}"
    wants_fa3 = config.use_fa3 and capability[0] == 9
    runtime_info["fa3_requested"] = config.use_fa3
    runtime_info["fa3_hopper_eligible"] = wants_fa3

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(
            wants_fa3 or config.allow_attention_fallback
        )
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(config.allow_attention_fallback)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(not wants_fa3)

    fa3_available = False
    if wants_fa3:
        try:
            import flash_attn_interface  # noqa: F401

            fa3_available = True
        except Exception:
            fa3_available = False
    runtime_info["fa3_available"] = fa3_available
    runtime_info["fa3_active"] = wants_fa3 and fa3_available
    runtime_info["attention_note"] = (
        "SeedVR uses PyTorch SDPA/varlen attention wrappers. "
        "FA3 is best-effort through installed kernels rather than a direct "
        "set_attention_backend(...) switch."
    )
    return runtime_info


def maybe_enable_mxfp8_training(
    model: torch.nn.Module, config: TrainingConfig
) -> tuple[torch.nn.Module, bool]:
    if not config.use_mxfp8:
        return model, False
    if convert_to_float8_training is None or Float8LinearConfig is None:
        print("[seedvr-hdr] torchao not available; skipping MXFP8 float8 training.")
        return model, False

    def module_filter_fn(mod: torch.nn.Module, fqn: str) -> bool:
        if isinstance(mod, torch.nn.Linear):
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
        return True

    try:
        float8_config = Float8LinearConfig(pad_inner_dim=True)
        model = convert_to_float8_training(
            model,
            module_filter_fn=module_filter_fn,
            config=float8_config,
        )
    except Exception as exc:  # pragma: no cover - runtime fallback
        print(f"[seedvr-hdr] MXFP8 float8 training disabled after failure: {exc}")
        return model, False
    print("[seedvr-hdr] MXFP8 float8 training enabled for DiT linear layers.")
    return model, True


def maybe_compile_dit(
    model: torch.nn.Module, config: TrainingConfig
) -> tuple[torch.nn.Module, bool]:
    if not config.use_torch_compile or not hasattr(torch, "compile"):
        return model, False
    try:
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except Exception:
            pass
        try:
            import torch._inductor.config as inductor_config

            for attr in ("shape_padding", "comprehensive_padding", "force_shape_pad"):
                if hasattr(inductor_config, attr):
                    setattr(inductor_config, attr, False)
        except Exception:
            pass
        compiled = torch.compile(
            model,
            mode=config.torch_compile_mode,
            fullgraph=config.torch_compile_fullgraph,
        )
    except Exception as exc:  # pragma: no cover - runtime fallback
        print(f"[seedvr-hdr] torch.compile disabled after failure: {exc}")
        return model, False
    return compiled, True


def maybe_init_wandb(config: TrainingConfig) -> Any | None:
    if not config.use_wandb:
        return None
    if wandb is None:
        print("[seedvr-hdr] wandb requested but not installed; continuing without it.")
        return None
    wandb_dir = Path(config.output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{re.sub(r'[^A-Za-z0-9_-]+', '-', config.experiment_name)[:96]}-{uuid.uuid4().hex[:8]}"
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")
    for attempt in range(1, 4):
        try:
            return wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name or config.experiment_name,
                id=run_id,
                resume="never",
                reinit=True,
                dir=str(wandb_dir),
                tags=config.wandb_tags or [],
                config=config.to_dict(),
            )
        except Exception as exc:  # pragma: no cover - network/runtime fallback
            print(
                "[seedvr-hdr] wandb init failed "
                f"attempt={attempt}/3 error={type(exc).__name__}: {exc}"
            )
            try:
                wandb.finish(exit_code=1, quiet=True)
            except Exception:
                pass
            if attempt < 3:
                time.sleep(5 * attempt)
    print("[seedvr-hdr] disabling wandb for this run after init failures.")
    return None


def build_wandb_train_metrics(
    *,
    step: int,
    final_metrics: dict[str, float],
    include_step_timing: bool,
) -> dict[str, float]:
    metrics = {
        "step": step,
        "loss": float(final_metrics["loss"]),
        "denoise_loss": float(final_metrics["denoise_loss"]),
        "lr": float(final_metrics["lr"]),
    }
    for key in (
        "denoise_loss_weight",
        "lpips_loss",
        "lpips_loss_weight",
        "dwt_hf_loss",
        "dwt_hf_loss_weight",
        "fft_hf_loss",
        "fft_hf_loss_weight",
        "tv_lpips_loss",
        "tv_lpips_loss_weight",
    ):
        if key in final_metrics:
            metrics[key] = float(final_metrics[key])
    if include_step_timing and "step_seconds" in final_metrics:
        metrics["step_seconds"] = float(final_metrics["step_seconds"])
    return metrics


def metric_safe_name(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip().lower()).strip("_")
    return safe or "extra"


def get_cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    stats = torch.cuda.memory_stats(device)
    gib = float(1024**3)
    return {
        "cuda_allocated_gb": float(torch.cuda.memory_allocated(device) / gib),
        "cuda_reserved_gb": float(torch.cuda.memory_reserved(device) / gib),
        "cuda_max_allocated_gb": float(torch.cuda.max_memory_allocated(device) / gib),
        "cuda_max_reserved_gb": float(torch.cuda.max_memory_reserved(device) / gib),
        "cuda_active_gb": float(stats.get("active_bytes.all.current", 0) / gib),
        "cuda_inactive_split_gb": float(
            stats.get("inactive_split_bytes.all.current", 0) / gib
        ),
    }


def log_cuda_memory(
    prefix: str, device: torch.device, step: int | None = None
) -> dict[str, float]:
    stats = get_cuda_memory_stats(device)
    if stats:
        step_label = f" step={step}" if step is not None else ""
        print(f"[seedvr-hdr] {prefix}{step_label} cuda_memory={stats}")
    return stats


def cleanup_cuda_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def encode_images_to_latents(
    runner: VideoDiffusionInfer,
    images: torch.Tensor,
    *,
    use_tiling: bool,
    use_tqdm: bool,
) -> list[torch.Tensor]:
    if images.numel() == 0:
        return []

    device = images.device
    dtype = getattr(torch, runner.config.vae.dtype)
    scale = runner.config.vae.scaling_factor
    shift = runner.config.vae.get("shifting_factor", 0.0)

    if isinstance(scale, ListConfig):
        scale = torch.tensor(scale, device=device, dtype=dtype)
    if isinstance(shift, ListConfig):
        shift = torch.tensor(shift, device=device, dtype=dtype)

    sample = images.to(device=device, dtype=dtype)
    if sample.ndim == 5:
        sample = rearrange(sample, "b t c h w -> b c t h w")
    if hasattr(runner.vae, "preprocess"):
        sample = runner.vae.preprocess(sample)

    encoded = runner.vae.encode(
        sample,
        use_tiling=use_tiling,
        use_tqdm=use_tqdm,
    ).latent
    encoded = encoded.unsqueeze(2) if encoded.ndim == 4 else encoded
    encoded = rearrange(encoded, "b c ... -> b ... c")
    encoded = (encoded - shift) * scale
    return [latent for latent in encoded]


def log_validation_previews_to_wandb(
    wandb_run: Any | None,
    preview_paths: list[Path],
    step: int,
    preview_captions: list[str] | None = None,
) -> None:
    if wandb_run is None or wandb is None or not preview_paths:
        return
    images = [
        wandb.Image(
            str(path),
            caption=(
                preview_captions[idx]
                if preview_captions is not None and idx < len(preview_captions)
                else f"step={step} preview={idx}"
            ),
        )
        for idx, path in enumerate(preview_paths)
    ]
    wandb.log({"step": step, "validation_previews": images})


def log_dataset_samples_to_wandb(
    wandb_run: Any | None,
    config: TrainingConfig,
    dataset: Dataset,
    *,
    split_name: str,
    step: int = 0,
) -> None:
    if (
        wandb_run is None
        or wandb is None
        or config.wandb_dataset_sample_count <= 0
        or len(dataset) == 0
    ):
        return
    sample_dir = config.output_path / "dataset_samples" / split_name
    sample_paths: list[Path] = []
    captions: list[str] = []
    indices = build_random_quality_preview_indices(
        dataset=dataset,
        config=config,
        num_previews=config.wandb_dataset_sample_count,
        step=step,
        split_name=split_name,
    )
    for preview_index, sample_index in enumerate(indices):
        sample = dataset[sample_index]
        preview_path = save_dataset_sample_preview(
            sample_dir / f"{split_name}_{preview_index:02d}_idx_{sample_index:06d}.png",
            sample["input_sdr"],
            sample["target"],
            config.target_representation,
        )
        sample_paths.append(preview_path)
        captions.append(
            " ".join(
                part
                for part in [
                    f"split={split_name}",
                    f"index={sample_index}",
                    f"scene={sample.get('scene_id', '')}",
                    f"sample={sample.get('sample_id', '')}",
                    f"source={sample.get('source_dataset', '')}",
                ]
                if part
            )
        )
    images = [
        wandb.Image(str(path), caption=captions[index])
        for index, path in enumerate(sample_paths)
    ]
    wandb.log({"step": step, f"dataset_samples/{split_name}": images})


def build_preview_indices(
    *,
    dataset_size: int,
    num_previews: int,
    step: int,
    validate_every: int,
) -> list[int]:
    if dataset_size <= 0 or num_previews <= 0:
        return []
    count = min(dataset_size, num_previews)
    if dataset_size <= count:
        return list(range(dataset_size))

    base_indices = [
        min(dataset_size - 1, int((index * dataset_size) / count))
        for index in range(count)
    ]
    return [int(base_index) for base_index in base_indices]


def build_random_quality_preview_indices(
    *,
    dataset: Dataset,
    config: TrainingConfig,
    num_previews: int,
    step: int,
    split_name: str,
) -> list[int]:
    dataset_size = len(dataset)
    if dataset_size <= 0 or num_previews <= 0:
        return []
    count = min(dataset_size, num_previews)
    cached_good_indices = list(getattr(dataset, "quality_good_indices", []) or [])
    seed_material = f"{config.seed}:{step}:{split_name}:{dataset_size}:dataset_samples"
    rng = random.Random(
        int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    )
    if cached_good_indices:
        rng.shuffle(cached_good_indices)
        selected = [int(index) for index in cached_good_indices[:count]]
        print(
            "[seedvr-hdr][stage=dataset_sample_log] "
            f"split={split_name} selected={selected} source=quality_cache_random"
        )
        return selected

    candidates = list(range(dataset_size))
    rng.shuffle(candidates)
    scan_limit = min(dataset_size, max(count, count * 64))
    selected: list[int] = []
    rejected = 0
    for index in candidates[:scan_limit]:
        try:
            sample = dataset[index]
            usable, _stats = validation_sample_is_preview_usable(sample, config)
        except Exception:
            rejected += 1
            continue
        if usable:
            selected.append(index)
            if len(selected) >= count:
                break
        else:
            rejected += 1
    print(
        "[seedvr-hdr][stage=dataset_sample_log] "
        f"split={split_name} selected={selected} rejected={rejected} "
        "source=random_quality_scan"
    )
    return selected


def _preview_chw_frame(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float()
    if tensor.ndim == 4:
        return tensor[len(tensor) // 2]
    if tensor.ndim == 3:
        return tensor
    raise ValueError(f"Expected CHW/TCHW preview tensor, got {tuple(tensor.shape)}")


def _luma_stats(image: torch.Tensor) -> dict[str, float]:
    image = image.detach().float()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got {tuple(image.shape)}")
    image = image.clamp(0.0, 1.0).unsqueeze(0)
    if max(image.shape[-2:]) > 768:
        scale = 768.0 / float(max(image.shape[-2:]))
        image = F.interpolate(
            image,
            size=(
                max(1, int(round(image.shape[-2] * scale))),
                max(1, int(round(image.shape[-1] * scale))),
            ),
            mode="area",
        )
    image = image.squeeze(0)
    luma = 0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]
    std = float(torch.std(luma, unbiased=False).item())
    dx = torch.abs(luma[:, 1:] - luma[:, :-1]).mean()
    dy = torch.abs(luma[1:, :] - luma[:-1, :]).mean()
    hf = float(((dx + dy) * 0.5).item())
    return {
        "luma_std": std,
        "luma_hf": hf,
        "noise_hf_ratio": float(hf / max(std, 1.0e-6)),
    }


def _target_preview_image(
    target: torch.Tensor, target_representation: str
) -> torch.Tensor:
    target_frame = _preview_chw_frame(target)
    linear = linear_hdr_from_target_tensor(target_frame, target_representation)
    linear = torch.clamp(
        torch.nan_to_num(linear.float(), nan=0.0, posinf=0.0, neginf=0.0), min=0.0
    )
    flat = linear.reshape(-1)
    scale = torch.quantile(flat, 0.995) if flat.numel() else linear.new_tensor(1.0)
    scale = torch.clamp(scale, min=1.0e-6)
    preview = torch.log1p(linear) / torch.log1p(scale)
    return preview.clamp(0.0, 1.0)


def _input_preview_image(input_sdr: torch.Tensor) -> torch.Tensor:
    frame = _preview_chw_frame(input_sdr)
    return frame.clamp(-1.0, 1.0).add(1.0).mul(0.5)


def validation_sample_is_preview_usable(
    sample: dict[str, Any],
    config: TrainingConfig,
) -> tuple[bool, dict[str, float]]:
    input_stats = _luma_stats(_input_preview_image(sample["input_sdr"]))
    target_stats = _luma_stats(
        _target_preview_image(sample["target"], config.target_representation)
    )
    stats = {
        "input_luma_std": input_stats["luma_std"],
        "input_luma_hf": input_stats["luma_hf"],
        "input_noise_hf_ratio": input_stats["noise_hf_ratio"],
        "target_luma_std": target_stats["luma_std"],
        "target_luma_hf": target_stats["luma_hf"],
        "target_noise_hf_ratio": target_stats["noise_hf_ratio"],
    }
    usable = (
        input_stats["luma_std"] >= config.validation_preview_min_luma_std
        and target_stats["luma_std"] >= config.validation_preview_min_luma_std
        and input_stats["luma_hf"] <= config.validation_preview_max_luma_hf
        and target_stats["luma_hf"] <= config.validation_preview_max_luma_hf
        and input_stats["noise_hf_ratio"]
        <= config.validation_preview_max_noise_hf_ratio
        and target_stats["noise_hf_ratio"]
        <= config.validation_preview_max_noise_hf_ratio
    )
    return usable, stats


def build_quality_preview_indices(
    *,
    dataset: Dataset,
    config: TrainingConfig,
    num_previews: int,
    step: int,
    validate_every: int,
    split_name: str,
) -> list[int]:
    dataset_size = len(dataset)
    if dataset_size <= 0 or num_previews <= 0:
        return []
    count = min(dataset_size, num_previews)
    scan_count = min(
        dataset_size,
        max(count, count * max(1, int(config.validation_preview_scan_multiplier))),
    )
    candidates = build_preview_indices(
        dataset_size=dataset_size,
        num_previews=scan_count,
        step=step,
        validate_every=validate_every,
    )
    selected: list[int] = []
    rejected: list[tuple[int, str]] = []
    for index in candidates:
        if index in selected:
            continue
        try:
            sample = dataset[index]
            usable, stats = validation_sample_is_preview_usable(sample, config)
        except Exception as exc:
            rejected.append((index, f"error={type(exc).__name__}:{exc}"))
            continue
        if usable:
            selected.append(index)
            if len(selected) >= count:
                break
        else:
            rejected.append(
                (
                    index,
                    " ".join(
                        f"{key}={value:.4f}" for key, value in sorted(stats.items())
                    ),
                )
            )
    if rejected:
        preview = "; ".join(f"idx={index} {reason}" for index, reason in rejected[:5])
        suffix = "" if len(rejected) <= 5 else f"; +{len(rejected) - 5} more"
        print(
            "[seedvr-hdr][stage=validation_preview_filter] "
            f"split={split_name} selected={selected} rejected={len(rejected)} "
            f"{preview}{suffix}"
        )
    return selected[:count]


def validation_noise_seed(config: TrainingConfig, sample_index: int) -> int:
    return int(config.seed + 17_017 + (sample_index * 1_009))


def sampler_validation_noise_seed(config: TrainingConfig, sample_index: int) -> int:
    return int(config.sampler_validation_seed + 23_017 + (sample_index * 1_009))


def collect_base_preview_indices(
    config: TrainingConfig,
    dataset: Dataset,
    *,
    num_validation_samples: int | None = None,
    sampler_validation_samples: int | None = None,
    split_name: str = "val",
) -> list[int]:
    dataset_size = len(dataset)
    if dataset_size <= 0:
        return []
    validation_count = (
        config.num_validation_samples
        if num_validation_samples is None
        else num_validation_samples
    )
    sampler_count = (
        config.sampler_validation_samples
        if sampler_validation_samples is None
        else sampler_validation_samples
    )
    indices: set[int] = set()
    sampler_preview_count = min(sampler_count, dataset_size)
    indices.update(
        build_quality_preview_indices(
            dataset=dataset,
            config=config,
            num_previews=validation_count,
            step=1,
            validate_every=config.validate_every,
            split_name=split_name,
        )
    )
    indices.update(
        build_quality_preview_indices(
            dataset=dataset,
            config=config,
            num_previews=sampler_preview_count,
            step=1,
            validate_every=config.validate_every,
            split_name=f"{split_name}/sampler",
        )
    )
    return sorted(indices)


def single_sample_to_batch(sample: dict[str, Any]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
    return batch


def aggregate_metric_rows(
    rows: list[dict[str, float]],
    *,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metric_names = sorted(set().union(*(set(row) for row in rows)))
    for name in metric_names:
        values = [float(row[name]) for row in rows if name in row]
        finite_values = [value for value in values if np.isfinite(value)]
        if finite_values:
            metrics[f"{prefix}_{name}"] = float(np.mean(finite_values))
        invalid_count = len(rows) - len(finite_values)
        if invalid_count:
            metrics[f"{prefix}_{name}_invalid_count"] = float(invalid_count)
    return metrics


def evaluate_validation_batch(
    *,
    config: TrainingConfig,
    runner: VideoDiffusionInfer,
    text_pos_embeds: torch.Tensor,
    text_pos_shapes: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
    noise_seed: int | None = None,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    input_images = batch["input_sdr"].to(device)
    target_images = batch["target"].to(device)
    target_latents = encode_images_to_latents(
        runner,
        target_images,
        use_tiling=config.vae_use_tiling,
        use_tqdm=config.vae_use_tqdm,
    )
    input_latents = encode_images_to_latents(
        runner,
        input_images,
        use_tiling=config.vae_use_tiling,
        use_tqdm=config.vae_use_tqdm,
    )
    noisy_inputs = [
        add_condition_noise(runner, latent, runner.config.condition.noise_scale)
        for latent in input_latents
    ]
    conditions = [
        runner.get_condition(target_latent, input_latent, task="sr")
        for target_latent, input_latent in zip(target_latents, noisy_inputs)
    ]

    latents_flat, latent_shapes = na.flatten(target_latents)
    cond_flat, _ = na.flatten(conditions)
    if noise_seed is None:
        noise = torch.randn_like(latents_flat)
    else:
        generator = torch.Generator(device=device).manual_seed(int(noise_seed))
        noise = torch.randn(
            latents_flat.shape,
            device=latents_flat.device,
            dtype=latents_flat.dtype,
            generator=generator,
        )
    timesteps = torch.full(
        (len(target_latents),),
        runner.schedule.T * 0.5,
        device=device,
        dtype=latents_flat.dtype,
    )
    timesteps = runner.timestep_transform(timesteps, latent_shapes).to(
        latents_flat.dtype
    )
    diffusion_timesteps = expand_timesteps_to_latents(timesteps, latent_shapes).to(
        latents_flat.dtype
    )
    x_t = runner.schedule.forward(latents_flat, noise, diffusion_timesteps)
    target = runner.schedule.convert_to_pred(
        latents_flat,
        noise,
        diffusion_timesteps,
        PredictionType.v_lerp,
    )
    prediction = run_dit_forward(
        runner=runner,
        config=config,
        vid=torch.cat([x_t, cond_flat], dim=-1),
        txt=text_pos_embeds,
        vid_shape=latent_shapes,
        txt_shape=text_pos_shapes,
        timestep=timesteps,
        stage="validation",
    ).vid_sample
    denoise = denoise_loss(prediction, target).item()
    pred_x0, _ = runner.schedule.convert_from_pred(
        prediction,
        PredictionType.v_lerp,
        x_t,
        diffusion_timesteps,
    )
    predicted_images = decode_latents_to_images(runner, pred_x0, latent_shapes)
    hdr_metrics = compute_hdr_metrics(
        predicted_image=predicted_images[0],
        target_image=target_images[0],
        target_representation=config.target_representation,
    )

    del (
        target_latents,
        input_latents,
        noisy_inputs,
        conditions,
        latents_flat,
        latent_shapes,
        cond_flat,
        noise,
        timesteps,
        diffusion_timesteps,
        x_t,
        target,
        prediction,
        pred_x0,
    )
    return denoise, hdr_metrics, input_images, target_images, predicted_images


def build_base_validation_prediction_cache(
    *,
    config: TrainingConfig,
    runner: VideoDiffusionInfer,
    positive_embeddings: tuple[torch.Tensor, torch.Tensor],
    val_loader: DataLoader,
    device: torch.device,
    num_validation_samples: int | None = None,
    sampler_validation_samples: int | None = None,
    split_name: str = "base_cache",
) -> dict[int, torch.Tensor]:
    dataset = val_loader.dataset
    indices = collect_base_preview_indices(
        config,
        dataset,
        num_validation_samples=num_validation_samples,
        sampler_validation_samples=sampler_validation_samples,
        split_name=split_name,
    )
    if not indices:
        return {}
    text_pos_embeds, text_pos_shapes = positive_embeddings
    cache: dict[int, torch.Tensor] = {}
    was_training = runner.dit.training
    runner.dit.eval()
    print("[seedvr-hdr] caching base validation predictions " f"count={len(indices)}")
    with torch.no_grad():
        for sample_index in indices:
            sample = dataset[sample_index]
            batch = single_sample_to_batch(sample)
            _denoise, _hdr_metrics, input_images, target_images, predicted_images = (
                evaluate_validation_batch(
                    config=config,
                    runner=runner,
                    text_pos_embeds=text_pos_embeds,
                    text_pos_shapes=text_pos_shapes,
                    batch=batch,
                    device=device,
                    noise_seed=validation_noise_seed(config, sample_index),
                )
            )
            cache[sample_index] = predicted_images[0].detach().cpu().to(torch.float16)
            del input_images, target_images, predicted_images
            cleanup_cuda_memory(device)
    if was_training:
        runner.dit.train()
    return cache


def _sampler_prediction_to_target_layout(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.detach().float()
    if target.ndim == 3:
        if prediction.ndim == 4 and prediction.shape[0] == 3:
            return prediction[:, 0]
        if prediction.ndim == 4 and prediction.shape[1] == 3:
            return prediction[0]
        return prediction
    if target.ndim == 4:
        if prediction.ndim == 4 and prediction.shape[0] == 3:
            return prediction.permute(1, 0, 2, 3).contiguous()
        return prediction
    return prediction


def evaluate_sampler_validation_batch(
    *,
    config: TrainingConfig,
    runner: VideoDiffusionInfer,
    positive_embeddings: tuple[torch.Tensor, torch.Tensor],
    batch: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    input_images = batch["input_sdr"].to(device)
    target_images = batch["target"].to(device)
    text_pos_embeds, text_pos_shapes = positive_embeddings
    text_positive = na.unflatten(text_pos_embeds, text_pos_shapes)[0]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_latents = encode_images_to_latents(
        runner,
        input_images,
        use_tiling=config.vae_use_tiling,
        use_tqdm=False,
    )
    dit_dtype = next(runner.dit.parameters()).dtype
    input_latents = [latent.to(dtype=dit_dtype) for latent in input_latents]
    noisy_inputs = [
        add_condition_noise(runner, latent, runner.config.condition.noise_scale)
        for latent in input_latents
    ]
    conditions = [
        runner.get_condition(latent, noisy, task="sr").to(dtype=dit_dtype)
        for latent, noisy in zip(input_latents, noisy_inputs)
    ]
    noises = [torch.randn_like(latent, dtype=dit_dtype) for latent in input_latents]
    with torch.autocast(device_type="cuda", dtype=dit_dtype):
        predicted_samples = runner.inference(
            noises=noises,
            conditions=conditions,
            texts_pos=[text_positive],
            texts_neg=[text_positive],
            cfg_scale=config.sampler_validation_guidance_scale,
        )
    predicted_image = _sampler_prediction_to_target_layout(
        predicted_samples[0],
        target_images[0],
    ).to(device=device, dtype=torch.float32)
    hdr_metrics = compute_hdr_metrics(
        predicted_image=predicted_image,
        target_image=target_images[0],
        target_representation=config.target_representation,
    )
    del input_latents, noisy_inputs, conditions, noises, predicted_samples
    return hdr_metrics, input_images, target_images, predicted_image.unsqueeze(0)


def select_trainable_parameters(model: torch.nn.Module, strategy: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def enable_if(predicate) -> None:
        for name, parameter in model.named_parameters():
            if predicate(name):
                parameter.requires_grad_(True)

    def block_index(name: str) -> int | None:
        if not name.startswith("blocks."):
            return None
        parts = name.split(".")
        if len(parts) < 2 or not parts[1].isdigit():
            return None
        return int(parts[1])

    def is_io_adapter(name: str) -> bool:
        return name.startswith(
            ("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in")
        )

    def enable_block_range(
        start: int, end: int, module_filter: str | None = None
    ) -> None:
        if start < 0 or end < start:
            raise ValueError(f"Invalid trainable block range: {start}-{end}")
        if num_blocks <= 0:
            raise ValueError("DiT model exposes no blocks for block trainable_strategy")
        if start >= num_blocks:
            raise ValueError(
                f"Block index {start} is outside model block range 0-{num_blocks - 1}"
            )
        end = min(end, num_blocks - 1)

        def predicate(name: str) -> bool:
            index = block_index(name)
            if index is None:
                return is_io_adapter(name)
            if index < start or index > end:
                return False
            return module_filter is None or module_filter in name

        enable_if(predicate)

    def parse_block_suffix(prefix: str) -> int:
        suffix = strategy.removeprefix(prefix)
        if not suffix.isdigit():
            raise ValueError(f"Invalid trainable_strategy {strategy!r}")
        return int(suffix)

    block_count = getattr(model, "blocks", None)
    num_blocks = len(block_count) if block_count is not None else 0
    if num_blocks == 0:
        block_indices = [
            index
            for name, _parameter in model.named_parameters()
            if (index := block_index(name)) is not None
        ]
        num_blocks = max(block_indices) + 1 if block_indices else 0
    if (
        strategy in {"top8", "top16", "attention_top16", "mlp_top16"}
        and num_blocks <= 0
    ):
        raise ValueError(
            f"DiT model exposes no blocks for trainable_strategy {strategy!r}"
        )
    top8_start = max(0, num_blocks - 8)
    top16_start = max(0, num_blocks - 16)

    def block_at_or_after(name: str, start: int) -> bool:
        index = block_index(name)
        return index is not None and index >= start

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif strategy == "top8":
        enable_if(
            lambda name: (block_at_or_after(name, top8_start) or is_io_adapter(name))
        )
    elif strategy == "top16":
        enable_if(
            lambda name: (block_at_or_after(name, top16_start) or is_io_adapter(name))
        )
    elif strategy == "emb_out":
        enable_if(
            lambda name: name.startswith(
                ("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in")
            )
        )
    elif strategy == "attention_top16":
        enable_if(
            lambda name: (block_at_or_after(name, top16_start) and "attn" in name)
            or is_io_adapter(name)
        )
    elif strategy == "mlp_top16":
        enable_if(
            lambda name: (block_at_or_after(name, top16_start) and "mlp" in name)
            or is_io_adapter(name)
        )
    elif strategy.startswith("block_"):
        index = parse_block_suffix("block_")
        enable_block_range(index, index)
    elif strategy.startswith("blocks_"):
        parts = strategy.split("_")
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            raise ValueError(f"Invalid trainable_strategy {strategy!r}")
        enable_block_range(int(parts[1]), int(parts[2]))
    elif strategy.startswith("attention_block_"):
        index = parse_block_suffix("attention_block_")
        enable_block_range(index, index, module_filter="attn")
    elif strategy.startswith("mlp_block_"):
        index = parse_block_suffix("mlp_block_")
        enable_block_range(index, index, module_filter="mlp")
    else:
        raise ValueError(f"Unsupported trainable_strategy '{strategy}'")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_trainable == 0:
        raise RuntimeError(f"Trainable strategy '{strategy}' disabled every parameter.")
    return num_trainable


def build_optimizer(
    config: TrainingConfig,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    trainable_params = build_trainable_param_groups(config, model)
    if config.optimizer_type.startswith("heavyball_"):
        import heavyball

        common_kwargs = {
            "lr": config.learning_rate,
            "betas": (config.adam_beta1, config.adam_beta2),
            "weight_decay": config.weight_decay,
        }
        if config.optimizer_type == "heavyball_adamw":
            return heavyball.AdamW(trainable_params, **common_kwargs)
        if config.optimizer_type == "heavyball_laprop":
            return heavyball.LaProp(trainable_params, **common_kwargs)
        if config.optimizer_type == "heavyball_muonadamw":
            return heavyball.MuonAdamW(trainable_params, **common_kwargs)
        if config.optimizer_type == "heavyball_sfadamw":
            return heavyball.SFAdamW(
                trainable_params,
                warmup_steps=config.warmup_steps,
                **common_kwargs,
            )
        if config.optimizer_type == "heavyball_soap":
            return heavyball.SOAP(
                trainable_params,
                warmup_steps=config.warmup_steps,
                **common_kwargs,
            )
        if config.optimizer_type == "heavyball_psgdkron":
            return heavyball.PSGDKron(
                trainable_params,
                warmup_steps=config.warmup_steps,
                **common_kwargs,
            )
        if config.optimizer_type == "heavyball_lather":
            return heavyball.LATHER(
                trainable_params,
                warmup_steps=config.warmup_steps,
                **common_kwargs,
            )

    if config.optimizer_type == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
    if config.optimizer_type == "adamw_fused":
        return torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
            fused=True,
        )
    raise ValueError(f"Unsupported optimizer_type '{config.optimizer_type}'")


def build_trainable_param_groups(
    config: TrainingConfig,
    model: torch.nn.Module,
) -> list[torch.nn.Parameter] | list[dict[str, object]]:
    trainable_named_params = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if config.layerwise_lr_decay > 1.0:
        raise ValueError("layerwise_lr_decay must be <= 1.0")
    if config.layerwise_lr_decay == 1.0:
        return [parameter for _, parameter in trainable_named_params]
    if config.layerwise_lr_decay <= 0.0:
        raise ValueError("layerwise_lr_decay must be > 0.0")

    blocks = getattr(model, "blocks", None)
    num_blocks = len(blocks) if blocks is not None else 0
    if num_blocks <= 0:
        return [parameter for _, parameter in trainable_named_params]

    grouped: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in trainable_named_params:
        block_idx = _dit_block_index_from_param_name(name)
        if block_idx is None:
            scale = 1.0
        else:
            # Later DiT blocks stay closest to the base LR; earlier blocks decay.
            depth_from_top = max(0, num_blocks - 1 - block_idx)
            scale = config.layerwise_lr_decay**depth_from_top
            scale = max(float(config.layerwise_lr_min_scale), float(scale))
        grouped.setdefault(round(float(scale), 8), []).append(parameter)

    return [
        {"params": params, "lr": config.learning_rate * scale}
        for scale, params in sorted(grouped.items())
    ]


def _dit_block_index_from_param_name(name: str) -> int | None:
    if not name.startswith("blocks."):
        return None
    parts = name.split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


def build_scheduler(
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    if config.optimizer_type == "heavyball_sfadamw":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    if config.scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    if config.scheduler_type == "cosine":

        def lr_lambda(step: int) -> float:
            if config.warmup_steps > 0 and step < config.warmup_steps:
                return max(1.0e-8, float(step + 1) / float(config.warmup_steps))
            total_steps = config.scheduler_total_steps or config.steps
            progress_denominator = max(1, total_steps - config.warmup_steps)
            progress = min(
                1.0,
                max(
                    0.0, float(step - config.warmup_steps) / float(progress_denominator)
                ),
            )
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unsupported scheduler_type '{config.scheduler_type}'")


def current_loss_weight(target_weight: float, warmup_steps: int, step: int) -> float:
    if target_weight <= 0.0:
        return 0.0
    if warmup_steps <= 0:
        return target_weight
    ramp = min(1.0, float(step) / float(warmup_steps))
    return target_weight * ramp


def build_lpips_model(config: TrainingConfig, device: torch.device):
    if config.lpips_loss_weight <= 0.0 and config.tv_lpips_loss_weight <= 0.0:
        return None
    try:
        import lpips  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "lpips_loss_weight > 0 but the lpips package is not installed."
        ) from exc

    model = lpips.LPIPS(net=config.lpips_net).eval().to(device)
    model.requires_grad_(False)
    return model


def _as_lpips_input(tensor: torch.Tensor, resize: int) -> torch.Tensor:
    if tensor.ndim == 5:
        b, t, c, h, w = tensor.shape
        tensor = tensor.reshape(b * t, c, h, w)
    elif tensor.ndim != 4:
        raise ValueError(
            f"Expected BCHW or BTCHW tensor for LPIPS, got {tuple(tensor.shape)}"
        )
    tensor = tensor[:, :3].float().clamp(0.0, 1.0)
    if resize > 0 and min(tensor.shape[-2:]) > resize:
        tensor = F.interpolate(
            tensor,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
    return tensor.mul(2.0).sub(1.0)


def lpips_reconstruction_loss(
    model,
    prediction: torch.Tensor,
    target: torch.Tensor,
    resize: int,
) -> torch.Tensor:
    pred = _as_lpips_input(prediction, resize=resize)
    target = _as_lpips_input(
        target.to(device=prediction.device, dtype=prediction.dtype),
        resize=resize,
    )
    return model(pred, target).mean()


def tv_lpips_reconstruction_loss(
    model,
    prediction: torch.Tensor,
    target: torch.Tensor,
    resize: int,
    gamma: float,
) -> torch.Tensor:
    pred_tv = total_variation_map(prediction)
    target_tv = total_variation_map(
        target.to(device=prediction.device, dtype=prediction.dtype)
    )
    gamma = max(0.1, float(gamma))
    pred_tv = pred_tv.clamp_min(0.0).pow(gamma)
    target_tv = target_tv.clamp_min(0.0).pow(gamma)
    return model(
        _as_lpips_input(pred_tv, resize=resize),
        _as_lpips_input(target_tv, resize=resize),
    ).mean()


def build_validation_dataloader(
    config: TrainingConfig,
    *,
    dataset_root: str,
    manifest_path: str,
    data_mode: str,
) -> DataLoader:
    dataset_cls = (
        SeedVRHDRVideoDataset if data_mode == "video" else SeedVRHDRImageDataset
    )
    val_dataset = dataset_cls(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        train_height=config.train_height,
        train_width=config.train_width,
        random_crop=False,
        seed=config.seed,
        target_representation=config.target_representation,
        runtime_cache_root=config.runtime_cache_root,
        download_timeout_seconds=config.remote_download_timeout_seconds,
        download_retries=config.remote_download_retries,
        cache_rendered_sdr_inputs=config.cache_rendered_sdr_inputs,
        cache_compressed_targets=config.cache_compressed_targets,
        phase_jitter=False,
        size_jitter_steps=0,
        jpeg_roundtrip_prob=0.0,
        filter_bad_samples=False,
        bad_sample_max_retries=config.bad_sample_max_retries,
        bad_sample_min_luma_std=config.bad_sample_min_luma_std,
        bad_sample_max_luma_hf=config.bad_sample_max_luma_hf,
        bad_sample_max_noise_hf_ratio=config.bad_sample_max_noise_hf_ratio,
        use_quality_cache=False,
        quality_cache_root=config.quality_cache_root,
        quality_cache_rebuild=False,
        quality_cache_build_on_init=False,
        quality_cache_workers=config.quality_cache_workers,
    )
    worker_count = min(2, config.num_workers)
    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=bool(
            config.dataloader_persistent_workers and worker_count > 0
        ),
        prefetch_factor=(
            config.dataloader_prefetch_factor if worker_count > 0 else None
        ),
    )


def build_extra_validation_dataloaders(
    config: TrainingConfig,
) -> list[tuple[ExtraValidationConfig, DataLoader]]:
    return [
        (
            extra,
            build_validation_dataloader(
                config,
                dataset_root=extra.dataset_root,
                manifest_path=extra.val_manifest,
                data_mode=extra.data_mode,
            ),
        )
        for extra in config.extra_validation_datasets
    ]


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    dataset_cls = (
        SeedVRHDRVideoDataset if config.data_mode == "video" else SeedVRHDRImageDataset
    )
    train_dataset = dataset_cls(
        dataset_root=config.dataset_root,
        manifest_path=config.train_manifest,
        train_height=config.train_height,
        train_width=config.train_width,
        random_crop=True,
        seed=config.seed,
        target_representation=config.target_representation,
        runtime_cache_root=config.runtime_cache_root,
        download_timeout_seconds=config.remote_download_timeout_seconds,
        download_retries=config.remote_download_retries,
        cache_rendered_sdr_inputs=config.cache_rendered_sdr_inputs,
        cache_compressed_targets=config.cache_compressed_targets,
        phase_jitter=config.phase_jitter,
        phase_jitter_max_pixels=config.phase_jitter_max_pixels,
        size_jitter_steps=config.size_jitter_steps,
        jpeg_roundtrip_prob=config.jpeg_roundtrip_prob,
        jpeg_roundtrip_min_quality=config.jpeg_roundtrip_min_quality,
        jpeg_roundtrip_max_quality=config.jpeg_roundtrip_max_quality,
        filter_bad_samples=config.filter_bad_samples,
        bad_sample_max_retries=config.bad_sample_max_retries,
        bad_sample_min_luma_std=config.bad_sample_min_luma_std,
        bad_sample_max_luma_hf=config.bad_sample_max_luma_hf,
        bad_sample_max_noise_hf_ratio=config.bad_sample_max_noise_hf_ratio,
        use_quality_cache=config.use_quality_cache,
        quality_cache_root=config.quality_cache_root,
        quality_cache_rebuild=config.quality_cache_rebuild,
        quality_cache_build_on_init=config.quality_cache_build_on_init,
        quality_cache_workers=config.quality_cache_workers,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=bool(
            config.dataloader_persistent_workers and config.num_workers > 0
        ),
        prefetch_factor=(
            config.dataloader_prefetch_factor if config.num_workers > 0 else None
        ),
    )
    val_loader = build_validation_dataloader(
        config,
        dataset_root=config.dataset_root,
        manifest_path=config.val_manifest,
        data_mode=config.data_mode,
    )
    return train_loader, val_loader


def build_runner(
    config: TrainingConfig,
    repo_root: Path,
    device: torch.device,
) -> tuple[
    VideoDiffusionInfer,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    dict[str, Any],
]:
    from seedvr.projects.video_diffusion_sr.infer import VideoDiffusionInfer

    if not config.freeze_vae:
        raise NotImplementedError(
            "Phase-1 SeedVR HDR training currently supports freeze_vae=True only."
        )

    dit_ckpt_path, vae_ckpt_path, model_config_path = download_checkpoints(
        config, repo_root
    )
    model_config = load_config(str(model_config_path), root_dir=str(repo_root))
    runner = VideoDiffusionInfer(model_config)
    runner.configure_dit_model(device=str(device), checkpoint=str(dit_ckpt_path))
    runner.dit.to(device=device, dtype=torch.bfloat16)
    runner.configure_vae_model(device=str(device), checkpoint=str(vae_ckpt_path))
    runner.configure_diffusion()
    runner.vae.requires_grad_(False).eval()
    runner.dit.train()
    num_trainable = select_trainable_parameters(runner.dit, config.trainable_strategy)

    runtime_info = configure_runtime_optimizations(config, device)
    runner.dit, mxfp8_active = maybe_enable_mxfp8_training(runner.dit, config)
    runtime_info["mxfp8_requested"] = config.use_mxfp8
    runtime_info["mxfp8_active"] = mxfp8_active
    runner.dit, compiled = maybe_compile_dit(runner.dit, config)
    runtime_info["torch_compile_active"] = compiled
    runtime_info["model_config_path"] = str(model_config_path)
    runtime_info["dit_checkpoint"] = str(dit_ckpt_path)
    runtime_info["vae_checkpoint"] = str(vae_ckpt_path)
    runtime_info["trainable_strategy"] = config.trainable_strategy
    runtime_info["num_trainable_parameters"] = num_trainable
    runtime_info["target_representation"] = config.target_representation
    runtime_info["data_mode"] = config.data_mode
    runtime_info["clip_length"] = config.clip_length
    runtime_info["frame_stride"] = config.frame_stride

    positive, negative = PrecomputedEmbeddings.default(
        device=device,
        dtype=next(runner.dit.parameters()).dtype,
    ).get()
    return runner, positive, negative, runtime_info


def decode_latents_to_images(
    runner: VideoDiffusionInfer,
    latents_flat: torch.Tensor,
    latent_shapes: torch.Tensor,
) -> torch.Tensor:
    latents = na.unflatten(latents_flat, latent_shapes)
    if not latents:
        return torch.empty(0, device=latents_flat.device, dtype=torch.float32)
    device = latents_flat.device
    dtype = getattr(torch, runner.config.vae.dtype)
    scale = runner.config.vae.scaling_factor
    shift = runner.config.vae.get("shifting_factor", 0.0)

    if isinstance(scale, ListConfig):
        scale = torch.tensor(scale, device=device, dtype=dtype)
    if isinstance(shift, ListConfig):
        shift = torch.tensor(shift, device=device, dtype=dtype)

    # Use a cloned batch tensor so the decode branch cannot mutate views
    # that autograd is still tracking for the denoise / latent losses.
    latent_batch = (
        torch.stack([latent for latent in latents], dim=0)
        .to(
            device=device,
            dtype=dtype,
        )
        .clone()
    )
    latent_batch = latent_batch / scale + shift
    latent_batch = rearrange(latent_batch, "b t h w c -> b c t h w")

    # The public VAE decode path uses tiled helpers that are wrapped in
    # @torch.no_grad() and perform in-place clamps. Bypass that path for
    # training-time image loss so gradients stay valid.
    if hasattr(runner.vae, "_decode"):
        sample = runner.vae._decode(latent_batch, memory_state=MemoryState.DISABLED)
    else:
        sample = runner.vae.decode(
            latent_batch,
            use_tiling=False,
            use_tqdm=False,
        ).sample
    if sample.ndim == 5:
        sample = rearrange(sample, "b c t h w -> b t c h w")
    return sample.to(device=device, dtype=torch.float32)


def run_validation(
    config: TrainingConfig,
    runner: VideoDiffusionInfer,
    positive_embeddings: tuple[torch.Tensor, torch.Tensor],
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    step: int,
    base_prediction_cache: dict[int, torch.Tensor] | None = None,
    metric_prefix: str = "val",
    sampler_metric_prefix: str = "sampler_val",
    preview_subdir: str = "validation",
    num_validation_samples: int | None = None,
    sampler_validation_samples: int | None = None,
) -> tuple[dict[str, float], list[Path], list[str]]:
    text_pos_embeds, text_pos_shapes = positive_embeddings
    preview_paths: list[Path] = []
    preview_captions: list[str] = []
    losses: list[float] = []
    hdr_metric_rows: list[dict[str, float]] = []
    sampler_metric_rows: list[dict[str, float]] = []
    validation_count = (
        config.num_validation_samples
        if num_validation_samples is None
        else num_validation_samples
    )
    sampler_count = (
        config.sampler_validation_samples
        if sampler_validation_samples is None
        else sampler_validation_samples
    )
    sampler_validate_every = int(config.sampler_validate_every or config.validate_every)
    run_sampler_validation = sampler_count > 0 and (
        step % sampler_validate_every == 0 or step == config.steps
    )
    if not run_sampler_validation:
        sampler_count = 0
    preview_dir = config.output_path / preview_subdir
    preview_dir.mkdir(parents=True, exist_ok=True)
    dataset = val_loader.dataset
    metric_indices = build_quality_preview_indices(
        dataset=dataset,
        config=config,
        num_previews=validation_count,
        step=step,
        validate_every=config.validate_every,
        split_name=preview_subdir,
    )

    runner.dit.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    if config.debug_cuda_memory:
        log_cuda_memory("validation_start", device, step)
    with torch.no_grad():
        for sample_index in metric_indices:
            sample = dataset[sample_index]
            batch = single_sample_to_batch(sample)

            denoise, hdr_metrics, input_images, target_images, predicted_images = (
                evaluate_validation_batch(
                    config=config,
                    runner=runner,
                    text_pos_embeds=text_pos_embeds,
                    text_pos_shapes=text_pos_shapes,
                    batch=batch,
                    device=device,
                    noise_seed=validation_noise_seed(config, sample_index),
                )
            )
            losses.append(denoise)
            hdr_metric_rows.append(hdr_metrics)
            preview_path = save_triptych(
                preview_dir / f"step_{step:06d}_{len(preview_paths):02d}.png",
                input_images[0].cpu(),
                predicted_images[0].cpu(),
                target_images[0].cpu(),
                target_representation=config.target_representation,
                base_predicted_image=(base_prediction_cache or {}).get(sample_index),
            )
            preview_paths.append(preview_path)
            preview_captions.append(
                " ".join(
                    part
                    for part in [
                        f"step={step}",
                        f"preview={len(preview_paths) - 1}",
                        f"index={sample_index}",
                        f"scene={sample.get('scene_id', '')}",
                        f"sample={sample.get('sample_id', '')}",
                        f"source={sample.get('source_dataset', '')}",
                    ]
                    if part
                )
            )
            if config.debug_cuda_memory:
                log_cuda_memory(f"validation_sample idx={sample_index}", device, step)

            del (
                input_images,
                target_images,
                predicted_images,
            )
            cleanup_cuda_memory(device)

        if sampler_count > 0:
            sampler_indices = build_quality_preview_indices(
                dataset=dataset,
                config=config,
                num_previews=min(sampler_count, len(dataset)),
                step=step,
                validate_every=config.validate_every,
                split_name=f"{preview_subdir}/sampler",
            )
            for sampler_index, sample_index in enumerate(sampler_indices):
                sample = dataset[sample_index]
                batch = single_sample_to_batch(sample)
                hdr_metrics, input_images, target_images, predicted_images = (
                    evaluate_sampler_validation_batch(
                        config=config,
                        runner=runner,
                        positive_embeddings=positive_embeddings,
                        batch=batch,
                        device=device,
                        seed=sampler_validation_noise_seed(config, sample_index),
                    )
                )
                sampler_metric_rows.append(hdr_metrics)
                preview_path = save_triptych(
                    preview_dir / f"step_{step:06d}_sampler_{sampler_index:02d}.png",
                    input_images[0].cpu(),
                    predicted_images[0].cpu(),
                    target_images[0].cpu(),
                    target_representation=config.target_representation,
                    base_predicted_image=(base_prediction_cache or {}).get(
                        sample_index
                    ),
                )
                preview_paths.append(preview_path)
                preview_captions.append(
                    " ".join(
                        part
                        for part in [
                            f"step={step}",
                            f"sampler_preview={sampler_index}",
                            f"index={sample_index}",
                            f"scene={sample.get('scene_id', '')}",
                            f"sample={sample.get('sample_id', '')}",
                            f"source={sample.get('source_dataset', '')}",
                            f"cfg={config.sampler_validation_guidance_scale}",
                        ]
                        if part
                    )
                )
                del input_images, target_images, predicted_images
                cleanup_cuda_memory(device)

    runner.dit.train()
    if hasattr(optimizer, "train"):
        optimizer.train()
    metrics = {
        f"{metric_prefix}_denoise_loss": float(np.mean(losses)) if losses else 0.0,
    }
    if hdr_metric_rows:
        metrics.update(aggregate_metric_rows(hdr_metric_rows, prefix=metric_prefix))
    if sampler_metric_rows:
        metrics.update(
            aggregate_metric_rows(sampler_metric_rows, prefix=sampler_metric_prefix)
        )
    if config.debug_cuda_memory:
        log_cuda_memory("validation_end", device, step)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    return metrics, preview_paths, preview_captions


def maybe_resume_training(
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    if not config.resume_from_checkpoint:
        return 1, {}

    checkpoint = load_checkpoint(
        checkpoint_path=config.resume_from_checkpoint,
        model=model,
        optimizer=optimizer if config.resume_optimizer_state else None,
        scheduler=scheduler if config.resume_scheduler_state else None,
        device=device,
        restore_rng=config.resume_rng_state,
    )
    resumed_step = int(checkpoint.get("step", 0))
    if resumed_step >= config.steps:
        print(
            f"Resume checkpoint is already at step {resumed_step}, "
            f"which is not below requested steps={config.steps}; writing result "
            "manifest without additional training."
        )
        return resumed_step + 1, checkpoint.get("metrics") or {}
    resumed_metrics = checkpoint.get("metrics") or {}
    print(
        f"[seedvr-hdr] resumed from checkpoint={config.resume_from_checkpoint} "
        f"step={resumed_step} "
        f"optimizer_state={config.resume_optimizer_state} "
        f"scheduler_state={config.resume_scheduler_state} "
        f"rng_state={config.resume_rng_state}"
    )
    return resumed_step + 1, resumed_metrics


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_path(args.config)
    repo_root = Path(__file__).resolve().parents[3]
    config.output_path.mkdir(parents=True, exist_ok=True)
    config_copy_path = config.output_path / "resolved_train_config.json"
    with open(config_copy_path, "w") as file:
        json.dump(config.to_dict(), file, indent=2)

    set_seed(config.seed)
    device = get_device()
    train_loader, val_loader = build_dataloaders(config)
    wandb_run = maybe_init_wandb(config)
    log_dataset_samples_to_wandb(
        wandb_run,
        config,
        train_loader.dataset,
        split_name="train",
        step=0,
    )
    log_dataset_samples_to_wandb(
        wandb_run,
        config,
        val_loader.dataset,
        split_name="val",
        step=0,
    )
    extra_validation_loaders = build_extra_validation_dataloaders(config)
    runner, positive_embeddings, _negative_embeddings, runtime_info = build_runner(
        config=config,
        repo_root=repo_root,
        device=device,
    )
    print("[seedvr-hdr] runtime info:")
    print(json.dumps(runtime_info, indent=2))
    base_validation_prediction_cache = build_base_validation_prediction_cache(
        config=config,
        runner=runner,
        positive_embeddings=positive_embeddings,
        val_loader=val_loader,
        device=device,
    )
    extra_base_prediction_caches: dict[str, dict[int, torch.Tensor]] = {}
    for extra, extra_loader in extra_validation_loaders:
        extra_name = metric_safe_name(extra.name)
        print(
            "[seedvr-hdr] preparing extra validation dataset "
            f"name={extra.name} rows={len(extra_loader.dataset)}"
        )
        extra_base_prediction_caches[extra_name] = (
            build_base_validation_prediction_cache(
                config=config,
                runner=runner,
                positive_embeddings=positive_embeddings,
                val_loader=extra_loader,
                device=device,
                num_validation_samples=extra.num_validation_samples,
                sampler_validation_samples=extra.sampler_validation_samples,
                split_name=f"{extra_name}/base_cache",
            )
        )

    optimizer = build_optimizer(config, runner.dit)
    optimizer_lrs = sorted({float(group["lr"]) for group in optimizer.param_groups})
    print(
        "[seedvr-hdr] optimizer parameter groups: "
        f"count={len(optimizer.param_groups)} "
        f"min_lr={optimizer_lrs[0]:.3e} max_lr={optimizer_lrs[-1]:.3e} "
        f"layerwise_lr_decay={config.layerwise_lr_decay}"
    )
    scheduler = build_scheduler(config, optimizer)
    start_step, resumed_metrics = maybe_resume_training(
        config=config,
        model=runner.dit,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    runtime_info["resume_from_checkpoint"] = config.resume_from_checkpoint
    runtime_info["start_step"] = start_step
    lpips_model = build_lpips_model(config, device)

    text_pos_embeds, text_pos_shapes = positive_embeddings
    train_iter = iter(train_loader)
    final_metrics: dict[str, float] = dict(resumed_metrics)
    latest_preview_paths: list[Path] = []
    latest_preview_captions: list[str] = []
    checkpoint_path: Path | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if config.debug_cuda_memory:
        log_cuda_memory("train_start", device)

    if start_step > config.steps and config.resume_from_checkpoint:
        final_metrics = dict(resumed_metrics)
        write_result_manifest(
            path=args.result_manifest,
            checkpoint_path=Path(config.resume_from_checkpoint),
            config_path=config_copy_path,
            validation_preview_paths=[],
            metrics=final_metrics,
            extra={"completed_from_resume": True},
        )
        if wandb_run is not None:
            try:
                wandb.log(final_metrics)
                wandb.finish()
            except Exception as exc:
                print(
                    "[seedvr-hdr] W&B finalization failed after completed "
                    f"resume manifest write: {exc}"
                )
        return

    for step in range(start_step, config.steps + 1):
        step_started_at = perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_images = batch["input_sdr"].to(device, non_blocking=True)
        target_images = batch["target"].to(device, non_blocking=True)

        encode_started_at = perf_counter()
        with torch.no_grad():
            target_latents = encode_images_to_latents(
                runner,
                target_images,
                use_tiling=config.vae_use_tiling,
                use_tqdm=config.vae_use_tqdm,
            )
            input_latents = encode_images_to_latents(
                runner,
                input_images,
                use_tiling=config.vae_use_tiling,
                use_tqdm=config.vae_use_tqdm,
            )
            noisy_inputs = [
                add_condition_noise(runner, latent, runner.config.condition.noise_scale)
                for latent in input_latents
            ]
            conditions = [
                runner.get_condition(target_latent, input_latent, task="sr")
                for target_latent, input_latent in zip(target_latents, noisy_inputs)
            ]
        encode_finished_at = perf_counter()

        latents_flat, latent_shapes = na.flatten(target_latents)
        cond_flat, _ = na.flatten(conditions)
        noise = torch.randn_like(latents_flat)
        timesteps = sample_training_timesteps(
            batch_size=len(target_latents),
            schedule_T=float(runner.schedule.T),
            device=device,
        ).to(latents_flat.dtype)
        timesteps = runner.timestep_transform(timesteps, latent_shapes).to(
            latents_flat.dtype
        )
        diffusion_timesteps = expand_timesteps_to_latents(timesteps, latent_shapes).to(
            latents_flat.dtype
        )
        x_t = runner.schedule.forward(latents_flat, noise, diffusion_timesteps)
        target = runner.schedule.convert_to_pred(
            latents_flat,
            noise,
            diffusion_timesteps,
            PredictionType.v_lerp,
        )

        optimizer.zero_grad(set_to_none=True)
        dit_started_at = perf_counter()
        prediction = run_dit_forward(
            runner=runner,
            config=config,
            vid=torch.cat([x_t, cond_flat], dim=-1),
            txt=text_pos_embeds,
            vid_shape=latent_shapes,
            txt_shape=text_pos_shapes,
            timestep=timesteps,
            stage="train",
            step=step,
        ).vid_sample

        denoise = denoise_loss(prediction, target)
        denoise_weight = current_loss_weight(
            config.denoise_loss_weight,
            warmup_steps=0,
            step=step,
        )
        total_loss = denoise * denoise_weight
        aux_metrics: dict[str, float] = {
            "denoise_loss": float(denoise.item()),
            "denoise_loss_weight": float(denoise_weight),
        }

        lpips_weight = current_loss_weight(
            config.lpips_loss_weight,
            config.lpips_loss_warmup_steps,
            step,
        )
        image_recon_weight = current_loss_weight(
            config.image_recon_loss_weight,
            config.image_loss_warmup_steps,
            step,
        )
        dwt_hf_weight = current_loss_weight(
            config.dwt_hf_loss_weight,
            config.dwt_hf_loss_warmup_steps,
            step,
        )
        fft_hf_weight = current_loss_weight(
            config.fft_hf_loss_weight,
            config.fft_hf_loss_warmup_steps,
            step,
        )
        tv_lpips_weight = current_loss_weight(
            config.tv_lpips_loss_weight,
            config.tv_lpips_loss_warmup_steps,
            step,
        )
        pred_x0 = None
        predicted_images = None
        lpips_value = None
        needs_decoded_prediction = (
            (lpips_model is not None and (lpips_weight > 0.0 or tv_lpips_weight > 0.0))
            or image_recon_weight > 0.0
            or dwt_hf_weight > 0.0
            or fft_hf_weight > 0.0
        )
        if needs_decoded_prediction:
            pred_x0, _ = runner.schedule.convert_from_pred(
                prediction,
                PredictionType.v_lerp,
                x_t,
                diffusion_timesteps,
            )
            predicted_images = decode_latents_to_images(runner, pred_x0, latent_shapes)
        if image_recon_weight > 0.0 and predicted_images is not None:
            image_recon_value = image_reconstruction_loss(
                predicted_images,
                target_images,
            )
            total_loss = total_loss + image_recon_value * image_recon_weight
            aux_metrics["image_recon_loss"] = float(image_recon_value.item())
            aux_metrics["image_recon_loss_weight"] = float(image_recon_weight)
        if (
            lpips_model is not None
            and lpips_weight > 0.0
            and predicted_images is not None
        ):
            lpips_value = lpips_reconstruction_loss(
                lpips_model,
                predicted_images,
                target_images,
                resize=config.lpips_resize,
            )
            total_loss = total_loss + lpips_value * lpips_weight
            aux_metrics["lpips_loss"] = float(lpips_value.item())
            aux_metrics["lpips_loss_weight"] = float(lpips_weight)
        if dwt_hf_weight > 0.0 and predicted_images is not None:
            dwt_hf_value = dwt_high_frequency_loss(
                predicted_images,
                target_images,
                levels=config.dwt_hf_levels,
            )
            total_loss = total_loss + dwt_hf_value * dwt_hf_weight
            aux_metrics["dwt_hf_loss"] = float(dwt_hf_value.item())
            aux_metrics["dwt_hf_loss_weight"] = float(dwt_hf_weight)
        if fft_hf_weight > 0.0 and predicted_images is not None:
            fft_hf_value = fft_high_frequency_loss(
                predicted_images,
                target_images,
                min_freq=config.fft_hf_min_freq,
            )
            total_loss = total_loss + fft_hf_value * fft_hf_weight
            aux_metrics["fft_hf_loss"] = float(fft_hf_value.item())
            aux_metrics["fft_hf_loss_weight"] = float(fft_hf_weight)
        if (
            lpips_model is not None
            and tv_lpips_weight > 0.0
            and predicted_images is not None
        ):
            tv_lpips_value = tv_lpips_reconstruction_loss(
                lpips_model,
                predicted_images,
                target_images,
                resize=config.lpips_resize,
                gamma=config.tv_lpips_gamma,
            )
            total_loss = total_loss + tv_lpips_value * tv_lpips_weight
            aux_metrics["tv_lpips_loss"] = float(tv_lpips_value.item())
            aux_metrics["tv_lpips_loss_weight"] = float(tv_lpips_weight)

        total_loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                runner.dit.parameters(), config.grad_clip_norm
            )
        optimizer.step()
        scheduler.step()
        step_finished_at = perf_counter()

        final_metrics = {
            "loss": float(total_loss.item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **aux_metrics,
        }
        if config.profile_step_time:
            final_metrics.update(
                {
                    "encode_seconds": float(encode_finished_at - encode_started_at),
                    "post_encode_seconds": float(step_finished_at - encode_finished_at),
                    "step_seconds": float(step_finished_at - step_started_at),
                }
            )
        if config.debug_cuda_memory:
            final_metrics.update(get_cuda_memory_stats(device))

        if step % config.log_every == 0 or step == 1:
            print(f"[seedvr-hdr] step={step} metrics={final_metrics}")
            if wandb_run is not None:
                wandb.log(
                    build_wandb_train_metrics(
                        step=step,
                        final_metrics=final_metrics,
                        include_step_timing=config.profile_step_time,
                    )
                )

        if step % config.validate_every == 0 or step == config.steps:
            val_metrics, latest_preview_paths, latest_preview_captions = run_validation(
                config=config,
                runner=runner,
                positive_embeddings=positive_embeddings,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                step=step,
                base_prediction_cache=base_validation_prediction_cache,
            )
            final_metrics.update(val_metrics)
            print(f"[seedvr-hdr] validation step={step} metrics={val_metrics}")
            if wandb_run is not None:
                wandb.log({"step": step, **val_metrics})
                log_validation_previews_to_wandb(
                    wandb_run=wandb_run,
                    preview_paths=latest_preview_paths,
                    step=step,
                    preview_captions=latest_preview_captions,
                )
            for extra, extra_loader in extra_validation_loaders:
                extra_name = metric_safe_name(extra.name)
                extra_metrics, extra_preview_paths, extra_preview_captions = (
                    run_validation(
                        config=config,
                        runner=runner,
                        positive_embeddings=positive_embeddings,
                        val_loader=extra_loader,
                        optimizer=optimizer,
                        device=device,
                        step=step,
                        base_prediction_cache=extra_base_prediction_caches.get(
                            extra_name
                        ),
                        metric_prefix=f"val_{extra_name}",
                        sampler_metric_prefix=f"sampler_val_{extra_name}",
                        preview_subdir=f"validation_{extra_name}",
                        num_validation_samples=extra.num_validation_samples,
                        sampler_validation_samples=extra.sampler_validation_samples,
                    )
                )
                final_metrics.update(extra_metrics)
                latest_preview_paths.extend(extra_preview_paths)
                latest_preview_captions.extend(extra_preview_captions)
                print(
                    "[seedvr-hdr] extra validation "
                    f"name={extra.name} step={step} metrics={extra_metrics}"
                )
                if wandb_run is not None:
                    wandb.log({"step": step, **extra_metrics})
                    log_validation_previews_to_wandb(
                        wandb_run=wandb_run,
                        preview_paths=extra_preview_paths,
                        step=step,
                        preview_captions=extra_preview_captions,
                    )

        model_checkpoint_every = int(config.model_checkpoint_every or 0)
        full_checkpoint_every = int(config.full_checkpoint_every or config.save_every)
        save_full_checkpoint = step % full_checkpoint_every == 0 or step == config.steps
        save_model_checkpoint = (
            model_checkpoint_every > 0
            and step % model_checkpoint_every == 0
            and not save_full_checkpoint
        )
        if save_model_checkpoint:
            save_checkpoint(
                output_dir=config.output_path / "checkpoints",
                step=step,
                model=runner.dit,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=final_metrics,
                config=config.to_dict() | runtime_info,
                include_optimizer=False,
                include_scheduler=False,
                include_rng=False,
                suffix="model",
            )
            if config.debug_cuda_memory:
                log_cuda_memory("model_checkpoint_saved", device, step)

        if save_full_checkpoint:
            checkpoint_path = save_checkpoint(
                output_dir=config.output_path / "checkpoints",
                step=step,
                model=runner.dit,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=final_metrics,
                config=config.to_dict() | runtime_info,
            )
            if config.debug_cuda_memory:
                log_cuda_memory("full_checkpoint_saved", device, step)

        del (
            batch,
            input_images,
            target_images,
            target_latents,
            input_latents,
            noisy_inputs,
            conditions,
            latents_flat,
            latent_shapes,
            cond_flat,
            noise,
            timesteps,
            diffusion_timesteps,
            x_t,
            target,
            prediction,
            denoise,
            pred_x0,
            predicted_images,
            lpips_value,
            total_loss,
        )
        if config.cuda_cleanup_every > 0 and step % config.cuda_cleanup_every == 0:
            cleanup_cuda_memory(device)
            if config.debug_cuda_memory:
                log_cuda_memory("post_cleanup", device, step)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

    assert checkpoint_path is not None
    write_result_manifest(
        path=args.result_manifest,
        checkpoint_path=checkpoint_path,
        config_path=config_copy_path,
        validation_preview_paths=latest_preview_paths,
        metrics=final_metrics,
        extra={},
    )
    if wandb_run is not None:
        try:
            wandb.log(final_metrics)
            wandb.finish()
        except Exception as exc:
            print(
                f"[seedvr-hdr] W&B finalization failed after result manifest write: {exc}"
            )


if __name__ == "__main__":
    main()
