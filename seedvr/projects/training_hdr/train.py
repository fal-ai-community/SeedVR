from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
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
from seedvr.projects.training_hdr.config import TrainingConfig
from seedvr.projects.training_hdr.dataset import (
    SeedVRHDRImageDataset,
    SeedVRHDRVideoDataset,
)
from seedvr.projects.training_hdr.losses import (
    color_constancy_loss,
    denoise_loss,
    detail_reconstruction_loss,
    edge_consistency_loss,
    flat_region_smoothness_loss,
    image_reconstruction_loss,
    latent_reconstruction_loss,
    low_frequency_banding_loss,
)
from seedvr.projects.training_hdr.validation import compute_hdr_metrics, save_triptych
from seedvr.projects.video_diffusion_sr.infer import VideoDiffusionInfer

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
    parser = argparse.ArgumentParser(description="Train SeedVR on image-first HDR pairs.")
    parser.add_argument("--config", required=True, help="Path to the JSON training config.")
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


def download_checkpoints(config: TrainingConfig, repo_root: Path) -> tuple[Path, Path, Path]:
    spec = config.resolve_checkpoint_spec()
    ckpt_dir = repo_root / "ckpts" / spec.repo_id.replace("/", "__")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
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


def configure_runtime_optimizations(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
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
        torch.backends.cuda.enable_flash_sdp(wants_fa3 or config.allow_attention_fallback)
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


def maybe_compile_dit(model: torch.nn.Module, config: TrainingConfig) -> tuple[torch.nn.Module, bool]:
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
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name or config.experiment_name,
        tags=config.wandb_tags or [],
        config=config.to_dict(),
    )
    return run


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
        "latent_recon_loss": float(final_metrics["latent_recon_loss"]),
        "image_recon_loss": float(final_metrics["image_recon_loss"]),
        "lr": float(final_metrics["lr"]),
    }
    if include_step_timing and "step_seconds" in final_metrics:
        metrics["step_seconds"] = float(final_metrics["step_seconds"])
    for loss_name in (
        "banding",
        "flat_smooth",
        "color_constancy",
        "detail",
        "edge_consistency",
    ):
        weight_key = f"{loss_name}_weight"
        loss_key = f"{loss_name}_loss"
        if float(final_metrics.get(weight_key, 0.0)) > 0.0:
            metrics[loss_key] = float(final_metrics[loss_key])
    return metrics


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
        "cuda_inactive_split_gb": float(stats.get("inactive_split_bytes.all.current", 0) / gib),
    }


def log_cuda_memory(prefix: str, device: torch.device, step: int | None = None) -> dict[str, float]:
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


def validation_noise_seed(config: TrainingConfig, sample_index: int) -> int:
    return int(config.seed + 17_017 + (sample_index * 1_009))


def sampler_validation_noise_seed(config: TrainingConfig, sample_index: int) -> int:
    return int(config.sampler_validation_seed + 23_017 + (sample_index * 1_009))


def collect_base_preview_indices(config: TrainingConfig, dataset_size: int) -> list[int]:
    if dataset_size <= 0:
        return []
    indices: set[int] = set()
    sampler_preview_count = min(config.sampler_validation_samples, dataset_size)
    indices.update(
        build_preview_indices(
            dataset_size=dataset_size,
            num_previews=config.num_validation_samples,
            step=1,
            validate_every=config.validate_every,
        )
    )
    indices.update(
        build_preview_indices(
            dataset_size=dataset_size,
            num_previews=sampler_preview_count,
            step=1,
            validate_every=config.validate_every,
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
    timesteps = runner.timestep_transform(timesteps, latent_shapes).to(latents_flat.dtype)
    diffusion_timesteps = expand_timesteps_to_latents(timesteps, latent_shapes).to(latents_flat.dtype)
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
) -> dict[int, torch.Tensor]:
    dataset = val_loader.dataset
    indices = collect_base_preview_indices(config, len(dataset))
    if not indices:
        return {}
    text_pos_embeds, text_pos_shapes = positive_embeddings
    cache: dict[int, torch.Tensor] = {}
    was_training = runner.dit.training
    runner.dit.eval()
    print(
        "[seedvr-hdr] caching base validation predictions "
        f"count={len(indices)}"
    )
    with torch.no_grad():
        for sample_index in indices:
            sample = dataset[sample_index]
            batch = single_sample_to_batch(sample)
            _denoise, _hdr_metrics, input_images, target_images, predicted_images = evaluate_validation_batch(
                config=config,
                runner=runner,
                text_pos_embeds=text_pos_embeds,
                text_pos_shapes=text_pos_shapes,
                batch=batch,
                device=device,
                noise_seed=validation_noise_seed(config, sample_index),
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
        return name.startswith(("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in"))

    def enable_block_range(start: int, end: int, module_filter: str | None = None) -> None:
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
    top8_start = max(0, num_blocks - 8)
    top16_start = max(0, num_blocks - 16)

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif strategy == "top8":
        enable_if(
            lambda name: (
                (name.startswith("blocks.") and int(name.split(".")[1]) >= top8_start)
                or name.startswith(("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in"))
            )
        )
    elif strategy == "top16":
        enable_if(
            lambda name: (
                (name.startswith("blocks.") and int(name.split(".")[1]) >= top16_start)
                or name.startswith(("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in"))
            )
        )
    elif strategy == "emb_out":
        enable_if(
            lambda name: name.startswith(("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in"))
        )
    elif strategy == "attention_top16":
        enable_if(
            lambda name: (
                name.startswith("blocks.")
                and int(name.split(".")[1]) >= top16_start
                and "attn" in name
            )
            or name.startswith(("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in"))
        )
    elif strategy == "mlp_top16":
        enable_if(
            lambda name: (
                name.startswith("blocks.")
                and int(name.split(".")[1]) >= top16_start
                and "mlp" in name
            )
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
            scale = config.layerwise_lr_decay ** depth_from_top
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
            progress_denominator = max(1, config.steps - config.warmup_steps)
            progress = min(
                1.0,
                max(0.0, float(step - config.warmup_steps) / float(progress_denominator)),
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


def build_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    dataset_cls = SeedVRHDRVideoDataset if config.data_mode == "video" else SeedVRHDRImageDataset
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
    )
    val_dataset = dataset_cls(
        dataset_root=config.dataset_root,
        manifest_path=config.val_manifest,
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
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=bool(config.dataloader_persistent_workers and config.num_workers > 0),
        prefetch_factor=(
            config.dataloader_prefetch_factor if config.num_workers > 0 else None
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=min(2, config.num_workers),
        pin_memory=True,
        persistent_workers=bool(config.dataloader_persistent_workers and min(2, config.num_workers) > 0),
        prefetch_factor=(
            config.dataloader_prefetch_factor if min(2, config.num_workers) > 0 else None
        ),
    )
    return train_loader, val_loader


def build_runner(
    config: TrainingConfig,
    repo_root: Path,
    device: torch.device,
) -> tuple[VideoDiffusionInfer, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    if not config.freeze_vae:
        raise NotImplementedError(
            "Phase-1 SeedVR HDR training currently supports freeze_vae=True only."
        )

    dit_ckpt_path, vae_ckpt_path, model_config_path = download_checkpoints(config, repo_root)
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
    latent_batch = torch.stack([latent for latent in latents], dim=0).to(
        device=device,
        dtype=dtype,
    ).clone()
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
) -> tuple[dict[str, float], list[Path], list[str]]:
    text_pos_embeds, text_pos_shapes = positive_embeddings
    preview_paths: list[Path] = []
    preview_captions: list[str] = []
    losses: list[float] = []
    hdr_metric_rows: list[dict[str, float]] = []
    sampler_metric_rows: list[dict[str, float]] = []
    preview_dir = config.output_path / "validation"
    preview_dir.mkdir(parents=True, exist_ok=True)
    metric_limit = config.num_validation_samples
    preview_index_set = set(
        build_preview_indices(
            dataset_size=len(val_loader.dataset),
            num_previews=config.num_validation_samples,
            step=step,
            validate_every=config.validate_every,
        )
    )

    runner.dit.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    if config.debug_cuda_memory:
        log_cuda_memory("validation_start", device, step)
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= metric_limit:
                break

            denoise, hdr_metrics, input_images, target_images, predicted_images = evaluate_validation_batch(
                config=config,
                runner=runner,
                text_pos_embeds=text_pos_embeds,
                text_pos_shapes=text_pos_shapes,
                batch=batch,
                device=device,
                noise_seed=validation_noise_seed(config, idx),
            )
            losses.append(denoise)
            hdr_metric_rows.append(hdr_metrics)
            if idx in preview_index_set:
                preview_path = save_triptych(
                    preview_dir / f"step_{step:06d}_{len(preview_paths):02d}.png",
                    input_images[0].cpu(),
                    predicted_images[0].cpu(),
                    target_images[0].cpu(),
                    target_representation=config.target_representation,
                    base_predicted_image=(
                        base_prediction_cache or {}
                    ).get(idx),
                )
                preview_paths.append(preview_path)
                scene_value = batch.get("scene_id", "")
                sample_value = batch.get("sample_id", "")
                if isinstance(scene_value, list):
                    scene_value = scene_value[0] if scene_value else ""
                if isinstance(sample_value, list):
                    sample_value = sample_value[0] if sample_value else ""
                preview_captions.append(
                    " ".join(
                        part
                        for part in [
                            f"step={step}",
                            f"preview={len(preview_paths) - 1}",
                            f"scene={scene_value}",
                            f"sample={sample_value}",
                        ]
                        if part
                    )
                )
            if config.debug_cuda_memory:
                log_cuda_memory(f"validation_sample idx={idx}", device, step)

            del (
                input_images,
                target_images,
                predicted_images,
            )
            cleanup_cuda_memory(device)

        dataset = val_loader.dataset
        extra_preview_indices = [
            index
            for index in build_preview_indices(
                dataset_size=len(dataset),
                num_previews=config.num_validation_samples,
                step=step,
                validate_every=config.validate_every,
            )
            if index >= metric_limit
        ]
        for sample_index in extra_preview_indices:
            sample = dataset[sample_index]
            batch = single_sample_to_batch(sample)
            _denoise, _hdr_metrics, input_images, target_images, predicted_images = evaluate_validation_batch(
                config=config,
                runner=runner,
                text_pos_embeds=text_pos_embeds,
                text_pos_shapes=text_pos_shapes,
                batch=batch,
                device=device,
                noise_seed=validation_noise_seed(config, sample_index),
            )
            preview_path = save_triptych(
                preview_dir / f"step_{step:06d}_{len(preview_paths):02d}.png",
                input_images[0].cpu(),
                predicted_images[0].cpu(),
                target_images[0].cpu(),
                target_representation=config.target_representation,
                base_predicted_image=(
                    base_prediction_cache or {}
                ).get(sample_index),
            )
            preview_paths.append(preview_path)
            preview_captions.append(
                " ".join(
                    part
                    for part in [
                        f"step={step}",
                        f"preview={len(preview_paths) - 1}",
                        f"scene={sample.get('scene_id', '')}",
                        f"sample={sample.get('sample_id', '')}",
                    ]
                    if part
                )
            )
            if config.debug_cuda_memory:
                log_cuda_memory(f"validation_preview sample_index={sample_index}", device, step)
            del input_images, target_images, predicted_images
            cleanup_cuda_memory(device)


        if config.sampler_validation_samples > 0:
            dataset = val_loader.dataset
            sampler_indices = build_preview_indices(
                dataset_size=len(dataset),
                num_previews=min(config.sampler_validation_samples, len(dataset)),
                step=step,
                validate_every=config.validate_every,
            )
            for sampler_index, sample_index in enumerate(sampler_indices):
                sample = dataset[sample_index]
                batch = single_sample_to_batch(sample)
                hdr_metrics, input_images, target_images, predicted_images = evaluate_sampler_validation_batch(
                    config=config,
                    runner=runner,
                    positive_embeddings=positive_embeddings,
                    batch=batch,
                    device=device,
                    seed=sampler_validation_noise_seed(config, sample_index),
                )
                sampler_metric_rows.append(hdr_metrics)
                preview_path = save_triptych(
                    preview_dir / f"step_{step:06d}_sampler_{sampler_index:02d}.png",
                    input_images[0].cpu(),
                    predicted_images[0].cpu(),
                    target_images[0].cpu(),
                    target_representation=config.target_representation,
                    base_predicted_image=(
                        base_prediction_cache or {}
                    ).get(sample_index),
                )
                preview_paths.append(preview_path)
                preview_captions.append(
                    " ".join(
                        part
                        for part in [
                            f"step={step}",
                            f"sampler_preview={sampler_index}",
                            f"scene={sample.get('scene_id', '')}",
                            f"sample={sample.get('sample_id', '')}",
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
        "val_denoise_loss": float(np.mean(losses)) if losses else 0.0,
    }
    if hdr_metric_rows:
        metric_names = sorted(set.intersection(*(set(row) for row in hdr_metric_rows)))
        for name in metric_names:
            metrics[f"val_{name}"] = float(np.mean([row[name] for row in hdr_metric_rows]))
    if sampler_metric_rows:
        metric_names = sorted(set.intersection(*(set(row) for row in sampler_metric_rows)))
        for name in metric_names:
            metrics[f"sampler_val_{name}"] = float(np.mean([row[name] for row in sampler_metric_rows]))
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
        raise ValueError(
            f"Resume checkpoint is already at step {resumed_step}, "
            f"which is not below requested steps={config.steps}."
        )
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

    optimizer = build_optimizer(config, runner.dit)
    optimizer_lrs = sorted(
        {float(group["lr"]) for group in optimizer.param_groups}
    )
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
    wandb_run = maybe_init_wandb(config)

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
        timesteps = runner.timestep_transform(timesteps, latent_shapes).to(latents_flat.dtype)
        diffusion_timesteps = expand_timesteps_to_latents(timesteps, latent_shapes).to(latents_flat.dtype)
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

        pred_x0, _ = runner.schedule.convert_from_pred(
            prediction,
            PredictionType.v_lerp,
            x_t,
            diffusion_timesteps,
        )
        denoise = denoise_loss(prediction, target)
        latent_recon = latent_reconstruction_loss(pred_x0, latents_flat)
        decode_started_at = perf_counter()
        predicted_images = decode_latents_to_images(runner, pred_x0, latent_shapes)
        image_recon = image_reconstruction_loss(predicted_images, target_images)
        banding = predicted_images.new_zeros(())
        banding_weight = current_loss_weight(
            config.banding_loss_weight,
            config.banding_loss_warmup_steps,
            step,
        )
        if banding_weight > 0.0:
            banding = low_frequency_banding_loss(
                predicted_images,
                target_images,
                flat_gradient_threshold=config.banding_loss_flat_gradient_threshold,
            )
        flat_smooth = predicted_images.new_zeros(())
        flat_smooth_weight = current_loss_weight(
            config.flat_smooth_loss_weight,
            config.flat_smooth_loss_warmup_steps,
            step,
        )
        if flat_smooth_weight > 0.0:
            flat_smooth = flat_region_smoothness_loss(
                predicted_images,
                target_images,
                flat_gradient_threshold=config.banding_loss_flat_gradient_threshold,
            )
        color_constancy = predicted_images.new_zeros(())
        color_constancy_weight = current_loss_weight(
            config.color_constancy_loss_weight,
            config.color_constancy_loss_warmup_steps,
            step,
        )
        if color_constancy_weight > 0.0:
            color_constancy = color_constancy_loss(predicted_images, target_images)
        detail = predicted_images.new_zeros(())
        detail_weight = current_loss_weight(
            config.detail_loss_weight,
            config.detail_loss_warmup_steps,
            step,
        )
        if detail_weight > 0.0:
            detail = detail_reconstruction_loss(
                predicted_images,
                target_images,
                blur_kernel=config.detail_loss_blur_kernel,
            )
        edge_consistency = predicted_images.new_zeros(())
        edge_consistency_weight = current_loss_weight(
            config.edge_consistency_loss_weight,
            config.edge_consistency_loss_warmup_steps,
            step,
        )
        if edge_consistency_weight > 0.0:
            edge_consistency = edge_consistency_loss(predicted_images, target_images)
        decode_finished_at = perf_counter()

        total_loss = (
            config.denoise_loss_weight * denoise
            + current_loss_weight(
                config.latent_recon_loss_weight,
                config.latent_loss_warmup_steps,
                step,
            )
            * latent_recon
            + current_loss_weight(
                config.image_recon_loss_weight,
                config.image_loss_warmup_steps,
                step,
            )
            * image_recon
            + banding_weight * banding
            + flat_smooth_weight * flat_smooth
            + color_constancy_weight * color_constancy
            + detail_weight * detail
            + edge_consistency_weight * edge_consistency
        )
        total_loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(runner.dit.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        step_finished_at = perf_counter()

        final_metrics = {
            "loss": float(total_loss.item()),
            "denoise_loss": float(denoise.item()),
            "latent_recon_loss": float(latent_recon.item()),
            "image_recon_loss": float(image_recon.item()),
            "banding_loss": float(banding.item()),
            "banding_weight": float(banding_weight),
            "flat_smooth_loss": float(flat_smooth.item()),
            "flat_smooth_weight": float(flat_smooth_weight),
            "color_constancy_loss": float(color_constancy.item()),
            "color_constancy_weight": float(color_constancy_weight),
            "detail_loss": float(detail.item()),
            "detail_weight": float(detail_weight),
            "edge_consistency_loss": float(edge_consistency.item()),
            "edge_consistency_weight": float(edge_consistency_weight),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "latent_recon_weight": float(
                current_loss_weight(
                    config.latent_recon_loss_weight,
                    config.latent_loss_warmup_steps,
                    step,
                )
            ),
            "image_recon_weight": float(
                current_loss_weight(
                    config.image_recon_loss_weight,
                    config.image_loss_warmup_steps,
                    step,
                )
            ),
        }
        if config.profile_step_time:
            final_metrics.update(
                {
                    "encode_seconds": float(encode_finished_at - encode_started_at),
                    "decode_seconds": float(decode_finished_at - decode_started_at),
                    "post_encode_seconds": float(
                        step_finished_at - encode_finished_at
                    ),
                    "post_decode_seconds": float(
                        step_finished_at - decode_finished_at
                    ),
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

        if step % config.save_every == 0 or step == config.steps:
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
                log_cuda_memory("checkpoint_saved", device, step)

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
            pred_x0,
            predicted_images,
            denoise,
            latent_recon,
            image_recon,
            banding,
            flat_smooth,
            color_constancy,
            detail,
            edge_consistency,
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
            print(f"[seedvr-hdr] W&B finalization failed after result manifest write: {exc}")


if __name__ == "__main__":
    main()
