from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from seedvr.common.config import load_config
from seedvr.common.diffusion.types import PredictionType
from seedvr.models.dit import na
from seedvr.models.embeds import PrecomputedEmbeddings
from seedvr.projects.training_hdr.checkpointing import save_checkpoint, write_result_manifest
from seedvr.projects.training_hdr.config import TrainingConfig
from seedvr.projects.training_hdr.dataset import SeedVRHDRImageDataset
from seedvr.projects.training_hdr.losses import (
    denoise_loss,
    image_reconstruction_loss,
    latent_reconstruction_loss,
)
from seedvr.projects.training_hdr.validation import save_triptych
from seedvr.projects.video_diffusion_sr.infer import VideoDiffusionInfer

try:
    import wandb
except ImportError:  # pragma: no cover - optional at runtime
    wandb = None


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


def maybe_compile_dit(model: torch.nn.Module, config: TrainingConfig) -> tuple[torch.nn.Module, bool]:
    if not config.use_torch_compile or not hasattr(torch, "compile"):
        return model, False
    try:
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


def select_trainable_parameters(model: torch.nn.Module, strategy: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def enable_if(predicate) -> None:
        for name, parameter in model.named_parameters():
            if predicate(name):
                parameter.requires_grad_(True)

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
            or name.startswith(("vid_out", "vid_out_norm", "vid_out_ada", "txt_in", "emb_in"))
        )
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
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
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


def build_scheduler(
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
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
    train_dataset = SeedVRHDRImageDataset(
        dataset_root=config.dataset_root,
        manifest_path=config.train_manifest,
        train_height=config.train_height,
        train_width=config.train_width,
        random_crop=True,
        seed=config.seed,
    )
    val_dataset = SeedVRHDRImageDataset(
        dataset_root=config.dataset_root,
        manifest_path=config.val_manifest,
        train_height=config.train_height,
        train_width=config.train_width,
        random_crop=False,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=min(2, config.num_workers),
        pin_memory=True,
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
    runner.configure_vae_model(device=str(device), checkpoint=str(vae_ckpt_path))
    runner.configure_diffusion()
    runner.vae.requires_grad_(False).eval()
    runner.dit.train()
    num_trainable = select_trainable_parameters(runner.dit, config.trainable_strategy)

    runtime_info = configure_runtime_optimizations(config, device)
    runner.dit, compiled = maybe_compile_dit(runner.dit, config)
    runtime_info["torch_compile_active"] = compiled
    runtime_info["model_config_path"] = str(model_config_path)
    runtime_info["dit_checkpoint"] = str(dit_ckpt_path)
    runtime_info["vae_checkpoint"] = str(vae_ckpt_path)
    runtime_info["trainable_strategy"] = config.trainable_strategy
    runtime_info["num_trainable_parameters"] = num_trainable

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
    decoded = runner.vae_decode([latent.to(runner.vae.dtype) for latent in latents])
    return torch.stack([sample.float() for sample in decoded], dim=0)


def run_validation(
    config: TrainingConfig,
    runner: VideoDiffusionInfer,
    positive_embeddings: tuple[torch.Tensor, torch.Tensor],
    val_loader: DataLoader,
    device: torch.device,
    step: int,
) -> tuple[dict[str, float], list[Path]]:
    text_pos_embeds, text_pos_shapes = positive_embeddings
    preview_paths: list[Path] = []
    losses: list[float] = []
    preview_dir = config.output_path / "validation"
    preview_dir.mkdir(parents=True, exist_ok=True)

    runner.dit.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= config.num_validation_samples:
                break

            input_images = batch["input_sdr"].to(device)
            target_images = batch["target"].to(device)
            target_latents = runner.vae_encode([sample for sample in target_images])
            input_latents = runner.vae_encode([sample for sample in input_images])
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
            noise = torch.randn_like(latents_flat)
            timesteps = torch.full(
                (len(target_latents),),
                runner.schedule.T * 0.5,
                device=device,
                dtype=latents_flat.dtype,
            )
            timesteps = runner.timestep_transform(timesteps, latent_shapes).to(latents_flat.dtype)
            x_t = runner.schedule.forward(latents_flat, noise, timesteps)
            target = runner.schedule.convert_to_pred(
                latents_flat,
                noise,
                timesteps,
                PredictionType.v_lerp,
            )
            prediction = runner.dit(
                vid=torch.cat([x_t, cond_flat], dim=-1),
                txt=text_pos_embeds,
                vid_shape=latent_shapes,
                txt_shape=text_pos_shapes,
                timestep=timesteps,
            ).vid_sample
            losses.append(denoise_loss(prediction, target).item())
            pred_x0, _ = runner.schedule.convert_from_pred(
                prediction,
                PredictionType.v_lerp,
                x_t,
                timesteps,
            )
            predicted_images = decode_latents_to_images(runner, pred_x0, latent_shapes)
            preview_path = save_triptych(
                preview_dir / f"step_{step:06d}_{idx:02d}.png",
                input_images[0].cpu(),
                predicted_images[0].cpu(),
                target_images[0].cpu(),
            )
            preview_paths.append(preview_path)

    runner.dit.train()
    metrics = {
        "val_denoise_loss": float(np.mean(losses)) if losses else 0.0,
    }
    return metrics, preview_paths


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

    optimizer = build_optimizer(config, runner.dit)
    scheduler = build_scheduler(config, optimizer)
    wandb_run = maybe_init_wandb(config)

    text_pos_embeds, text_pos_shapes = positive_embeddings
    train_iter = iter(train_loader)
    final_metrics: dict[str, float] = {}
    latest_preview_paths: list[Path] = []
    checkpoint_path: Path | None = None

    for step in range(1, config.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_images = batch["input_sdr"].to(device, non_blocking=True)
        target_images = batch["target"].to(device, non_blocking=True)

        with torch.no_grad():
            target_latents = runner.vae_encode([sample for sample in target_images])
            input_latents = runner.vae_encode([sample for sample in input_images])
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
        noise = torch.randn_like(latents_flat)
        timesteps = sample_training_timesteps(
            batch_size=len(target_latents),
            schedule_T=float(runner.schedule.T),
            device=device,
        ).to(latents_flat.dtype)
        timesteps = runner.timestep_transform(timesteps, latent_shapes).to(latents_flat.dtype)
        x_t = runner.schedule.forward(latents_flat, noise, timesteps)
        target = runner.schedule.convert_to_pred(
            latents_flat,
            noise,
            timesteps,
            PredictionType.v_lerp,
        )

        optimizer.zero_grad(set_to_none=True)
        prediction = runner.dit(
            vid=torch.cat([x_t, cond_flat], dim=-1),
            txt=text_pos_embeds,
            vid_shape=latent_shapes,
            txt_shape=text_pos_shapes,
            timestep=timesteps,
        ).vid_sample

        pred_x0, _ = runner.schedule.convert_from_pred(
            prediction,
            PredictionType.v_lerp,
            x_t,
            timesteps,
        )
        denoise = denoise_loss(prediction, target)
        latent_recon = latent_reconstruction_loss(pred_x0, latents_flat)

        with torch.no_grad():
            predicted_images = decode_latents_to_images(runner, pred_x0, latent_shapes)
        image_recon = image_reconstruction_loss(predicted_images, target_images)

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
        )
        total_loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(runner.dit.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        final_metrics = {
            "loss": float(total_loss.item()),
            "denoise_loss": float(denoise.item()),
            "latent_recon_loss": float(latent_recon.item()),
            "image_recon_loss": float(image_recon.item()),
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

        if step % config.log_every == 0 or step == 1:
            print(f"[seedvr-hdr] step={step} metrics={final_metrics}")
            if wandb_run is not None:
                wandb.log({"step": step, **final_metrics})

        if step % config.validate_every == 0 or step == config.steps:
            val_metrics, latest_preview_paths = run_validation(
                config=config,
                runner=runner,
                positive_embeddings=positive_embeddings,
                val_loader=val_loader,
                device=device,
                step=step,
            )
            final_metrics.update(val_metrics)
            print(f"[seedvr-hdr] validation step={step} metrics={val_metrics}")
            if wandb_run is not None:
                wandb.log({"step": step, **val_metrics})

        if step % config.save_every == 0 or step == config.steps:
            checkpoint_path = save_checkpoint(
                output_dir=config.output_path / "checkpoints",
                step=step,
                model=runner.dit,
                optimizer=optimizer,
                metrics=final_metrics,
                config=config.to_dict() | runtime_info,
            )

    assert checkpoint_path is not None
    write_result_manifest(
        path=args.result_manifest,
        checkpoint_path=checkpoint_path,
        config_path=config_copy_path,
        validation_preview_paths=latest_preview_paths,
        metrics=final_metrics,
    )
    if wandb_run is not None:
        wandb.log(final_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
