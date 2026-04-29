from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckpointSpec:
    repo_id: str
    dit_filename: str
    vae_filename: str
    config_path: str


BASE_MODEL_SPECS: dict[str, CheckpointSpec] = {
    "seedvr2_3b": CheckpointSpec(
        repo_id="ByteDance-Seed/SeedVR2-3B",
        dit_filename="seedvr2_ema_3b.pth",
        vae_filename="ema_vae.pth",
        config_path="configs_3b/main.yaml",
    ),
    "seedvr2_7b": CheckpointSpec(
        repo_id="ByteDance-Seed/SeedVR2-7B",
        dit_filename="seedvr2_ema_7b.pth",
        vae_filename="ema_vae.pth",
        config_path="configs_7b/main.yaml",
    ),
    "seedvr_3b": CheckpointSpec(
        repo_id="ByteDance-Seed/SeedVR-3B",
        dit_filename="seedvr_ema_3b.pth",
        vae_filename="ema_vae.pth",
        config_path="configs_3b/main.yaml",
    ),
    "seedvr_7b": CheckpointSpec(
        repo_id="ByteDance-Seed/SeedVR-7B",
        dit_filename="seedvr_ema_7b.pth",
        vae_filename="ema_vae.pth",
        config_path="configs_7b/main.yaml",
    ),
}

SUPPORTED_TARGET_REPRESENTATIONS = {"raw_hdr", "mu_law_mu5000", "log_hdr", "pq_1000", "logc3"}


@dataclass
class ExtraValidationConfig:
    name: str
    dataset_root: str
    val_manifest: str
    data_mode: str = "image"
    num_validation_samples: int | None = None
    sampler_validation_samples: int | None = None


@dataclass
class TrainingConfig:
    dataset_root: str
    train_manifest: str
    val_manifest: str
    output_dir: str
    experiment_name: str
    steps: int
    learning_rate: float
    batch_size: int
    train_width: int
    train_height: int
    seed: int
    training_mode: str
    trainable_strategy: str
    base_model: str
    freeze_vae: bool
    num_validation_samples: int
    compressed_target_space: str
    task_mode: str
    data_mode: str = "image"
    clip_length: int = 1
    frame_stride: int = 1
    target_representation: str = "mu_law_mu5000"
    resume_from_checkpoint: str | None = None
    resume_optimizer_state: bool = True
    resume_scheduler_state: bool = True
    resume_rng_state: bool = True
    checkpoint_repo_id: str | None = None
    dit_filename: str | None = None
    vae_filename: str | None = None
    seedvr_config_path: str | None = None
    use_wandb: bool = False
    wandb_project: str = "seedvr-hdr"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    use_torch_compile: bool = True
    torch_compile_mode: str = "max-autotune"
    torch_compile_fullgraph: bool = False
    use_fa3: bool = True
    allow_attention_fallback: bool = True
    use_mxfp8: bool = False
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    scheduler_total_steps: int | None = None
    layerwise_lr_decay: float = 1.0
    layerwise_lr_min_scale: float = 0.05
    num_workers: int = 4
    runtime_cache_root: str = "/data/seedvr_hdr_runtime_cache"
    remote_download_timeout_seconds: int = 120
    remote_download_retries: int = 5
    cache_rendered_sdr_inputs: bool = True
    cache_compressed_targets: bool = True
    denoise_loss_weight: float = 1.0
    latent_recon_loss_weight: float = 0.1
    image_recon_loss_weight: float = 0.0
    lpips_loss_weight: float = 0.0
    latent_loss_warmup_steps: int = 0
    image_loss_warmup_steps: int = 0
    lpips_loss_warmup_steps: int = 0
    lpips_net: str = "alex"
    lpips_resize: int = 256
    dwt_hf_loss_weight: float = 0.0
    dwt_hf_loss_warmup_steps: int = 0
    dwt_hf_levels: int = 2
    fft_hf_loss_weight: float = 0.0
    fft_hf_loss_warmup_steps: int = 0
    fft_hf_min_freq: float = 0.25
    tv_lpips_loss_weight: float = 0.0
    tv_lpips_loss_warmup_steps: int = 0
    tv_lpips_gamma: float = 0.5
    sampler_validation_samples: int = 0
    sampler_validation_guidance_scale: float = 1.0
    sampler_validation_seed: int = 2079280474
    banding_loss_weight: float = 0.0
    banding_loss_warmup_steps: int = 0
    banding_loss_flat_gradient_threshold: float = 0.08
    flat_smooth_loss_weight: float = 0.0
    flat_smooth_loss_warmup_steps: int = 0
    color_constancy_loss_weight: float = 0.0
    color_constancy_loss_warmup_steps: int = 0
    detail_loss_weight: float = 0.0
    detail_loss_warmup_steps: int = 0
    detail_loss_blur_kernel: int = 9
    edge_consistency_loss_weight: float = 0.0
    edge_consistency_loss_warmup_steps: int = 0
    phase_jitter: bool = False
    phase_jitter_max_pixels: int = 15
    size_jitter_steps: int = 0
    jpeg_roundtrip_prob: float = 0.0
    jpeg_roundtrip_min_quality: int = 75
    jpeg_roundtrip_max_quality: int = 95
    grad_clip_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    log_every: int = 10
    validate_every: int = 100
    save_every: int = 500
    debug_cuda_memory: bool = False
    cuda_cleanup_every: int = 100
    vae_use_tiling: bool = False
    vae_use_tqdm: bool = False
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    profile_step_time: bool = True
    extra_validation_datasets: list[ExtraValidationConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.target_representation not in SUPPORTED_TARGET_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported target_representation '{self.target_representation}'. "
                f"Expected one of: {sorted(SUPPORTED_TARGET_REPRESENTATIONS)}"
            )
        if self.data_mode not in {"image", "video"}:
            raise ValueError(
                f"Unsupported data_mode '{self.data_mode}'. Expected 'image' or 'video'."
            )
        if self.clip_length < 1:
            raise ValueError("clip_length must be >= 1")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as file:
            data = json.load(file)
        data["extra_validation_datasets"] = [
            item if isinstance(item, ExtraValidationConfig) else ExtraValidationConfig(**item)
            for item in data.get("extra_validation_datasets", [])
        ]
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def train_size(self) -> tuple[int, int]:
        return self.train_height, self.train_width

    def resolve_checkpoint_spec(self) -> CheckpointSpec:
        if self.base_model not in BASE_MODEL_SPECS:
            raise ValueError(
                f"Unsupported base_model '{self.base_model}'. "
                f"Expected one of: {sorted(BASE_MODEL_SPECS)}"
            )

        base = BASE_MODEL_SPECS[self.base_model]
        return CheckpointSpec(
            repo_id=self.checkpoint_repo_id or base.repo_id,
            dit_filename=self.dit_filename or base.dit_filename,
            vae_filename=self.vae_filename or base.vae_filename,
            config_path=self.seedvr_config_path or base.config_path,
        )
