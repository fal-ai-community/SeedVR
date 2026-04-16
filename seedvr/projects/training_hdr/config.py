from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    target_representation: str = "mu_law_mu5000"
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
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    num_workers: int = 4
    denoise_loss_weight: float = 1.0
    latent_recon_loss_weight: float = 0.1
    image_recon_loss_weight: float = 0.0
    latent_loss_warmup_steps: int = 0
    image_loss_warmup_steps: int = 0
    grad_clip_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.01
    log_every: int = 10
    validate_every: int = 250
    save_every: int = 500

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as file:
            data = json.load(file)
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
