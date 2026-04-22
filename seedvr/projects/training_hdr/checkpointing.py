from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def unwrap_compiled_module(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_orig_mod", module)


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"seedvr_hdr_step_{step:06d}.pt"
    model_to_save = unwrap_compiled_module(model)
    torch.save(
        {
            "step": step,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    rng_state = checkpoint.get("rng_state")
    if not rng_state:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    cuda_states = rng_state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_to_load = unwrap_compiled_module(model)
    model_to_load.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if device is not None:
            _move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        restore_rng_state(checkpoint)
    return checkpoint


def write_result_manifest(
    path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    validation_preview_paths: list[str | Path],
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "validation_preview_paths": [str(item) for item in validation_preview_paths],
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)
