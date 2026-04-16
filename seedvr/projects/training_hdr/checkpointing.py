from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def unwrap_compiled_module(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_orig_mod", module)


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
            "metrics": metrics,
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def write_result_manifest(
    path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    validation_preview_paths: list[str | Path],
    metrics: dict[str, float],
) -> None:
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "validation_preview_paths": [str(item) for item in validation_preview_paths],
        "metrics": metrics,
    }
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)

