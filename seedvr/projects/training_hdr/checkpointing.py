from __future__ import annotations

import atexit
import concurrent.futures as _cf
import copy
import json
import os
import re
import tempfile
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def unwrap_compiled_module(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "_orig_mod", module)


_SAVE_EXECUTOR: _cf.ThreadPoolExecutor | None = None
_PENDING: list[_cf.Future] = []
_ATEXIT_REGISTERED: bool = False
_ASYNC_SUFFIX: str = "async_val"
_ASYNC_FILENAME_RE = re.compile(rf"^seedvr_hdr_step_(\d+)_{_ASYNC_SUFFIX}\.pt$")


def _get_save_executor() -> _cf.ThreadPoolExecutor:
    """Single-worker executor so disk writes are serialized."""
    global _SAVE_EXECUTOR, _ATEXIT_REGISTERED
    if _SAVE_EXECUTOR is None:
        _SAVE_EXECUTOR = _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-save"
        )
    if not _ATEXIT_REGISTERED:
        atexit.register(drain_pending_async_saves)
        _ATEXIT_REGISTERED = True
    return _SAVE_EXECUTOR


def drain_pending_async_saves(timeout: float = 600.0) -> None:
    """Block until every pending async checkpoint finishes (or errors out)."""
    pending = list(_PENDING)
    _PENDING.clear()
    for fut in pending:
        try:
            fut.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[seedvr-hdr][async-ckpt] save failed: {exc}", flush=True)


def _snapshot_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Detach + copy tensors to CPU; releases GPU memory + lets caller resume training."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to(device="cpu", copy=True)
        else:
            out[k] = copy.deepcopy(v)
    return out


def save_checkpoint_async(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    metrics: dict[str, float],
    config: dict[str, Any],
    *,
    suffix: str = _ASYNC_SUFFIX,
    include_optimizer: bool = True,
    include_scheduler: bool = True,
    include_rng: bool = True,
    keep_last: int = 3,
) -> Path:
    """Snapshot weights to CPU now (blocking), persist on a background thread.

    The synchronous snapshot is fast (a few seconds at most for SeedVR2-7B); training
    resumes while the actual torch.save runs in a worker thread. ``keep_last``
    automatically prunes older async checkpoints so disk usage stays bounded.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_part = f"_{suffix}" if suffix else ""
    final_path = output_dir / f"seedvr_hdr_step_{step:06d}{suffix_part}.pt"

    model_to_save = unwrap_compiled_module(model)
    cpu_model_state = _snapshot_state_dict(model_to_save.state_dict())
    cpu_optim_state = (
        _snapshot_state_dict(optimizer.state_dict()) if include_optimizer and optimizer is not None else None
    )
    sched_state = (
        scheduler.state_dict() if include_scheduler and scheduler is not None else None
    )
    rng_state = None
    if include_rng:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    payload: dict[str, Any] = {
        "step": step,
        "model": cpu_model_state,
        "optimizer": cpu_optim_state,
        "scheduler": sched_state,
        "metrics": dict(metrics),
        "config": config,
        "rng_state": rng_state,
        "checkpoint_kind": "full_async" if cpu_optim_state is not None else "model_async",
    }

    def _write() -> None:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".pt", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, final_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        if keep_last > 0:
            _prune_old_async_checkpoints(output_dir, keep_last=keep_last)

    fut = _get_save_executor().submit(_write)
    _PENDING.append(fut)
    # Best-effort: also remove already-completed futures from the pending list.
    _PENDING[:] = [f for f in _PENDING if not f.done()] + [fut]
    return final_path


def _prune_old_async_checkpoints(output_dir: Path, *, keep_last: int) -> None:
    """Delete async checkpoints older than the ``keep_last`` most recent ones."""
    entries: list[tuple[int, Path]] = []
    for p in output_dir.glob("seedvr_hdr_step_*_" + _ASYNC_SUFFIX + ".pt"):
        m = _ASYNC_FILENAME_RE.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort()
    for _, path in entries[:-keep_last]:
        try:
            path.unlink()
        except OSError as exc:
            print(f"[seedvr-hdr][async-ckpt] prune failed for {path}: {exc}", flush=True)


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    metrics: dict[str, float],
    config: dict[str, Any],
    include_optimizer: bool = True,
    include_scheduler: bool = True,
    include_rng: bool = True,
    suffix: str = "",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_part = f"_{suffix}" if suffix else ""
    checkpoint_path = output_dir / f"seedvr_hdr_step_{step:06d}{suffix_part}.pt"
    model_to_save = unwrap_compiled_module(model)
    payload: dict[str, Any] = {
        "step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if include_optimizer else None,
        "scheduler": (
            scheduler.state_dict()
            if include_scheduler and scheduler is not None
            else None
        ),
        "metrics": metrics,
        "config": config,
        "checkpoint_kind": "full" if include_optimizer else "model",
    }
    if include_rng:
        payload["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".pt", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
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
    # These checkpoints are created by our own trainer and contain optimizer,
    # scheduler, config, and RNG state objects beyond plain tensor weights.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".json", delete=False
    ) as file:
        tmp_path = Path(file.name)
        json.dump(payload, file, indent=2)
    try:
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
