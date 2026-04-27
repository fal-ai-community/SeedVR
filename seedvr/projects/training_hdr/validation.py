from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


MU_LAW_MU = 5000.0
LOG_HDR_EPS = 1.0e-6
# Highlight-specific validation MAE was constant on this dataset, so do not log it.


def _input_tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().clamp(-1.0, 1.0).add(1.0).mul(127.5)
    array = array.round().to(torch.uint8).cpu().numpy()
    return np.transpose(array, (1, 2, 0))


def _select_preview_frame(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        frame_index = tensor.shape[0] // 2
        return tensor[frame_index]
    return tensor


def linear_hdr_from_target_tensor(
    tensor: torch.Tensor,
    target_representation: str,
) -> torch.Tensor:
    tensor = tensor.detach().float()
    if target_representation == "raw_hdr":
        return torch.clamp(tensor, min=0.0)
    if target_representation == "mu_law_mu5000":
        normalized = tensor.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        return torch.expm1(normalized * np.log1p(MU_LAW_MU)) / MU_LAW_MU
    if target_representation == "log_hdr":
        return torch.clamp(torch.exp(tensor) - LOG_HDR_EPS, min=0.0)
    if target_representation == "pq_1000":
        normalized = tensor.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        m1 = 2610.0 / 16384.0
        m2 = 2523.0 / 32.0
        c1 = 3424.0 / 4096.0
        c2 = 2413.0 / 128.0
        c3 = 2392.0 / 128.0
        powered = torch.pow(normalized, 1.0 / m2)
        numerator = torch.clamp(powered - c1, min=0.0)
        denominator = torch.clamp(c2 - c3 * powered, min=1.0e-8)
        nits = torch.pow(numerator / denominator, 1.0 / m1) * 10000.0
        return nits / 1000.0
    if target_representation == "logc3":
        normalized = tensor.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        cut = 0.010591
        a = 5.555556
        b = 0.052272
        c = 0.247190
        d = 0.385537
        e = 5.367655
        f = 0.092809
        cut_encoded = e * cut + f
        linear = torch.where(
            normalized > cut_encoded,
            (torch.pow(torch.tensor(10.0, device=normalized.device), (normalized - d) / c) - b) / a,
            (normalized - f) / e,
        )
        return torch.clamp(linear, min=0.0)
    raise ValueError(f"Unsupported target_representation: {target_representation}")


def _robust_tonemap(linear_hdr: torch.Tensor) -> torch.Tensor:
    linear_hdr = torch.clamp(linear_hdr, min=0.0)
    flat = linear_hdr.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(linear_hdr)
    percentile = torch.quantile(flat, 0.995)
    scale = percentile if percentile > 1.0e-6 else torch.clamp(flat.max(), min=1.0)
    normalized = linear_hdr / scale
    mapped = normalized / (1.0 + normalized)
    mapped = torch.clamp(mapped, 0.0, 1.0) ** (1.0 / 2.2)
    return mapped


def _preview_uint8_from_linear_hdr(linear_hdr: torch.Tensor) -> np.ndarray:
    preview = _robust_tonemap(linear_hdr)
    preview = preview.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    return np.transpose(preview, (1, 2, 0))


def save_triptych(
    output_path: str | Path,
    input_image: torch.Tensor,
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    target_representation: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_image = _select_preview_frame(input_image)
    predicted_image = _select_preview_frame(predicted_image)
    target_image = _select_preview_frame(target_image)
    predicted_linear = linear_hdr_from_target_tensor(predicted_image, target_representation)
    target_linear = linear_hdr_from_target_tensor(target_image, target_representation)
    canvas = np.concatenate(
        [
            _input_tensor_to_uint8_image(input_image),
            _preview_uint8_from_linear_hdr(predicted_linear),
            _preview_uint8_from_linear_hdr(target_linear),
        ],
        axis=1,
    )
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), canvas_bgr)
    return output_path


def _psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((prediction - target) ** 2).item()
    if mse <= 1.0e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def compute_hdr_metrics(
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    target_representation: str,
) -> dict[str, float]:
    if predicted_image.ndim == 4 and target_image.ndim == 4:
        frame_metrics = [
            compute_hdr_metrics(predicted_image[index], target_image[index], target_representation)
            for index in range(predicted_image.shape[0])
        ]
        metric_names = frame_metrics[0].keys()
        return {
            name: float(np.mean([row[name] for row in frame_metrics]))
            for name in metric_names
        }

    predicted_linear = linear_hdr_from_target_tensor(predicted_image, target_representation)
    target_linear = linear_hdr_from_target_tensor(target_image, target_representation)

    predicted_log = torch.log1p(torch.clamp(predicted_linear, min=0.0))
    target_log = torch.log1p(torch.clamp(target_linear, min=0.0))
    log_diff = torch.abs(predicted_log - target_log)

    preview_prediction = _robust_tonemap(predicted_linear)
    preview_target = _robust_tonemap(target_linear)

    return {
        "hdr_log_mae": float(log_diff.mean().item()),
        "hdr_log_psnr": _psnr(predicted_log, target_log, data_range=max(1.0, float(target_log.max().item()))),
        "tonemap_psnr": _psnr(preview_prediction, preview_target, data_range=1.0),
    }
