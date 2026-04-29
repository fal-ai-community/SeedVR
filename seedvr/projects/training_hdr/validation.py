from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


MU_LAW_MU = 5000.0
LOG_HDR_EPS = 1.0e-6
PU21_L_MIN = 0.005
PU21_L_MAX = 10000.0
PU21_PEAK_NITS = 1000.0
_LPIPS_MODEL = None
_LPIPS_UNAVAILABLE = False
_CVVDP_MODEL = None
_CVVDP_UNAVAILABLE = False
# Highlight-specific validation MAE was constant on this dataset, so do not log it.


def _finite_tensor(tensor: torch.Tensor, *, min_value: float | None = None) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=PU21_L_MAX, neginf=0.0)
    if min_value is not None:
        tensor = torch.clamp(tensor, min=min_value)
    return tensor


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
    tensor = _finite_tensor(tensor.detach())
    if target_representation == "raw_hdr":
        return torch.clamp(tensor, min=0.0)
    if target_representation == "mu_law_mu5000":
        normalized = tensor.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        return _finite_tensor(torch.expm1(normalized * np.log1p(MU_LAW_MU)) / MU_LAW_MU, min_value=0.0)
    if target_representation == "log_hdr":
        return _finite_tensor(torch.exp(tensor) - LOG_HDR_EPS, min_value=0.0)
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
        return _finite_tensor(nits / 1000.0, min_value=0.0)
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
        return _finite_tensor(linear, min_value=0.0)
    raise ValueError(f"Unsupported target_representation: {target_representation}")


def _robust_tonemap(linear_hdr: torch.Tensor) -> torch.Tensor:
    linear_hdr = _finite_tensor(linear_hdr, min_value=0.0)
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


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    return (
        0.2126 * rgb[0:1]
        + 0.7152 * rgb[1:2]
        + 0.0722 * rgb[2:3]
    )


def _linear_hdr_to_absolute_nits(linear_hdr: torch.Tensor) -> torch.Tensor:
    linear_hdr = _finite_tensor(linear_hdr, min_value=0.0)
    return torch.clamp(linear_hdr * PU21_PEAK_NITS, min=PU21_L_MIN, max=PU21_L_MAX)


def _pu21_encode(luminance_nits: torch.Tensor) -> torch.Tensor:
    # PU21 banding_glare parameters from gfxdisp/pu21, recommended by the authors.
    p = luminance_nits.new_tensor(
        [
            0.353487901,
            0.3734658629,
            8.277049286e-05,
            0.9062562627,
            0.09150303166,
            0.9099517204,
            596.3148142,
        ]
    )
    y = torch.clamp(
        torch.nan_to_num(luminance_nits.float(), nan=PU21_L_MIN, posinf=PU21_L_MAX, neginf=PU21_L_MIN),
        PU21_L_MIN,
        PU21_L_MAX,
    )
    encoded = p[6] * (((p[0] + p[1] * y.pow(p[3])) / (1.0 + p[2] * y.pow(p[3]))).pow(p[4]) - p[5])
    return torch.clamp(torch.nan_to_num(encoded, nan=0.0, posinf=1023.0, neginf=0.0), min=0.0)


def _finite_mse(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    diff = prediction.float() - target.float()
    valid = torch.isfinite(diff)
    if not bool(valid.any().item()):
        return None
    return float(torch.mean(diff[valid] ** 2).item())


def _psnr_from_mse(mse: float | None, data_range: float) -> float | None:
    if mse is None or not np.isfinite(mse):
        return None
    if mse <= 1.0e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def _pu21_psnr(predicted_linear: torch.Tensor, target_linear: torch.Tensor) -> float | None:
    pred_luminance = _linear_hdr_to_absolute_nits(_luminance(predicted_linear))
    target_luminance = _linear_hdr_to_absolute_nits(_luminance(target_linear))
    pred_pu = _pu21_encode(pred_luminance)
    target_pu = _pu21_encode(target_luminance)
    return _psnr_from_mse(_finite_mse(pred_pu, target_pu), data_range=1023.0)


def _lpips_distance(predicted_linear: torch.Tensor, target_linear: torch.Tensor) -> float | None:
    global _LPIPS_MODEL, _LPIPS_UNAVAILABLE
    if _LPIPS_UNAVAILABLE:
        return None
    try:
        import lpips  # type: ignore

        if _LPIPS_MODEL is None:
            _LPIPS_MODEL = lpips.LPIPS(net="alex").eval().to(predicted_linear.device)
        pred = _robust_tonemap(predicted_linear).mul(2.0).sub(1.0).unsqueeze(0)
        target = _robust_tonemap(target_linear).mul(2.0).sub(1.0).unsqueeze(0)
        with torch.no_grad():
            return float(_LPIPS_MODEL(pred, target).mean().item())
    except Exception as exc:
        _LPIPS_UNAVAILABLE = True
        print(f"[seedvr-hdr] LPIPS validation metric unavailable: {exc}")
        return None


def _colorvideovdp_jod(predicted_linear: torch.Tensor, target_linear: torch.Tensor) -> float | None:
    global _CVVDP_MODEL, _CVVDP_UNAVAILABLE
    if _CVVDP_UNAVAILABLE:
        return None
    try:
        import pycvvdp  # type: ignore

        if _CVVDP_MODEL is None:
            _CVVDP_MODEL = pycvvdp.cvvdp(display_name="standard_hdr_linear")
        pred = _linear_hdr_to_absolute_nits(predicted_linear).permute(1, 2, 0).detach().cpu().numpy()
        target = _linear_hdr_to_absolute_nits(target_linear).permute(1, 2, 0).detach().cpu().numpy()
        jod, _stats = _CVVDP_MODEL.predict(pred.astype(np.float32), target.astype(np.float32), dim_order="HWC")
        return float(jod)
    except Exception as exc:
        _CVVDP_UNAVAILABLE = True
        print(f"[seedvr-hdr] ColorVideoVDP JOD validation metric unavailable: {exc}")
        return None


def _add_panel_labels(panels: list[np.ndarray], labels: list[str]) -> list[np.ndarray]:
    labeled_panels: list[np.ndarray] = []
    for panel, label in zip(panels, labels):
        labeled = panel.copy()
        cv2.rectangle(labeled, (0, 0), (min(labeled.shape[1], 220), 28), (0, 0, 0), thickness=-1)
        cv2.putText(
            labeled,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        labeled_panels.append(labeled)
    return labeled_panels


def save_triptych(
    output_path: str | Path,
    input_image: torch.Tensor,
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    target_representation: str,
    base_predicted_image: torch.Tensor | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_image = _select_preview_frame(input_image)
    predicted_image = _select_preview_frame(predicted_image)
    target_image = _select_preview_frame(target_image)
    predicted_linear = linear_hdr_from_target_tensor(predicted_image, target_representation)
    target_linear = linear_hdr_from_target_tensor(target_image, target_representation)
    panels = [
        _input_tensor_to_uint8_image(input_image),
        _preview_uint8_from_linear_hdr(predicted_linear),
        _preview_uint8_from_linear_hdr(target_linear),
    ]
    labels = ["input", "prediction", "ground_truth"]
    if base_predicted_image is not None:
        base_predicted_image = _select_preview_frame(base_predicted_image)
        base_linear = linear_hdr_from_target_tensor(base_predicted_image, target_representation)
        panels.append(_preview_uint8_from_linear_hdr(base_linear))
        labels.append("base_seedvr")
    panels = _add_panel_labels(panels, labels)
    canvas = np.concatenate(panels, axis=1)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), canvas_bgr)
    return output_path


def _psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    psnr = _psnr_from_mse(_finite_mse(prediction, target), data_range=data_range)
    return 0.0 if psnr is None else psnr


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
        metric_names = sorted(set.intersection(*(set(row) for row in frame_metrics)))
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

    metrics = {
        "hdr_log_mae": float(log_diff.mean().item()),
        "hdr_log_psnr": _psnr(predicted_log, target_log, data_range=max(1.0, float(target_log.max().item()))),
        "tonemap_psnr": _psnr(preview_prediction, preview_target, data_range=1.0),
    }
    pu21_psnr = _pu21_psnr(predicted_linear, target_linear)
    if pu21_psnr is not None:
        metrics["pu21_psnr"] = pu21_psnr
    lpips_value = _lpips_distance(predicted_linear, target_linear)
    if lpips_value is not None:
        metrics["lpips"] = lpips_value
    jod_value = _colorvideovdp_jod(predicted_linear, target_linear)
    if jod_value is not None:
        metrics["jod"] = jod_value
    return metrics
