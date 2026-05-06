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
    frames = _as_chw_frames(tensor)
    return frames[len(frames) // 2]


def _as_chw_frames(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Normalize validation tensors to a list of C,H,W RGB frames."""
    tensor = tensor.detach().float()
    if tensor.ndim == 5:
        if tensor.shape[2] == 3:
            batch, frames, channels, height, width = tensor.shape
            return list(tensor.reshape(batch * frames, channels, height, width))
        if tensor.shape[1] == 3:
            batch, channels, frames, height, width = tensor.shape
            tensor = tensor.permute(0, 2, 1, 3, 4).contiguous()
            return list(tensor.reshape(batch * frames, channels, height, width))
    if tensor.ndim == 4:
        if tensor.shape[1] == 3:
            return list(tensor)
        if tensor.shape[0] == 3:
            return list(tensor.permute(1, 0, 2, 3).contiguous())
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        return [tensor]
    raise ValueError(
        "Expected RGB tensor in CHW, TCHW, BTCHW, or BCTHW layout, "
        f"got {tuple(tensor.shape)}"
    )


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


_BT2020_TO_BT709 = (
    (1.66049100, -0.58764114, -0.07284986),
    (-0.12455047, 1.13289990, -0.00834942),
    (-0.01815076, -0.10057890, 1.11872966),
)


def _bt2020_linear_to_bt709_linear(linear_bt2020: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.2087 matrix; clip out-of-gamut to non-negative."""
    m = torch.tensor(_BT2020_TO_BT709, dtype=linear_bt2020.dtype, device=linear_bt2020.device)
    if linear_bt2020.ndim == 3:
        c, h, w = linear_bt2020.shape
        flat = linear_bt2020.reshape(c, h * w)
        out = m @ flat
        return out.clamp(min=0.0).reshape(c, h, w)
    flat = linear_bt2020.reshape(linear_bt2020.shape[0], -1)
    out = m @ flat
    return out.clamp(min=0.0).reshape(linear_bt2020.shape)


def _preview_uint8_from_linear_hdr(linear_hdr: torch.Tensor) -> np.ndarray:
    bt709 = _bt2020_linear_to_bt709_linear(linear_hdr)
    preview = _robust_tonemap(bt709)
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


def save_triptych_video(
    output_path: str | Path,
    input_image: torch.Tensor,
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    target_representation: str,
    base_predicted_image: torch.Tensor | None = None,
    fps: int = 4,
) -> Path | None:
    """Render a per-frame triptych (input | prediction | ground_truth | base) as
    an MP4. Returns None for single-frame inputs (use save_triptych instead).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_frames = _as_chw_frames(input_image)
    pred_frames = _as_chw_frames(predicted_image)
    target_frames = _as_chw_frames(target_image)
    base_frames = (
        _as_chw_frames(base_predicted_image)
        if base_predicted_image is not None
        else None
    )
    T = min(len(input_frames), len(pred_frames), len(target_frames))
    if T <= 1:
        return None

    composed_frames: list[np.ndarray] = []
    for t in range(T):
        pred_lin = linear_hdr_from_target_tensor(pred_frames[t], target_representation)
        target_lin = linear_hdr_from_target_tensor(target_frames[t], target_representation)
        panels = [
            _input_tensor_to_uint8_image(input_frames[t]),
            _preview_uint8_from_linear_hdr(pred_lin),
            _preview_uint8_from_linear_hdr(target_lin),
        ]
        labels = ["input", "prediction", "ground_truth"]
        if base_frames is not None and t < len(base_frames):
            base_lin = linear_hdr_from_target_tensor(
                base_frames[t], target_representation
            )
            panels.append(_preview_uint8_from_linear_hdr(base_lin))
            labels.append("base_seedvr")
        panels = _add_panel_labels(panels, labels)
        composed_frames.append(np.concatenate(panels, axis=1))

    # Encode H.264 via PyAV/libx264. cv2's mp4v fourcc produces MPEG-4 Part 2
    # which HTML5 <video> (and thus WandB's web preview) cannot decode — that
    # was rendering as solid colors / pure noise.
    import av  # type: ignore

    height, width, _ = composed_frames[0].shape
    # libx264 requires even dimensions
    even_h = height - (height % 2)
    even_w = width - (width % 2)
    container = av.open(str(output_path), mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = even_w
        stream.height = even_h
        stream.pix_fmt = "yuv420p"
        # Keep CRF reasonable for small previews; faststart so WandB streams.
        stream.options = {"crf": "20", "preset": "medium", "movflags": "+faststart"}
        for frame_rgb in composed_frames:
            if (frame_rgb.shape[0], frame_rgb.shape[1]) != (even_h, even_w):
                frame_rgb = frame_rgb[:even_h, :even_w]
            video_frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frame_rgb), format="rgb24"
            )
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception:
        try:
            container.close()
        except Exception:
            pass
        return None
    container.close()
    return output_path


# ---------------------------------------------------------------------
# HDR10 PQ-encoded MP4 preview (HEVC main10, BT.2020 NCL, SMPTE 2084 PQ).
# Plays correctly on HDR-capable players; regular browsers render it
# tonemapped at best, but the file preserves the actual HDR signal.
# ---------------------------------------------------------------------
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0
_HDR_PEAK_NITS = 1000.0
_PQ_OETF_PEAK_NITS = 10_000.0
# BT.709 → BT.2020 chromaticity matrix (linear light)
_BT709_TO_BT2020 = (
    (0.6274040, 0.3292820, 0.0433136),
    (0.0690970, 0.9195400, 0.0113612),
    (0.0163916, 0.0880132, 0.8955950),
)


def _apply_pq_oetf_np(linear: np.ndarray) -> np.ndarray:
    """SMPTE ST 2084 PQ OETF. 1.0 input = 10,000 cd/m²."""
    y = np.clip(linear.astype(np.float32, copy=False), 0.0, 1.0)
    ym = np.power(y, _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * ym) / (1.0 + _PQ_C3 * ym), _PQ_M2)


def _input_tensor_to_linear_bt2020_np(tensor: torch.Tensor) -> np.ndarray:
    """SDR input ([-1, 1]) → linear sRGB → BT.2020 linear, modest HDR
    brightness (~100 nits) so it sits next to the HDR panels without
    clipping the display."""
    arr = tensor.detach().float().clamp(-1.0, 1.0).add(1.0).mul(0.5).cpu().numpy()
    # CHW → HWC
    arr = np.transpose(arr, (1, 2, 0))
    # Approx sRGB → linear with gamma 2.2 (close enough for preview use)
    linear_srgb = np.power(np.clip(arr, 0.0, 1.0), 2.2)
    m = np.array(_BT709_TO_BT2020, dtype=np.float32)
    h, w, _ = linear_srgb.shape
    linear_bt2020 = (linear_srgb.reshape(-1, 3) @ m.T).reshape(h, w, 3)
    # Scale to 100 nits / 1000 = 0.1 of peak so input is dim vs HDR panels
    return np.clip(linear_bt2020, 0.0, None) * 0.1


def _hdr_panel_label(canvas: np.ndarray, label: str) -> np.ndarray:
    """Burn-in label tinted to remain readable in HDR (low-brightness white)."""
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        return canvas
    # Draw label in 8-bit space then composite at ~150 nits white.
    overlay_u8 = np.zeros((min(canvas.shape[0], 28), min(canvas.shape[1], 220), 3), dtype=np.uint8)
    cv2.rectangle(overlay_u8, (0, 0), (overlay_u8.shape[1], overlay_u8.shape[0]), (0, 0, 0), thickness=-1)
    cv2.putText(
        overlay_u8, label, (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
        thickness=1, lineType=cv2.LINE_AA,
    )
    label_linear = (overlay_u8.astype(np.float32) / 255.0) ** 2.2 * 0.15
    canvas[: overlay_u8.shape[0], : overlay_u8.shape[1]] = label_linear
    return canvas


def save_triptych_video_hdr(
    output_path: str | Path,
    input_image: torch.Tensor,
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    target_representation: str,
    base_predicted_image: torch.Tensor | None = None,
    fps: int = 4,
    peak_nits: float = _HDR_PEAK_NITS,
) -> Path | None:
    """HDR10 PQ-encoded MP4 of the per-frame triptych. Input panel is
    sRGB→BT.2020 at ~100 nits; HDR panels are linear BT.2020 [0, 1]
    treated as 0–``peak_nits`` cd/m². T<=1 returns None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_frames = _as_chw_frames(input_image)
    pred_frames = _as_chw_frames(predicted_image)
    target_frames = _as_chw_frames(target_image)
    base_frames = (
        _as_chw_frames(base_predicted_image)
        if base_predicted_image is not None
        else None
    )
    T = min(len(input_frames), len(pred_frames), len(target_frames))
    if T <= 1:
        return None

    composed_u16: list[np.ndarray] = []
    for t in range(T):
        in_lin = _input_tensor_to_linear_bt2020_np(input_frames[t])
        pred_lin = (
            linear_hdr_from_target_tensor(pred_frames[t], target_representation)
            .detach().cpu().numpy().transpose(1, 2, 0)
        )
        target_lin = (
            linear_hdr_from_target_tensor(target_frames[t], target_representation)
            .detach().cpu().numpy().transpose(1, 2, 0)
        )
        panels = [in_lin, pred_lin, target_lin]
        labels = ["input", "prediction", "ground_truth"]
        if base_frames is not None and t < len(base_frames):
            base_lin = (
                linear_hdr_from_target_tensor(base_frames[t], target_representation)
                .detach().cpu().numpy().transpose(1, 2, 0)
            )
            panels.append(base_lin)
            labels.append("base_seedvr")
        # Burn-in labels (linear-space)
        panels = [_hdr_panel_label(np.ascontiguousarray(p, dtype=np.float32), lab) for p, lab in zip(panels, labels)]
        canvas = np.concatenate(panels, axis=1)
        # Linear → PQ OETF: scale by peak_nits/10_000 so 1.0 = peak_nits cd/m²
        canvas_for_oetf = np.clip(canvas, 0.0, None) * np.float32(peak_nits / _PQ_OETF_PEAK_NITS)
        encoded = _apply_pq_oetf_np(canvas_for_oetf)
        u16 = np.clip(encoded * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
        composed_u16.append(u16)

    import av  # type: ignore

    height, width, _ = composed_u16[0].shape
    even_h = height - (height % 2)
    even_w = width - (width % 2)
    container = av.open(str(output_path), mode="w", format="mp4")
    try:
        stream = container.add_stream("libx265", rate=int(fps))
        stream.width = even_w
        stream.height = even_h
        stream.pix_fmt = "yuv420p10le"
        stream.options = {
            "x265-params": (
                "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"
                ":range=limited:hdr10=1:hdr10-opt=1"
            ),
            "profile": "main10",
            "crf": "22",
            "preset": "medium",
            "movflags": "+faststart",
            "tag:v": "hvc1",
        }
        for u16_frame in composed_u16:
            if (u16_frame.shape[0], u16_frame.shape[1]) != (even_h, even_w):
                u16_frame = u16_frame[:even_h, :even_w]
            video_frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(u16_frame), format="rgb48le"
            )
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception:
        try:
            container.close()
        except Exception:
            pass
        return None
    container.close()
    return output_path


def save_dataset_sample_preview(
    output_path: str | Path,
    input_image: torch.Tensor,
    target_image: torch.Tensor,
    target_representation: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_image = _select_preview_frame(input_image)
    target_image = _select_preview_frame(target_image)
    target_linear = linear_hdr_from_target_tensor(target_image, target_representation)
    panels = [
        _input_tensor_to_uint8_image(input_image),
        _preview_uint8_from_linear_hdr(target_linear),
    ]
    panels = _add_panel_labels(panels, ["input", "ground_truth"])
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
    predicted_frames = _as_chw_frames(predicted_image)
    target_frames = _as_chw_frames(target_image)
    if len(predicted_frames) != len(target_frames):
        if len(predicted_frames) == 1:
            predicted_frames = predicted_frames * len(target_frames)
        elif len(target_frames) == 1:
            target_frames = target_frames * len(predicted_frames)
        else:
            raise ValueError(
                "Validation prediction/target frame counts do not match: "
                f"{len(predicted_frames)} vs {len(target_frames)}"
            )

    if len(predicted_frames) > 1:
        frame_metrics = [
            _compute_hdr_metrics_chw(predicted_frame, target_frame, target_representation)
            for predicted_frame, target_frame in zip(predicted_frames, target_frames)
        ]
        metric_names = sorted(set.intersection(*(set(row) for row in frame_metrics)))
        aggregated = {
            name: float(np.mean([row[name] for row in frame_metrics]))
            for name in metric_names
        }
        # Video-temporal metrics (paper §4: F2F-PSNR + Flicker). Require T>1.
        aggregated.update(
            _compute_video_metrics(
                predicted_frames, target_frames, target_representation
            )
        )
        return aggregated

    return _compute_hdr_metrics_chw(predicted_frames[0], target_frames[0], target_representation)


def _compute_video_metrics(
    predicted_frames: list[torch.Tensor],
    target_frames: list[torch.Tensor],
    target_representation: str,
) -> dict[str, float]:
    """F2F-PSNR (consecutive-frame PSNR in tonemapped sRGB) + Flicker
    (luminance temporal stddev / mean), computed on linear HDR values
    after BT.2020→BT.709 + Reinhard tonemap. Returns empty dict for T<=1.
    """
    if len(predicted_frames) <= 1:
        return {}
    pred_tm = torch.stack(
        [
            _robust_tonemap(
                _bt2020_linear_to_bt709_linear(
                    linear_hdr_from_target_tensor(f, target_representation)
                )
            )
            for f in predicted_frames
        ],
        dim=0,
    )
    tgt_tm = torch.stack(
        [
            _robust_tonemap(
                _bt2020_linear_to_bt709_linear(
                    linear_hdr_from_target_tensor(f, target_representation)
                )
            )
            for f in target_frames
        ],
        dim=0,
    )
    # F2F-PSNR: PSNR between consecutive frames within each sequence
    pred_f2f = [_psnr(pred_tm[t], pred_tm[t + 1], data_range=1.0) for t in range(pred_tm.shape[0] - 1)]
    tgt_f2f = [_psnr(tgt_tm[t], tgt_tm[t + 1], data_range=1.0) for t in range(tgt_tm.shape[0] - 1)]
    # Flicker: per-pixel temporal stddev of luminance / mean luminance
    pred_lum = (0.2126 * pred_tm[:, 0] + 0.7152 * pred_tm[:, 1] + 0.0722 * pred_tm[:, 2])
    tgt_lum = (0.2126 * tgt_tm[:, 0] + 0.7152 * tgt_tm[:, 1] + 0.0722 * tgt_tm[:, 2])
    eps = 1.0e-3
    pred_flicker = float((pred_lum.std(dim=0) / (pred_lum.mean(dim=0) + eps)).mean().item())
    tgt_flicker = float((tgt_lum.std(dim=0) / (tgt_lum.mean(dim=0) + eps)).mean().item())
    return {
        "f2f_psnr_pred": float(np.mean(pred_f2f)),
        "f2f_psnr_target": float(np.mean(tgt_f2f)),
        "f2f_psnr_diff": float(np.mean(pred_f2f) - np.mean(tgt_f2f)),
        "flicker_pred": pred_flicker,
        "flicker_target": tgt_flicker,
    }


def _compute_hdr_metrics_chw(
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    target_representation: str,
) -> dict[str, float]:
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
