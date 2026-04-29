from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-3,
) -> torch.Tensor:
    return torch.sqrt((prediction - target) ** 2 + eps**2).mean()


def denoise_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction.float(), target.float())


def latent_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction.float(), target.float())


def image_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    return F.l1_loss(prediction.float(), target.float())



def _flatten_btchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(1)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(f"Expected BCHW or BTCHW tensor, got {tuple(tensor.shape)}")


def _luminance(tensor: torch.Tensor) -> torch.Tensor:
    coeffs = tensor.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 1, 3, 1, 1)
    return (tensor[:, :, :3] * coeffs).sum(dim=2)


def _smooth_profile(profile: torch.Tensor, kernel_size: int = 33) -> torch.Tensor:
    if profile.shape[-1] < 3:
        return profile
    kernel_size = min(kernel_size, profile.shape[-1] if profile.shape[-1] % 2 == 1 else profile.shape[-1] - 1)
    kernel_size = max(3, kernel_size)
    padding = kernel_size // 2
    return F.avg_pool1d(profile.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=padding).squeeze(1)


def _btchw_as_bchw(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _flatten_btchw(tensor.float())
    b, t, c, h, w = tensor.shape
    return tensor.reshape(b * t, c, h, w)


def _luminance_bchw(tensor: torch.Tensor) -> torch.Tensor:
    coeffs = tensor.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (tensor[:, :3] * coeffs).sum(dim=1, keepdim=True)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _gradient_xy(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = F.pad(tensor[..., :, 1:] - tensor[..., :, :-1], (1, 0, 0, 0))
    grad_y = F.pad(tensor[..., 1:, :] - tensor[..., :-1, :], (0, 0, 1, 0))
    return grad_x, grad_y


def color_constancy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pool_size: int = 32,
) -> torch.Tensor:
    """Preserve low-frequency chroma to reduce blue/purple and cyan shifts."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred_rgb = prediction[:, :3]
    target_rgb = target[:, :3]
    pred_luma = _luminance_bchw(pred_rgb)
    target_luma = _luminance_bchw(target_rgb)
    pred_chroma = pred_rgb - pred_luma
    target_chroma = target_rgb - target_luma
    kernel = max(1, min(pool_size, pred_chroma.shape[-2], pred_chroma.shape[-1]))
    if kernel > 1:
        pred_chroma = F.avg_pool2d(pred_chroma, kernel_size=kernel, stride=1, padding=kernel // 2)
        target_chroma = F.avg_pool2d(target_chroma, kernel_size=kernel, stride=1, padding=kernel // 2)
        if pred_chroma.shape[-2:] != target.shape[-2:]:
            pred_chroma = pred_chroma[..., : target.shape[-2], : target.shape[-1]]
            target_chroma = target_chroma[..., : target.shape[-2], : target.shape[-1]]
    mean_loss = charbonnier_loss(pred_chroma, target_chroma)
    pred_std = pred_chroma.flatten(2).std(dim=-1)
    target_std = target_chroma.flatten(2).std(dim=-1)
    return mean_loss + F.l1_loss(pred_std, target_std)


def _high_pass(tensor: torch.Tensor, blur_kernel: int = 9) -> torch.Tensor:
    blur_kernel = max(3, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_kernel = min(blur_kernel, tensor.shape[-2], tensor.shape[-1])
    if blur_kernel % 2 == 0:
        blur_kernel -= 1
    if blur_kernel < 3:
        return tensor - tensor.mean(dim=(-2, -1), keepdim=True)
    blurred = F.avg_pool2d(
        tensor,
        kernel_size=blur_kernel,
        stride=1,
        padding=blur_kernel // 2,
    )
    if blurred.shape[-2:] != tensor.shape[-2:]:
        blurred = blurred[..., : tensor.shape[-2], : tensor.shape[-1]]
    return tensor - blurred


def detail_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    blur_kernel: int = 9,
) -> torch.Tensor:
    """Match high-frequency RGB/luma residuals so fine texture does not wash out."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred_rgb = prediction[:, :3]
    target_rgb = target[:, :3]
    rgb_loss = charbonnier_loss(
        _high_pass(pred_rgb, blur_kernel=blur_kernel),
        _high_pass(target_rgb, blur_kernel=blur_kernel),
    )
    luma_loss = charbonnier_loss(
        _high_pass(_luminance_bchw(pred_rgb), blur_kernel=blur_kernel),
        _high_pass(_luminance_bchw(target_rgb), blur_kernel=blur_kernel),
    )
    return 0.5 * rgb_loss + 0.5 * luma_loss


def dwt_high_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    levels: int = 2,
) -> torch.Tensor:
    """Match Haar wavelet high-frequency bands across one or more scales."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred = prediction[:, :3].float()
    tgt = target[:, :3].float()
    levels = max(1, int(levels))
    total = pred.new_tensor(0.0)
    count = 0
    for _level in range(levels):
        height = min(pred.shape[-2], tgt.shape[-2])
        width = min(pred.shape[-1], tgt.shape[-1])
        height = height - (height % 2)
        width = width - (width % 2)
        if height < 2 or width < 2:
            break
        pred = pred[..., :height, :width]
        tgt = tgt[..., :height, :width]
        filters = pred.new_tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ]
        ).view(4, 1, 2, 2)
        channels = pred.shape[1]
        filters = filters.repeat(channels, 1, 1, 1)
        pred_bands = F.conv2d(pred, filters, stride=2, groups=channels)
        tgt_bands = F.conv2d(tgt, filters, stride=2, groups=channels)
        pred_bands = pred_bands.view(pred.shape[0], channels, 4, pred_bands.shape[-2], pred_bands.shape[-1])
        tgt_bands = tgt_bands.view(tgt.shape[0], channels, 4, tgt_bands.shape[-2], tgt_bands.shape[-1])
        total = total + charbonnier_loss(pred_bands[:, :, 1:], tgt_bands[:, :, 1:])
        count += 1
        pred = pred_bands[:, :, 0]
        tgt = tgt_bands[:, :, 0]
    return total / max(1, count)


def fft_high_frequency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    min_freq: float = 0.25,
) -> torch.Tensor:
    """Compare high-frequency log-magnitude spectra while ignoring low-frequency color/exposure."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred_luma = _luminance_bchw(prediction[:, :3].float())
    target_luma = _luminance_bchw(target[:, :3].float())
    height, width = pred_luma.shape[-2:]
    fy = torch.fft.fftfreq(height, device=pred_luma.device).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=pred_luma.device).view(1, width // 2 + 1)
    radius = torch.sqrt(fx.pow(2) + fy.pow(2))
    mask = (radius >= float(min_freq)).to(dtype=pred_luma.dtype).view(1, 1, height, width // 2 + 1)
    if not bool(mask.any().item()):
        return pred_luma.new_tensor(0.0)
    pred_fft = torch.fft.rfft2(pred_luma, norm="ortho")
    target_fft = torch.fft.rfft2(target_luma, norm="ortho")
    pred_mag = torch.log1p(pred_fft.abs())
    target_mag = torch.log1p(target_fft.abs())
    return _masked_mean((pred_mag - target_mag).abs(), mask)


def total_variation_map(tensor: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Return a 3-channel local gradient magnitude map for perceptual TV losses."""
    tensor = _btchw_as_bchw(tensor)
    rgb = tensor[:, :3].float()
    dx, dy = _gradient_xy(rgb)
    return torch.sqrt(dx.square() + dy.square() + eps)


def edge_consistency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Preserve source/target structure and reduce painterly texture hallucination."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred_luma = _luminance_bchw(prediction[:, :3])
    target_luma = _luminance_bchw(target[:, :3])
    pred_dx, pred_dy = _gradient_xy(pred_luma)
    target_dx, target_dy = _gradient_xy(target_luma)
    return charbonnier_loss(pred_dx, target_dx) + charbonnier_loss(pred_dy, target_dy)


def flat_region_smoothness_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    flat_gradient_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalize local second-derivative artifacts in smooth skies/UI panels."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred_luma = _luminance_bchw(prediction[:, :3])
    target_luma = _luminance_bchw(target[:, :3])
    target_dx, target_dy = _gradient_xy(target_luma)
    flat_mask = ((target_dx.abs() + target_dy.abs()) < flat_gradient_threshold).to(
        dtype=prediction.dtype
    )
    d2x = pred_luma[..., :, 2:] - 2.0 * pred_luma[..., :, 1:-1] + pred_luma[..., :, :-2]
    d2y = pred_luma[..., 2:, :] - 2.0 * pred_luma[..., 1:-1, :] + pred_luma[..., :-2, :]
    mask_x = flat_mask[..., :, 1:-1]
    mask_y = flat_mask[..., 1:-1, :]
    return _masked_mean(d2x.abs(), mask_x) + _masked_mean(d2y.abs(), mask_y)


def low_frequency_banding_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    flat_gradient_threshold: float = 0.08,
) -> torch.Tensor:
    prediction = _flatten_btchw(prediction.float())
    target = _flatten_btchw(target.to(device=prediction.device, dtype=prediction.dtype).float())
    pred_luma = _luminance(prediction)
    target_luma = _luminance(target)

    grad_x = F.pad((target_luma[..., :, 1:] - target_luma[..., :, :-1]).abs(), (1, 0, 0, 0))
    grad_y = F.pad((target_luma[..., 1:, :] - target_luma[..., :-1, :]).abs(), (0, 0, 1, 0))
    flat_mask = ((grad_x + grad_y) < flat_gradient_threshold).float()
    residual = (pred_luma - target_luma) * flat_mask

    col_den = flat_mask.sum(dim=-2).clamp_min(1.0)
    row_den = flat_mask.sum(dim=-1).clamp_min(1.0)
    col_profile = residual.sum(dim=-2) / col_den
    row_profile = residual.sum(dim=-1) / row_den
    col_profile = col_profile.reshape(-1, col_profile.shape[-1])
    row_profile = row_profile.reshape(-1, row_profile.shape[-1])
    col_profile = col_profile - col_profile.mean(dim=-1, keepdim=True)
    row_profile = row_profile - row_profile.mean(dim=-1, keepdim=True)
    col_profile = _smooth_profile(col_profile)
    row_profile = _smooth_profile(row_profile)
    return col_profile.abs().mean() + row_profile.abs().mean()
