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
    return charbonnier_loss(prediction.float(), target.float())



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


def _text_region_crop_batch(
    tensor: torch.Tensor,
    boxes: torch.Tensor,
    mask: torch.Tensor,
    crop_height: int,
    crop_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = _flatten_btchw(tensor.float())
    b, t, c, h, w = images.shape
    flat_images = images.reshape(b * t, c, h, w)
    if boxes.ndim == 3:
        boxes = boxes.unsqueeze(1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(1)
    boxes = boxes.to(device=tensor.device, dtype=tensor.dtype).reshape(b * t, -1, 4)
    mask = mask.to(device=tensor.device, dtype=torch.bool).reshape(b * t, -1)
    crops: list[torch.Tensor] = []
    crop_masks: list[torch.Tensor] = []
    ys = torch.linspace(-1.0, 1.0, crop_height, device=tensor.device, dtype=tensor.dtype)
    xs = torch.linspace(-1.0, 1.0, crop_width, device=tensor.device, dtype=tensor.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    for image_index in range(flat_images.shape[0]):
        valid_indices = torch.nonzero(mask[image_index], as_tuple=False).flatten()
        for region_index in valid_indices:
            x0, y0, x1, y1 = boxes[image_index, region_index].unbind()
            if (x1 - x0) <= 1.0e-4 or (y1 - y0) <= 1.0e-4:
                continue
            cx = (x0 + x1) - 1.0
            cy = (y0 + y1) - 1.0
            sx = (x1 - x0).clamp_min(1.0e-4)
            sy = (y1 - y0).clamp_min(1.0e-4)
            grid = base_grid.clone()
            grid[..., 0] = grid[..., 0] * sx + cx
            grid[..., 1] = grid[..., 1] * sy + cy
            crop = F.grid_sample(
                flat_images[image_index : image_index + 1],
                grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0]
            crops.append(crop)
            crop_masks.append(tensor.new_ones(()))
    if not crops:
        return tensor.new_zeros((0, images.shape[2], crop_height, crop_width)), tensor.new_zeros((0,))
    return torch.stack(crops, dim=0), torch.stack(crop_masks, dim=0)


def extract_text_region_crops(
    tensor: torch.Tensor,
    boxes: torch.Tensor,
    mask: torch.Tensor,
    crop_height: int = 32,
    crop_width: int = 128,
) -> torch.Tensor:
    crops, _ = _text_region_crop_batch(tensor, boxes, mask, crop_height, crop_width)
    return crops


def text_region_detail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    boxes: torch.Tensor,
    mask: torch.Tensor,
    crop_height: int = 32,
    crop_width: int = 128,
    blur_kernel: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-box-local high-pass/edge loss for OCR-positive regions."""
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    pred_crops = extract_text_region_crops(prediction, boxes, mask, crop_height, crop_width)
    target_crops = extract_text_region_crops(target, boxes, mask, crop_height, crop_width)
    if pred_crops.numel() == 0:
        return prediction.new_zeros(()), prediction.new_zeros(())
    pred_luma = _luminance_bchw(pred_crops[:, :3])
    target_luma = _luminance_bchw(target_crops[:, :3])
    pred_dx, pred_dy = _gradient_xy(pred_luma)
    target_dx, target_dy = _gradient_xy(target_luma)
    edge_loss = charbonnier_loss(pred_dx, target_dx) + charbonnier_loss(pred_dy, target_dy)
    detail_loss_value = charbonnier_loss(
        _high_pass(pred_crops[:, :3], blur_kernel=blur_kernel),
        _high_pass(target_crops[:, :3], blur_kernel=blur_kernel),
    )
    return edge_loss + 0.5 * detail_loss_value, prediction.new_tensor(float(pred_crops.shape[0]))


def edge_consistency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Preserve source/target structure and reduce painterly texture hallucination."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred_luma = _luminance_bchw(prediction[:, :3])
    target_luma = _luminance_bchw(target[:, :3])
    pred_dx, pred_dy = _gradient_xy(pred_luma)
    target_dx, target_dy = _gradient_xy(target_luma)
    return charbonnier_loss(pred_dx, target_dx) + charbonnier_loss(pred_dy, target_dy)


def text_layout_edge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_threshold: float = 0.12,
) -> torch.Tensor:
    """Edge-gated structure loss for text/UI layout without relying on OCR tokens."""
    prediction = _btchw_as_bchw(prediction)
    target = _btchw_as_bchw(target.to(device=prediction.device, dtype=prediction.dtype))
    pred_luma = _luminance_bchw(prediction[:, :3])
    target_luma = _luminance_bchw(target[:, :3])
    pred_dx, pred_dy = _gradient_xy(pred_luma)
    target_dx, target_dy = _gradient_xy(target_luma)
    target_edge = target_dx.abs() + target_dy.abs()
    mask = (target_edge > edge_threshold).to(dtype=prediction.dtype)
    if mask.sum() < 1:
        return prediction.new_zeros(())
    return _masked_mean((pred_dx - target_dx).abs() + (pred_dy - target_dy).abs(), mask)


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
