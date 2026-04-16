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
