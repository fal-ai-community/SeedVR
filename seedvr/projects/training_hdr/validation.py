from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().clamp(-1.0, 1.0).add(1.0).mul(127.5)
    array = array.round().to(torch.uint8).cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return array


def save_triptych(
    output_path: str | Path,
    input_image: torch.Tensor,
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.concatenate(
        [
            tensor_to_uint8_image(input_image),
            tensor_to_uint8_image(predicted_image),
            tensor_to_uint8_image(target_image),
        ],
        axis=1,
    )
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), canvas_bgr)
    return output_path

