from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    scene_id: str
    split: str
    input_sdr_path: str
    compressed_target_path: str
    width: int
    height: int
    variant_id: str | None = None
    target_hdr_path: str | None = None
    clip_mask_path: str | None = None
    saturation_mask_path: str | None = None
    metadata_path: str | None = None


def load_manifest(path: str | Path) -> list[ManifestSample]:
    rows: list[ManifestSample] = []
    with open(path) as file:
        for line in file:
            rows.append(ManifestSample(**json.loads(line)))
    return rows


def _read_rgb_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _read_compressed_target(path: Path) -> np.ndarray:
    target = np.load(path)
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB compressed target at {path}, got {target.shape}")
    target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(target.astype(np.float32), 0.0, 1.0)


def _resize_to_min_side(image: np.ndarray, min_side: int, interpolation: int) -> np.ndarray:
    height, width = image.shape[:2]
    if min(height, width) >= min_side:
        return image
    scale = float(min_side) / float(min(height, width))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)


def _aligned_size(size: int, alignment: int = 16) -> int:
    size = max(size, alignment)
    return size - (size % alignment)


def _crop_pair(
    input_image: np.ndarray,
    target_image: np.ndarray,
    crop_size: int,
    random_crop: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = input_image.shape[:2]
    crop_size = min(crop_size, height, width)
    crop_size = _aligned_size(crop_size)
    crop_size = min(crop_size, height, width)
    if crop_size <= 0:
        raise ValueError(f"Invalid crop size {crop_size} for image shape {(height, width)}")

    if height == crop_size:
        top = 0
    elif random_crop:
        top = rng.randint(0, height - crop_size)
    else:
        top = (height - crop_size) // 2

    if width == crop_size:
        left = 0
    elif random_crop:
        left = rng.randint(0, width - crop_size)
    else:
        left = (width - crop_size) // 2

    bottom = top + crop_size
    right = left + crop_size
    return (
        input_image[top:bottom, left:right],
        target_image[top:bottom, left:right],
    )


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return tensor.mul(2.0).sub(1.0)


class SeedVRHDRImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        resolution: int,
        random_crop: bool,
        seed: int,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples = load_manifest(manifest_path)
        self.resolution = resolution
        self.random_crop = random_crop
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rng = random.Random((self.seed * 10_000) + index)

        input_path = self.dataset_root / sample.input_sdr_path
        target_path = self.dataset_root / sample.compressed_target_path
        input_image = _read_rgb_png(input_path)
        target_image = _read_compressed_target(target_path)

        if input_image.shape[:2] != target_image.shape[:2]:
            raise ValueError(
                f"Input/target shape mismatch for {sample.sample_id}: "
                f"{input_image.shape[:2]} vs {target_image.shape[:2]}"
            )

        input_image = _resize_to_min_side(
            input_image, self.resolution, interpolation=cv2.INTER_AREA
        )
        target_image = _resize_to_min_side(
            target_image, self.resolution, interpolation=cv2.INTER_CUBIC
        )

        input_image, target_image = _crop_pair(
            input_image=input_image,
            target_image=target_image,
            crop_size=self.resolution,
            random_crop=self.random_crop,
            rng=rng,
        )

        return {
            "input_sdr": _to_tensor(input_image),
            "target": _to_tensor(target_image),
            "scene_id": sample.scene_id,
            "sample_id": sample.sample_id,
            "variant_id": sample.variant_id or "",
        }

