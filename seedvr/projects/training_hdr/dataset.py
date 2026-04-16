from __future__ import annotations

import json
import random
from dataclasses import fields
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
    width: int
    height: int
    variant_id: str | None = None
    target_hdr_path: str | None = None
    target_hdr_npy_path: str | None = None
    target_mu_law_path: str | None = None
    target_log_hdr_path: str | None = None
    compressed_target_path: str | None = None
    clip_mask_path: str | None = None
    saturation_mask_path: str | None = None
    metadata_path: str | None = None

    @classmethod
    def from_row(cls, row: dict) -> "ManifestSample":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in row.items() if key in allowed}
        return cls(**filtered)


def load_manifest(path: str | Path) -> list[ManifestSample]:
    rows: list[ManifestSample] = []
    with open(path) as file:
        for line in file:
            rows.append(ManifestSample.from_row(json.loads(line)))
    return rows


def _read_rgb_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _read_target_array(path: Path) -> np.ndarray:
    target = np.load(path)
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB target at {path}, got {target.shape}")
    return np.nan_to_num(target.astype(np.float32), nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _resize_to_cover(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    interpolation: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    if height >= target_height and width >= target_width:
        return image
    scale = max(float(target_height) / float(height), float(target_width) / float(width))
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)


def _aligned_size(size: int, alignment: int = 16) -> int:
    size = max(size, alignment)
    return size - (size % alignment)


def _crop_pair(
    input_image: np.ndarray,
    target_image: np.ndarray,
    crop_height: int,
    crop_width: int,
    random_crop: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = input_image.shape[:2]
    crop_height = min(_aligned_size(crop_height), height)
    crop_width = min(_aligned_size(crop_width), width)
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError(
            f"Invalid crop size {(crop_height, crop_width)} for image shape {(height, width)}"
        )

    if height == crop_height:
        top = 0
    elif random_crop:
        top = rng.randint(0, height - crop_height)
    else:
        top = (height - crop_height) // 2

    if width == crop_width:
        left = 0
    elif random_crop:
        left = rng.randint(0, width - crop_width)
    else:
        left = (width - crop_width) // 2

    bottom = top + crop_height
    right = left + crop_width
    return (
        input_image[top:bottom, left:right],
        target_image[top:bottom, left:right],
    )


def _to_input_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return tensor.mul(2.0).sub(1.0)


def _to_target_tensor(image: np.ndarray, target_representation: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    if target_representation == "mu_law_mu5000":
        return tensor.mul(2.0).sub(1.0)
    if target_representation in {"raw_hdr", "log_hdr"}:
        return tensor
    raise ValueError(f"Unsupported target_representation: {target_representation}")


class SeedVRHDRImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        train_height: int,
        train_width: int,
        random_crop: bool,
        seed: int,
        target_representation: str,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples = load_manifest(manifest_path)
        self.train_height = train_height
        self.train_width = train_width
        self.random_crop = random_crop
        self.seed = seed
        self.target_representation = target_representation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rng = random.Random((self.seed * 10_000) + index)

        input_path = self.dataset_root / sample.input_sdr_path
        target_rel = self._target_path_for_sample(sample)
        target_path = self.dataset_root / target_rel
        input_image = _read_rgb_png(input_path)
        target_image = _read_target_array(target_path)

        if input_image.shape[:2] != target_image.shape[:2]:
            raise ValueError(
                f"Input/target shape mismatch for {sample.sample_id}: "
                f"{input_image.shape[:2]} vs {target_image.shape[:2]}"
            )

        input_image = _resize_to_cover(
            input_image,
            self.train_height,
            self.train_width,
            interpolation=cv2.INTER_AREA,
        )
        target_image = _resize_to_cover(
            target_image,
            self.train_height,
            self.train_width,
            interpolation=cv2.INTER_CUBIC,
        )

        input_image, target_image = _crop_pair(
            input_image=input_image,
            target_image=target_image,
            crop_height=self.train_height,
            crop_width=self.train_width,
            random_crop=self.random_crop,
            rng=rng,
        )

        return {
            "input_sdr": _to_input_tensor(input_image),
            "target": _to_target_tensor(target_image, self.target_representation),
            "scene_id": sample.scene_id,
            "sample_id": sample.sample_id,
            "variant_id": sample.variant_id or "",
        }

    def _target_path_for_sample(self, sample: ManifestSample) -> str:
        if self.target_representation == "raw_hdr":
            if sample.target_hdr_npy_path:
                return sample.target_hdr_npy_path
            if sample.target_hdr_path and sample.target_hdr_path.endswith(".npy"):
                return sample.target_hdr_path
        elif self.target_representation == "mu_law_mu5000":
            if sample.target_mu_law_path:
                return sample.target_mu_law_path
        elif self.target_representation == "log_hdr":
            if sample.target_log_hdr_path:
                return sample.target_log_hdr_path
        else:
            raise ValueError(f"Unsupported target_representation: {self.target_representation}")

        if sample.compressed_target_path:
            return sample.compressed_target_path
        raise ValueError(
            f"Sample {sample.sample_id} does not provide a target path for {self.target_representation}"
        )
