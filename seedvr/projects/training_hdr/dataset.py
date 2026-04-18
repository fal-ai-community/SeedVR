from __future__ import annotations

import json
import random
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    scene_id: str
    split: str
    input_sdr_path: str | None = None
    input_sdr_paths: list[str] | None = None
    width: int = 0
    height: int = 0
    variant_id: str | None = None
    target_hdr_path: str | None = None
    target_hdr_paths: list[str] | None = None
    target_hdr_npy_path: str | None = None
    target_hdr_npy_paths: list[str] | None = None
    target_mu_law_path: str | None = None
    target_mu_law_paths: list[str] | None = None
    target_log_hdr_path: str | None = None
    target_log_hdr_paths: list[str] | None = None
    compressed_target_path: str | None = None
    compressed_target_paths: list[str] | None = None
    clip_mask_path: str | None = None
    saturation_mask_path: str | None = None
    metadata_path: str | None = None
    frame_indices: list[int] | None = None

    @classmethod
    def from_row(cls, row: dict) -> "ManifestSample":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in row.items() if key in allowed}
        return cls(**filtered)

    @property
    def is_video(self) -> bool:
        return bool(self.input_sdr_paths)


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


def _sample_crop_box(
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
    random_crop: bool,
    rng: random.Random,
) -> tuple[int, int, int, int]:
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

    return top, left, top + crop_height, left + crop_width


def _crop_pair(
    input_image: np.ndarray,
    target_image: np.ndarray,
    crop_height: int,
    crop_width: int,
    random_crop: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    top, left, bottom, right = _sample_crop_box(
        height=input_image.shape[0],
        width=input_image.shape[1],
        crop_height=crop_height,
        crop_width=crop_width,
        random_crop=random_crop,
        rng=rng,
    )
    return (
        input_image[top:bottom, left:right],
        target_image[top:bottom, left:right],
    )


def _crop_pair_sequence(
    input_images: list[np.ndarray],
    target_images: list[np.ndarray],
    crop_height: int,
    crop_width: int,
    random_crop: bool,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not input_images or not target_images:
        raise ValueError("Expected non-empty frame sequences.")
    top, left, bottom, right = _sample_crop_box(
        height=input_images[0].shape[0],
        width=input_images[0].shape[1],
        crop_height=crop_height,
        crop_width=crop_width,
        random_crop=random_crop,
        rng=rng,
    )
    cropped_inputs = [image[top:bottom, left:right] for image in input_images]
    cropped_targets = [image[top:bottom, left:right] for image in target_images]
    return cropped_inputs, cropped_targets


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


def _stack_input_frames(frames: list[np.ndarray]) -> torch.Tensor:
    tensors = [_to_input_tensor(frame) for frame in frames]
    return torch.stack(tensors, dim=0)


def _stack_target_frames(frames: list[np.ndarray], target_representation: str) -> torch.Tensor:
    tensors = [_to_target_tensor(frame, target_representation) for frame in frames]
    return torch.stack(tensors, dim=0)


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

        input_path = self.dataset_root / self._single_path(sample, "input_sdr")
        target_path = self.dataset_root / self._single_target_path(sample)
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

    def _single_path(self, sample: ManifestSample, field_name: str) -> str:
        value = getattr(sample, f"{field_name}_path", None)
        if not value:
            raise ValueError(f"Sample {sample.sample_id} does not provide {field_name}_path")
        return value

    def _single_target_path(self, sample: ManifestSample) -> str:
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


class SeedVRHDRVideoDataset(Dataset):
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
        input_paths = self._path_list(sample, "input_sdr")
        target_paths = self._target_path_list(sample)

        input_images = [_read_rgb_png(self.dataset_root / path) for path in input_paths]
        target_images = [_read_target_array(self.dataset_root / path) for path in target_paths]
        if len(input_images) != len(target_images):
            raise ValueError(
                f"Input/target clip length mismatch for {sample.sample_id}: "
                f"{len(input_images)} vs {len(target_images)}"
            )
        if not input_images:
            raise ValueError(f"Empty clip for {sample.sample_id}")

        reference_shape = input_images[0].shape[:2]
        for frame_index, (input_image, target_image) in enumerate(zip(input_images, target_images)):
            if input_image.shape[:2] != target_image.shape[:2]:
                raise ValueError(
                    f"Frame {frame_index} mismatch for {sample.sample_id}: "
                    f"{input_image.shape[:2]} vs {target_image.shape[:2]}"
                )
            if input_image.shape[:2] != reference_shape:
                raise ValueError(
                    f"Inconsistent input frame size in clip {sample.sample_id}: "
                    f"{input_image.shape[:2]} vs {reference_shape}"
                )

        input_images = [
            _resize_to_cover(
                image,
                self.train_height,
                self.train_width,
                interpolation=cv2.INTER_AREA,
            )
            for image in input_images
        ]
        target_images = [
            _resize_to_cover(
                image,
                self.train_height,
                self.train_width,
                interpolation=cv2.INTER_CUBIC,
            )
            for image in target_images
        ]
        input_images, target_images = _crop_pair_sequence(
            input_images=input_images,
            target_images=target_images,
            crop_height=self.train_height,
            crop_width=self.train_width,
            random_crop=self.random_crop,
            rng=rng,
        )

        return {
            "input_sdr": _stack_input_frames(input_images),
            "target": _stack_target_frames(target_images, self.target_representation),
            "scene_id": sample.scene_id,
            "sample_id": sample.sample_id,
            "variant_id": sample.variant_id or "",
        }

    def _path_list(self, sample: ManifestSample, field_name: str) -> list[str]:
        plural = getattr(sample, f"{field_name}_paths", None)
        if plural:
            return plural
        singular = getattr(sample, f"{field_name}_path", None)
        if singular:
            return [singular]
        raise ValueError(f"Sample {sample.sample_id} does not provide {field_name}_path(s)")

    def _target_path_list(self, sample: ManifestSample) -> list[str]:
        if self.target_representation == "raw_hdr":
            if sample.target_hdr_npy_paths:
                return sample.target_hdr_npy_paths
            if sample.target_hdr_npy_path:
                return [sample.target_hdr_npy_path]
        elif self.target_representation == "mu_law_mu5000":
            if sample.target_mu_law_paths:
                return sample.target_mu_law_paths
            if sample.target_mu_law_path:
                return [sample.target_mu_law_path]
        elif self.target_representation == "log_hdr":
            if sample.target_log_hdr_paths:
                return sample.target_log_hdr_paths
            if sample.target_log_hdr_path:
                return [sample.target_log_hdr_path]
        else:
            raise ValueError(f"Unsupported target_representation: {self.target_representation}")

        if sample.compressed_target_paths:
            return sample.compressed_target_paths
        if sample.compressed_target_path:
            return [sample.compressed_target_path]
        raise ValueError(
            f"Sample {sample.sample_id} does not provide a target path list for {self.target_representation}"
        )
