from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


MU_LAW_MU = 5000.0
LOG_HDR_EPS = 1.0e-6
LOCK_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    scene_id: str
    split: str
    input_sdr_path: str | None = None
    input_sdr_paths: list[str] | None = None
    input_sdr_url: str | None = None
    input_sdr_urls: list[str] | None = None
    width: int = 0
    height: int = 0
    variant_id: str | None = None
    source_hdr_path: str | None = None
    source_hdr_paths: list[str] | None = None
    source_hdr_url: str | None = None
    source_hdr_urls: list[str] | None = None
    target_hdr_path: str | None = None
    target_hdr_paths: list[str] | None = None
    target_hdr_url: str | None = None
    target_hdr_urls: list[str] | None = None
    target_hdr_npy_path: str | None = None
    target_hdr_npy_paths: list[str] | None = None
    target_hdr_npy_url: str | None = None
    target_hdr_npy_urls: list[str] | None = None
    target_mu_law_path: str | None = None
    target_mu_law_paths: list[str] | None = None
    target_mu_law_url: str | None = None
    target_mu_law_urls: list[str] | None = None
    target_log_hdr_path: str | None = None
    target_log_hdr_paths: list[str] | None = None
    target_log_hdr_url: str | None = None
    target_log_hdr_urls: list[str] | None = None
    compressed_target_path: str | None = None
    compressed_target_paths: list[str] | None = None
    compressed_target_url: str | None = None
    compressed_target_urls: list[str] | None = None
    clip_mask_path: str | None = None
    saturation_mask_path: str | None = None
    metadata_path: str | None = None
    frame_indices: list[int] | None = None
    camera_response: str | None = None
    exposure_ev: float | None = None
    gamma: float | None = None
    quantization_bits: int | None = None
    jpeg_quality: int | None = None

    @classmethod
    def from_row(cls, row: dict) -> "ManifestSample":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in row.items() if key in allowed}
        return cls(**filtered)

    @property
    def is_video(self) -> bool:
        return bool(self.input_sdr_paths or self.input_sdr_urls or self.source_hdr_paths or self.source_hdr_urls)


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


def _read_hdr_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read HDR image: {path}")
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.nan_to_num(image, nan=0.0, posinf=1.0e6, neginf=0.0)


def _compress_target_from_hdr(hdr: np.ndarray, target_representation: str) -> np.ndarray:
    hdr = np.maximum(hdr.astype(np.float32), 0.0)
    if target_representation == "raw_hdr":
        return hdr
    if target_representation == "mu_law_mu5000":
        return np.log1p(MU_LAW_MU * hdr) / np.log1p(MU_LAW_MU)
    if target_representation == "log_hdr":
        return np.log(hdr + LOG_HDR_EPS)
    raise ValueError(f"Unsupported target_representation: {target_representation}")


def _read_target_for_representation(path: Path, target_representation: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return _read_target_array(path)
    if path.suffix.lower() in {".exr", ".hdr"}:
        return _compress_target_from_hdr(_read_hdr_image(path), target_representation)
    raise ValueError(f"Unsupported target file extension for {path}")


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


def _robust_normalize_hdr(hdr: np.ndarray) -> np.ndarray:
    hdr = np.maximum(hdr.astype(np.float32), 0.0)
    percentile = float(np.percentile(hdr, 99.5))
    scale = percentile if percentile > 1.0e-6 else max(float(hdr.max()), 1.0)
    return hdr / scale


def _apply_exposure_ev(image: np.ndarray, ev: float) -> np.ndarray:
    return image * (2.0**ev)


def _apply_camera_response(image: np.ndarray, crf_name: str, gamma: float) -> np.ndarray:
    image = np.clip(image, 0.0, None)
    if crf_name == "linear_clip":
        return np.clip(image, 0.0, 1.0)
    if crf_name == "gamma22":
        return np.clip(image, 0.0, 1.0) ** (1.0 / gamma)
    if crf_name == "filmic_simple":
        x = np.clip(image, 0.0, None)
        mapped = x / (1.0 + x)
        return np.clip(mapped, 0.0, 1.0) ** (1.0 / gamma)
    raise ValueError(f"Unknown camera response: {crf_name}")


def _quantize_sdr(image: np.ndarray, bits: int) -> np.ndarray:
    levels = float((2**bits) - 1)
    return np.round(np.clip(image, 0.0, 1.0) * levels) / levels


def _apply_jpeg_roundtrip(image: np.ndarray, quality: int | None) -> np.ndarray:
    if quality is None:
        return image
    bgr = cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return decoded.astype(np.float32) / 255.0


def _render_sdr_from_hdr(
    hdr: np.ndarray,
    *,
    camera_response: str,
    exposure_ev: float,
    gamma: float,
    quantization_bits: int,
    jpeg_quality: int | None,
) -> np.ndarray:
    normalized = _robust_normalize_hdr(hdr)
    exposed = _apply_exposure_ev(normalized, exposure_ev)
    mapped = _apply_camera_response(exposed, camera_response, gamma)
    quantized = _quantize_sdr(mapped, quantization_bits)
    quantized = _apply_jpeg_roundtrip(quantized, jpeg_quality)
    return np.clip(quantized, 0.0, 1.0).astype(np.float32)


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _suffix_hint(value: str | None, default: str) -> str:
    if not value:
        return default
    parsed = urllib.parse.urlparse(value)
    suffix = Path(parsed.path if parsed.scheme else value).suffix.lower()
    return suffix or default


def _validate_cached_asset(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".exr", ".hdr"}:
        _read_hdr_image(path)
        return
    if suffix in {".png", ".jpg", ".jpeg"}:
        _read_rgb_png(path)
        return
    if suffix == ".npy":
        _read_target_array(path)
        return
    if not path.exists():
        raise FileNotFoundError(f"Cached asset does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Cached asset is empty: {path}")


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        bgr = cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(tmp_path), bgr):
            raise ValueError(f"Failed to write PNG cache: {tmp_path}")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        np.save(tmp_path, array.astype(np.float32))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@contextmanager
def _file_lock(lock_path: Path, timeout_seconds: float = LOCK_TIMEOUT_SECONDS):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if lock_path.exists():
                age = time.time() - lock_path.stat().st_mtime
                if age > timeout_seconds:
                    try:
                        lock_path.unlink()
                        continue
                    except FileNotFoundError:
                        continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(0.25)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class _RuntimeAssetCache:
    def __init__(
        self,
        dataset_root: Path,
        runtime_cache_root: str | Path,
        download_timeout_seconds: int,
        download_retries: int,
    ) -> None:
        self.dataset_root = dataset_root
        self.runtime_cache_root = Path(runtime_cache_root)
        self.download_timeout_seconds = download_timeout_seconds
        self.download_retries = download_retries
        self.runtime_cache_root.mkdir(parents=True, exist_ok=True)

    def resolve_dataset_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else self.dataset_root / path

    def resolve_local_or_remote(
        self,
        raw_path: str | None,
        raw_url: str | None,
        *,
        kind: str,
        cache_key: str,
        default_suffix: str,
        force_refresh: bool = False,
    ) -> Path:
        if raw_path:
            candidate = self.resolve_dataset_path(raw_path)
            if candidate.exists():
                return candidate
        if not raw_url:
            missing = raw_path or "<none>"
            raise FileNotFoundError(f"Missing local asset and no fallback url for {kind}: {missing}")

        suffix = _suffix_hint(raw_path or raw_url, default_suffix)
        digest = _sha1_text(f"{kind}|{cache_key}|{raw_url}")
        dest = self.runtime_cache_root / "remote" / kind / digest[:2] / f"{digest}{suffix}"
        if dest.exists() and not force_refresh:
            return dest

        lock_path = dest.with_suffix(dest.suffix + ".lock")
        with _file_lock(lock_path):
            if dest.exists() and not force_refresh:
                return dest
            if force_refresh:
                try:
                    dest.unlink()
                except FileNotFoundError:
                    pass
            self._download_url(raw_url, dest)
        return dest

    def derived_path(self, *, kind: str, cache_key: str, suffix: str) -> Path:
        digest = _sha1_text(f"{kind}|{cache_key}")
        return self.runtime_cache_root / "derived" / kind / digest[:2] / f"{digest}{suffix}"

    def _download_url(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, self.download_retries + 1):
            with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            try:
                with urllib.request.urlopen(url, timeout=self.download_timeout_seconds) as response, tmp_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                os.replace(tmp_path, dest)
                _validate_cached_asset(dest)
                return
            except Exception as exc:  # pragma: no cover - exercised in runner
                last_error = exc
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    dest.unlink()
                except FileNotFoundError:
                    pass
                if attempt >= self.download_retries:
                    break
                time.sleep(min(30.0, 2.0 * attempt))
        raise RuntimeError(f"Failed to download {url} after {self.download_retries} attempts: {last_error}")


class _SeedVRHDRDatasetBase(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        train_height: int,
        train_width: int,
        random_crop: bool,
        seed: int,
        target_representation: str,
        runtime_cache_root: str | Path,
        download_timeout_seconds: int,
        download_retries: int,
        cache_rendered_sdr_inputs: bool,
        cache_compressed_targets: bool,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples = load_manifest(manifest_path)
        self.train_height = train_height
        self.train_width = train_width
        self.random_crop = random_crop
        self.seed = seed
        self.target_representation = target_representation
        self.cache_rendered_sdr_inputs = cache_rendered_sdr_inputs
        self.cache_compressed_targets = cache_compressed_targets
        self.asset_cache = _RuntimeAssetCache(
            dataset_root=self.dataset_root,
            runtime_cache_root=runtime_cache_root,
            download_timeout_seconds=download_timeout_seconds,
            download_retries=download_retries,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _optional_field_pair(
        self,
        sample: ManifestSample,
        field_name: str,
    ) -> tuple[str | None, str | None]:
        return (
            getattr(sample, f"{field_name}_path", None),
            getattr(sample, f"{field_name}_url", None),
        )

    def _pair_list(
        self,
        sample: ManifestSample,
        field_name: str,
    ) -> list[tuple[str | None, str | None]]:
        plural_paths = getattr(sample, f"{field_name}_paths", None) or []
        plural_urls = getattr(sample, f"{field_name}_urls", None) or []
        singular_path = getattr(sample, f"{field_name}_path", None)
        singular_url = getattr(sample, f"{field_name}_url", None)
        if plural_paths or plural_urls:
            length = max(len(plural_paths), len(plural_urls))
            return [
                (
                    plural_paths[index] if index < len(plural_paths) else None,
                    plural_urls[index] if index < len(plural_urls) else None,
                )
                for index in range(length)
            ]
        if singular_path or singular_url:
            return [(singular_path, singular_url)]
        return []

    def _cache_identity(
        self,
        sample: ManifestSample,
        *,
        purpose: str,
        frame_index: int | None = None,
        extra: str = "",
    ) -> str:
        return "|".join(
            [
                sample.sample_id,
                sample.variant_id or "",
                purpose,
                "" if frame_index is None else str(frame_index),
                self.target_representation,
                extra,
                sample.camera_response or "",
                "" if sample.exposure_ev is None else f"{sample.exposure_ev:.8f}",
                "" if sample.gamma is None else f"{sample.gamma:.8f}",
                "" if sample.quantization_bits is None else str(sample.quantization_bits),
                "" if sample.jpeg_quality is None else str(sample.jpeg_quality),
            ]
        )

    def _resolve_pair(
        self,
        sample: ManifestSample,
        pair: tuple[str | None, str | None],
        *,
        kind: str,
        cache_key: str,
        default_suffix: str,
        force_refresh: bool = False,
    ) -> Path:
        raw_path, raw_url = pair
        return self.asset_cache.resolve_local_or_remote(
            raw_path,
            raw_url,
            kind=kind,
            cache_key=cache_key,
            default_suffix=default_suffix,
            force_refresh=force_refresh,
        )

    def _hdr_source_candidates(
        self,
        sample: ManifestSample,
        *,
        frame_index: int | None = None,
    ) -> list[tuple[str | None, str | None]]:
        candidates = []
        if frame_index is None:
            candidates.extend(
                [
                    self._optional_field_pair(sample, "source_hdr"),
                    self._optional_field_pair(sample, "target_hdr"),
                ]
            )
        else:
            for field_name in ("source_hdr", "target_hdr"):
                pairs = self._pair_list(sample, field_name)
                if frame_index < len(pairs):
                    candidates.append(pairs[frame_index])
        return candidates

    def _resolve_hdr_source_path(
        self,
        sample: ManifestSample,
        *,
        frame_index: int | None = None,
        force_refresh: bool = False,
    ) -> Path:
        candidates = self._hdr_source_candidates(sample, frame_index=frame_index)
        for raw_path, raw_url in candidates:
            if raw_path or raw_url:
                return self._resolve_pair(
                    sample,
                    (raw_path, raw_url),
                    kind="hdr_source",
                    cache_key=self._cache_identity(
                        sample,
                        purpose="hdr_source",
                        frame_index=frame_index,
                        extra=f"{raw_path or ''}|{raw_url or ''}",
                    ),
                    default_suffix=".exr",
                    force_refresh=force_refresh,
                )
        raise ValueError(f"Sample {sample.sample_id} does not provide an HDR source path/url")

    def _load_hdr_source(
        self,
        sample: ManifestSample,
        *,
        frame_index: int | None = None,
    ) -> tuple[np.ndarray, Path]:
        candidates = self._hdr_source_candidates(sample, frame_index=frame_index)
        last_error: Exception | None = None
        for raw_path, raw_url in candidates:
            if not raw_path and not raw_url:
                continue
            source_path = self._resolve_pair(
                sample,
                (raw_path, raw_url),
                kind="hdr_source",
                cache_key=self._cache_identity(
                    sample,
                    purpose="hdr_source",
                    frame_index=frame_index,
                    extra=f"{raw_path or ''}|{raw_url or ''}",
                ),
                default_suffix=".exr",
            )
            try:
                return _read_hdr_image(source_path), source_path
            except Exception as exc:
                last_error = exc
                if not raw_url:
                    continue
                refreshed_path = self._resolve_pair(
                    sample,
                    (raw_path, raw_url),
                    kind="hdr_source",
                    cache_key=self._cache_identity(
                        sample,
                        purpose="hdr_source",
                        frame_index=frame_index,
                        extra=f"{raw_path or ''}|{raw_url or ''}",
                    ),
                    default_suffix=".exr",
                    force_refresh=True,
                )
                try:
                    return _read_hdr_image(refreshed_path), refreshed_path
                except Exception as refresh_exc:
                    last_error = refresh_exc
                    continue
        if last_error is not None:
            raise FileNotFoundError(
                f"Failed to load HDR source for {sample.sample_id} "
                f"(frame_index={frame_index}): {last_error}"
            ) from last_error
        raise ValueError(f"Sample {sample.sample_id} does not provide an HDR source path/url")

    def _load_cached_or_rendered_input(
        self,
        sample: ManifestSample,
        *,
        hdr: np.ndarray,
        hdr_source_path: Path,
        frame_index: int | None = None,
    ) -> np.ndarray:
        if sample.camera_response is None or sample.exposure_ev is None:
            raise ValueError(
                f"Sample {sample.sample_id} is missing camera_response/exposure_ev required for SDR synthesis"
            )
        gamma = sample.gamma if sample.gamma is not None else 2.2
        quantization_bits = sample.quantization_bits if sample.quantization_bits is not None else 8
        cache_key = self._cache_identity(
            sample,
            purpose="rendered_input",
            frame_index=frame_index,
            extra=str(hdr_source_path),
        )
        if self.cache_rendered_sdr_inputs:
            cached_path = self.asset_cache.derived_path(
                kind="rendered_input",
                cache_key=cache_key,
                suffix=".png",
            )
            if cached_path.exists():
                return _read_rgb_png(cached_path)
            lock_path = cached_path.with_suffix(".png.lock")
            with _file_lock(lock_path):
                if cached_path.exists():
                    return _read_rgb_png(cached_path)
                rendered = _render_sdr_from_hdr(
                    hdr,
                    camera_response=sample.camera_response,
                    exposure_ev=float(sample.exposure_ev),
                    gamma=float(gamma),
                    quantization_bits=int(quantization_bits),
                    jpeg_quality=sample.jpeg_quality,
                )
                _atomic_write_png(cached_path, rendered)
                return rendered
        return _render_sdr_from_hdr(
            hdr,
            camera_response=sample.camera_response,
            exposure_ev=float(sample.exposure_ev),
            gamma=float(gamma),
            quantization_bits=int(quantization_bits),
            jpeg_quality=sample.jpeg_quality,
        )

    def _load_cached_or_compressed_target(
        self,
        sample: ManifestSample,
        *,
        hdr: np.ndarray,
        hdr_source_path: Path,
        frame_index: int | None = None,
    ) -> np.ndarray:
        cache_key = self._cache_identity(
            sample,
            purpose="compressed_target",
            frame_index=frame_index,
            extra=str(hdr_source_path),
        )
        if self.cache_compressed_targets and self.target_representation != "raw_hdr":
            cached_path = self.asset_cache.derived_path(
                kind=f"target_{self.target_representation}",
                cache_key=cache_key,
                suffix=".npy",
            )
            if cached_path.exists():
                return _read_target_array(cached_path)
            lock_path = cached_path.with_suffix(".npy.lock")
            with _file_lock(lock_path):
                if cached_path.exists():
                    return _read_target_array(cached_path)
                compressed = _compress_target_from_hdr(hdr, self.target_representation)
                _atomic_write_npy(cached_path, compressed)
                return compressed
        return _compress_target_from_hdr(hdr, self.target_representation)


class SeedVRHDRImageDataset(_SeedVRHDRDatasetBase):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        train_height: int,
        train_width: int,
        random_crop: bool,
        seed: int,
        target_representation: str,
        runtime_cache_root: str | Path,
        download_timeout_seconds: int,
        download_retries: int,
        cache_rendered_sdr_inputs: bool,
        cache_compressed_targets: bool,
    ) -> None:
        super().__init__(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            train_height=train_height,
            train_width=train_width,
            random_crop=random_crop,
            seed=seed,
            target_representation=target_representation,
            runtime_cache_root=runtime_cache_root,
            download_timeout_seconds=download_timeout_seconds,
            download_retries=download_retries,
            cache_rendered_sdr_inputs=cache_rendered_sdr_inputs,
            cache_compressed_targets=cache_compressed_targets,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rng = random.Random((self.seed * 10_000) + index)

        direct_input_pair = self._optional_field_pair(sample, "input_sdr")
        direct_target_path = self._resolve_direct_target_path(sample)

        hdr = None
        hdr_source_path = None
        if direct_target_path is None or not any(direct_input_pair):
            hdr, hdr_source_path = self._load_hdr_source(sample)

        if any(direct_input_pair):
            input_path = self._resolve_pair(
                sample,
                direct_input_pair,
                kind="input_sdr",
                cache_key=self._cache_identity(
                    sample,
                    purpose="input_sdr",
                    extra="|".join(value or "" for value in direct_input_pair),
                ),
                default_suffix=".png",
            )
            input_image = _read_rgb_png(input_path)
        else:
            assert hdr is not None and hdr_source_path is not None
            input_image = self._load_cached_or_rendered_input(
                sample,
                hdr=hdr,
                hdr_source_path=hdr_source_path,
            )

        if direct_target_path is not None:
            target_image = _read_target_for_representation(direct_target_path, self.target_representation)
        else:
            assert hdr is not None and hdr_source_path is not None
            target_image = self._load_cached_or_compressed_target(
                sample,
                hdr=hdr,
                hdr_source_path=hdr_source_path,
            )

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

    def _resolve_direct_target_path(self, sample: ManifestSample) -> Path | None:
        candidates: list[tuple[str | None, str | None]] = []
        if self.target_representation == "raw_hdr":
            candidates.extend(
                [
                    self._optional_field_pair(sample, "target_hdr_npy"),
                    self._optional_field_pair(sample, "target_hdr"),
                ]
            )
        elif self.target_representation == "mu_law_mu5000":
            candidates.extend(
                [
                    self._optional_field_pair(sample, "target_mu_law"),
                    self._optional_field_pair(sample, "compressed_target"),
                ]
            )
        elif self.target_representation == "log_hdr":
            candidates.extend(
                [
                    self._optional_field_pair(sample, "target_log_hdr"),
                    self._optional_field_pair(sample, "compressed_target"),
                ]
            )
        else:
            raise ValueError(f"Unsupported target_representation: {self.target_representation}")

        for raw_path, raw_url in candidates:
            if raw_path or raw_url:
                return self._resolve_pair(
                    sample,
                    (raw_path, raw_url),
                    kind=f"target_{self.target_representation}",
                    cache_key=self._cache_identity(
                        sample,
                        purpose=f"target_{self.target_representation}",
                        extra=f"{raw_path or ''}|{raw_url or ''}",
                    ),
                    default_suffix=".npy" if (raw_path or raw_url or "").endswith(".npy") else ".exr",
                )
        return None


class SeedVRHDRVideoDataset(_SeedVRHDRDatasetBase):
    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        train_height: int,
        train_width: int,
        random_crop: bool,
        seed: int,
        target_representation: str,
        runtime_cache_root: str | Path,
        download_timeout_seconds: int,
        download_retries: int,
        cache_rendered_sdr_inputs: bool,
        cache_compressed_targets: bool,
    ) -> None:
        super().__init__(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            train_height=train_height,
            train_width=train_width,
            random_crop=random_crop,
            seed=seed,
            target_representation=target_representation,
            runtime_cache_root=runtime_cache_root,
            download_timeout_seconds=download_timeout_seconds,
            download_retries=download_retries,
            cache_rendered_sdr_inputs=cache_rendered_sdr_inputs,
            cache_compressed_targets=cache_compressed_targets,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rng = random.Random((self.seed * 10_000) + index)

        input_images: list[np.ndarray] = []
        target_images: list[np.ndarray] = []

        input_pairs = self._pair_list(sample, "input_sdr")
        target_pairs = self._direct_target_pairs(sample)
        frame_count = max(
            len(input_pairs),
            len(target_pairs),
            len(self._pair_list(sample, "source_hdr")),
            len(self._pair_list(sample, "target_hdr")),
        )
        if frame_count == 0:
            raise ValueError(f"Empty clip for {sample.sample_id}")

        if not input_pairs:
            input_pairs = [(None, None)] * frame_count
        if not target_pairs:
            target_pairs = [(None, None)] * frame_count

        for frame_index in range(frame_count):
            hdr = None
            hdr_source_path = None

            if any(input_pairs[frame_index]):
                input_path = self._resolve_pair(
                    sample,
                    input_pairs[frame_index],
                    kind="input_sdr",
                    cache_key=self._cache_identity(
                        sample,
                        purpose="input_sdr",
                        frame_index=frame_index,
                        extra="|".join(value or "" for value in input_pairs[frame_index]),
                    ),
                    default_suffix=".png",
                )
                input_image = _read_rgb_png(input_path)
            else:
                hdr, hdr_source_path = self._load_hdr_source(sample, frame_index=frame_index)
                input_image = self._load_cached_or_rendered_input(
                    sample,
                    hdr=hdr,
                    hdr_source_path=hdr_source_path,
                    frame_index=frame_index,
                )

            if any(target_pairs[frame_index]):
                target_path = self._resolve_pair(
                    sample,
                    target_pairs[frame_index],
                    kind=f"target_{self.target_representation}",
                    cache_key=self._cache_identity(
                        sample,
                        purpose=f"target_{self.target_representation}",
                        frame_index=frame_index,
                        extra="|".join(value or "" for value in target_pairs[frame_index]),
                    ),
                    default_suffix=".npy",
                )
                target_image = _read_target_for_representation(target_path, self.target_representation)
            else:
                if hdr is None or hdr_source_path is None:
                    hdr, hdr_source_path = self._load_hdr_source(sample, frame_index=frame_index)
                target_image = self._load_cached_or_compressed_target(
                    sample,
                    hdr=hdr,
                    hdr_source_path=hdr_source_path,
                    frame_index=frame_index,
                )

            if input_image.shape[:2] != target_image.shape[:2]:
                raise ValueError(
                    f"Frame {frame_index} mismatch for {sample.sample_id}: "
                    f"{input_image.shape[:2]} vs {target_image.shape[:2]}"
                )
            input_images.append(input_image)
            target_images.append(target_image)

        reference_shape = input_images[0].shape[:2]
        for frame_index, input_image in enumerate(input_images):
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

    def _direct_target_pairs(self, sample: ManifestSample) -> list[tuple[str | None, str | None]]:
        candidates: list[list[tuple[str | None, str | None]]] = []
        if self.target_representation == "raw_hdr":
            candidates.extend(
                [
                    self._pair_list(sample, "target_hdr_npy"),
                    self._pair_list(sample, "target_hdr"),
                ]
            )
        elif self.target_representation == "mu_law_mu5000":
            candidates.extend(
                [
                    self._pair_list(sample, "target_mu_law"),
                    self._pair_list(sample, "compressed_target"),
                ]
            )
        elif self.target_representation == "log_hdr":
            candidates.extend(
                [
                    self._pair_list(sample, "target_log_hdr"),
                    self._pair_list(sample, "compressed_target"),
                ]
            )
        else:
            raise ValueError(f"Unsupported target_representation: {self.target_representation}")

        for pairs in candidates:
            if pairs:
                return pairs
        return []
