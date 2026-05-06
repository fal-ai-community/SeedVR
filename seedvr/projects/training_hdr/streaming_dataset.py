"""HTTP-streaming HDR video dataset for SeedVR HDR training.

Activated by the trainer config field `streaming_dataset_url`. When set
(e.g., `hf://lpj990/hdr-video-dataset`), the dataloader bypasses the
filesystem-backed video dataset and streams paired HDR/SDR mp4 clips
directly from the HuggingFace CDN, decodes them in-memory in DataLoader
workers, and yields tensors in the same shape as `SeedVRHDRVideoDataset`.

Key properties:
- ZERO bulk pre-download. Tar parts streamed on demand. Disk usage ~10 MB
  (just metadata.csv cache).
- HDR-correct: PyAV decodes HDR clips to gbrp16le (16-bit planar), applies BT.2100
  PQ inverse EOTF, robust-normalizes per clip, then encodes to the
  configured target representation (LogC3 / mu_law / etc.). The SDR side
  decodes to rgb24 (8-bit, BT.709, the original encoding) for the input
  pair.
- Worker-friendly: PyTorch IterableDataset shards tar parts across workers
  so multiple workers don't redundantly download the same bytes. Decoding
  happens in worker processes, off the GPU critical path.

Format note: returns dicts compatible with `SeedVRHDRVideoDataset`:
  input_sdr: torch.Tensor (T, 3, H, W) float32 in [0, 1]
  target:    torch.Tensor (T, 3, H, W) float32 in encoded space
  scene_id, sample_id, variant_id: str
"""
from __future__ import annotations

import csv
import io
import json
import os
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Reuse the same target-encoding helpers the rest of the project uses, so
# `streaming_dataset` and the disk-backed dataset stay numerically identical.
from .dataset import (
    _fal_logc3_encode,
    _fal_pq_oetf,
    MU_LAW_MU,
    LOG_HDR_EPS,
)


HF_CDN_REPO = "lpj990/hdr-video-dataset"
HDR_PARTS = [
    f"Processed_Scenes/HDR/Processed_Scenes_HDR.tar.part-{i:02d}" for i in range(5)
]
SDR_PARTS = [
    f"Processed_Scenes/SDR/Processed_Scenes_SDR.tar.part-{i:02d}" for i in range(9)
]


# ---------------------------------------------------------------------
# BT.2100 PQ inverse EOTF — converts PQ-encoded values in [0,1] to
# linear, scaled so 1.0 = 100 nits (matches `npl=100` zscale convention).
# ---------------------------------------------------------------------
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0


def _pq_to_linear_npl100(pq: np.ndarray) -> np.ndarray:
    pq = np.clip(pq.astype(np.float32), 0.0, 1.0)
    p = np.power(pq, 1.0 / _PQ_M2)
    num = np.maximum(p - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * p
    e = np.power(num / np.maximum(den, 1e-10), 1.0 / _PQ_M1)
    return (e * 100.0).astype(np.float32)


def _robust_normalize_hdr(hdr: np.ndarray) -> np.ndarray:
    """99.5-percentile normalize an HDR clip to [0, ~1]; clip caller."""
    hdr = np.maximum(hdr.astype(np.float32), 0.0)
    pct = float(np.percentile(hdr, 99.5))
    scale = pct if pct > 1e-6 else max(float(hdr.max()), 1.0)
    return hdr / scale


def _resize_clip(arr: np.ndarray, target_h: int) -> np.ndarray:
    """Resize a (T, H, W, 3) clip preserving aspect ratio. Uses cv2."""
    if arr.shape[1] <= target_h:
        return arr
    import cv2

    src_h, src_w = arr.shape[1:3]
    target_w = int(round(src_w * (target_h / src_h)))
    target_w -= target_w % 2
    out = np.empty(
        (arr.shape[0], target_h, target_w, arr.shape[3]), dtype=arr.dtype
    )
    interp = cv2.INTER_AREA
    for t in range(arr.shape[0]):
        out[t] = cv2.resize(arr[t], (target_w, target_h), interpolation=interp)
    return out


def _subsample_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if num_frames <= max_frames:
        return np.arange(num_frames)
    return np.linspace(0, num_frames - 1, max_frames).round().astype(int)


def _decode_video(blob: bytes, *, pix_fmt: str) -> np.ndarray:
    """PyAV-decode an mp4 blob to a stacked frame array."""
    import av

    bio = io.BytesIO(blob)
    frames: list[np.ndarray] = []
    with av.open(bio) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format=pix_fmt))
    if not frames:
        return np.empty((0, 0, 0, 0), dtype=np.uint16 if "16" in pix_fmt else np.uint8)
    return np.stack(frames, axis=0)


def _hf_url(filename: str, repo: str = HF_CDN_REPO) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id=repo, repo_type="dataset", filename=filename)


def _load_captions_for_repo(repo: str = HF_CDN_REPO) -> dict[str, str]:
    """Download metadata.csv (~10 MB) once; HF caches it locally afterwards."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, repo_type="dataset", filename="metadata.csv")
    out: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            video = (row.get("video") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if video:
                out[Path(video).name] = prompt
    return out


# ---------------------------------------------------------------------
# Concatenated HTTP stream over multiple HF tar parts
# ---------------------------------------------------------------------
class _ConcatHTTPStream(io.RawIOBase):
    """Concatenate HF tar parts into one read-only stream with byte-range
    resumption on transient CloudFront drops.

    Tracks Content-Length and bytes-read per part. If `urlopen.read()`
    returns short while the current part has more bytes left, the stream
    automatically reconnects using `Range: bytes=<offset>-` and continues
    from the last byte. Only advances to the next part when the current
    one's entire Content-Length has been consumed.
    """

    def __init__(
        self,
        filenames: list[str],
        repo: str,
        timeout: int = 120,
        max_retries: int = 5,
    ):
        self._urls = [_hf_url(n, repo=repo) for n in filenames]
        self._timeout = timeout
        self._max_retries = max_retries
        self._idx = -1                 # current part index (-1 = before first)
        self._cur: io.IOBase | None = None
        self._cur_len: int | None = None  # Content-Length of current part
        self._cur_pos = 0              # bytes already read from current part
        self._advance_part()

    def _advance_part(self) -> None:
        import sys, time
        if self._cur is not None:
            try:
                self._cur.close()
            except Exception:
                pass
            self._cur = None
        self._cur_len = None
        self._cur_pos = 0
        self._idx += 1
        while self._idx < len(self._urls):
            url = self._urls[self._idx]
            try:
                t0 = time.monotonic()
                resp = urllib.request.urlopen(url, timeout=self._timeout)
                self._cur = resp
                length = resp.headers.get("Content-Length")
                self._cur_len = int(length) if length else None
                print(
                    f"[stream-http] opened part_idx={self._idx} "
                    f"size={self._cur_len} in {time.monotonic()-t0:.2f}s",
                    file=sys.stderr, flush=True,
                )
                return
            except urllib.error.URLError as exc:
                print(
                    f"[stream-http] fail open part_idx={self._idx}: {exc}",
                    file=sys.stderr, flush=True,
                )
                self._idx += 1
        self._cur = None

    def _resume_part(self) -> None:
        """Reconnect to the current part starting at self._cur_pos via Range."""
        if self._idx < 0 or self._idx >= len(self._urls):
            self._cur = None
            return
        url = self._urls[self._idx]
        for attempt in range(self._max_retries):
            try:
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={self._cur_pos}-"}
                )
                resp = urllib.request.urlopen(req, timeout=self._timeout)
                if self._cur is not None:
                    try:
                        self._cur.close()
                    except Exception:
                        pass
                self._cur = resp
                # Don't trust new Content-Length on a Range response; keep _cur_len.
                return
            except urllib.error.URLError as exc:
                print(
                    f"  [stream] resume attempt {attempt+1} part {self._idx} "
                    f"@offset={self._cur_pos} failed: {exc}",
                    flush=True,
                )
        # All retries failed; abandon this part
        self._cur = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        out = bytearray()
        while size < 0 or len(out) < size:
            if self._cur is None:
                break
            need = -1 if size < 0 else (size - len(out))
            try:
                chunk = self._cur.read(need if need > 0 else -1)
            except Exception as exc:
                print(
                    f"  [stream] read() raised on part {self._idx} "
                    f"@offset={self._cur_pos}: {exc}; resuming",
                    flush=True,
                )
                chunk = b""
            if chunk:
                out.extend(chunk)
                self._cur_pos += len(chunk)
                continue
            # Empty chunk: end-of-response. Decide if part is fully drained.
            if self._cur_len is not None and self._cur_pos < self._cur_len:
                # Truncated mid-part — resume via Range.
                self._resume_part()
                if self._cur is None:
                    break
            else:
                self._advance_part()
        return bytes(out)

    def close(self) -> None:
        if self._cur is not None:
            try:
                self._cur.close()
            except Exception:
                pass
            self._cur = None
        super().close()


# ---------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------
def _build_target_representation(linear_norm: np.ndarray, repr_name: str) -> np.ndarray:
    """linear_norm: (..., 3) in [0,1]. Returns encoded float32 (...,3)."""
    linear_norm = np.maximum(linear_norm.astype(np.float32), 0.0)
    if repr_name == "raw_hdr":
        return linear_norm
    if repr_name == "mu_law_mu5000":
        return (np.log1p(MU_LAW_MU * linear_norm) / np.log1p(MU_LAW_MU)).astype(np.float32)
    if repr_name == "log_hdr":
        return np.log(linear_norm + LOG_HDR_EPS).astype(np.float32)
    if repr_name == "logc3":
        return _fal_logc3_encode(linear_norm)
    if repr_name == "pq_1000":
        clamped = np.clip(linear_norm, 0.0, 1.0)
        encoded = _fal_pq_oetf(clamped * 1000.0)
        return np.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"Unsupported target_representation: {repr_name}")


@dataclass
class _StreamingConfig:
    target_representation: str
    train_height: int
    train_width: int
    frames_per_clip: int
    repo: str = HF_CDN_REPO
    hdr_parts: tuple[str, ...] = tuple(HDR_PARTS)
    sdr_parts: tuple[str, ...] = tuple(SDR_PARTS)
    max_clips_per_worker: int | None = None


class StreamingHDRVideoDataset(IterableDataset):
    """Streams paired HDR/SDR mp4 clips from a HuggingFace dataset.

    Output per __iter__ step (compatible with `SeedVRHDRVideoDataset`):
        {
            "input_sdr": torch.float32, shape (T, 3, H, W), in [0, 1]
            "target":    torch.float32, shape (T, 3, H, W), encoded
            "scene_id":  str  (the clip basename, e.g. "p00042")
            "sample_id": str
            "variant_id": ""
        }
    """

    def __init__(
        self,
        *,
        target_representation: str,
        train_height: int,
        train_width: int,
        frames_per_clip: int = 25,
        streaming_dataset_url: str = "hf://lpj990/hdr-video-dataset",
        max_clips_per_worker: int | None = None,
    ):
        super().__init__()
        if not streaming_dataset_url.startswith("hf://"):
            raise ValueError(
                f"streaming_dataset_url must start with 'hf://'; got {streaming_dataset_url!r}"
            )
        repo = streaming_dataset_url[len("hf://") :]
        self._cfg = _StreamingConfig(
            target_representation=target_representation,
            train_height=train_height,
            train_width=train_width,
            frames_per_clip=frames_per_clip,
            repo=repo,
            max_clips_per_worker=max_clips_per_worker,
        )
        # Cached caption table — downloaded in main process; pickled to workers.
        self._captions = _load_captions_for_repo(repo)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        import sys
        import time

        def _log(msg: str) -> None:
            # Force-flush so DataLoader worker stdout reaches the runner log.
            print(f"[stream {wid}] {msg}", file=sys.stderr, flush=True)
            sys.stderr.flush()

        try:
            worker_info = torch.utils.data.get_worker_info()
        except Exception:
            worker_info = None
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        wid = f"w{worker_id}/{num_workers}"
        _log(
            f"__iter__ entered: target_repr={self._cfg.target_representation} "
            f"target={self._cfg.train_width}x{self._cfg.train_height} "
            f"frames={self._cfg.frames_per_clip} repo={self._cfg.repo}"
        )

        # Shard HDR parts across workers
        hdr_parts = list(self._cfg.hdr_parts)
        n_parts = len(hdr_parts)
        per_worker = max(1, (n_parts + num_workers - 1) // num_workers)
        start = worker_id * per_worker
        end = min(n_parts, start + per_worker)
        if start >= n_parts:
            return
        my_hdr_parts = hdr_parts[start:end]
        _log(f"shard: hdr_parts[{start}:{end}] = {my_hdr_parts}")

        # Infinite iteration: when one full pass through the tar parts ends,
        # reopen the streams and start over. Training step counts (config.steps)
        # bound total work, not the iterator.
        total_yielded = 0
        epoch = 0
        last_log_ts = time.monotonic()
        # Per-reason skip counters; keep cumulative across epochs.
        skip_counts: dict[str, int] = {
            "no_sdr_match": 0,
            "decode_exception": 0,
            "empty_decode": 0,
            "extract_error": 0,
            "sdr_tar_error": 0,
        }
        first_skip_logged: set[str] = set()

        def _log_skip(reason: str, detail: str) -> None:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            if reason not in first_skip_logged:
                first_skip_logged.add(reason)
                _log(f"first skip[{reason}]: {detail}")

        while True:
            epoch += 1
            _log(f"epoch={epoch} opening streams")
            t0 = time.monotonic()
            hdr_stream = _ConcatHTTPStream(my_hdr_parts, repo=self._cfg.repo)
            sdr_stream = _ConcatHTTPStream(
                list(self._cfg.sdr_parts), repo=self._cfg.repo
            )
            _log(f"epoch={epoch} streams opened in {time.monotonic()-t0:.2f}s")
            hdr_tf = None
            sdr_tf = None
            sdr_pending: dict[str, bytes] = {}
            try:
                t0 = time.monotonic()
                hdr_tf = tarfile.open(fileobj=hdr_stream, mode="r|")
                sdr_tf = tarfile.open(fileobj=sdr_stream, mode="r|")
                _log(f"epoch={epoch} tar headers parsed in {time.monotonic()-t0:.2f}s")
                sdr_iter = iter(sdr_tf)

                def _advance_sdr_until(target_name: str) -> bytes | None:
                    if target_name in sdr_pending:
                        return sdr_pending.pop(target_name)
                    for sm in sdr_iter:
                        if not sm.isfile() or not sm.name.lower().endswith(".mp4"):
                            continue
                        f = sdr_tf.extractfile(sm)
                        if f is None:
                            continue
                        blob = f.read()
                        name = Path(sm.name).name
                        if name == target_name:
                            return blob
                        sdr_pending[name] = blob
                        if len(sdr_pending) > 64:
                            sdr_pending.pop(next(iter(sdr_pending)))
                    return None

                tar_iter = iter(hdr_tf)
                while True:
                    try:
                        hm = next(tar_iter)
                    except StopIteration:
                        break
                    except (tarfile.ReadError, OSError) as exc:
                        _log(
                            f"hdr tar read error after {total_yielded} "
                            f"clips total: {exc!r}; ending epoch {epoch}"
                        )
                        break
                    if not hm.isfile() or not hm.name.lower().endswith(".mp4"):
                        continue
                    try:
                        f = hdr_tf.extractfile(hm)
                        if f is None:
                            continue
                        hdr_blob = f.read()
                    except Exception as exc:
                        _log_skip("extract_error", f"{hm.name}: {exc!r}")
                        continue
                    hdr_name = Path(hm.name).name
                    try:
                        sdr_blob = _advance_sdr_until(hdr_name)
                    except (tarfile.ReadError, OSError) as exc:
                        _log_skip(
                            "sdr_tar_error", f"{hdr_name}: {exc!r}"
                        )
                        continue
                    if sdr_blob is None:
                        _log_skip(
                            "no_sdr_match",
                            f"{hdr_name} (sdr_pending size={len(sdr_pending)})",
                        )
                        continue
                    try:
                        sample, decode_fail = self._decode_one(
                            hdr_name, hdr_blob, sdr_blob
                        )
                    except Exception as exc:
                        _log_skip(
                            "decode_exception",
                            f"{hdr_name}: outer {exc!r}",
                        )
                        continue
                    if sample is None:
                        bucket = (decode_fail or "decode_exception").split(":", 1)[0]
                        _log_skip(bucket, f"{hdr_name}: {decode_fail}")
                        continue
                    now = time.monotonic()
                    if total_yielded < 5 or (now - last_log_ts) > 30:
                        _log(
                            f"yield #{total_yielded+1} clip={sample['scene_id']} "
                            f"epoch={epoch} dt_since_last={now-last_log_ts:.1f}s "
                            f"skips={dict(skip_counts)} "
                            f"target_shape={tuple(sample['target'].shape)}"
                        )
                        last_log_ts = now
                    yield sample
                    total_yielded += 1
                    if (
                        self._cfg.max_clips_per_worker
                        and total_yielded >= self._cfg.max_clips_per_worker
                    ):
                        return
            finally:
                _log(
                    f"epoch={epoch} ended: yielded_total={total_yielded} "
                    f"skips={dict(skip_counts)} "
                    f"sdr_pending_size={len(sdr_pending)}"
                )
                if hdr_tf is not None:
                    try:
                        hdr_tf.close()
                    except Exception:
                        pass
                if sdr_tf is not None:
                    try:
                        sdr_tf.close()
                    except Exception:
                        pass
                hdr_stream.close()
                sdr_stream.close()

    def _decode_one(
        self, hdr_name: str, hdr_blob: bytes, sdr_blob: bytes
    ) -> tuple[dict[str, torch.Tensor | str] | None, str | None]:
        """Returns (sample, fail_reason). On success, fail_reason is None."""
        try:
            # NOTE: PyAV's "rgb48le" / "rgb48be" produce mangled output for
            # 10-bit HEVC (yuv420p10le) HDR sources — channels are not in the
            # documented R-G-B order and the decode is silently corrupt.
            # gbrp16le (planar G-B-R 16-bit) round-trips correctly through
            # libswscale and PyAV's to_ndarray remaps it to (H, W, 3) with
            # values matching an 8-bit rgb24 decode of the same frame.
            hdr_u16 = _decode_video(hdr_blob, pix_fmt="gbrp16le")
        except Exception as exc:
            return None, f"hdr_decode_exception: {type(exc).__name__}: {exc}"
        try:
            sdr_u8 = _decode_video(sdr_blob, pix_fmt="rgb24")
        except Exception as exc:
            return None, f"sdr_decode_exception: {type(exc).__name__}: {exc}"
        if hdr_u16.shape[0] == 0 or sdr_u8.shape[0] == 0:
            return (
                None,
                f"empty_decode: hdr_T={hdr_u16.shape[0]} sdr_T={sdr_u8.shape[0]} "
                f"hdr_blob={len(hdr_blob)} sdr_blob={len(sdr_blob)}",
            )
        T = min(hdr_u16.shape[0], sdr_u8.shape[0])
        hdr_u16 = hdr_u16[:T]
        sdr_u8 = sdr_u8[:T]

        # PQ -> linear -> normalize
        pq = hdr_u16.astype(np.float32) / 65535.0
        linear = _pq_to_linear_npl100(pq)
        normalized = np.clip(_robust_normalize_hdr(linear), 0.0, 1.0)

        # Resize to train resolution (maintain aspect via height target)
        target_h = self._cfg.train_height
        if target_h and target_h < normalized.shape[1]:
            normalized = _resize_clip(normalized, target_h)
            sdr_u8 = _resize_clip(sdr_u8, target_h)

        # Subsample frames
        idx = _subsample_indices(normalized.shape[0], self._cfg.frames_per_clip)
        normalized = normalized[idx]
        sdr_u8 = sdr_u8[idx]

        # Crop / pad to exact (train_height, train_width). Center crop on width.
        Tn, Hn, Wn, _ = normalized.shape
        target_w = self._cfg.train_width
        if Wn > target_w:
            x0 = (Wn - target_w) // 2
            normalized = normalized[:, :, x0 : x0 + target_w, :]
            sdr_u8 = sdr_u8[:, :, x0 : x0 + target_w, :]
        elif Wn < target_w:
            # pad with edge
            pad_left = (target_w - Wn) // 2
            pad_right = target_w - Wn - pad_left
            normalized = np.pad(
                normalized,
                ((0, 0), (0, 0), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )
            sdr_u8 = np.pad(
                sdr_u8,
                ((0, 0), (0, 0), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )

        # Encode target
        target = _build_target_representation(normalized, self._cfg.target_representation)

        # To tensors. Output (T, C, H, W).
        target_t = torch.from_numpy(target).permute(0, 3, 1, 2).contiguous().float()
        # Match disk-backed dataset.py:_to_target_tensor normalization: bounded
        # representations are mapped from [0, 1] → [-1, 1] before training.
        # Without this, the validator's inverse transform (clamp(-1,1)+1)*0.5
        # double-undoes the missing scale, producing wrong linear-HDR values
        # and color-cast previews.
        if self._cfg.target_representation in {"mu_law_mu5000", "pq_1000", "logc3"}:
            target_t = target_t.mul(2.0).sub(1.0)
        input_t = torch.from_numpy(sdr_u8).permute(0, 3, 1, 2).contiguous().float() / 255.0

        clip_id = hdr_name.replace(".mp4", "")
        sample = {
            "input_sdr": input_t,
            "target": target_t,
            "scene_id": clip_id,
            "sample_id": clip_id,
            "variant_id": "",
            "caption": self._captions.get(hdr_name, ""),
        }
        return sample, None


__all__ = ["StreamingHDRVideoDataset"]
