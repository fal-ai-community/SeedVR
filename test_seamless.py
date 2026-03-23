"""
Test seamless tiling support for SeedVR.

Usage:
    python test_seamless.py <input_image> [--target-area 2048x2048] [--pad 256] [--seed 42]

Outputs:
    - <name>_seamless.png     — seamless upscaled result
    - <name>_normal.png       — normal (non-seamless) upscaled result for comparison
    - <name>_tiled_seamless.png — 2x2 tiled seamless result (visual seam check)
    - <name>_tiled_normal.png  — 2x2 tiled normal result (visual seam check)
"""

import argparse
import math
import sys
import datetime

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image

from seedvr.pipeline import SeedVRPipeline
from seedvr.common.distributed import (
    init_torch,
    get_world_size,
    get_device,
)
from seedvr.common.distributed.advanced import init_sequence_parallel

# Match production noise scale decay
DEFAULT_COND_NOISE_SCALE = 0.10
NOISE_SCALE_EXPONENT = 1.10515


def get_noise_scale(noise_scale: float, target_area: int) -> float:
    """Match production noise scale decay."""
    media_area_megapixels = target_area / 1_000_000
    return noise_scale * (media_area_megapixels**-NOISE_SCALE_EXPONENT)


def load_image_as_media(path: str) -> torch.Tensor:
    """Load an image and convert to media tensor (C, F=1, H, W) in [-1, 1].
    Matches production: torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(1)
    """
    img = Image.open(path).convert("RGB")
    # Match production exactly: int tensor -> float32 -> normalize
    media = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(1)  # C, 1, H, W
    media = media.to(torch.float32) / 255.0
    media = media * 2.0 - 1.0
    return media


def media_to_image(media: torch.Tensor) -> Image.Image:
    """Convert pipeline output (C, F, H, W) uint8 tensor to PIL Image.
    Matches production: Image.fromarray(media[:, 0].permute(1, 2, 0).numpy())
    """
    return Image.fromarray(media[:, 0].permute(1, 2, 0).numpy())


def tile_2x2(img: Image.Image) -> Image.Image:
    """Create a 2x2 tiled version for visual seam inspection."""
    w, h = img.size
    tiled = Image.new("RGB", (w * 2, h * 2))
    tiled.paste(img, (0, 0))
    tiled.paste(img, (w, 0))
    tiled.paste(img, (0, h))
    tiled.paste(img, (w, h))
    return tiled


def main():
    parser = argparse.ArgumentParser(description="Test SeedVR seamless tiling")
    parser.add_argument("input", help="Path to input tileable texture image")
    parser.add_argument(
        "--target-area",
        default=None,
        help="Target area as WxH (default: 2x input area, matching production factor=2)",
    )
    parser.add_argument(
        "--upscale-factor",
        type=float,
        default=2.0,
        help="Upscale factor (default: 2.0, used when --target-area not specified)",
    )
    parser.add_argument(
        "--pad", type=int, default=256, help="Seamless padding in pixels (default: 256)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--model",
        default="fal/SeedVR2-7B-FlashPack",
        help="Model path or HF repo (default: fal/SeedVR2-7B-FlashPack)",
    )
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="Skip the non-seamless comparison run",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=DEFAULT_COND_NOISE_SCALE,
        help=f"Base noise scale (default: {DEFAULT_COND_NOISE_SCALE}, will be decayed by target area)",
    )
    args = parser.parse_args()

    # Load image first to compute target area
    print(f"Loading image from {args.input}...")
    media = load_image_as_media(args.input)
    _, _, h, w = media.shape
    print(f"  Input shape: {media.shape} (C, F={media.shape[1]}, H={h}, W={w})")
    input_area = h * w

    # Parse target area
    if args.target_area:
        if "x" in args.target_area:
            tw, th = map(int, args.target_area.split("x"))
            target_area = tw * th
        else:
            target_area = int(args.target_area)
    else:
        target_area = int(input_area * args.upscale_factor**2)

    # Compute noise scale matching production
    noise_scale = get_noise_scale(args.noise_scale, target_area)
    print(f"  Target area: {target_area} ({math.isqrt(target_area)}² approx)")
    print(f"  Noise scale: {noise_scale:.6f} (base {args.noise_scale}, decayed)")

    # Determine if tiling needed (match production logic)
    use_tiling = target_area > 1920 * 1080
    print(f"  Use tiling: {use_tiling}")

    # Init distributed if needed
    if get_world_size() > 1:
        init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        init_sequence_parallel(get_world_size())

    # Load pipeline (match production: bfloat16)
    print(f"\nLoading pipeline from {args.model}...")
    pipeline = SeedVRPipeline.from_pretrained_flashpack(
        args.model,
        device=get_device(),
        dtype=torch.bfloat16,
        use_distributed_loading=get_world_size() > 1,
    )

    stem = args.input.rsplit(".", 1)[0].rsplit("/", 1)[-1]

    # Run seamless
    print(f"\n=== Seamless run (pad={args.pad}, seed={args.seed}) ===")
    result_seamless = pipeline(
        media=media,
        target_area=target_area,
        seed=args.seed,
        use_tiling=use_tiling,
        cond_noise_scale=noise_scale,
        seamless=True,
        seamless_pad=args.pad,
    )
    print(f"  Output shape: {result_seamless.shape}")

    img_seamless = media_to_image(result_seamless)
    img_seamless.save(f"{stem}_seamless.png")
    print(f"  Saved: {stem}_seamless.png")

    tiled_seamless = tile_2x2(img_seamless)
    tiled_seamless.save(f"{stem}_tiled_seamless.png")
    print(f"  Saved: {stem}_tiled_seamless.png (2x2 tiled for seam check)")

    # Run normal for comparison
    if not args.skip_normal:
        print(f"\n=== Normal run (seed={args.seed}) ===")
        result_normal = pipeline(
            media=media,
            target_area=target_area,
            seed=args.seed,
            use_tiling=use_tiling,
            cond_noise_scale=noise_scale,
            seamless=False,
        )
        print(f"  Output shape: {result_normal.shape}")

        img_normal = media_to_image(result_normal)
        img_normal.save(f"{stem}_normal.png")
        print(f"  Saved: {stem}_normal.png")

        tiled_normal = tile_2x2(img_normal)
        tiled_normal.save(f"{stem}_tiled_normal.png")
        print(f"  Saved: {stem}_tiled_normal.png (2x2 tiled for seam check)")

    print("\nDone! Compare the _tiled_ images to check seam quality.")


if __name__ == "__main__":
    main()
