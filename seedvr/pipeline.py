import math
import os
import random
from collections.abc import Callable
from typing import Any, Optional

import torch
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from flashpack.integrations.diffusers import FlashPackDiffusionPipeline
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from seedvr.common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
)
from seedvr.common.diffusion.samplers.base import Sampler
from seedvr.common.distributed import get_device
from seedvr.common.utils import filter_kwargs_for_method, maybe_use_tqdm
from seedvr.data.image.transforms.area_resize import area_resize
from seedvr.models.dit.na import flatten, pack, unflatten, unpack
from seedvr.models.dit.nadit import NaDiT
from seedvr.models.embeds import PrecomputedEmbeddings
from seedvr.models.video_vae_v3.modules.attn_video_vae import (
    DEFAULT_LATENT_TILE_SIZE,
    DEFAULT_LATENT_TILE_STRIDE,
    DEFAULT_PIXEL_TILE_SIZE,
    DEFAULT_PIXEL_TILE_STRIDE,
    VideoAutoencoderKLWrapper,
)
from seedvr.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr.projects.video_diffusion_sr.color_fix import (
    get_wavelet_kernel,
    wavelet_reconstruction,
)

# from common.fs import download


class SeedVRPipeline(FlashPackDiffusionPipeline):
    @register_to_config
    def __init__(
        self,
        dit: NaDiT,
        vae: VideoAutoencoderKLWrapper,
        sampler: Sampler,
        embeds: Optional[PrecomputedEmbeddings] = None,
        transform_timesteps: bool = True,
    ) -> None:
        super().__init__()
        embeds = embeds or PrecomputedEmbeddings.default(device=get_device(), dtype=dit.dtype)
        self.register_modules(
            dit=dit,
            vae=vae,
            sampler=sampler,
            embeds=embeds,
        )
        self.wavelet_kernel = get_wavelet_kernel(self.vae.dtype, in_device=self.dit.device)
        self.transform_timesteps = transform_timesteps

    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    @classmethod
    def from_original_pretrained(
        cls,
        pretrained_model_name_or_path: str = "ByteDance-Seed/SeedVR2-7B",
        dit_filename: str = "seedvr2_ema_7b.pth",
        vae_filename: str = "ema_vae.pth",
        device: str | torch.device | int | None = None,
        schedule_type: str = "lerp",
        schedule_t: float = 1000.0,
        sampler_type: str = "euler",
        sampler_prediction_type: str = "v_lerp",
        timesteps_sampling_type: str = "uniform_trailing",
        timesteps_sampling_steps: int = 1,
        **kwargs: Any,
    ) -> "SeedVRPipeline":
        """
        Load the pipeline from a pretrained model repository.
        """
        if not os.path.isdir(pretrained_model_name_or_path):
            download_kwargs = filter_kwargs_for_method(snapshot_download, kwargs)
            download_kwargs["allow_patterns"] = [dit_filename, vae_filename]
            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path, **download_kwargs
            )

        dit_path = os.path.join(pretrained_model_name_or_path, dit_filename)
        vae_path = os.path.join(pretrained_model_name_or_path, vae_filename)

        dit = NaDiT.from_single_file(dit_path, device=device)
        vae = VideoAutoencoderKLWrapper.from_single_file(vae_path, device=device)
        sampler = create_sampler_from_config(
            config=DictConfig({"type": sampler_type, "prediction_type": sampler_prediction_type}),
            schedule_type=schedule_type,
            schedule_t=schedule_t,
            timesteps_type=timesteps_sampling_type,
            timesteps_steps=timesteps_sampling_steps,
            device=device,
        )
        return cls(dit=dit, vae=vae, sampler=sampler)

    @torch.no_grad()
    def vae_encode(
        self,
        samples: list[Tensor],
        use_sample: bool = True,
        use_tiling: bool = True,
        use_tqdm: bool = True,
        tile_size: tuple[int, int] = DEFAULT_PIXEL_TILE_SIZE,
        tile_stride: tuple[int, int] = DEFAULT_PIXEL_TILE_STRIDE,
    ) -> list[Tensor]:
        latents = []
        if len(samples) > 0:
            device = get_device()
            dtype = self.vae.dtype
            scale = self.vae.config.scaling_factor
            shift = getattr(self.vae.config, "shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device="cpu", dtype=dtype, pin_memory=True).to(
                    device=device,
                    non_blocking=True,
                )
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device="cpu", dtype=dtype, pin_memory=True).to(
                    device=device,
                    non_blocking=True,
                )

            # Group samples of the same shape to batches if enabled.
            if self.vae.grouping:
                batches, indices = pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # Vae process by each group.
            for sample in batches:
                sample = sample.to(device, dtype)
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)
                if use_sample:
                    latent = self.vae.encode(
                        sample,
                        use_tiling=use_tiling,
                        use_tqdm=use_tqdm,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    ).latent
                else:
                    # Deterministic vae encode, only used for i2v inference (optionally)
                    latent = (
                        self.vae.encode(
                            sample,
                            use_tiling=use_tiling,
                            use_tqdm=use_tqdm,
                            tile_size=tile_size,
                            tile_stride=tile_stride,
                        )
                        .posterior.mode()
                        .squeeze(2)
                    )
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                latent = (latent - shift) * scale
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.vae.grouping:
                latents = unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]

        return latents

    @torch.no_grad()
    def vae_decode(
        self,
        latents: list[Tensor],
        use_tiling: bool = True,
        use_tqdm: bool = True,
        tile_size: tuple[int, int] = DEFAULT_LATENT_TILE_SIZE,
        tile_stride: tuple[int, int] = DEFAULT_LATENT_TILE_STRIDE,
    ) -> list[Tensor]:
        samples = []
        if len(latents) > 0:
            device = get_device()
            dtype = self.vae.dtype
            scale = self.vae.config.scaling_factor
            shift = getattr(self.vae.config, "shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device="cpu", dtype=dtype, pin_memory=True).to(
                    device=device,
                    non_blocking=True,
                )
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device="cpu", dtype=dtype, pin_memory=True).to(
                    device=device,
                    non_blocking=True,
                )

            # Group latents of the same shape to batches if enabled.
            if self.vae.grouping:
                latents, indices = pack(latents)
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            # Vae process by each group.
            for latent in latents:
                latent = latent.to(device, dtype)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                latent = latent.squeeze(2)
                sample = self.vae.decode(
                    latent,
                    use_tiling=use_tiling,
                    use_tqdm=use_tqdm,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).sample
                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)
                samples.append(sample)

            # Ungroup back to individual sample with the original order.
            if self.vae.grouping:
                samples = unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]

        return samples

    @torch.no_grad()
    def inference(
        self,
        noises: list[Tensor],
        conditions: list[Tensor],
        cfg_scale: float,
        cfg_rescale: float = 0.0,
        dit_offload: bool = False,
        use_tiling: bool = True,
        use_tqdm: bool = True,
        tile_size_pixel: tuple[int, int] = DEFAULT_PIXEL_TILE_SIZE,
        tile_stride_pixel: tuple[int, int] = DEFAULT_PIXEL_TILE_STRIDE,
        tile_size_latent: tuple[int, int] = DEFAULT_LATENT_TILE_SIZE,
        tile_stride_latent: tuple[int, int] = DEFAULT_LATENT_TILE_STRIDE,
    ) -> list[Tensor]:
        assert len(noises) == len(conditions)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        (text_pos_embeds, text_pos_shapes), (text_neg_embeds, text_neg_shapes) = self.embeds.get()
        latents, latents_shapes = flatten(noises)
        latents_cond, _ = flatten(conditions)

        # Enter eval mode.
        was_training = self.dit.training
        self.dit.eval()

        # Sampling.
        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(cfg_scale if (args.i + 1) / len(self.sampler.timesteps) <= 1.0 else 1.0),
                rescale=cfg_rescale,
            ),
        )

        # Exit eval mode.
        self.dit.train(was_training)

        # Unflatten.
        latents = unflatten(latents, latents_shapes)
        latents = [latent.to(self.vae.dtype) for latent in latents]

        # Vae decode.
        samples = self.vae_decode(
            latents,
            use_tiling=use_tiling,
            use_tqdm=use_tqdm,
            tile_size=tile_size_latent,
            tile_stride=tile_stride_latent,
        )

        return samples

    def get_linear_shift_function(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Callable[[float], float]:
        """
        Get a linear shift function.
        """
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def timestep_transform(
        self,
        timesteps: Tensor,
        latents_shapes: Tensor,
    ) -> Tensor:
        # Skip if not needed.
        if not self.transform_timesteps:
            return timesteps

        # Compute resolution.
        frames = (latents_shapes[:, 0] - 1) * self.vae.temporal_downsample_factor + 1
        heights = latents_shapes[:, 1] * self.vae.spatial_downsample_factor
        widths = latents_shapes[:, 2] * self.vae.spatial_downsample_factor

        img_shift_fn = self.get_linear_shift_function(
            x1=256 * 256,
            y1=1.0,
            x2=1024 * 1024,
            y2=3.2,
        )
        vid_shift_fn = self.get_linear_shift_function(
            x1=256 * 256 * 37,
            y1=1.0,
            x2=1280 * 720 * 145,
            y2=5.0,
        )

        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.sampler.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.sampler.schedule.T
        return timesteps

    def add_noise(
        self,
        x: torch.Tensor,
        aug_noise: torch.Tensor,
        cond_noise_scale: float = 0.25,
    ) -> torch.Tensor:
        """
        Add noise to the input.
        """
        t = (
            torch.tensor([self.sampler.timesteps.T], device="cpu", pin_memory=True)
            .to(device=x.device, non_blocking=True)
            * cond_noise_scale
        )
        shape = (
            torch.tensor(x.shape[1:], device="cpu", pin_memory=True)
            .to(device=x.device, non_blocking=True)
            .unsqueeze(0)
        )
        t = self.timestep_transform(t, shape)
        x = self.sampler.schedule.forward(x, aug_noise, t)
        return x

    def random_seeded_like(
        self,
        tensor: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.randn(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
            generator=generator,
        )

    def _clear_module_memory(self, module: torch.nn.Module) -> None:
        if isinstance(module, InflatedCausalConv3d):
            module.memory = None
        for child in module.children():
            self._clear_module_memory(child)

    def _clear_vae_memory(self) -> None:
        self._clear_module_memory(self.vae)

    @torch.no_grad()
    def __call__(
        self,
        media: torch.Tensor,
        target_area: int,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 1.0,
        seed: int | None = None,
        batch_size: int = 1,
        temporal_overlap: int = 0,
        cond_noise_scale: float = 0.05,
        use_tqdm: bool = True,
        use_tiling: bool = True,
        tile_size_latent: tuple[int, int] = (48, 48),
        tile_size_pixel: tuple[int, int] = (384, 384),
        tile_stride_latent: tuple[int, int] = (32, 32),
        tile_stride_pixel: tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        Generate a video from a media.
        """
        assert media.ndim == 4, "Media must be in CFHW format"
        c, f, h, w = media.shape

        overlap = 0 if f == 1 else temporal_overlap
        batch_size = max(batch_size, overlap + 1)
        if batch_size % 4 != 1:
            batch_size = math.ceil(batch_size / 4) * 4 + 1

        step_size = batch_size - overlap
        if step_size % 4 != 1:
            step_size = math.ceil(step_size / 4) * 4 + 1

        # Set up reproducibility.
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device=get_device()).manual_seed(seed)
        # Prepare media for inference.
        media_area = h * w
        scale = math.sqrt(target_area / media_area)
        media = area_resize(media, scale)

        # Update h, w
        h, w = media.shape[2:]

        # Now iterate over the media in batches.
        output_samples = []

        if overlap >= f:
            overlap = 0
            step_size = f

        batch_indices = list(range(0, f - overlap, step_size))
        num_batches = len(batch_indices)

        for batch_idx in maybe_use_tqdm(
            batch_indices,
            desc="Upsampling",
            use_tqdm=use_tqdm,
        ):
            batch_media = media[:, batch_idx : batch_idx + batch_size]
            num_padded_frames = 0
            if batch_media.shape[1] % 4 != 1:
                num_padded_frames = 4 - (batch_media.shape[1] % 4) + 1
                batch_media = torch.cat(
                    [batch_media] + [batch_media[:, -1:]] * num_padded_frames, dim=1
                )

            latents = self.vae_encode(
                [batch_media],
                use_tiling=use_tiling,
                use_tqdm=use_tqdm and num_batches == 1,
                tile_size=tile_size_pixel,
                tile_stride=tile_stride_pixel,
            )
            latents = [latent.to(self.dit.dtype) for latent in latents]
            noises = [self.random_seeded_like(latent, generator) for latent in latents]
            aug_noises = [self.random_seeded_like(latent, generator) for latent in latents]
            conditions = [
                self.get_condition(
                    noise,
                    self.add_noise(latent, aug_noise, cond_noise_scale),
                    task="sr",
                )
                for noise, aug_noise, latent in zip(noises, aug_noises, latents)
            ]
            samples = self.inference(
                noises,
                conditions,
                cfg_scale,
                cfg_rescale,
                use_tiling=use_tiling,
                use_tqdm=use_tqdm and num_batches == 1,
                tile_size_pixel=tile_size_pixel,
                tile_stride_pixel=tile_stride_pixel,
                tile_size_latent=tile_size_latent,
                tile_stride_latent=tile_stride_latent,
            )
            samples = [sample.unsqueeze(1) if sample.ndim == 3 else sample for sample in samples]

            batch_media = rearrange(batch_media, "c t h w -> t c h w").to(
                self.wavelet_kernel.device,
                dtype=self.wavelet_kernel.dtype,
                non_blocking=True,
            )
            samples = [
                rearrange(sample, "c t h w -> t c h w").to(
                    self.wavelet_kernel.device, dtype=self.wavelet_kernel.dtype
                )
                for sample in samples
            ]
            if num_padded_frames > 0:
                batch_media = batch_media[:-num_padded_frames]
                samples = [sample[:-num_padded_frames] for sample in samples]
            samples = [
                wavelet_reconstruction(sample, batch_media, self.wavelet_kernel)
                for sample in samples
            ]
            samples = [
                sample.clamp(-1.0, 1.0)
                .mul_(0.5)
                .add_(0.5)
                .mul_(255)
                .round()
                .to(torch.uint8)
                .detach()
                .cpu()
                for sample in samples
            ]
            # if batch_idx > 0 and overlap > 0:
            #    samples = [sample[overlap:] for sample in samples]

            output_samples.append(samples)
            self._clear_vae_memory()

        if len(output_samples) == 0 or len(output_samples[0]) == 0:
            return []

        # output_samples = [torch.cat(samples, dim=0) for samples in zip(*output_samples)]
        # torch.cuda.empty_cache()
        # return output_samples[0].permute(1, 0, 2, 3)

        # Simple approach: just concatenate all frames, skipping overlaps after the first batch
        all_frames = []

        for i, samples in enumerate(output_samples):
            sample_tensor = samples[0]

            if i == 0:
                # First batch - take all frames
                all_frames.append(sample_tensor)
            else:
                # Subsequent batches - skip the first 'overlap' frames
                if overlap > 0 and overlap < sample_tensor.shape[0]:
                    all_frames.append(sample_tensor[overlap:])
                else:
                    all_frames.append(sample_tensor)

        # Concatenate all frames
        combined_video = torch.cat(all_frames, dim=0)

        # Trim to original length if necessary
        if combined_video.shape[0] > f:
            combined_video = combined_video[:f]

        # t c h w -> c t h w
        return combined_video.permute(1, 0, 2, 3)
