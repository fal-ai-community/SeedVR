from __future__ import annotations

import inspect
import importlib
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM

from seedvr.projects.training_hdr.validation import linear_hdr_from_target_tensor


def _ensure_falcon_ocr_flex_attention_compat() -> None:
    import torch.nn.attention.flex_attention as flex_attention_mod

    if hasattr(flex_attention_mod, "AuxRequest"):
        return

    original_flex_attention = getattr(flex_attention_mod, "flex_attention", None)
    if original_flex_attention is None:
        return

    signature = inspect.signature(original_flex_attention)
    if "return_lse" not in signature.parameters:
        return

    @dataclass(frozen=True)
    class AuxRequestCompat:
        lse: bool = False
        max_scores: bool = False

    @dataclass(frozen=True)
    class AuxOutputCompat:
        lse: torch.Tensor | None = None
        max_scores: torch.Tensor | None = None

    @wraps(original_flex_attention)
    def compat_flex_attention(*args: Any, return_aux: Any = None, **kwargs: Any) -> Any:
        wants_aux = return_aux is not None
        wants_lse = bool(getattr(return_aux, "lse", False))
        if wants_aux and "return_aux" not in signature.parameters and "return_lse" not in kwargs:
            kwargs["return_lse"] = wants_lse

        result = original_flex_attention(*args, **kwargs)
        if not wants_aux:
            return result

        if isinstance(result, tuple) and len(result) == 2:
            output, aux_or_lse = result
            if hasattr(aux_or_lse, "lse") or hasattr(aux_or_lse, "max_scores"):
                return output, aux_or_lse
            return output, AuxOutputCompat(
                lse=aux_or_lse if wants_lse else None,
                max_scores=None,
            )

        return result, AuxOutputCompat(
            lse=None,
            max_scores=None,
        )

    flex_attention_mod.AuxRequest = AuxRequestCompat
    flex_attention_mod.AuxOutput = AuxOutputCompat
    flex_attention_mod.flex_attention = compat_flex_attention
    print(
        "[seedvr-hdr] enabled Falcon-OCR flex_attention compatibility shim "
        "for torch versions without AuxRequest"
    )


def _disable_falcon_ocr_compiled_attention(model_module: Any) -> None:
    attention_module = importlib.import_module(
        model_module.__name__.replace("modeling_falcon_ocr", "attention")
    )
    eager_flex_attention = attention_module.flex_attention
    attention_module.compiled_flex_attn_decode = eager_flex_attention
    attention_module.compiled_flex_attn_prefill = eager_flex_attention
    model_module.compiled_flex_attn_decode = eager_flex_attention
    model_module.compiled_flex_attn_prefill = eager_flex_attention
    print(
        "[seedvr-hdr] disabled Falcon-OCR compiled flex attention wrappers; "
        "using eager flex_attention for trainer compatibility"
    )


def _select_preview_frames(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5:
        return images[:, images.shape[1] // 2]
    return images


def _robust_tonemap_for_ocr(linear_hdr: torch.Tensor) -> torch.Tensor:
    linear_hdr = torch.clamp(linear_hdr, min=0.0)
    if linear_hdr.numel() == 0:
        return torch.zeros_like(linear_hdr)
    flat = linear_hdr.reshape(linear_hdr.shape[0], -1)
    percentile = torch.quantile(flat, 0.995, dim=1, keepdim=True)
    maximum = flat.max(dim=1, keepdim=True).values.clamp(min=1.0)
    scale = torch.where(percentile > 1.0e-6, percentile, maximum)
    scale = scale.view(linear_hdr.shape[0], 1, 1, 1)
    normalized = linear_hdr / scale
    mapped = normalized / (1.0 + normalized)
    return torch.clamp(mapped, 0.0, 1.0) ** (1.0 / 2.2)


def build_ocr_preview_batch(
    images: torch.Tensor,
    *,
    target_representation: str,
) -> torch.Tensor:
    images = _select_preview_frames(images)
    linear = linear_hdr_from_target_tensor(images, target_representation)
    return _robust_tonemap_for_ocr(linear)


def preview_batch_to_pil_images(images: torch.Tensor) -> list[Image.Image]:
    uint8 = (
        images.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    return [Image.fromarray(array) for array in uint8]


@dataclass
class FalconOCRConfig:
    model_id: str
    max_new_tokens: int
    min_dimension: int
    max_dimension: int
    min_chars: int = 3
    min_tokens: int = 8
    loss_normalizer_min_tokens: int = 16
    require_alnum: bool = True


class FalconOCRTextTeacher:
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        config: FalconOCRConfig,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.config = config
        self._disabled_reason: str | None = None
        self._last_status: str = "init"
        self._last_generated_texts: list[str] = []
        self._last_valid_text_count: int = 0
        self._last_char_count: int = 0

        _ensure_falcon_ocr_flex_attention_compat()
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        model.requires_grad_(False)
        if hasattr(model, "_ensure_device_buffers"):
            model._ensure_device_buffers()
        self.model = model
        self.tokenizer = model._get_tokenizer() if hasattr(model, "_get_tokenizer") else None

        model_module = importlib.import_module(model.__class__.__module__)
        _disable_falcon_ocr_compiled_attention(model_module)
        processing_module = importlib.import_module(
            model.__class__.__module__.replace("modeling_falcon_ocr", "processing_falcon_ocr")
        )
        self.KVCache = model_module.KVCache
        self.process_batch = processing_module.process_batch
        category_prompts = getattr(model_module, "CATEGORY_PROMPTS")
        self.prompt = f"<|image|>{category_prompts['text']}\n<|OCR_PLAIN|>"
        self.stop_token = "<|end_of_query|>"

    @property
    def disabled(self) -> bool:
        return self._disabled_reason is not None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            print(f"[seedvr-hdr] disabling Falcon-OCR text reconstruction loss: {reason}")
        self._disabled_reason = reason
        self._last_status = "disabled"

    def debug_state(self) -> dict[str, Any]:
        return {
            "disabled": self.disabled,
            "disabled_reason": self.disabled_reason,
            "last_status": self._last_status,
            "last_valid_text_count": self._last_valid_text_count,
            "last_char_count": self._last_char_count,
            "last_generated_texts": self._last_generated_texts[:4],
        }

    @torch.no_grad()
    def generate_texts(self, target_previews: torch.Tensor) -> list[str]:
        target_pils = preview_batch_to_pil_images(target_previews)
        return self.model.generate(
            target_pils,
            category="text",
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
            compile=False,
            seed=42,
            min_dimension=self.config.min_dimension,
            max_dimension=self.config.max_dimension,
        )

    def _build_batch_inputs(
        self,
        image_pils: list[Image.Image],
        prompts: list[str],
    ) -> dict[str, Any]:
        batch = self.process_batch(
            self.tokenizer,
            self.model.config,
            list(zip(image_pils, prompts)),
            max_length=self.model.config.max_seq_len,
            min_dimension=self.config.min_dimension,
            max_dimension=self.config.max_dimension,
            patch_size=self.model.config.spatial_patch_size,
            merge_size=1,
        )
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _build_differentiable_pixel_values(
        self,
        previews: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_temporal, max_height, max_width = pixel_mask.shape
        pixel_values = previews.new_zeros(
            (batch_size, max_temporal, max_height, max_width, 3),
            dtype=self.dtype,
        )
        valid_mask = pixel_mask.to(device=previews.device, dtype=previews.dtype)
        for index in range(batch_size):
            sample_mask = valid_mask[index]
            valid_h = int(sample_mask.any(dim=0).any(dim=-1).sum().item())
            valid_w = int(sample_mask.any(dim=0).any(dim=-2).sum().item())
            if valid_h <= 0 or valid_w <= 0:
                continue
            resized = F.interpolate(
                previews[index : index + 1],
                size=(valid_h, valid_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
            # Falcon-OCR normalizes RGB pixels to [-1, 1].
            resized = resized.clamp(0.0, 1.0).mul(2.0).sub(1.0)
            pixel_values[index, 0, :valid_h, :valid_w] = resized.permute(1, 2, 0).to(self.dtype)
        return pixel_values

    def _forward_teacher_forced(
        self,
        *,
        previews: torch.Tensor,
        image_pils: list[Image.Image],
        prompts: list[str],
    ) -> dict[str, Any]:
        batch_inputs = self._build_batch_inputs(image_pils, prompts)
        pixel_values = self._build_differentiable_pixel_values(previews, batch_inputs["pixel_mask"])
        self.model._pad_token_id = batch_inputs["pad_token_id"]
        attention_mask = self.model.get_attention_mask(
            batch_inputs["tokens"],
            max_len=batch_inputs["tokens"].shape[1],
        )
        kv_cache = self.KVCache(
            max_batch_size=batch_inputs["tokens"].shape[0],
            max_seq_length=batch_inputs["tokens"].shape[1],
            n_heads=self.model.config.n_heads,
            head_dim=self.model.config.head_dim,
            num_layers=self.model.config.n_layers,
        )
        logits = self.model(
            tokens=batch_inputs["tokens"],
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            rope_pos_t=batch_inputs["pos_t"],
            rope_pos_hw=batch_inputs["pos_hw"],
            pixel_values=pixel_values,
            pixel_mask=batch_inputs["pixel_mask"],
        )
        return {
            "logits": logits,
            "tokens": batch_inputs["tokens"],
            "pad_token_id": int(batch_inputs["pad_token_id"]),
        }

    def compute_loss(
        self,
        *,
        predicted_images: torch.Tensor,
        target_images: torch.Tensor,
        target_representation: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = predicted_images.new_zeros(())
        if self.disabled:
            return zero, {
                "text_recon_loss": 0.0,
                "text_recon_active": 0.0,
                "text_recon_tokens": 0.0,
                "text_recon_chars": 0.0,
                "text_recon_normalizer_tokens": 0.0,
                "text_recon_skipped_short": 0.0,
            }

        try:
            predicted_previews = build_ocr_preview_batch(
                predicted_images,
                target_representation=target_representation,
            )
            target_previews = build_ocr_preview_batch(
                target_images,
                target_representation=target_representation,
            )
            target_texts = self.generate_texts(target_previews)
            self._last_generated_texts = [
                text if isinstance(text, str) else repr(type(text))
                for text in target_texts[:8]
            ]
            valid_indices = [
                idx
                for idx, text in enumerate(target_texts)
                if isinstance(text, str)
                and len(text.strip()) >= self.config.min_chars
                and (
                    not self.config.require_alnum
                    or any(ch.isalnum() for ch in text.strip())
                )
            ]
            self._last_valid_text_count = len(valid_indices)
            self._last_char_count = int(
                sum(len(text.strip()) for text in target_texts if isinstance(text, str))
            )
            if not valid_indices:
                self._last_status = "no_text"
                return zero, {
                    "text_recon_loss": 0.0,
                    "text_recon_active": 0.0,
                    "text_recon_tokens": 0.0,
                    "text_recon_chars": 0.0,
                    "text_recon_normalizer_tokens": 0.0,
                    "text_recon_skipped_short": 0.0,
                }

            predicted_previews = predicted_previews[valid_indices]
            target_texts = [target_texts[idx].strip() for idx in valid_indices]
            image_pils = preview_batch_to_pil_images(predicted_previews)
            full_prompts = [f"{self.prompt}{text}{self.stop_token}" for text in target_texts]
            prefix_prompts = [self.prompt for _ in target_texts]

            full_batch = self._forward_teacher_forced(
                previews=predicted_previews,
                image_pils=image_pils,
                prompts=full_prompts,
            )
            prefix_batch = self._build_batch_inputs(image_pils, prefix_prompts)
            prefix_lengths = (prefix_batch["tokens"] != full_batch["pad_token_id"]).sum(dim=1)

            logits = full_batch["logits"]
            tokens = full_batch["tokens"]
            pad_token_id = full_batch["pad_token_id"]

            shift_logits = logits[:, :-1].transpose(1, 2).float()
            shift_labels = tokens[:, 1:].clone()

            total_lengths = (tokens != pad_token_id).sum(dim=1)
            sequence_length = tokens.shape[1]
            target_start = sequence_length - total_lengths + prefix_lengths
            label_positions = torch.arange(1, sequence_length, device=tokens.device).unsqueeze(0)
            valid_label_mask = (
                (label_positions >= target_start.unsqueeze(1))
                & (shift_labels != pad_token_id)
            )
            valid_token_count = valid_label_mask.sum()
            valid_char_count = float(sum(len(text) for text in target_texts))
            if not torch.any(valid_label_mask):
                self._last_status = "no_label_tokens"
                return zero, {
                    "text_recon_loss": 0.0,
                    "text_recon_active": 0.0,
                    "text_recon_tokens": 0.0,
                    "text_recon_chars": valid_char_count,
                    "text_recon_normalizer_tokens": 0.0,
                    "text_recon_skipped_short": 0.0,
                }
            if int(valid_token_count.detach().item()) < self.config.min_tokens:
                self._last_status = "short_text"
                return zero, {
                    "text_recon_loss": 0.0,
                    "text_recon_active": 0.0,
                    "text_recon_tokens": float(valid_token_count.detach().item()),
                    "text_recon_chars": valid_char_count,
                    "text_recon_normalizer_tokens": float(
                        self.config.loss_normalizer_min_tokens
                    ),
                    "text_recon_skipped_short": 1.0,
                }

            shift_labels[~valid_label_mask] = -100
            loss_sum = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction="sum",
            )
            normalizer = torch.clamp(
                valid_token_count.to(dtype=loss_sum.dtype),
                min=float(self.config.loss_normalizer_min_tokens),
            )
            loss = loss_sum / normalizer
            self._last_status = "active"
            return loss, {
                "text_recon_loss": float(loss.detach().item()),
                "text_recon_active": 1.0,
                "text_recon_tokens": float(valid_token_count.detach().item()),
                "text_recon_chars": valid_char_count,
                "text_recon_normalizer_tokens": float(normalizer.detach().item()),
                "text_recon_skipped_short": 0.0,
            }
        except Exception as exc:
            self.disable(str(exc))
            return zero, {
                "text_recon_loss": 0.0,
                "text_recon_active": 0.0,
                "text_recon_tokens": 0.0,
                "text_recon_chars": 0.0,
                "text_recon_normalizer_tokens": 0.0,
                "text_recon_skipped_short": 0.0,
            }
