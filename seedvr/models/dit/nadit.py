# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from flashpack.integrations.diffusers import FlashPackDiffusersModelMixin
from seedvr.common.cache import Cache
from seedvr.common.distributed.ops import slice_inputs
from seedvr.common.utils import get_torch_dtype
from torch import nn

from ..utils import PretrainedMixin
from . import na
from .embedding import TimeEmbedding
from .modulation import get_ada_layer
from .nablocks import get_nablock
from .normalization import get_norm_layer
from .patch import NaPatchIn, NaPatchOut


def gradient_checkpointing(
    module: Callable | nn.Module,
    *args,
    enabled: bool,
    use_reentrant: bool = False,
    preserve_rng_state: bool = True,
    **kwargs,
):
    """Wrap ``module`` with torch.utils.checkpoint when ``enabled`` is True.

    Defaults to the modern non-reentrant impl which supports non-tensor kwargs
    (NaDiT block forward takes ``vid_shape``/``txt_shape``/``cache``).
    Inference and disabled-checkpoint paths bypass the wrapper for zero overhead.
    """
    if not enabled:
        return module(*args, **kwargs)
    from torch.utils.checkpoint import checkpoint as _torch_checkpoint

    return _torch_checkpoint(
        module,
        *args,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
        **kwargs,
    )


@dataclass
class NaDiTOutput:
    vid_sample: torch.Tensor


class NaDiT(PretrainedMixin, FlashPackDiffusersModelMixin, ConfigMixin):
    """
    Native Resolution Diffusers Transformer (NaDiT)
    """

    config_name = "config.json"
    # Per-block selector. Empty set or False means disabled. Use
    # ``set_gradient_checkpointing`` to populate.
    gradient_checkpointing: "set[int] | bool" = False
    gradient_checkpointing_use_reentrant: bool = False
    # Optional sub-module restriction: when set to "mlp_only" / "attn_only",
    # the per-block forward in NaMMSRTransformerBlock will route only the named
    # sub-call through torch.utils.checkpoint instead of wrapping the whole
    # block. None means whole-block (default).
    gradient_checkpointing_submodule: "str | None" = None

    @register_to_config
    def __init__(
        self,
        vid_in_channels: int = 33,
        vid_out_channels: int = 16,
        vid_dim: int = 3072,
        txt_in_dim: int | None = 5120,
        txt_dim: int | None = 3072,
        emb_dim: int = 18432,
        heads: int = 24,
        head_dim: int = 128,
        expand_ratio: int = 4,
        norm: str | None = "fusedrms",
        norm_eps: float = 1e-5,
        ada: str = "single",
        qk_bias: bool = False,
        qk_rope: bool = True,
        qk_norm: str | None = "fusedrms",
        patch_size: int | tuple[int, int, int] = (1, 2, 2),
        num_layers: int = 36,
        block_type: str | tuple[str] = ("mmdit_sr",) * 36,
        shared_qkv: bool = False,
        shared_mlp: bool = False,
        mlp_type: str = "normal",
        window: tuple | None = ((4, 3, 3),) * 36,
        window_method: tuple[str] | None = (
            "720pwin_by_size_bysize",
            "720pswin_by_size_bysize",
        )
        * 18,
        temporal_window_size: int | None = None,
        temporal_shifted: bool = False,
        dtype: str | torch.dtype = "float32",
        **kwargs: Any,
    ) -> None:
        ada = get_ada_layer(ada)
        norm = get_norm_layer(norm)
        qk_norm = get_norm_layer(qk_norm)
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )
        self.txt_in = (
            nn.Linear(txt_in_dim, txt_dim)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers
        if temporal_window_size is None or isinstance(temporal_window_size, int):
            temporal_window_size = [temporal_window_size] * num_layers
        if temporal_shifted is None or isinstance(temporal_shifted, bool):
            temporal_shifted = [temporal_shifted] * num_layers

        self.blocks = nn.ModuleList(
            [
                get_nablock(block_type[i])(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_rope=qk_rope,
                    qk_norm=qk_norm,
                    shared_qkv=shared_qkv,
                    shared_mlp=shared_mlp,
                    mlp_type=mlp_type,
                    window=window[i],
                    window_method=window_method[i],
                    temporal_window_size=temporal_window_size[i],
                    temporal_shifted=temporal_shifted[i],
                    dtype=get_torch_dtype(dtype),
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

        self.need_txt_repeat = block_type[0] in [
            "mmdit_stwin",
            "mmdit_stwin_spatial",
            "mmdit_stwin_3d_spatial",
        ]

    def set_gradient_checkpointing(
        self,
        enable: "bool | set[int] | list[int]",
        *,
        use_reentrant: bool = False,
        submodule: "str | None" = None,
    ):
        """Enable activation checkpointing on a subset of trunk blocks.

        Args:
            enable: ``True``/``False`` for whole-trunk on/off, or an explicit
                set/list of block indices to checkpoint. Out-of-range indices
                are silently dropped (caller already validated).
            use_reentrant: forwarded to torch.utils.checkpoint.
            submodule: ``None`` (default) wraps the whole NaMMSRTransformerBlock.
                ``"mlp_only"`` / ``"attn_only"`` route only that sub-call through
                checkpoint inside the block forward.
        """
        if isinstance(enable, bool):
            self.gradient_checkpointing = enable
        else:
            n = len(self.blocks)
            self.gradient_checkpointing = {i for i in enable if 0 <= i < n}
        self.gradient_checkpointing_use_reentrant = bool(use_reentrant)
        self.gradient_checkpointing_submodule = submodule

    def _block_checkpoint_enabled(self, block_index: int) -> bool:
        flag = self.gradient_checkpointing
        if isinstance(flag, bool):
            return flag and self.training
        return bool(flag) and (block_index in flag) and self.training

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        timestep: int | float | torch.IntTensor | torch.FloatTensor,  # b
        disable_cache: bool = True,  # for test
    ):
        # Text input.
        if txt_shape.size(-1) == 1 and self.need_txt_repeat:
            txt, txt_shape = na.repeat(
                txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0]
            )
        # slice vid after patching in when using sequence parallelism
        txt = slice_inputs(txt, dim=0)
        txt = self.txt_in(txt)

        # Video input.
        # Sequence parallel slicing is done inside patching class.
        vid, vid_shape = self.vid_in(vid, vid_shape)

        # Embedding input.
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        # Body
        cache = Cache(disable=disable_cache)
        for i, block in enumerate(self.blocks):
            vid, txt, vid_shape, txt_shape = gradient_checkpointing(
                enabled=self._block_checkpoint_enabled(i),
                use_reentrant=self.gradient_checkpointing_use_reentrant,
                module=block,
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
            )

        vid, vid_shape = self.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)


class NaDiTUpscaler(nn.Module):
    """
    Native Resolution Diffusers Transformer (NaDiT)
    """

    gradient_checkpointing: "set[int] | bool" = False
    gradient_checkpointing_use_reentrant: bool = False
    gradient_checkpointing_submodule: "str | None" = None

    def __init__(
        self,
        vid_in_channels: int,
        vid_out_channels: int,
        vid_dim: int,
        txt_in_dim: int | None,
        txt_dim: int | None,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: str | None,
        norm_eps: float,
        ada: str,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: str | None,
        patch_size: int | tuple[int, int, int],
        num_layers: int,
        block_type: str | tuple[str],
        shared_qkv: bool = False,
        shared_mlp: bool = False,
        mlp_type: str = "normal",
        window: tuple | None = None,
        window_method: tuple[str] | None = None,
        temporal_window_size: int = None,
        temporal_shifted: bool = False,
        **kwargs,
    ):
        ada = get_ada_layer(ada)
        norm = get_norm_layer(norm)
        qk_norm = get_norm_layer(qk_norm)
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )
        self.txt_in = (
            nn.Linear(txt_in_dim, txt_dim)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
        )

        self.emb_scale = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers
        if temporal_window_size is None or isinstance(temporal_window_size, int):
            temporal_window_size = [temporal_window_size] * num_layers
        if temporal_shifted is None or isinstance(temporal_shifted, bool):
            temporal_shifted = [temporal_shifted] * num_layers

        self.blocks = nn.ModuleList(
            [
                get_nablock(block_type[i])(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_rope=qk_rope,
                    qk_norm=qk_norm,
                    shared_qkv=shared_qkv,
                    shared_mlp=shared_mlp,
                    mlp_type=mlp_type,
                    window=window[i],
                    window_method=window_method[i],
                    temporal_window_size=temporal_window_size[i],
                    temporal_shifted=temporal_shifted[i],
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

        self.need_txt_repeat = block_type[0] in [
            "mmdit_stwin",
            "mmdit_stwin_spatial",
            "mmdit_stwin_3d_spatial",
        ]

    def set_gradient_checkpointing(
        self,
        enable: "bool | set[int] | list[int]",
        *,
        use_reentrant: bool = False,
        submodule: "str | None" = None,
    ):
        if isinstance(enable, bool):
            self.gradient_checkpointing = enable
        else:
            n = len(self.blocks)
            self.gradient_checkpointing = {i for i in enable if 0 <= i < n}
        self.gradient_checkpointing_use_reentrant = bool(use_reentrant)
        self.gradient_checkpointing_submodule = submodule

    def _block_checkpoint_enabled(self, block_index: int) -> bool:
        flag = self.gradient_checkpointing
        if isinstance(flag, bool):
            return flag and self.training
        return bool(flag) and (block_index in flag) and self.training

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        timestep: int | float | torch.IntTensor | torch.FloatTensor,  # b
        downscale: int | float | torch.IntTensor | torch.FloatTensor,  # b
        disable_cache: bool = False,  # for test
    ):

        # Text input.
        if txt_shape.size(-1) == 1 and self.need_txt_repeat:
            txt, txt_shape = na.repeat(
                txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0]
            )
        # slice vid after patching in when using sequence parallelism
        txt = slice_inputs(txt, dim=0)
        txt = self.txt_in(txt)

        # Video input.
        # Sequence parallel slicing is done inside patching class.
        vid, vid_shape = self.vid_in(vid, vid_shape)

        # Embedding input.
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)
        emb_scale = self.emb_scale(downscale, device=vid.device, dtype=vid.dtype)
        emb = emb + emb_scale

        # Body
        cache = Cache(disable=disable_cache)
        for i, block in enumerate(self.blocks):
            vid, txt, vid_shape, txt_shape = gradient_checkpointing(
                enabled=self._block_checkpoint_enabled(i),
                use_reentrant=self.gradient_checkpointing_use_reentrant,
                module=block,
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
            )

        vid, vid_shape = self.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)
