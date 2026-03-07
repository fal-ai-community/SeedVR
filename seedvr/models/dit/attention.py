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

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.varlen import varlen_attn


class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert (
            len(args) == 0 or len(args) > 2
        ), "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        query = kwargs.pop("query", kwargs.pop("q", None))
        key = kwargs.pop("key", kwargs.pop("k", None))
        value = kwargs.pop("value", kwargs.pop("v", None))
        if query is None or key is None or value is None:
            raise ValueError("query, key, value must be provided")

        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
        )


class VarlenAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        query = kwargs.pop("query", kwargs.pop("q", None))
        key = kwargs.pop("key", kwargs.pop("k", None))
        value = kwargs.pop("value", kwargs.pop("v", None))
        if query is None and len(args) > 0:
            query = args[0]
        if key is None and len(args) > 1:
            key = args[1]
        if value is None and len(args) > 2:
            value = args[2]
        if query is None or key is None or value is None:
            raise ValueError("query, key, value must be provided")

        return varlen_attn(
            query=query,
            key=key,
            value=value,
            cu_seq_q=kwargs.pop("cu_seqlens_q"),
            cu_seq_k=kwargs.pop("cu_seqlens_k"),
            max_q=kwargs.pop("max_seqlen_q"),
            max_k=kwargs.pop("max_seqlen_k"),
            is_causal=kwargs.pop("is_causal", False),
            return_aux=kwargs.pop("return_aux", None),
        )
