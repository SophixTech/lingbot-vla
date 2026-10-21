# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn.attention.flex_attention import flex_attention


# FP32 Q/K/V, GQA and fused attention match lingbot-vla. Do not materialize
# or pad an attention mask for every head/layer: reuse the [B, Q, K] mask.
_flex_attention = torch.compile(
    flex_attention, mode="max-autotune-no-cudagraphs", dynamic=False
)


def flex_attention_forward(query_states, key_states, value_states, attention_mask, scaling=None):
    """Fused joint attention, including rectangular ECOT label-K/V masks.

    Mask semantics and all-masked padding rows match our eager implementation.
    Keeping the finite eager mask sentinel avoids changing the training math;
    all-masked rows use zero scores (the same uniform distribution) to avoid
    loss of precision in the fused kernel's log-sum-exp backward calculation.
    """
    # PyTorch 2.8 CUDA autotuning cannot handle symbolic dimensions captured
    # by score_mod. Specialize this kernel even under outer model compilation;
    # dynamic=False on the inner compile alone does not cover that case.
    for tensor in (query_states, key_states, value_states, attention_mask):
        if tensor is not None:
            torch._dynamo.mark_static(tensor)
    batch_size, query_len, num_heads, head_dim = query_states.shape
    key_len = key_states.shape[1]
    original_dtype = query_states.dtype
    q = query_states.transpose(1, 2).float()
    k = key_states.transpose(1, 2).float()
    v = value_states.transpose(1, 2).float()
    score_mod = None
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool or attention_mask.ndim != 3:
            raise ValueError("Joint attention requires a boolean [B, Q, K] mask")
        if attention_mask.shape != (batch_size, query_len, key_len):
            raise ValueError("Attention mask must include every query and key, including label K/V")
        row_has_keys = attention_mask.any(dim=-1)

        def score_mod(score, b, h, q_idx, kv_idx):
            # Kernels may evaluate padded tile indices for non-multiple lengths.
            qi = q_idx.clamp(max=query_len - 1)
            ki = kv_idx.clamp(max=key_len - 1)
            masked = torch.where(attention_mask[b, qi, ki], score, -2.3819763e38)
            return torch.where(row_has_keys[b, qi], masked, 0.0)

    output = _flex_attention(
        q, k, v, score_mod=score_mod, enable_gqa=True,
        scale=head_dim**-0.5 if scaling is None else scaling,
        # Use the training kernel even for short prefix/test sequences.
        kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
    )
    return output.to(original_dtype).transpose(1, 2).contiguous().reshape(
        batch_size, query_len, num_heads * head_dim
    )
