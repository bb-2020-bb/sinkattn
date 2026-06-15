from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F

from sinkattention.timing import AttentionTimingCollector
from wanx.runtime.fast_misc import apply_fast_qk_norm, apply_fast_rotary_qk


class WanTimedDenseAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanTimedDenseAttnProcessor2_0 requires PyTorch 2.0+.")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attention_mask
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        timing = getattr(getattr(attn, "inner_attention", None), "timing", None)
        with timing.section("qk_norm") if timing is not None and timing.is_enabled else nullcontext():
            query, key = apply_fast_qk_norm(attn, query, key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            with timing.section("rotary_apply") if timing is not None and timing.is_enabled else nullcontext():
                query, key = apply_fast_rotary_qk(query, key, rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3).type_as(query)

        hidden_states = attn.inner_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class TimedDenseAttentionRuntimeState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.timing = AttentionTimingCollector(label="dense")

    def reset_runtime_timing(self) -> None:
        self.timing.reset()

    def summarize_timing(self) -> dict:
        return self.timing.summarize()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        with self.timing.section("inner_total") if self.timing.is_enabled else nullcontext():
            with self.timing.section("dense_main") if self.timing.is_enabled else nullcontext():
                return F.scaled_dot_product_attention(q, k, v)


def set_timed_dense_attn_wanx(model, verbose: bool = False) -> None:
    inner_attn = TimedDenseAttentionRuntimeState()
    model._dense_runtime_state = inner_attn
    for block in model.blocks:
        block.attn1.verbose = verbose
        block.attn1.inner_attention = inner_attn
        origin_processor = block.attn1.get_processor()
        block.attn1.set_processor(WanTimedDenseAttnProcessor2_0())
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor
