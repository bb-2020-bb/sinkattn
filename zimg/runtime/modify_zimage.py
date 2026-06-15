from __future__ import annotations

from typing import Any, Optional

import torch


def apply_zimage_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


def prepare_zimage_attention_tensors(
    attn: Any,
    hidden_states: torch.Tensor,
    *,
    freqs_cis: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if freqs_cis is not None:
        query = apply_zimage_rotary_emb(query, freqs_cis)
        key = apply_zimage_rotary_emb(key, freqs_cis)

    dtype = value.dtype
    query = query.to(dtype)
    key = key.to(dtype)
    value = value.to(dtype)
    return query, key, value


__all__ = [
    "apply_zimage_rotary_emb",
    "prepare_zimage_attention_tensors",
]
