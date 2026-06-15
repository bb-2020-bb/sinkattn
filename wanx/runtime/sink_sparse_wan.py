from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sinkattention.sparse_attention import (
    SinkMaskCalibrator,
    SinkRuntimeState,
    SinkSparseConfig,
    _BLOCK_SPARSE_KERNEL_BLOCK_SIZE,
    _adapt_block_mask_for_kernel,
    _block_sparse_attn,
    _build_offline_required_mask,
    _ceil_div,
    _dense_short_attention,
    _dense_short_attention_with_lse,
    _device_cache_key,
    _make_full_block_mask,
    _mean_pool_sequence_preserve_prefix,
    _mix_attention_outputs,
    load_sink_mask_package,
)
from wanx.runtime.fast_misc import apply_fast_qk_norm, apply_fast_rotary_qk


class WanSinkSparseInnerAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        sink_mask: torch.Tensor,
        config: SinkSparseConfig,
        runtime_state: Optional[SinkRuntimeState] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.runtime_state = runtime_state
        self.register_buffer("sink_mask", sink_mask.bool().cpu())
        self._runtime_mask_cache: Dict[tuple[str, tuple[str, int], int, int, int, int], torch.Tensor] = {}
        self._kernel_runtime_mask_cache: Dict[tuple[str, tuple[str, int], int, int, int, int, int], torch.Tensor] = {}
        self._dynamic_mask_cache: Dict[tuple[tuple[str, int], int, int, int, int], torch.Tensor] = {}
        self._local_band_mask_cache: Dict[tuple[tuple[str, int], int, int, int, int, int], torch.Tensor] = {}

    def _resolve_dynamic_plan(self) -> tuple[str, int]:
        dynamic_mode = str(self.config.dynamic_mode).strip().lower()
        if dynamic_mode not in {"none", "pooled_kv"}:
            raise ValueError(
                "Unsupported Sink dynamic_mode. The maintained runtime only keeps the two-branch "
                f"compensation path: none | pooled_kv. Got: {self.config.dynamic_mode}"
            )

        if dynamic_mode == "none":
            return "none", max(int(self.config.sample_gap), 1)
        return dynamic_mode, max(int(self.config.sample_gap), 1)

    def _get_runtime_mask(
        self,
        sink_mask: torch.Tensor,
        mask_tag: str,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        expected_q_blocks = _ceil_div(seq_len_q, self.config.block_size)
        expected_k_blocks = _ceil_div(seq_len_k, self.config.block_size)
        if (
            sink_mask.size(0) != num_heads
            or sink_mask.size(1) != expected_q_blocks
            or sink_mask.size(2) != expected_k_blocks
        ):
            raise ValueError(
                "Sink mask shape does not match runtime sequence layout. "
                f"Expected [{num_heads}, {expected_q_blocks}, {expected_k_blocks}], "
                f"got {list(sink_mask.shape)}. Re-calibrate the mask for this Wan setup."
            )

        cache_key = (mask_tag, _device_cache_key(device), batch_size, num_heads, seq_len_q, seq_len_k)
        cached = self._runtime_mask_cache.get(cache_key)
        if cached is None or cached.device != device:
            base_mask = sink_mask if sink_mask.device == device else sink_mask.to(device=device)
            required_mask = _build_offline_required_mask(
                num_query_blocks=expected_q_blocks,
                num_key_blocks=expected_k_blocks,
                config=self.config,
                device=device,
            )
            if required_mask is not None:
                base_mask = (base_mask | required_mask.unsqueeze(0).expand(num_heads, -1, -1)).contiguous()
            if batch_size == 1:
                cached = base_mask.unsqueeze(0).contiguous()
            else:
                cached = base_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            self._runtime_mask_cache[cache_key] = cached
        runtime_mask = cached

        local_band_mask = self._get_runtime_local_band_mask(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            device=device,
        )
        if local_band_mask is None:
            source_runtime_mask = runtime_mask
        else:
            source_runtime_mask = (runtime_mask | local_band_mask).contiguous()

        kernel_cache_key = (
            mask_tag,
            _device_cache_key(device),
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_k,
            int(self.config.block_size),
        )
        kernel_cached = self._kernel_runtime_mask_cache.get(kernel_cache_key)
        if kernel_cached is None or kernel_cached.device != device:
            kernel_cached = _adapt_block_mask_for_kernel(
                source_runtime_mask,
                source_block_size=int(self.config.block_size),
                seq_len_q=seq_len_q,
                seq_len_k=seq_len_k,
            )
            self._kernel_runtime_mask_cache[kernel_cache_key] = kernel_cached
        return kernel_cached

    def _get_dynamic_mask(
        self,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (_device_cache_key(device), batch_size, num_heads, seq_len_q, seq_len_k)
        cached = self._dynamic_mask_cache.get(cache_key)
        if cached is None or cached.device != device:
            cached = _make_full_block_mask(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len_q=seq_len_q,
                seq_len_k=seq_len_k,
                block_size=_BLOCK_SPARSE_KERNEL_BLOCK_SIZE,
                device=device,
            )
            self._dynamic_mask_cache[cache_key] = cached
        return cached

    def _get_runtime_local_band_mask(
        self,
        batch_size: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        band_width = max(int(self.config.runtime_diagonal_band_width), 0)
        if band_width <= 0:
            return None

        num_query_blocks = _ceil_div(seq_len_q, self.config.block_size)
        num_key_blocks = _ceil_div(seq_len_k, self.config.block_size)
        cache_key = (_device_cache_key(device), batch_size, num_heads, num_query_blocks, num_key_blocks, band_width)
        cached = self._local_band_mask_cache.get(cache_key)
        if cached is None or cached.device != device:
            q_positions = torch.arange(num_query_blocks, device=device).view(1, 1, num_query_blocks, 1)
            k_positions = torch.arange(num_key_blocks, device=device).view(1, 1, 1, num_key_blocks)
            cached = (q_positions - k_positions).abs() <= band_width
            if batch_size != 1 or num_heads != 1:
                cached = cached.expand(batch_size, num_heads, -1, -1).contiguous()
            self._local_band_mask_cache[cache_key] = cached
        return cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        time_section = self.runtime_state.time_section if self.runtime_state is not None else None
        with (time_section("inner_total") if time_section is not None else nullcontext()):
            q_work, k_work, v_work = q, k, v

            with (time_section("mask_prepare") if time_section is not None else nullcontext()):
                sink_mask_tag = "__base__"
                resolved_dynamic_mode, dynamic_sample_gap = self._resolve_dynamic_plan()
                base_runtime_mask = self._get_runtime_mask(
                    sink_mask=self.sink_mask,
                    mask_tag=sink_mask_tag,
                    batch_size=q_work.size(0),
                    num_heads=q_work.size(1),
                    seq_len_q=q_work.size(2),
                    seq_len_k=k_work.size(2),
                    device=q_work.device,
                )
            topup_mask = None
            dynamic_seq_len_k = None

            if resolved_dynamic_mode == "none":
                with (time_section("sparse_main") if time_section is not None else nullcontext()):
                    mixed_out, _ = _block_sparse_attn(
                        q_work,
                        k_work,
                        v_work,
                        base_runtime_mask,
                        return_lse=False,
                    )
            elif resolved_dynamic_mode == "pooled_kv":
                need_lse = self.config.mixing_mode == "adaptive"
                with (time_section("sparse_main") if time_section is not None else nullcontext()):
                    sink_out, sink_lse = _block_sparse_attn(
                        q_work,
                        k_work,
                        v_work,
                        base_runtime_mask,
                        return_lse=need_lse,
                    )
                with (time_section("comp_pool") if time_section is not None else nullcontext()):
                    prefix_tokens = max(int(self.config.text_length), 0)
                    k_dynamic = _mean_pool_sequence_preserve_prefix(
                        k_work,
                        dynamic_sample_gap,
                        prefix_tokens=prefix_tokens,
                    )
                    v_dynamic = _mean_pool_sequence_preserve_prefix(
                        v_work,
                        dynamic_sample_gap,
                        prefix_tokens=prefix_tokens,
                    )
                    dynamic_seq_len_k = int(k_dynamic.size(2))
                with (time_section("comp_attn") if time_section is not None else nullcontext()):
                    if self.config.dynamic_kernel == "dense" or (
                        self.config.dynamic_kernel == "auto" and self.config.mixing_mode != "adaptive"
                    ):
                        if self.config.mixing_mode == "adaptive":
                            dynamic_out, dynamic_lse = _dense_short_attention_with_lse(q_work, k_dynamic, v_dynamic)
                        else:
                            dynamic_out = _dense_short_attention(q_work, k_dynamic, v_dynamic)
                            dynamic_lse = None
                    else:
                        dynamic_mask = self._get_dynamic_mask(
                            batch_size=q_work.size(0),
                            num_heads=q_work.size(1),
                            seq_len_q=q_work.size(2),
                            seq_len_k=k_dynamic.size(2),
                            device=q_work.device,
                        )
                        dynamic_out, dynamic_lse = _block_sparse_attn(
                            q_work,
                            k_dynamic,
                            v_dynamic,
                            dynamic_mask,
                            return_lse=self.config.mixing_mode == "adaptive",
                        )
                with (time_section("mix") if time_section is not None else nullcontext()):
                    mixed_out = _mix_attention_outputs(
                        sink_out=sink_out,
                        sink_lse=sink_lse,
                        dynamic_out=dynamic_out,
                        dynamic_lse=dynamic_lse,
                        config=self.config,
                        dynamic_sample_gap=dynamic_sample_gap,
                    )
            else:
                raise ValueError(f"Unsupported dynamic_mode: {resolved_dynamic_mode}")

            if self.runtime_state is not None:
                with self.runtime_state.time_section("stats"):
                    self.runtime_state.record_mask_stats(
                        layer_idx=self.layer_idx,
                        sink_mask_tag=sink_mask_tag,
                        dynamic_mode=resolved_dynamic_mode,
                        dynamic_sample_gap=dynamic_sample_gap,
                        base_runtime_mask=base_runtime_mask,
                        topup_mask=topup_mask,
                        dynamic_seq_len_k=dynamic_seq_len_k,
                        total_seq_len_k=int(k_work.size(2)),
                    )

            return mixed_out


def _prepare_attention_tensors(
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    rotary_emb: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    timing = getattr(getattr(attn, "inner_attention", None), "timing", None)
    encoder_hidden_states_img = None
    if attn.add_k_proj is not None and encoder_hidden_states is not None:
        encoder_hidden_states_img = encoder_hidden_states[:, :257]
        encoder_hidden_states = encoder_hidden_states[:, 257:]

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    with (timing.section("qk_norm") if timing is not None and timing.is_enabled else nullcontext()):
        query, key = apply_fast_qk_norm(attn, query, key)

    query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

    if rotary_emb is not None:
        with (timing.section("rotary_apply") if timing is not None and timing.is_enabled else nullcontext()):
            query, key = apply_fast_rotary_qk(query, key, rotary_emb)

    return query, key, value, encoder_hidden_states_img


def _compute_image_branch(attn, query: torch.Tensor, encoder_hidden_states_img: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if encoder_hidden_states_img is None:
        return None

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
    return hidden_states_img.transpose(1, 2).flatten(2, 3).type_as(query)


class WanSinkSparseAttnProcessor2_0:
    def __init__(self, layer_idx: int, runtime_state: Optional[SinkRuntimeState] = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanSinkSparseAttnProcessor2_0 requires PyTorch 2.0+.")
        self.layer_idx = layer_idx
        self.runtime_state = runtime_state

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attention_mask
        if self.runtime_state is not None:
            self.runtime_state.enter_layer(self.layer_idx)

        query, key, value, encoder_hidden_states_img = _prepare_attention_tensors(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=rotary_emb,
        )
        hidden_states_img = _compute_image_branch(attn, query, encoder_hidden_states_img)

        hidden_states = attn.inner_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanSinkCalibrationProcessor2_0:
    def __init__(self, calibrator: SinkMaskCalibrator, layer_idx: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanSinkCalibrationProcessor2_0 requires PyTorch 2.0+.")
        self.calibrator = calibrator
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attention_mask

        query, key, value, encoder_hidden_states_img = _prepare_attention_tensors(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=rotary_emb,
        )

        self.calibrator.record(self.layer_idx, query, key, value)
        hidden_states_img = _compute_image_branch(attn, query, encoder_hidden_states_img)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_sink_calibration_attn_wanx(model, calibrator: SinkMaskCalibrator, verbose: bool = False) -> SinkMaskCalibrator:
    for layer_idx, block in enumerate(model.blocks):
        block.attn1.verbose = verbose
        origin_processor = block.attn1.get_processor()
        block.attn1.set_processor(WanSinkCalibrationProcessor2_0(calibrator=calibrator, layer_idx=layer_idx))
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor
    return calibrator


def set_sink_sparse_attn_wanx(
    model,
    sink_mask_path: str,
    config_overrides: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    package = load_sink_mask_package(sink_mask_path)
    config = SinkSparseConfig.from_dict(package.get("config")).merged(config_overrides)
    sink_masks = package.get("sink_masks", {})
    runtime_state = SinkRuntimeState()
    model._sink_runtime_state = runtime_state

    for layer_idx, block in enumerate(model.blocks):
        if str(layer_idx) not in sink_masks:
            raise KeyError(f"Layer {layer_idx} does not exist in sink mask package: {sink_mask_path}")

        sink_mask = sink_masks[str(layer_idx)]
        block.attn1.verbose = verbose
        block.attn1.inner_attention = WanSinkSparseInnerAttention(
            layer_idx=layer_idx,
            sink_mask=sink_mask,
            config=config,
            runtime_state=runtime_state,
        )
        origin_processor = block.attn1.get_processor()
        block.attn1.set_processor(WanSinkSparseAttnProcessor2_0(layer_idx=layer_idx, runtime_state=runtime_state))
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor

    return {
        "config": config,
        "num_layers": len(model.blocks),
        "mask_path": sink_mask_path,
    }
