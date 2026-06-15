from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sinkattention.sparse_attention import (
    SinkRuntimeState,
    SinkSparseConfig,
    _block_sparse_attn,
    _build_direct_mean_coverage_mask,
    _ceil_div,
    _compute_block_scores,
    _dense_short_attention_with_lse,
    _mean_pool_sequence,
    _mix_attention_outputs,
    load_sink_mask_package,
)
from zimg.runtime.modify_zimage import prepare_zimage_attention_tensors


SEQ_MULTI_OF = 32
DEFAULT_TEXT_LENGTH = 512


def _pad_len_to_multiple(value: int, multiple: int = SEQ_MULTI_OF) -> int:
    return _ceil_div(int(value), int(multiple)) * int(multiple)


def infer_zimg_token_layout(*, height: int, width: int) -> tuple[int, int, int]:
    token_width = max(int(width) // 16, 1)
    token_height = max(int(height) // 16, 1)
    token_depth = 1
    return token_width, token_height, token_depth


def _default_zimg_sink_config_overrides() -> dict:
    return {
        "token_depth": 1,
        "dynamic_mode": "none",
        "runtime_diagonal_band_width": 0,
        "force_text_global_attention": True,
    }


def build_zimg_sink_config(
    *,
    package_config: Optional[dict] = None,
    config_overrides: Optional[dict] = None,
) -> SinkSparseConfig:
    data = asdict(SinkSparseConfig())
    data.update(_default_zimg_sink_config_overrides())
    data["text_length"] = DEFAULT_TEXT_LENGTH
    if package_config:
        data.update({key: value for key, value in package_config.items() if value is not None})
    if config_overrides:
        data.update({key: value for key, value in config_overrides.items() if value is not None})
    config = SinkSparseConfig.from_dict(data)
    config.force_text_global_attention = True
    dynamic_mode = str(config.dynamic_mode).strip().lower()
    if dynamic_mode not in {"none", "pooled_kv"}:
        raise ValueError(
            "Unsupported Z-Image Sink dynamic_mode. "
            f"Expected one of: none, pooled_kv. Got: {config.dynamic_mode}"
        )
    mixing_mode = str(config.mixing_mode).strip().lower()
    if mixing_mode != "adaptive":
        raise ValueError(
            "Unsupported Z-Image Sink mixing_mode. "
            f"Expected adaptive. Got: {config.mixing_mode}"
        )
    config.dynamic_kernel = str(config.dynamic_kernel or "auto").strip().lower()
    return config


def _resolve_forced_dense_step_list(config: SinkSparseConfig, runtime_state: Optional[SinkRuntimeState]) -> set[int]:
    raw_steps = config.force_dense_steps
    if not raw_steps:
        return set()

    total_steps = None if runtime_state is None else runtime_state.total_steps
    resolved: set[int] = set()
    for raw_step in raw_steps:
        step_value = int(raw_step)
        if step_value < 0:
            if total_steps is None:
                continue
            step_value = int(total_steps) + step_value
        if step_value >= 0:
            resolved.add(step_value)
    return resolved


def _resolve_forced_dense_layer_list(
    config: SinkSparseConfig,
    *,
    total_layers: Optional[int] = None,
) -> set[int]:
    raw_layers = config.force_dense_layers
    if not raw_layers:
        return set()

    resolved: set[int] = set()
    for raw_layer in raw_layers:
        layer_value = int(raw_layer)
        if layer_value < 0:
            if total_layers is None:
                continue
            layer_value = int(total_layers) + layer_value
        if layer_value >= 0:
            resolved.add(layer_value)
    return resolved


def _image_token_count(config: SinkSparseConfig) -> int:
    raw = int(config.token_width) * int(config.token_height) * int(config.token_depth)
    return _pad_len_to_multiple(raw)


def _max_text_token_count(config: SinkSparseConfig) -> int:
    return _pad_len_to_multiple(int(config.text_length))


def _fixed_block_shape(config: SinkSparseConfig) -> tuple[int, int, int]:
    image_tokens = _image_token_count(config)
    image_blocks = _ceil_div(image_tokens, int(config.block_size))
    max_text_blocks = _ceil_div(_max_text_token_count(config), int(config.block_size))
    total_blocks = image_blocks + max_text_blocks
    return image_tokens, image_blocks, total_blocks


def _align_scores_image_text_suffix(
    scores: torch.Tensor,
    *,
    image_blocks: int,
    current_text_blocks: int,
    total_blocks: int,
) -> torch.Tensor:
    if scores.dim() != 3:
        raise ValueError(f"Expected [heads, blocks_q, blocks_k] scores, got {tuple(scores.shape)}")
    expected_blocks = image_blocks + current_text_blocks
    if scores.size(1) != expected_blocks or scores.size(2) != expected_blocks:
        raise ValueError(
            "Unexpected Z-Image block score shape. "
            f"Expected {expected_blocks}x{expected_blocks}, got {scores.size(1)}x{scores.size(2)}."
        )
    aligned = torch.zeros(
        scores.size(0),
        total_blocks,
        total_blocks,
        dtype=scores.dtype,
        device=scores.device,
    )
    aligned[:, :image_blocks, :image_blocks] = scores[:, :image_blocks, :image_blocks]
    if current_text_blocks <= 0:
        return aligned
    text_dst_start = total_blocks - current_text_blocks
    aligned[:, :image_blocks, text_dst_start:] = scores[:, :image_blocks, image_blocks:]
    aligned[:, text_dst_start:, :image_blocks] = scores[:, image_blocks:, :image_blocks]
    aligned[:, text_dst_start:, text_dst_start:] = scores[:, image_blocks:, image_blocks:]
    return aligned


def _alignment_presence_mask(
    *,
    num_heads: int,
    image_blocks: int,
    current_text_blocks: int,
    total_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(num_heads, total_blocks, total_blocks, dtype=torch.float32, device=device)
    mask[:, :image_blocks, :image_blocks] = 1.0
    if current_text_blocks <= 0:
        return mask
    text_dst_start = total_blocks - current_text_blocks
    mask[:, :image_blocks, text_dst_start:] = 1.0
    mask[:, text_dst_start:, :image_blocks] = 1.0
    mask[:, text_dst_start:, text_dst_start:] = 1.0
    return mask


def _slice_runtime_sink_mask(
    sink_mask: torch.Tensor,
    *,
    image_blocks: int,
    current_text_blocks: int,
    total_blocks: int,
) -> torch.Tensor:
    if sink_mask.dim() != 3:
        raise ValueError(f"Expected [heads, blocks_q, blocks_k] sink mask, got {tuple(sink_mask.shape)}")
    if sink_mask.size(1) != total_blocks or sink_mask.size(2) != total_blocks:
        raise ValueError(
            "Z-Image sink mask shape does not match the calibrated fixed block layout. "
            f"Expected {total_blocks}x{total_blocks}, got {sink_mask.size(1)}x{sink_mask.size(2)}."
        )
    if current_text_blocks <= 0:
        return sink_mask[:, :image_blocks, :image_blocks].contiguous()
    text_start = total_blocks - current_text_blocks
    block_indices = torch.cat(
        [
            torch.arange(image_blocks, device=sink_mask.device),
            torch.arange(text_start, total_blocks, device=sink_mask.device),
        ],
        dim=0,
    )
    return sink_mask.index_select(1, block_indices).index_select(2, block_indices).contiguous()


def _force_text_dense_runtime_mask(
    runtime_mask: torch.Tensor,
    *,
    image_blocks: int,
    current_text_blocks: int,
) -> torch.Tensor:
    if current_text_blocks <= 0:
        return runtime_mask
    text_start = int(image_blocks)
    dense_mask = runtime_mask.clone()
    dense_mask[..., :, text_start:] = True
    dense_mask[..., text_start:, :] = True
    return dense_mask


def _dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )


class ZImageSinkMaskCalibrator:
    def __init__(self, config: SinkSparseConfig):
        self.config = build_zimg_sink_config(package_config=asdict(config))
        self.current_step = -1
        self.score_sums: Dict[int, torch.Tensor] = {}
        self.score_sq_sums: Dict[int, torch.Tensor] = {}
        self.score_presence_sums: Dict[int, torch.Tensor] = {}
        self.score_counts: Dict[int, int] = {}
        self.image_token_count, self.image_blocks, self.total_blocks = _fixed_block_shape(self.config)
        self.max_text_tokens = _max_text_token_count(self.config)

    def reset_generation(self) -> None:
        self.current_step = -1

    def _accumulate(
        self,
        *,
        layer_idx: int,
        scores: torch.Tensor,
        presence: torch.Tensor,
    ) -> None:
        if layer_idx not in self.score_sums:
            self.score_sums[layer_idx] = scores
            self.score_sq_sums[layer_idx] = scores.square()
            self.score_presence_sums[layer_idx] = presence
            self.score_counts[layer_idx] = 1
            return
        self.score_sums[layer_idx] += scores
        self.score_sq_sums[layer_idx] += scores.square()
        self.score_presence_sums[layer_idx] += presence
        self.score_counts[layer_idx] += 1

    @torch.no_grad()
    def record(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        text_token_count: int,
    ) -> None:
        if layer_idx == 0:
            self.current_step += 1
        if text_token_count < 0:
            raise ValueError(f"Invalid Z-Image text token count: {text_token_count}")
        if text_token_count > self.max_text_tokens:
            raise ValueError(
                "Observed Z-Image text token count exceeds calibrated maximum. "
                f"Observed {text_token_count}, configured max {self.max_text_tokens}."
            )

        score_samples = _compute_block_scores(q, k, v, self.config).detach().float()
        if score_samples.size(0) != 1:
            raise ValueError("Z-Image calibration expects one sample at a time after valid-token slicing.")
        scores = score_samples.mean(dim=0).cpu()

        current_text_blocks = _ceil_div(int(text_token_count), int(self.config.block_size))
        aligned_scores = _align_scores_image_text_suffix(
            scores,
            image_blocks=self.image_blocks,
            current_text_blocks=current_text_blocks,
            total_blocks=self.total_blocks,
        )
        presence = _alignment_presence_mask(
            num_heads=int(aligned_scores.size(0)),
            image_blocks=self.image_blocks,
            current_text_blocks=current_text_blocks,
            total_blocks=self.total_blocks,
            device=aligned_scores.device,
        ).cpu()
        self._accumulate(layer_idx=layer_idx, scores=aligned_scores, presence=presence)

    def build_package(self, include_scores: bool = True, config_override: Optional[SinkSparseConfig] = None) -> dict:
        export_config = self.config if config_override is None else build_zimg_sink_config(package_config=asdict(config_override))
        mean_scores = {}
        std_scores = {}
        sink_masks = {}
        counts = {}

        for layer_idx in sorted(self.score_sums.keys()):
            key = str(layer_idx)
            presence = self.score_presence_sums[layer_idx].clamp_min_(1.0)
            mean_score = self.score_sums[layer_idx] / presence
            second_moment = self.score_sq_sums[layer_idx] / presence
            variance = (second_moment - mean_score.square()).clamp_min_(0.0)
            std_score = torch.sqrt(variance)
            mean_scores[key] = mean_score
            std_scores[key] = std_score
            sink_masks[key] = _build_direct_mean_coverage_mask(mean_score, export_config)
            counts[key] = int(self.score_counts[layer_idx])

        return {
            "version": 3,
            "config": asdict(export_config),
            "mean_scores": mean_scores if include_scores else {},
            "std_scores": std_scores if include_scores else {},
            "sink_masks": sink_masks,
            "counts": counts,
            "meta": {
                "route": "zimg_static_only",
                "text_layout": "image_prefix_text_suffix_aligned",
                "image_token_count": self.image_token_count,
                "max_text_token_count": self.max_text_tokens,
                "total_blocks": self.total_blocks,
            },
        }

    def save(
        self,
        output_path: str,
        include_scores: bool = True,
        config_override: Optional[SinkSparseConfig] = None,
    ) -> dict:
        package = self.build_package(include_scores=include_scores, config_override=config_override)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(package, output_file)
        return package


class ZImageSinkCalibrationProcessor:
    def __init__(self, calibrator: ZImageSinkMaskCalibrator, layer_idx: int, origin_processor: Any):
        self.calibrator = calibrator
        self.layer_idx = int(layer_idx)
        self.origin_processor = origin_processor

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise AssertionError("Z-Image uses single-stream attention only.")
        if self.origin_processor is None:
            raise RuntimeError("Z-Image calibration requires access to the original dense attention processor.")

        query, key, value = prepare_zimage_attention_tensors(attn, hidden_states, freqs_cis=freqs_cis)
        batch_size = int(hidden_states.size(0))
        image_token_count = self.calibrator.image_token_count

        if attention_mask is None:
            valid_lengths = [int(hidden_states.size(1))] * batch_size
        else:
            valid_lengths = [int(mask.sum().item()) for mask in attention_mask]

        for batch_idx, valid_len in enumerate(valid_lengths):
            if valid_len < image_token_count:
                raise ValueError(
                    "Observed valid sequence shorter than the padded image token prefix. "
                    f"valid_len={valid_len}, image_token_count={image_token_count}"
                )
            text_token_count = valid_len - image_token_count
            q_i = query[batch_idx : batch_idx + 1, :valid_len].permute(0, 2, 1, 3).contiguous()
            k_i = key[batch_idx : batch_idx + 1, :valid_len].permute(0, 2, 1, 3).contiguous()
            v_i = value[batch_idx : batch_idx + 1, :valid_len].permute(0, 2, 1, 3).contiguous()
            self.calibrator.record(self.layer_idx, q_i, k_i, v_i, text_token_count=text_token_count)

        return self.origin_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
        )


class ZImageStaticSinkProcessor:
    def __init__(
        self,
        *,
        layer_idx: int,
        total_layers: Optional[int],
        sink_mask: torch.Tensor,
        config: SinkSparseConfig,
        runtime_state: Optional[SinkRuntimeState] = None,
    ):
        self.layer_idx = int(layer_idx)
        self.total_layers = None if total_layers is None else int(total_layers)
        self.sink_mask = sink_mask.bool().cpu()
        self.config = config
        self.runtime_state = runtime_state
        self.image_token_count, self.image_blocks, self.total_blocks = _fixed_block_shape(self.config)
        self.max_text_tokens = _max_text_token_count(self.config)

    def _resolve_dynamic_plan(self) -> tuple[str, int]:
        dynamic_mode = str(self.config.dynamic_mode).strip().lower()
        if dynamic_mode not in {"none", "pooled_kv"}:
            raise ValueError(
                "Unsupported Z-Image Sink dynamic_mode. "
                f"Expected one of: none, pooled_kv. Got: {self.config.dynamic_mode}"
            )

        sample_gap = max(int(self.config.sample_gap), 1)
        if dynamic_mode == "none":
            return "none", sample_gap

        return dynamic_mode, sample_gap

    def _should_force_dense_current_step(self) -> bool:
        if self.runtime_state is None:
            return False
        current_step = int(self.runtime_state.current_step)
        if current_step < 0:
            return False
        forced_dense_steps = _resolve_forced_dense_step_list(self.config, self.runtime_state)
        return current_step in forced_dense_steps

    def _should_force_dense_current_layer(self) -> bool:
        forced_dense_layers = _resolve_forced_dense_layer_list(
            self.config,
            total_layers=self.total_layers,
        )
        return self.layer_idx in forced_dense_layers

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise AssertionError("Z-Image uses single-stream attention only.")

        if self.runtime_state is not None:
            self.runtime_state.enter_layer(self.layer_idx)
        timing = self.runtime_state.time_section if self.runtime_state is not None else None

        with (timing("inner_total") if timing is not None else nullcontext()):
            with (timing("qkv") if timing is not None else nullcontext()):
                query, key, value = prepare_zimage_attention_tensors(attn, hidden_states, freqs_cis=freqs_cis)

            batch_size = int(hidden_states.size(0))
            max_seq_len = int(hidden_states.size(1))
            head_dim = int(query.size(-1))
            out_padded = hidden_states.new_zeros((batch_size, max_seq_len, attn.heads, head_dim))
            resolved_dynamic_mode, dynamic_sample_gap = self._resolve_dynamic_plan()
            force_dense_step = self._should_force_dense_current_step()
            force_dense_layer = self._should_force_dense_current_layer()
            force_dense = force_dense_step or force_dense_layer

            if attention_mask is None:
                valid_lengths = [max_seq_len] * batch_size
            else:
                valid_lengths = [int(mask.sum().item()) for mask in attention_mask]

            for batch_idx, valid_len in enumerate(valid_lengths):
                if valid_len < self.image_token_count:
                    raise ValueError(
                        "Observed valid sequence shorter than the padded image token prefix. "
                        f"valid_len={valid_len}, image_token_count={self.image_token_count}"
                    )
                text_token_count = valid_len - self.image_token_count
                if text_token_count > self.max_text_tokens:
                    raise ValueError(
                        "Observed Z-Image text token count exceeds the mask package maximum. "
                        f"Observed {text_token_count}, configured max {self.max_text_tokens}."
                    )

                current_text_blocks = _ceil_div(int(text_token_count), int(self.config.block_size))
                runtime_mask = _slice_runtime_sink_mask(
                    self.sink_mask.to(device=query.device),
                    image_blocks=self.image_blocks,
                    current_text_blocks=current_text_blocks,
                    total_blocks=self.total_blocks,
                )
                runtime_mask = _force_text_dense_runtime_mask(
                    runtime_mask,
                    image_blocks=self.image_blocks,
                    current_text_blocks=current_text_blocks,
                ).unsqueeze(0)

                q_i = query[batch_idx : batch_idx + 1, :valid_len].permute(0, 2, 1, 3).contiguous()
                k_i = key[batch_idx : batch_idx + 1, :valid_len].permute(0, 2, 1, 3).contiguous()
                v_i = value[batch_idx : batch_idx + 1, :valid_len].permute(0, 2, 1, 3).contiguous()

                dynamic_seq_len_k = None
                if force_dense:
                    with (timing("dense_override") if timing is not None else nullcontext()):
                        out_i = _dense_attention(q_i, k_i, v_i)
                    runtime_mask = torch.ones_like(runtime_mask, dtype=torch.bool)
                    resolved_runtime_mode = "forced_dense_step" if force_dense_step else "forced_dense_layer"
                else:
                    need_lse = resolved_dynamic_mode == "pooled_kv"
                    with (timing("sparse_main") if timing is not None else nullcontext()):
                        out_i, sink_lse = _block_sparse_attn(q_i, k_i, v_i, runtime_mask, return_lse=need_lse)
                    resolved_runtime_mode = resolved_dynamic_mode

                if not force_dense and resolved_dynamic_mode == "pooled_kv":
                    image_len = self.image_token_count
                    image_q = q_i[:, :, :image_len, :]
                    image_k = k_i[:, :, :image_len, :]
                    image_v = v_i[:, :, :image_len, :]
                    text_k = k_i[:, :, image_len:valid_len, :]
                    text_v = v_i[:, :, image_len:valid_len, :]

                    with (timing("comp_pool") if timing is not None else nullcontext()):
                        pooled_image_k = _mean_pool_sequence(image_k, dynamic_sample_gap)
                        pooled_image_v = _mean_pool_sequence(image_v, dynamic_sample_gap)
                        if text_token_count > 0:
                            dynamic_k = torch.cat([pooled_image_k, text_k], dim=2)
                            dynamic_v = torch.cat([pooled_image_v, text_v], dim=2)
                        else:
                            dynamic_k = pooled_image_k
                            dynamic_v = pooled_image_v
                        dynamic_seq_len_k = int(dynamic_k.size(2))

                    with (timing("comp_attn") if timing is not None else nullcontext()):
                        if self.config.dynamic_kernel == "dense":
                            dynamic_image_out, dynamic_lse = _dense_short_attention_with_lse(
                                image_q,
                                dynamic_k,
                                dynamic_v,
                            )
                        else:
                            dynamic_mask = torch.ones(
                                image_q.size(0),
                                image_q.size(1),
                                _ceil_div(image_q.size(2), int(self.config.block_size)),
                                _ceil_div(dynamic_k.size(2), int(self.config.block_size)),
                                dtype=torch.bool,
                                device=image_q.device,
                            )
                            dynamic_image_out, dynamic_lse = _block_sparse_attn(
                                image_q,
                                dynamic_k,
                                dynamic_v,
                                dynamic_mask,
                                return_lse=True,
                            )

                    with (timing("mix") if timing is not None else nullcontext()):
                        mixed_image = _mix_attention_outputs(
                            sink_out=out_i[:, :, :image_len, :],
                            sink_lse=sink_lse[:, :, :image_len, :],
                            dynamic_out=dynamic_image_out,
                            dynamic_lse=dynamic_lse,
                            config=self.config,
                            dynamic_sample_gap=dynamic_sample_gap,
                        )
                        out_i = out_i.clone()
                        out_i[:, :, :image_len, :] = mixed_image

                out_padded[batch_idx, :valid_len] = out_i.transpose(1, 2)[0]
                if self.runtime_state is not None:
                    self.runtime_state.record_mask_stats(
                        layer_idx=self.layer_idx,
                        sink_mask_tag="__base__",
                        dynamic_mode=resolved_runtime_mode,
                        dynamic_sample_gap=0 if force_dense else int(dynamic_sample_gap),
                        base_runtime_mask=runtime_mask,
                        topup_mask=None,
                        dynamic_seq_len_k=dynamic_seq_len_k,
                        total_seq_len_k=valid_len,
                    )

            with (timing("proj_out") if timing is not None else nullcontext()):
                hidden_states = out_padded.flatten(2, 3).to(dtype=hidden_states.dtype)
                hidden_states = attn.to_out[0](hidden_states)
                if len(attn.to_out) > 1:
                    hidden_states = attn.to_out[1](hidden_states)
            return hidden_states


def set_sink_calibration_attn_zimage(
    model: Any,
    calibrator: ZImageSinkMaskCalibrator,
) -> ZImageSinkMaskCalibrator:
    build_zimg_sink_config(package_config=asdict(calibrator.config))
    for layer_idx, block in enumerate(model.layers):
        origin_processor = block.attention.get_processor()
        block.attention.set_processor(
            ZImageSinkCalibrationProcessor(
                calibrator=calibrator,
                layer_idx=layer_idx,
                origin_processor=origin_processor,
            )
        )
        if not hasattr(block.attention, "origin_processor"):
            block.attention.origin_processor = origin_processor
    return calibrator


def set_sink_sparse_attn_zimage(
    model: Any,
    sink_mask_path: str,
    config_overrides: Optional[dict] = None,
) -> dict:
    package = load_sink_mask_package(sink_mask_path)
    config = build_zimg_sink_config(
        package_config=package.get("config"),
        config_overrides=config_overrides,
    )
    runtime_state = SinkRuntimeState()
    model._sink_runtime_state = runtime_state

    sink_masks = package.get("sink_masks", {})
    num_layers = len(model.layers)
    for layer_idx, block in enumerate(model.layers):
        layer_key = str(layer_idx)
        if layer_key not in sink_masks:
            raise KeyError(f"Layer {layer_idx} does not exist in sink mask package: {sink_mask_path}")
        origin_processor = block.attention.get_processor()
        block.attention.set_processor(
            ZImageStaticSinkProcessor(
                layer_idx=layer_idx,
                total_layers=num_layers,
                sink_mask=sink_masks[layer_key],
                config=config,
                runtime_state=runtime_state,
            )
        )
        if not hasattr(block.attention, "origin_processor"):
            block.attention.origin_processor = origin_processor

    return {
        "config": config,
        "num_layers": num_layers,
        "mask_path": sink_mask_path,
        "route": "zimg_static_only" if str(config.dynamic_mode).strip().lower() == "none" else "zimg_static_comp",
    }


def reset_sink_runtime_state_zimage(
    model: Any,
    num_inference_steps: Optional[int] = None,
) -> None:
    runtime_state = getattr(model, "_sink_runtime_state", None)
    if runtime_state is not None:
        runtime_state.reset_generation(total_steps=num_inference_steps)


__all__ = [
    "DEFAULT_TEXT_LENGTH",
    "SinkRuntimeState",
    "SinkSparseConfig",
    "ZImageSinkMaskCalibrator",
    "build_zimg_sink_config",
    "infer_zimg_token_layout",
    "reset_sink_runtime_state_zimage",
    "set_sink_calibration_attn_zimage",
    "set_sink_sparse_attn_zimage",
]
