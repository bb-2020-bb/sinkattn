from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from sinkattention.timing import AttentionTimingCollector

try:
    from sinkattention.pooling_kernel import attn_with_pooling

    _HAS_TRITON_POOLING = True
except Exception:
    attn_with_pooling = None
    _HAS_TRITON_POOLING = False

try:
    from block_sparse_attn import block_sparse_attn_func

    _HAS_BLOCK_SPARSE = True
except Exception:
    block_sparse_attn_func = None
    _HAS_BLOCK_SPARSE = False

_BLOCK_SPARSE_RUNTIME_CACHE: Dict[Tuple[str, int, int, int, int, int], tuple[torch.Tensor, ...]] = {}
_EXACT_BLOCK_SCORE_KEY_CHUNK_BLOCKS = 16
_BLOCK_SPARSE_KERNEL_BLOCK_SIZE = 128


@dataclass
class SinkSparseConfig:
    block_size: int = 128
    sample_gap: int = 30
    token_width: int = 52
    token_height: int = 30
    token_depth: int = 21
    text_length: int = 0
    force_text_global_attention: bool = False
    offline_text_video_coverage_mode: str = "joint"
    keep_diagonal: bool = True
    runtime_diagonal_band_width: int = 0
    pooling_mode: str = "triton"
    dynamic_mode: str = "pooled_kv"
    dynamic_kernel: str = "auto"
    mixing_mode: str = "adaptive"
    force_dense_steps: Optional[list[int]] = None
    force_dense_layers: Optional[list[int]] = None
    offline_direct_coverage: float = 0.85

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SinkSparseConfig":
        if not data:
            return cls()
        valid_keys = {field_name for field_name in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def merged(self, overrides: Optional[dict] = None) -> "SinkSparseConfig":
        if not overrides:
            return SinkSparseConfig.from_dict(asdict(self))
        data = asdict(self)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return SinkSparseConfig.from_dict(data)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _device_cache_key(device: torch.device) -> tuple[str, int]:
    return device.type, -1 if device.index is None else int(device.index)


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    length = x.size(2)
    remainder = length % multiple
    if remainder == 0:
        return x
    pad_len = multiple - remainder
    return F.pad(x, (0, 0, 0, pad_len), mode="replicate")


def _mean_pool_sequence(x: torch.Tensor, pool_size: int) -> torch.Tensor:
    if pool_size <= 1:
        return x
    x = _pad_to_multiple(x, pool_size)
    bsz, num_heads, seq_len, dim = x.shape
    x = x.view(bsz, num_heads, seq_len // pool_size, pool_size, dim)
    return x.mean(dim=-2)


def _mean_pool_sequence_preserve_prefix(
    x: torch.Tensor,
    pool_size: int,
    *,
    prefix_tokens: int = 0,
) -> torch.Tensor:
    if pool_size <= 1:
        return x
    prefix_tokens = min(max(int(prefix_tokens), 0), int(x.size(2)))
    if prefix_tokens <= 0:
        return _mean_pool_sequence(x, pool_size)
    if prefix_tokens >= int(x.size(2)):
        return x

    prefix = x[:, :, :prefix_tokens, :]
    tail = x[:, :, prefix_tokens:, :]
    if tail.size(2) == 0:
        return prefix
    return torch.cat([prefix, _mean_pool_sequence(tail, pool_size)], dim=2)


def _compute_pooled_qk_block_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: SinkSparseConfig,
) -> torch.Tensor:
    if config.pooling_mode == "triton" and _HAS_TRITON_POOLING and q.is_cuda:
        try:
            _, pooled = attn_with_pooling(
                q,
                k,
                v,
                False,
                1.0 / math.sqrt(q.size(-1)),
                config.block_size,
            )
            return pooled.float()
        except RuntimeError:
            pass

    q_pool = _mean_pool_sequence(q.float(), config.block_size)
    k_pool = _mean_pool_sequence(k.float(), config.block_size)
    scores = torch.matmul(q_pool, k_pool.transpose(-1, -2))
    scores = scores * (1.0 / math.sqrt(q.size(-1)))
    return torch.softmax(scores, dim=-1)


def _compute_exact_block_mass_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    config: SinkSparseConfig,
) -> torch.Tensor:
    batch_size, num_heads, q_len, head_dim = q.shape
    k_len = int(k.size(2))
    block_size = int(config.block_size)
    num_query_blocks = _ceil_div(int(q_len), block_size)
    num_key_blocks = _ceil_div(int(k_len), block_size)
    key_chunk_blocks = max(1, min(int(_EXACT_BLOCK_SCORE_KEY_CHUNK_BLOCKS), num_key_blocks))
    scale = 1.0 / math.sqrt(head_dim)

    q_work = q.float()
    k_work = k.float()
    scores = torch.empty(
        batch_size,
        num_heads,
        num_query_blocks,
        num_key_blocks,
        device=q.device,
        dtype=torch.float32,
    )

    for query_block_idx in range(num_query_blocks):
        q_start = query_block_idx * block_size
        q_end = min(q_start + block_size, int(q_len))
        q_block = q_work[:, :, q_start:q_end, :]
        query_tokens = int(q_block.size(2))

        row_max = torch.full(
            (batch_size, num_heads, query_tokens),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )

        for key_block_start in range(0, num_key_blocks, key_chunk_blocks):
            chunk_block_count = min(key_chunk_blocks, num_key_blocks - key_block_start)
            k_start = key_block_start * block_size
            k_end = min(k_start + chunk_block_count * block_size, k_len)
            k_chunk = k_work[:, :, k_start:k_end, :]
            logits = torch.matmul(q_block, k_chunk.transpose(-1, -2))
            logits = logits * scale
            row_max = torch.maximum(row_max, logits.amax(dim=-1))

        row_sum = torch.zeros_like(row_max)
        query_block_mass = torch.empty(
            batch_size,
            num_heads,
            query_tokens,
            num_key_blocks,
            device=q.device,
            dtype=torch.float32,
        )

        for key_block_start in range(0, num_key_blocks, key_chunk_blocks):
            chunk_block_count = min(key_chunk_blocks, num_key_blocks - key_block_start)
            k_start = key_block_start * block_size
            k_end = min(k_start + chunk_block_count * block_size, k_len)
            k_chunk = k_work[:, :, k_start:k_end, :]
            logits = torch.matmul(q_block, k_chunk.transpose(-1, -2))
            logits = logits * scale
            probs_unnorm = torch.exp(logits - row_max.unsqueeze(-1))
            row_sum += probs_unnorm.sum(dim=-1)

            chunk_token_count = int(k_chunk.size(2))
            padded_chunk_tokens = chunk_block_count * block_size
            if padded_chunk_tokens != chunk_token_count:
                probs_unnorm = F.pad(probs_unnorm, (0, padded_chunk_tokens - chunk_token_count), value=0.0)
            block_mass = probs_unnorm.view(
                batch_size,
                num_heads,
                query_tokens,
                chunk_block_count,
                block_size,
            ).sum(dim=-1)
            query_block_mass[:, :, :, key_block_start:key_block_start + chunk_block_count] = block_mass

        query_block_mass = query_block_mass / row_sum.clamp_min_(1e-6).unsqueeze(-1)
        scores[:, :, query_block_idx, :] = query_block_mass.mean(dim=2)

    return scores


def _compute_block_scores(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: SinkSparseConfig) -> torch.Tensor:
    pooling_mode = str(config.pooling_mode).strip().lower()
    if pooling_mode == "exact":
        return _compute_exact_block_mass_scores(q, k, config)
    if pooling_mode in {"triton", "mean"}:
        return _compute_pooled_qk_block_scores(q, k, v, config)
    raise ValueError(
        f"Unsupported pooling_mode: {config.pooling_mode}. Expected one of: exact, triton, mean."
    )


def _build_offline_required_mask(
    *,
    num_query_blocks: int,
    num_key_blocks: int,
    config: SinkSparseConfig,
    device: torch.device,
) -> Optional[torch.Tensor]:
    required_mask = None

    if bool(getattr(config, "force_text_global_attention", False)) and int(config.text_length) > 0:
        text_blocks = _ceil_div(int(config.text_length), int(config.block_size))
        text_query_blocks = min(text_blocks, int(num_query_blocks))
        text_key_blocks = min(text_blocks, int(num_key_blocks))
        if text_query_blocks > 0 and text_key_blocks > 0:
            text_required = torch.zeros(
                num_query_blocks,
                num_key_blocks,
                dtype=torch.bool,
                device=device,
            )
            text_required[:, :text_key_blocks] = True
            text_required[:text_query_blocks, :] = True
            required_mask = text_required

    return required_mask


def _build_direct_mean_coverage_mask(
    mean_scores: torch.Tensor,
    config: SinkSparseConfig,
) -> torch.Tensor:
    if mean_scores.dim() != 3:
        raise ValueError(f"Expected [heads, blocks_q, blocks_k] scores, got {tuple(mean_scores.shape)}")

    num_heads, num_query_blocks, num_key_blocks = mean_scores.shape
    coverage = min(max(float(config.offline_direct_coverage), 0.0), 1.0)
    text_video_mode = str(getattr(config, "offline_text_video_coverage_mode", "joint")).strip().lower()
    required_mask = _build_offline_required_mask(
        num_query_blocks=num_query_blocks,
        num_key_blocks=num_key_blocks,
        config=config,
        device=mean_scores.device,
    )

    scores = mean_scores.float()
    if (
        text_video_mode in {"video_residual", "video_residual_only", "residual_video"}
        and bool(getattr(config, "force_text_global_attention", False))
        and int(getattr(config, "text_length", 0)) > 0
    ):
        text_blocks = min(_ceil_div(int(config.text_length), int(config.block_size)), num_query_blocks, num_key_blocks)
        if text_blocks > 0:
            mask = torch.zeros_like(mean_scores, dtype=torch.bool)
            if required_mask is not None:
                mask |= required_mask.unsqueeze(0).expand(num_heads, -1, -1)

            video_query_count = max(num_query_blocks - text_blocks, 0)
            video_key_count = max(num_key_blocks - text_blocks, 0)
            if video_query_count > 0 and video_key_count > 0:
                video_scores = scores[:, text_blocks:, text_blocks:]
                video_total_scores = video_scores.sum(dim=-1).clamp_min_(1e-6)
                video_target_scores = coverage * video_total_scores
                sorted_scores, sorted_indices = torch.sort(video_scores, dim=-1, descending=True)
                cumulative_scores = torch.cumsum(sorted_scores, dim=-1)
                reached = cumulative_scores >= video_target_scores.unsqueeze(-1)
                k_indices = torch.where(
                    video_target_scores > 1e-6,
                    torch.where(
                        reached.any(dim=-1),
                        reached.float().argmax(dim=-1) + 1,
                        torch.full(
                            (num_heads, video_query_count),
                            video_key_count,
                            dtype=torch.int64,
                            device=mean_scores.device,
                        ),
                    ),
                    torch.zeros((num_heads, video_query_count), dtype=torch.int64, device=mean_scores.device),
                )
                k_indices = k_indices.clamp(min=0, max=video_key_count)
                positions = torch.arange(video_key_count, device=mean_scores.device).view(1, 1, -1)
                keep_in_sorted_order = positions < k_indices.unsqueeze(-1)
                video_mask = torch.zeros_like(video_scores, dtype=torch.bool)
                video_mask.scatter_(-1, sorted_indices, keep_in_sorted_order)
                mask[:, text_blocks:, text_blocks:] |= video_mask

            empty_rows = ~mask.any(dim=-1)
            if empty_rows.any():
                top1 = scores.argmax(dim=-1, keepdim=True)
                mask.scatter_(-1, top1, empty_rows.unsqueeze(-1))

            return mask

    total_scores = scores.sum(dim=-1).clamp_min_(1e-6)
    mask = torch.zeros_like(mean_scores, dtype=torch.bool)
    if required_mask is not None:
        mask |= required_mask.unsqueeze(0).expand(num_heads, -1, -1)

    selected_scores = (scores * mask.float()).sum(dim=-1)
    target_scores = coverage * total_scores
    residual_target = (target_scores - selected_scores).clamp_min_(0.0)

    candidate_scores = scores.masked_fill(mask, float("-inf"))
    sorted_scores, sorted_indices = torch.sort(candidate_scores, dim=-1, descending=True)
    sorted_scores = torch.where(torch.isfinite(sorted_scores), sorted_scores, torch.zeros_like(sorted_scores))
    cumulative_scores = torch.cumsum(sorted_scores, dim=-1)
    reached = cumulative_scores >= residual_target.unsqueeze(-1)
    candidate_counts = (~mask).sum(dim=-1).to(dtype=torch.int64)
    k_indices = torch.where(
        residual_target > 1e-6,
        torch.where(
            reached.any(dim=-1),
            reached.float().argmax(dim=-1) + 1,
            candidate_counts,
        ),
        torch.zeros_like(candidate_counts),
    )
    k_indices = k_indices.clamp(min=0, max=num_key_blocks)

    positions = torch.arange(num_key_blocks, device=mean_scores.device).view(1, 1, -1)
    keep_in_sorted_order = positions < k_indices.unsqueeze(-1)
    extra_mask = torch.zeros_like(mean_scores, dtype=torch.bool)
    extra_mask.scatter_(-1, sorted_indices, keep_in_sorted_order)
    mask |= extra_mask

    if config.keep_diagonal:
        diag_len = min(mask.size(-2), mask.size(-1))
        diag = torch.arange(diag_len, device=mask.device)
        mask[:, diag, diag] = True

    empty_rows = ~mask.any(dim=-1)
    if empty_rows.any():
        top1 = sorted_indices[..., :1]
        mask.scatter_(-1, top1, empty_rows.unsqueeze(-1))

    return mask


class SinkRuntimeState:
    def __init__(self):
        self.current_step = -1
        self.total_steps: Optional[int] = None
        self.sparsity_events: list[dict] = []
        self.timing = AttentionTimingCollector(label="sink")

    def reset_generation(self, total_steps: Optional[int] = None) -> None:
        self.current_step = -1
        self.total_steps = total_steps
        self.sparsity_events = []
        self.timing.reset()

    def time_section(self, name: str):
        return self.timing.section(name)

    def summarize_timing(self) -> dict:
        return self.timing.summarize()

    def enter_layer(self, layer_idx: int) -> None:
        if layer_idx == 0:
            self.current_step += 1

    @staticmethod
    def _mask_density(mask: Optional[torch.Tensor]) -> float:
        if mask is None:
            return 0.0
        total = mask.numel()
        if total <= 0:
            return 0.0
        selected = int(mask.sum().item())
        return float(selected) / float(total)

    @staticmethod
    def _mean_metric(events: list[dict], key: str) -> float:
        if not events:
            return 0.0
        return float(sum(float(event[key]) for event in events) / len(events))

    def record_mask_stats(
        self,
        *,
        layer_idx: int,
        sink_mask_tag: str,
        dynamic_mode: str,
        dynamic_sample_gap: int,
        base_runtime_mask: torch.Tensor,
        topup_mask: Optional[torch.Tensor] = None,
        dynamic_seq_len_k: Optional[int] = None,
        total_seq_len_k: Optional[int] = None,
    ) -> None:
        base_density = self._mask_density(base_runtime_mask)
        topup_density = self._mask_density(topup_mask)
        combined_density = self._mask_density(
            base_runtime_mask if topup_mask is None else (base_runtime_mask | topup_mask)
        )
        dynamic_key_ratio = 0.0
        if dynamic_seq_len_k is not None and total_seq_len_k is not None and total_seq_len_k > 0:
            dynamic_key_ratio = float(dynamic_seq_len_k) / float(total_seq_len_k)
        dense_equivalent_density = combined_density + dynamic_key_ratio
        self.sparsity_events.append(
            {
                'step': int(self.current_step),
                'layer_idx': int(layer_idx),
                'sink_mask_tag': str(sink_mask_tag),
                'dynamic_mode': str(dynamic_mode),
                'dynamic_sample_gap': int(dynamic_sample_gap),
                'base_density': base_density,
                'base_sparsity': 1.0 - base_density,
                'topup_density': topup_density,
                'combined_density': combined_density,
                'combined_sparsity': 1.0 - combined_density,
                'dynamic_key_ratio': dynamic_key_ratio,
                'dense_equivalent_density': dense_equivalent_density,
                'dense_equivalent_sparsity': 1.0 - dense_equivalent_density,
                'dynamic_seq_len_k': None if dynamic_seq_len_k is None else int(dynamic_seq_len_k),
                'total_seq_len_k': None if total_seq_len_k is None else int(total_seq_len_k),
            }
        )

    def _summarize_events(self, events: list[dict]) -> dict:
        if not events:
            return {}
        return {
            'count': len(events),
            'base_density_mean': self._mean_metric(events, 'base_density'),
            'base_sparsity_mean': self._mean_metric(events, 'base_sparsity'),
            'topup_density_mean': self._mean_metric(events, 'topup_density'),
            'combined_density_mean': self._mean_metric(events, 'combined_density'),
            'combined_sparsity_mean': self._mean_metric(events, 'combined_sparsity'),
            'dynamic_key_ratio_mean': self._mean_metric(events, 'dynamic_key_ratio'),
            'dense_equivalent_density_mean': self._mean_metric(events, 'dense_equivalent_density'),
            'dense_equivalent_sparsity_mean': self._mean_metric(events, 'dense_equivalent_sparsity'),
        }

    def summarize_sparsity(self) -> dict:
        if not self.sparsity_events:
            return {
                'num_events': 0,
                'total_steps_expected': self.total_steps,
                'steps_recorded': [],
                'layers_recorded': [],
                'overall': {},
                'by_step': {},
                'by_layer': {},
                'by_dynamic_mode': {},
                'by_sink_mask_tag': {},
            }

        steps = sorted({int(event['step']) for event in self.sparsity_events})
        layers = sorted({int(event['layer_idx']) for event in self.sparsity_events})
        dynamic_modes = sorted({str(event['dynamic_mode']) for event in self.sparsity_events})
        sink_mask_tags = sorted({str(event['sink_mask_tag']) for event in self.sparsity_events})

        return {
            'num_events': len(self.sparsity_events),
            'total_steps_expected': self.total_steps,
            'steps_recorded': steps,
            'layers_recorded': layers,
            'overall': self._summarize_events(self.sparsity_events),
            'by_step': {
                str(step): self._summarize_events(
                    [event for event in self.sparsity_events if int(event['step']) == step]
                )
                for step in steps
            },
            'by_layer': {
                str(layer_idx): self._summarize_events(
                    [event for event in self.sparsity_events if int(event['layer_idx']) == layer_idx]
                )
                for layer_idx in layers
            },
            'by_dynamic_mode': {
                mode: self._summarize_events(
                    [event for event in self.sparsity_events if str(event['dynamic_mode']) == mode]
                )
                for mode in dynamic_modes
            },
            'by_sink_mask_tag': {
                tag: self._summarize_events(
                    [event for event in self.sparsity_events if str(event['sink_mask_tag']) == tag]
                )
                for tag in sink_mask_tags
            },
        }


class SinkMaskCalibrator:
    def __init__(self, config: SinkSparseConfig):
        self.config = config
        self.current_step = -1
        self.score_sums: Dict[int, torch.Tensor] = {}
        self.score_sq_sums: Dict[int, torch.Tensor] = {}
        self.score_counts: Dict[int, int] = {}

    def reset_generation(self) -> None:
        self.current_step = -1

    def _accumulate_score(
        self,
        layer_idx: int,
        scores: torch.Tensor,
    ) -> None:
        if layer_idx not in self.score_sums:
            self.score_sums[layer_idx] = scores
            self.score_sq_sums[layer_idx] = scores.square()
            self.score_counts[layer_idx] = 1
        else:
            self.score_sums[layer_idx] += scores
            self.score_sq_sums[layer_idx] += scores.square()
            self.score_counts[layer_idx] += 1

    def _build_mean_and_masks(
        self,
        config: Optional[SinkSparseConfig] = None,
    ) -> tuple[dict, dict, dict, dict]:
        config = self.config if config is None else config
        mean_scores = {}
        std_scores = {}
        sink_masks = {}
        counts = {}

        for layer_idx in sorted(self.score_sums.keys()):
            key = str(layer_idx)
            mean_score = self.score_sums[layer_idx] / self.score_counts[layer_idx]
            second_moment = self.score_sq_sums[layer_idx] / self.score_counts[layer_idx]
            variance = (second_moment - mean_score.square()).clamp_min_(0.0)
            std_score = torch.sqrt(variance)

            mean_scores[key] = mean_score
            std_scores[key] = std_score
            sink_masks[key] = _build_direct_mean_coverage_mask(mean_score, config)
            counts[key] = int(self.score_counts[layer_idx])

        return mean_scores, std_scores, sink_masks, counts

    @torch.no_grad()
    def record(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if layer_idx == 0:
            self.current_step += 1

        score_samples = _compute_block_scores(q, k, v, self.config).detach().float()
        scores = score_samples.mean(dim=0).cpu()
        self._accumulate_score(layer_idx, scores)

    def build_package(self, include_scores: bool = True, config_override: Optional[SinkSparseConfig] = None) -> dict:
        export_config = self.config if config_override is None else config_override
        mean_scores, std_scores, sink_masks, counts = self._build_mean_and_masks(config=export_config)

        return {
            "version": 3,
            "config": asdict(export_config),
            "mean_scores": mean_scores if include_scores else {},
            "std_scores": std_scores if include_scores else {},
            "sink_masks": sink_masks,
            "counts": counts,
            "meta": {"route": "wanx_direct_coverage"},
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


def load_sink_mask_package(mask_path: str) -> dict:
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Sink mask file not found: {mask_path}")
    try:
        package = torch.load(mask_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        try:
            with open(mask_path, "rb") as handle:
                prefix = handle.read(128)
        except OSError:
            prefix = b""
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                "Sink mask file is still a Git LFS pointer, not the real package: "
                f"{mask_path}. Download the actual .pt asset before running Sink-Sparse."
            ) from exc
        raise RuntimeError(f"Failed to load sink mask package: {mask_path}: {exc}") from exc

    if not isinstance(package, dict):
        raise TypeError(
            f"Sink mask package must deserialize to a dict, got {type(package).__name__}: {mask_path}"
        )

    sink_masks = package.get("sink_masks")
    if not isinstance(sink_masks, dict) or not sink_masks:
        raise ValueError(f"Sink mask package does not contain usable sink_masks: {mask_path}")

    raw_config = package.get("config") or {}
    if not isinstance(raw_config, dict):
        raise TypeError(f"Sink mask package config must be a dict: {mask_path}")

    def _reject(reason: str) -> None:
        raise ValueError(
            "Incompatible Sink mask package for the current SinkAttention runtime: "
            f"{mask_path}. {reason} Current runtime only supports one shared-step mask in "
            "the original token order. Re-calibrate or re-export a no-rearrange shared-step package."
        )

    if bool(raw_config.get("use_rearrange", False)):
        _reject("The package was calibrated with use_rearrange=True.")

    if package.get("step_groups"):
        _reject("The package still contains step-group metadata.")

    if package.get("step_group_sink_masks"):
        _reject("The package still contains per-step-group mask banks.")

    if bool(raw_config.get("use_step_mask_bank", False)):
        _reject("The package config still enables the legacy step mask bank.")

    for legacy_key in ("step_group_boundaries", "step_group_names"):
        legacy_value = raw_config.get(legacy_key)
        if legacy_value not in (None, False, "", [], {}):
            _reject(f"The package config still carries populated legacy field {legacy_key}.")

    return package


def _ensure_block_sparse_available() -> None:
    if not _HAS_BLOCK_SPARSE:
        raise ImportError(
            "Sink-Sparse online inference requires the Block-Sparse-Attention package. "
            "Install it before using --attn_mode sink."
        )


def _get_block_sparse_runtime_tensors(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (*_device_cache_key(device), batch_size, num_heads, seq_len_q, seq_len_k)
    cached = _BLOCK_SPARSE_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * seq_len_q,
        step=seq_len_q,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * seq_len_k,
        step=seq_len_k,
        dtype=torch.int32,
        device=device,
    )
    head_mask_type = torch.ones(num_heads, dtype=torch.int32, device=device)
    streaming_info = torch.zeros(2 * num_heads, dtype=torch.int32, device=device)
    cached = (cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info)
    _BLOCK_SPARSE_RUNTIME_CACHE[cache_key] = cached
    return cached


def _block_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    return_lse: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    _ensure_block_sparse_available()

    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape

    q_unpad = q.transpose(1, 2).reshape(batch_size * seq_len_q, num_heads, head_dim)
    k_unpad = k.transpose(1, 2).reshape(batch_size * seq_len_k, num_heads, head_dim)
    v_unpad = v.transpose(1, 2).reshape(batch_size * seq_len_k, num_heads, head_dim)

    cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info = _get_block_sparse_runtime_tensors(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        device=q.device,
    )

    if return_lse:
        out_unpad, lse, _ = block_sparse_attn_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            block_mask.contiguous(),
            seq_len_q,
            seq_len_k,
            p_dropout=0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=True,
        )
    else:
        out_unpad = block_sparse_attn_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            streaming_info,
            block_mask.contiguous(),
            seq_len_q,
            seq_len_k,
            p_dropout=0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        )
        lse = None
    out = out_unpad.reshape(batch_size, seq_len_q, num_heads, head_dim).permute(0, 2, 1, 3)
    if lse is None:
        return out, None
    return out, lse.unsqueeze(-1).to(device=q.device, dtype=q.dtype)


def _make_full_block_mask(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.ones(
        batch_size,
        num_heads,
        _ceil_div(seq_len_q, block_size),
        _ceil_div(seq_len_k, block_size),
        dtype=torch.bool,
        device=device,
    )


def _adapt_block_mask_for_kernel(
    block_mask: torch.Tensor,
    *,
    source_block_size: int,
    seq_len_q: int,
    seq_len_k: int,
    kernel_block_size: int = _BLOCK_SPARSE_KERNEL_BLOCK_SIZE,
) -> torch.Tensor:
    """Map an offline block mask to the fixed tile size required by block_sparse_attn."""
    source_block_size = int(source_block_size)
    kernel_block_size = int(kernel_block_size)
    if source_block_size <= 0 or kernel_block_size <= 0:
        raise ValueError(f"Invalid block sizes: source={source_block_size}, kernel={kernel_block_size}")

    target_q_blocks = _ceil_div(int(seq_len_q), kernel_block_size)
    target_k_blocks = _ceil_div(int(seq_len_k), kernel_block_size)
    if source_block_size == kernel_block_size:
        if block_mask.size(-2) != target_q_blocks or block_mask.size(-1) != target_k_blocks:
            raise ValueError(
                "Runtime block mask is already at kernel tile size but has the wrong shape: "
                f"expected [..., {target_q_blocks}, {target_k_blocks}], got {list(block_mask.shape)}"
            )
        return block_mask.contiguous()

    if source_block_size > kernel_block_size:
        if source_block_size % kernel_block_size != 0:
            raise ValueError(
                "Cannot adapt a coarse Sink mask to the block-sparse kernel because "
                f"source block size {source_block_size} is not a multiple of kernel block size {kernel_block_size}."
            )
        repeat = source_block_size // kernel_block_size
        adapted = block_mask.repeat_interleave(repeat, dim=-2).repeat_interleave(repeat, dim=-1)
        return adapted[..., :target_q_blocks, :target_k_blocks].contiguous()

    if kernel_block_size % source_block_size != 0:
        raise ValueError(
            "Cannot adapt a fine Sink mask to the block-sparse kernel because "
            f"kernel block size {kernel_block_size} is not a multiple of source block size {source_block_size}."
        )

    group = kernel_block_size // source_block_size
    q_blocks = int(block_mask.size(-2))
    k_blocks = int(block_mask.size(-1))
    pad_q = (-q_blocks) % group
    pad_k = (-k_blocks) % group
    if pad_q or pad_k:
        block_mask = F.pad(block_mask, (0, pad_k, 0, pad_q), value=False)
    grouped = block_mask.view(
        *block_mask.shape[:-2],
        (q_blocks + pad_q) // group,
        group,
        (k_blocks + pad_k) // group,
        group,
    )
    adapted = grouped.any(dim=-1).any(dim=-2)
    return adapted[..., :target_q_blocks, :target_k_blocks].contiguous()


def _dense_short_attention(
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


def _dense_short_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(q.size(-1))
    score_elements = int(q.size(0)) * int(q.size(1)) * int(q.size(2)) * int(k.size(2))
    max_score_elements = int(os.environ.get("SINK_DENSE_LSE_MAX_SCORE_ELEMENTS", "67108864"))
    if score_elements <= max_score_elements:
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs.to(dtype=v.dtype), v)
        lse = torch.logsumexp(scores, dim=-1, keepdim=True).to(device=q.device, dtype=q.dtype)
        return out.to(dtype=q.dtype), lse

    chunk_size = max(1, max_score_elements // max(int(q.size(0)) * int(q.size(1)) * int(k.size(2)), 1))
    out_chunks = []
    lse_chunks = []
    k_t = k.float().transpose(-1, -2)
    for start in range(0, int(q.size(2)), int(chunk_size)):
        end = min(start + int(chunk_size), int(q.size(2)))
        scores = torch.matmul(q[:, :, start:end, :].float(), k_t) * scale
        probs = torch.softmax(scores, dim=-1)
        out_chunks.append(torch.matmul(probs.to(dtype=v.dtype), v).to(dtype=q.dtype))
        lse_chunks.append(torch.logsumexp(scores, dim=-1, keepdim=True).to(device=q.device, dtype=q.dtype))
    return torch.cat(out_chunks, dim=2), torch.cat(lse_chunks, dim=2)


def _mix_attention_outputs(
    sink_out: torch.Tensor,
    sink_lse: Optional[torch.Tensor],
    dynamic_out: torch.Tensor,
    dynamic_lse: Optional[torch.Tensor],
    config: SinkSparseConfig,
    dynamic_sample_gap: Optional[int] = None,
) -> torch.Tensor:
    if config.mixing_mode != "adaptive":
        raise ValueError(f"Unsupported mixing_mode: {config.mixing_mode}")
    if sink_lse is None or dynamic_lse is None:
        raise ValueError("Adaptive Sink-Sparse mixing requires both sink_lse and dynamic_lse.")

    resolved_sample_gap = config.sample_gap if dynamic_sample_gap is None else int(dynamic_sample_gap)
    log_sample_gap = sink_lse.new_tensor(float(max(1, int(resolved_sample_gap)))).log()
    log_weight_sink = sink_lse
    log_weight_dynamic = dynamic_lse + log_sample_gap
    max_log_weight = torch.maximum(log_weight_sink, log_weight_dynamic)
    exp_sink = torch.exp(log_weight_sink - max_log_weight)
    exp_dynamic = torch.exp(log_weight_dynamic - max_log_weight)
    alpha = exp_sink / (exp_sink + exp_dynamic)
    return sink_out * alpha + dynamic_out * (1.0 - alpha)


def summarize_sink_package(package: dict) -> str:
    counts = package.get("counts", {})
    sink_masks = package.get("sink_masks", {})
    mask_count = len(sink_masks)
    total_calls = sum(counts.values()) if counts else 0
    config = package.get("config", {})
    densities = []
    for sink_mask in sink_masks.values():
        mask_tensor = sink_mask.float()
        densities.append(float(mask_tensor.mean().item()))
    mean_density = sum(densities) / len(densities) if densities else 0.0
    direct_coverage = config.get("offline_direct_coverage")
    density_str = f", mean_density={mean_density:.4f}" if densities else ""
    coverage_str = f", coverage={float(direct_coverage):.3f}" if direct_coverage is not None else ""
    return (
        f"sink_masks={mask_count}, mask_strategy=direct_mean_coverage{coverage_str}, "
        f"aggregated_calls={total_calls}{density_str}"
    )


def parse_int_list(values: Optional[str]) -> Optional[list[int]]:
    if values is None or values == "":
        return None
    return [int(item.strip()) for item in values.split(",") if item.strip()]


def iter_seed_list(seed_values: Iterable[int]) -> list[int]:
    return [int(seed) for seed in seed_values]
