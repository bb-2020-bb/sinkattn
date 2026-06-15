import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

DEFAULT_HF_CACHE_ROOT = Path.home() / ".cache" / "sinkattention" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DIFFUSERS_CACHE", str(DEFAULT_HF_CACHE_ROOT / "diffusers"))

from sinkattention.timing import aggregate_timing_summaries, extract_attention_time_ms

WAN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = os.environ.get("WAN_MODEL")
DEFAULT_PROMPT_FILE = WAN_ROOT / "prompts" / "inference_prompts.txt"
DEFAULT_OUTPUT_DIR = WAN_ROOT / "outputs" / "inference" / "wanx_batch"
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_SINK_RUNTIME_OVERRIDES = {
    "dynamic_mode": "pooled_kv",
    "dynamic_kernel": "block_sparse",
    "mixing_mode": "adaptive",
    "sample_gap": 30,
    "runtime_diagonal_band_width": 1,
}

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def load_prompts(prompt_file):
    """Load a list of prompts from a file."""
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                prompts.append(line)
    return prompts


def require_model_path(model_path: Optional[str], env_var: str) -> str:
    if model_path is not None and str(model_path).strip():
        return str(model_path)
    raise ValueError(f"Pass --model_path or set {env_var} to a local Diffusers-format model directory.")


def resolve_device(gpu: int) -> str:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


def resolve_sink_mask_path(sink_mask_path: Optional[str]) -> Optional[str]:
    if sink_mask_path is None:
        return None

    normalized = str(sink_mask_path).strip()
    if not normalized:
        return None
    return normalized


def set_attention_timing_enabled(enabled: bool) -> None:
    value = "1" if enabled else "0"
    os.environ["SINK_ATTN_ENABLE_TIMING"] = value
    os.environ["WAN_ATTN_ENABLE_TIMING"] = value


def resolve_wan_fast_kernel_mode(
    *,
    attn_mode: str,
    enable_wan_fast_misc_fusion: bool,
    enable_wan_fast_rotary: bool,
    enable_wan_fast_kernels: bool,
    disable_wan_fast_kernels: bool,
) -> str:
    if disable_wan_fast_kernels:
        return "disabled"
    if enable_wan_fast_kernels:
        return "fast_kernels"
    if enable_wan_fast_rotary:
        return "fast_rotary"
    if enable_wan_fast_misc_fusion:
        return "misc_fusion"
    if str(attn_mode).strip().lower() == "sink":
        return "fast_kernels"
    return "disabled"


class PipelineWallClockCollector:
    def __init__(self, *, label: str = "pipeline", enabled: bool = True):
        self.label = label
        self.enabled = bool(enabled)
        self._total_by_section_s: dict[str, float] = defaultdict(float)
        self._count_by_section: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self._total_by_section_s.clear()
        self._count_by_section.clear()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_t = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start_t
            self._total_by_section_s[name] += float(elapsed_s)
            self._count_by_section[name] += 1

    def summarize(self) -> dict:
        if not self.enabled:
            return {
                "enabled": False,
                "label": self.label,
                "num_records": 0,
                "by_section": {},
                "total_s": 0.0,
            }

        ordered_names = [
            name
            for name, _ in sorted(self._total_by_section_s.items(), key=lambda item: (-item[1], item[0]))
        ]
        by_section = {
            name: {
                "total_s": float(self._total_by_section_s[name]),
                "mean_s": float(self._total_by_section_s[name] / max(self._count_by_section[name], 1)),
                "count": int(self._count_by_section[name]),
            }
            for name in ordered_names
        }
        return {
            "enabled": True,
            "label": self.label,
            "num_records": int(sum(self._count_by_section.values())),
            "by_section": by_section,
            "total_s": float(sum(self._total_by_section_s.values())),
        }


def _wrap_instance_method_with_collector(instance, method_name: str, collector: PipelineWallClockCollector, section_name: str) -> None:
    original_method = getattr(instance, method_name)
    if getattr(original_method, "_sink_timing_wrapped", False):
        return

    underlying_func = getattr(original_method, "__func__", None)
    if underlying_func is None:
        def wrapped_method(self, *args, **kwargs):
            with collector.section(section_name):
                return original_method(*args, **kwargs)
        wrapped_method._sink_timing_wrapped = True
        wrapped_method._sink_timing_original = original_method
        wrapped = MethodType(wrapped_method, instance)
    else:
        def wrapped_method(self, *args, **kwargs):
            with collector.section(section_name):
                return underlying_func(self, *args, **kwargs)
        wrapped_method._sink_timing_wrapped = True
        wrapped_method._sink_timing_original = original_method
        wrapped = MethodType(wrapped_method, instance)

    setattr(instance, method_name, wrapped)


def enable_pipeline_runtime_profiling(pipe) -> PipelineWallClockCollector:
    collector = PipelineWallClockCollector()
    pipe._pipeline_runtime_collector = collector
    _wrap_instance_method_with_collector(pipe, "encode_prompt", collector, "encode_prompt")
    _wrap_instance_method_with_collector(pipe.transformer, "forward", collector, "transformer_total")
    _wrap_instance_method_with_collector(pipe.scheduler, "step", collector, "scheduler_step")
    _wrap_instance_method_with_collector(pipe.vae, "decode", collector, "vae_decode")
    _wrap_instance_method_with_collector(pipe.video_processor, "postprocess_video", collector, "postprocess_video")
    _wrap_instance_method_with_collector(pipe.transformer.rope, "forward", collector, "transformer_rope")
    _wrap_instance_method_with_collector(
        pipe.transformer.patch_embedding,
        "forward",
        collector,
        "transformer_patch_embedding",
    )
    _wrap_instance_method_with_collector(
        pipe.transformer.condition_embedder,
        "forward",
        collector,
        "transformer_condition_embedder",
    )
    _wrap_instance_method_with_collector(pipe.transformer.norm_out, "forward", collector, "transformer_norm_out")
    _wrap_instance_method_with_collector(pipe.transformer.proj_out, "forward", collector, "transformer_proj_out")
    for block in pipe.transformer.blocks:
        _wrap_instance_method_with_collector(block.attn1, "forward", collector, "transformer_self_attn_total")
        _wrap_instance_method_with_collector(block.attn2, "forward", collector, "transformer_cross_attn_total")
        _wrap_instance_method_with_collector(block.ffn, "forward", collector, "transformer_ffn_total")
    return collector


def apply_attention_backend(
    pipe,
    attn_mode: str,
    enable_attention_timing: bool = False,
    sink_mask_path: Optional[str] = None,
    sink_sample_gap: Optional[int] = None,
    sink_dynamic_mode: Optional[str] = None,
    sink_mixing_mode: Optional[str] = None,
    sink_dynamic_kernel: Optional[str] = None,
    sink_disable_dynamic_branch: bool = False,
) -> Optional[dict]:
    if attn_mode == 'sink':
        if not sink_mask_path:
            raise ValueError('--sink_mask_path is required when --attn_mode sink is used')

        from wanx.runtime.sink_sparse_wan import set_sink_sparse_attn_wanx

        explicit_overrides = {
            'sample_gap': sink_sample_gap,
            'dynamic_mode': 'none' if sink_disable_dynamic_branch else sink_dynamic_mode,
            'mixing_mode': sink_mixing_mode,
            'dynamic_kernel': sink_dynamic_kernel,
        }
        config_overrides = dict(DEFAULT_SINK_RUNTIME_OVERRIDES)
        config_overrides.update({key: value for key, value in explicit_overrides.items() if value is not None})
        metadata = set_sink_sparse_attn_wanx(
            pipe.transformer,
            sink_mask_path=sink_mask_path,
            config_overrides=config_overrides,
        )
        return metadata

    if attn_mode == 'dense':
        if enable_attention_timing:
            from wanx.runtime.modify_wan import set_timed_dense_attn_wanx

            set_timed_dense_attn_wanx(pipe.transformer)
        return None

    raise ValueError(f'Unsupported attention backend: {attn_mode}')


def build_wan_pipeline(
    model_path: str,
    device: str,
    attn_mode: str,
    enable_attention_timing: bool = False,
    enable_wan_fast_misc_fusion: bool = False,
    enable_wan_fast_rotary: bool = False,
    enable_wan_fast_kernels: bool = False,
    disable_wan_fast_kernels: bool = False,
    flow_shift: float = 3.0,
    sink_mask_path: Optional[str] = None,
    sink_sample_gap: Optional[int] = None,
    sink_dynamic_mode: Optional[str] = None,
    sink_mixing_mode: Optional[str] = None,
    sink_dynamic_kernel: Optional[str] = None,
    sink_disable_dynamic_branch: bool = False,
) -> tuple[object, Optional[dict]]:
    from wanx.runtime.wan_model_loader import load_wan_pipeline

    pipe, model_info = load_wan_pipeline(model_path=model_path, flow_shift=flow_shift)

    set_attention_timing_enabled(enable_attention_timing)
    metadata = apply_attention_backend(
        pipe=pipe,
        attn_mode=attn_mode,
        enable_attention_timing=enable_attention_timing,
        sink_mask_path=sink_mask_path,
        sink_sample_gap=sink_sample_gap,
        sink_dynamic_mode=sink_dynamic_mode,
        sink_mixing_mode=sink_mixing_mode,
        sink_dynamic_kernel=sink_dynamic_kernel,
        sink_disable_dynamic_branch=sink_disable_dynamic_branch,
    )

    from wanx.runtime.fast_misc import disable_wan_fast_kernels as reset_wan_fast_kernels

    reset_wan_fast_kernels(pipe.transformer)

    resolved_fast_kernel_mode = resolve_wan_fast_kernel_mode(
        attn_mode=attn_mode,
        enable_wan_fast_misc_fusion=enable_wan_fast_misc_fusion,
        enable_wan_fast_rotary=enable_wan_fast_rotary,
        enable_wan_fast_kernels=enable_wan_fast_kernels,
        disable_wan_fast_kernels=disable_wan_fast_kernels,
    )

    if resolved_fast_kernel_mode == "fast_kernels":
        from wanx.runtime.fast_misc import enable_wan_fast_kernels

        enable_wan_fast_kernels(pipe.transformer)
    elif resolved_fast_kernel_mode == "fast_rotary":
        from wanx.runtime.fast_misc import enable_wan_fast_rotary

        enable_wan_fast_rotary(pipe.transformer)
    elif resolved_fast_kernel_mode == "misc_fusion":
        from wanx.runtime.fast_misc import enable_wan_fast_misc_fusion

        enable_wan_fast_misc_fusion(pipe.transformer)

    pipe.to(device)
    if enable_attention_timing:
        enable_pipeline_runtime_profiling(pipe)
    return pipe, metadata


def prepare_generation(pipe, num_inference_steps: int) -> None:
    runtime_state = getattr(pipe.transformer, "_sink_runtime_state", None)
    if runtime_state is not None:
        runtime_state.reset_generation(total_steps=num_inference_steps)
    dense_runtime_state = getattr(pipe.transformer, "_dense_runtime_state", None)
    if dense_runtime_state is not None and hasattr(dense_runtime_state, "reset_runtime_timing"):
        dense_runtime_state.reset_runtime_timing()
    pipeline_runtime_collector = getattr(pipe, "_pipeline_runtime_collector", None)
    if pipeline_runtime_collector is not None and hasattr(pipeline_runtime_collector, "reset"):
        pipeline_runtime_collector.reset()


def get_sink_runtime_summary(pipe) -> Optional[dict]:
    runtime_state = getattr(pipe.transformer, "_sink_runtime_state", None)
    if runtime_state is None or not hasattr(runtime_state, "summarize_sparsity"):
        return None
    summary = runtime_state.summarize_sparsity()
    if not summary or summary.get("num_events", 0) <= 0:
        return None
    return summary


def get_attention_runtime_summary(pipe) -> Optional[dict]:
    runtime_states = [
        getattr(pipe.transformer, "_sink_runtime_state", None),
        getattr(pipe.transformer, "_dense_runtime_state", None),
    ]
    for runtime_state in runtime_states:
        if runtime_state is None or not hasattr(runtime_state, "summarize_timing"):
            continue
        summary = runtime_state.summarize_timing()
        if not summary or not summary.get("enabled", False):
            continue
        if summary.get("num_records", 0) <= 0:
            continue
        return summary
    return None


def get_pipeline_runtime_summary(pipe) -> Optional[dict]:
    collector = getattr(pipe, "_pipeline_runtime_collector", None)
    if collector is None or not hasattr(collector, "summarize"):
        return None
    summary = collector.summarize()
    if not summary or not summary.get("enabled", False):
        return None
    if summary.get("num_records", 0) <= 0:
        return None
    return summary


def get_section_total_s(summary: Optional[dict], section_name: str) -> Optional[float]:
    if not summary:
        return None
    by_section = summary.get("by_section") or {}
    section_summary = by_section.get(section_name)
    if section_summary is None:
        return None
    total_s = section_summary.get("total_s")
    if total_s is None:
        return None
    return float(total_s)


def summarize_scalar_series(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.pstdev(values) if len(values) > 1 else 0.0),
    }


def aggregate_pipeline_summaries(summaries: list[Optional[dict]]) -> Optional[dict]:
    valid_summaries = [summary for summary in summaries if summary and summary.get("enabled", False)]
    if not valid_summaries:
        return None

    values_by_section: dict[str, list[float]] = defaultdict(list)
    total_s_values: list[float] = []
    for summary in valid_summaries:
        total_s = summary.get("total_s")
        if total_s is not None:
            total_s_values.append(float(total_s))
        for section_name, section_summary in (summary.get("by_section") or {}).items():
            section_total_s = section_summary.get("total_s")
            if section_total_s is None:
                continue
            values_by_section[str(section_name)].append(float(section_total_s))

    return {
        "count": len(valid_summaries),
        "labels": sorted({str(summary.get("label", "")) for summary in valid_summaries if summary.get("label")}),
        "mean_total_s": float(sum(total_s_values) / len(total_s_values)) if total_s_values else None,
        "max_total_s": float(max(total_s_values)) if total_s_values else None,
        "mean_by_section_s": {
            name: float(sum(values) / len(values))
            for name, values in sorted(values_by_section.items(), key=lambda item: item[0])
            if values
        },
        "max_by_section_s": {
            name: float(max(values))
            for name, values in sorted(values_by_section.items(), key=lambda item: item[0])
            if values
        },
        "total_by_section_s": {
            name: float(sum(values))
            for name, values in sorted(values_by_section.items(), key=lambda item: item[0])
            if values
        },
    }


def collect_generation_runtime_profile(
    pipe,
    *,
    pipe_time_s: float,
    peak_memory_gb: Optional[float] = None,
) -> Optional[dict]:
    attention_timing = get_attention_runtime_summary(pipe)
    pipeline_timing = get_pipeline_runtime_summary(pipe)
    sink_runtime_sparsity = get_sink_runtime_summary(pipe)
    attention_time_ms = extract_attention_time_ms(attention_timing)
    attention_time_s = None if attention_time_ms is None else float(attention_time_ms) / 1000.0
    attention_share_of_pipe = None
    if attention_time_s is not None and pipe_time_s > 0:
        attention_share_of_pipe = attention_time_s / float(pipe_time_s)

    encode_prompt_s = get_section_total_s(pipeline_timing, "encode_prompt")
    transformer_total_s = get_section_total_s(pipeline_timing, "transformer_total")
    scheduler_step_s = get_section_total_s(pipeline_timing, "scheduler_step")
    vae_decode_s = get_section_total_s(pipeline_timing, "vae_decode")
    postprocess_video_s = get_section_total_s(pipeline_timing, "postprocess_video")
    transformer_rope_s = get_section_total_s(pipeline_timing, "transformer_rope")
    transformer_patch_embedding_s = get_section_total_s(pipeline_timing, "transformer_patch_embedding")
    transformer_condition_embedder_s = get_section_total_s(pipeline_timing, "transformer_condition_embedder")
    transformer_self_attn_total_s = get_section_total_s(pipeline_timing, "transformer_self_attn_total")
    transformer_cross_attn_total_s = get_section_total_s(pipeline_timing, "transformer_cross_attn_total")
    transformer_ffn_total_s = get_section_total_s(pipeline_timing, "transformer_ffn_total")
    transformer_norm_out_s = get_section_total_s(pipeline_timing, "transformer_norm_out")
    transformer_proj_out_s = get_section_total_s(pipeline_timing, "transformer_proj_out")
    transformer_input_prep_s = None
    if any(
        value is not None
        for value in (
            transformer_rope_s,
            transformer_patch_embedding_s,
            transformer_condition_embedder_s,
        )
    ):
        transformer_input_prep_s = float(
            sum(
                value
                for value in (
                    transformer_rope_s,
                    transformer_patch_embedding_s,
                    transformer_condition_embedder_s,
                )
                if value is not None
            )
        )
    transformer_output_head_s = None
    if transformer_norm_out_s is not None or transformer_proj_out_s is not None:
        transformer_output_head_s = float(
            sum(
                value
                for value in (
                    transformer_norm_out_s,
                    transformer_proj_out_s,
                )
                if value is not None
            )
        )
    transformer_non_attention_s = None
    if transformer_total_s is not None and attention_time_s is not None:
        transformer_non_attention_s = max(float(transformer_total_s) - float(attention_time_s), 0.0)
    transformer_self_attn_overhead_s = None
    if transformer_self_attn_total_s is not None and attention_time_s is not None:
        transformer_self_attn_overhead_s = max(float(transformer_self_attn_total_s) - float(attention_time_s), 0.0)
    transformer_other_s = None
    if transformer_total_s is not None:
        accounted_parts = [
            transformer_input_prep_s,
            transformer_self_attn_total_s,
            transformer_cross_attn_total_s,
            transformer_ffn_total_s,
            transformer_output_head_s,
        ]
        transformer_other_s = max(
            float(transformer_total_s) - float(sum(value for value in accounted_parts if value is not None)),
            0.0,
        )

    if (
        attention_timing is None
        and pipeline_timing is None
        and sink_runtime_sparsity is None
        and peak_memory_gb is None
    ):
        return None

    return {
        "pipe_time_s": float(pipe_time_s),
        "attention_time_ms": attention_time_ms,
        "attention_time_s": attention_time_s,
        "attention_share_of_pipe": attention_share_of_pipe,
        "encode_prompt_s": encode_prompt_s,
        "transformer_total_s": transformer_total_s,
        "transformer_non_attention_s": transformer_non_attention_s,
        "scheduler_step_s": scheduler_step_s,
        "vae_decode_s": vae_decode_s,
        "postprocess_video_s": postprocess_video_s,
        "transformer_rope_s": transformer_rope_s,
        "transformer_patch_embedding_s": transformer_patch_embedding_s,
        "transformer_condition_embedder_s": transformer_condition_embedder_s,
        "transformer_input_prep_s": transformer_input_prep_s,
        "transformer_self_attn_total_s": transformer_self_attn_total_s,
        "transformer_self_attn_overhead_s": transformer_self_attn_overhead_s,
        "transformer_cross_attn_total_s": transformer_cross_attn_total_s,
        "transformer_ffn_total_s": transformer_ffn_total_s,
        "transformer_norm_out_s": transformer_norm_out_s,
        "transformer_proj_out_s": transformer_proj_out_s,
        "transformer_output_head_s": transformer_output_head_s,
        "transformer_other_s": transformer_other_s,
        "peak_memory_gb": None if peak_memory_gb is None else float(peak_memory_gb),
        "attention_timing": attention_timing,
        "pipeline_timing": pipeline_timing,
        "sink_runtime_sparsity": sink_runtime_sparsity,
    }


def summarize_runtime_profiles(runtime_profiles: list[dict]) -> Optional[dict]:
    valid_profiles = [profile for profile in runtime_profiles if profile]
    if not valid_profiles:
        return None

    attention_time_values = [
        float(profile["attention_time_s"])
        for profile in valid_profiles
        if profile.get("attention_time_s") is not None
    ]
    attention_share_values = [
        float(profile["attention_share_of_pipe"])
        for profile in valid_profiles
        if profile.get("attention_share_of_pipe") is not None
    ]
    peak_memory_values = [
        float(profile["peak_memory_gb"])
        for profile in valid_profiles
        if profile.get("peak_memory_gb") is not None
    ]
    encode_prompt_values = [
        float(profile["encode_prompt_s"])
        for profile in valid_profiles
        if profile.get("encode_prompt_s") is not None
    ]
    transformer_total_values = [
        float(profile["transformer_total_s"])
        for profile in valid_profiles
        if profile.get("transformer_total_s") is not None
    ]
    transformer_non_attention_values = [
        float(profile["transformer_non_attention_s"])
        for profile in valid_profiles
        if profile.get("transformer_non_attention_s") is not None
    ]
    transformer_rope_values = [
        float(profile["transformer_rope_s"])
        for profile in valid_profiles
        if profile.get("transformer_rope_s") is not None
    ]
    transformer_patch_embedding_values = [
        float(profile["transformer_patch_embedding_s"])
        for profile in valid_profiles
        if profile.get("transformer_patch_embedding_s") is not None
    ]
    transformer_condition_embedder_values = [
        float(profile["transformer_condition_embedder_s"])
        for profile in valid_profiles
        if profile.get("transformer_condition_embedder_s") is not None
    ]
    transformer_input_prep_values = [
        float(profile["transformer_input_prep_s"])
        for profile in valid_profiles
        if profile.get("transformer_input_prep_s") is not None
    ]
    transformer_self_attn_total_values = [
        float(profile["transformer_self_attn_total_s"])
        for profile in valid_profiles
        if profile.get("transformer_self_attn_total_s") is not None
    ]
    transformer_self_attn_overhead_values = [
        float(profile["transformer_self_attn_overhead_s"])
        for profile in valid_profiles
        if profile.get("transformer_self_attn_overhead_s") is not None
    ]
    transformer_cross_attn_total_values = [
        float(profile["transformer_cross_attn_total_s"])
        for profile in valid_profiles
        if profile.get("transformer_cross_attn_total_s") is not None
    ]
    transformer_ffn_values = [
        float(profile["transformer_ffn_total_s"])
        for profile in valid_profiles
        if profile.get("transformer_ffn_total_s") is not None
    ]
    transformer_norm_out_values = [
        float(profile["transformer_norm_out_s"])
        for profile in valid_profiles
        if profile.get("transformer_norm_out_s") is not None
    ]
    transformer_proj_out_values = [
        float(profile["transformer_proj_out_s"])
        for profile in valid_profiles
        if profile.get("transformer_proj_out_s") is not None
    ]
    transformer_output_head_values = [
        float(profile["transformer_output_head_s"])
        for profile in valid_profiles
        if profile.get("transformer_output_head_s") is not None
    ]
    transformer_other_values = [
        float(profile["transformer_other_s"])
        for profile in valid_profiles
        if profile.get("transformer_other_s") is not None
    ]
    scheduler_step_values = [
        float(profile["scheduler_step_s"])
        for profile in valid_profiles
        if profile.get("scheduler_step_s") is not None
    ]
    vae_decode_values = [
        float(profile["vae_decode_s"])
        for profile in valid_profiles
        if profile.get("vae_decode_s") is not None
    ]
    postprocess_video_values = [
        float(profile["postprocess_video_s"])
        for profile in valid_profiles
        if profile.get("postprocess_video_s") is not None
    ]
    attention_timing_summaries = [
        profile.get("attention_timing")
        for profile in valid_profiles
        if profile.get("attention_timing") is not None
    ]
    pipeline_timing_summaries = [
        profile.get("pipeline_timing")
        for profile in valid_profiles
        if profile.get("pipeline_timing") is not None
    ]

    return {
        "count": len(valid_profiles),
        "attention_time_s": summarize_scalar_series(attention_time_values),
        "attention_share_of_pipe": summarize_scalar_series(attention_share_values),
        "encode_prompt_s": summarize_scalar_series(encode_prompt_values),
        "transformer_total_s": summarize_scalar_series(transformer_total_values),
        "transformer_non_attention_s": summarize_scalar_series(transformer_non_attention_values),
        "transformer_rope_s": summarize_scalar_series(transformer_rope_values),
        "transformer_patch_embedding_s": summarize_scalar_series(transformer_patch_embedding_values),
        "transformer_condition_embedder_s": summarize_scalar_series(transformer_condition_embedder_values),
        "transformer_input_prep_s": summarize_scalar_series(transformer_input_prep_values),
        "transformer_self_attn_total_s": summarize_scalar_series(transformer_self_attn_total_values),
        "transformer_self_attn_overhead_s": summarize_scalar_series(transformer_self_attn_overhead_values),
        "transformer_cross_attn_total_s": summarize_scalar_series(transformer_cross_attn_total_values),
        "transformer_ffn_total_s": summarize_scalar_series(transformer_ffn_values),
        "transformer_norm_out_s": summarize_scalar_series(transformer_norm_out_values),
        "transformer_proj_out_s": summarize_scalar_series(transformer_proj_out_values),
        "transformer_output_head_s": summarize_scalar_series(transformer_output_head_values),
        "transformer_other_s": summarize_scalar_series(transformer_other_values),
        "scheduler_step_s": summarize_scalar_series(scheduler_step_values),
        "vae_decode_s": summarize_scalar_series(vae_decode_values),
        "postprocess_video_s": summarize_scalar_series(postprocess_video_values),
        "peak_memory_gb": summarize_scalar_series(peak_memory_values),
        "attention_timing": aggregate_timing_summaries(attention_timing_summaries),
        "pipeline_timing": aggregate_pipeline_summaries(pipeline_timing_summaries),
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Wan train-free Sink-Sparse inference')
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='Path to the local Wan model directory.',
    )
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify the GPU device ID to use (default: 0)')
    parser.add_argument(
        '--prompt_file',
        type=str,
        default=str(DEFAULT_PROMPT_FILE),
        help='Path to the prompt file used for inference.',
    )
    parser.add_argument(
        '--num_prompts',
        type=int,
        default=None,
        help='Optional limit on the number of prompts loaded from --prompt_file.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory where generated videos will be saved.',
    )
    parser.add_argument(
        '--attn_mode',
        type=str,
        default='sink',
        choices=['dense', 'sink'],
        help='Attention backend used during inference.',
    )
    parser.add_argument(
        '--sink_mask_path',
        type=str,
        default=None,
        help=(
            'Sink mask package path. Required for --attn_mode sink. Generate a mask '
            'with wanx.offline.calibrate_sink_sparse and pass that file explicitly.'
        ),
    )
    parser.add_argument(
        '--sink_sample_gap',
        type=int,
        default=None,
        help='Optional runtime override for the Sink-Sparse dynamic branch sample gap.',
    )
    parser.add_argument(
        '--sink_dynamic_mode',
        type=str,
        default=None,
        choices=['none', 'pooled_kv'],
        help='Optional runtime override for the Sink-Sparse dynamic branch mode.',
    )
    parser.add_argument(
        '--sink_mixing_mode',
        type=str,
        default=None,
        choices=['adaptive'],
        help='Optional runtime override for Sink-Sparse branch fusion.',
    )
    parser.add_argument(
        '--sink_dynamic_kernel',
        type=str,
        default=None,
        choices=['auto', 'block_sparse', 'dense'],
        help='Optional runtime override for the Sink-Sparse dynamic branch kernel.',
    )
    parser.add_argument(
        '--sink_disable_dynamic_branch',
        action='store_true',
        help='Disable the lightweight dynamic branch in Sink-Sparse mode.',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Output video height.',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=832,
        help='Output video width.',
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=81,
        help='Number of frames to generate.',
    )
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help='Number of denoising steps.',
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help='Classifier-free guidance scale.',
    )
    parser.add_argument(
        '--seed_base',
        type=int,
        default=8888,
        help='Base seed. Prompt i uses seed_base + i.',
    )
    parser.add_argument(
        '--profile_runtime',
        action='store_true',
        help='Collect attention runtime, end-to-end pipe latency, and peak memory per generated sample.',
    )
    parser.add_argument(
        '--enable_wan_fast_misc_fusion',
        action='store_true',
        dest='enable_wan_fast_misc_fusion',
        help='Enable the optional fused Wan residual-gate path for self-attn/FFN residual updates.',
    )
    parser.add_argument(
        '--enable_wan_fast_rotary',
        action='store_true',
        dest='enable_wan_fast_rotary',
        help='Enable the fast Wan rotary_emb application path only, without layernorm/modulate/misc/qk_norm fusion.',
    )
    parser.add_argument(
        '--enable_wan_fast_kernels',
        action='store_true',
        dest='enable_wan_fast_kernels',
        help='Force-enable optional fast layernorm/modulate/misc/qk_norm/rotary kernels on the Wan path.',
    )
    parser.add_argument(
        '--disable_wan_fast_kernels',
        action='store_true',
        dest='disable_wan_fast_kernels',
        help='Force-disable optional Wan fast kernels. By default, sink enables them and dense leaves them off.',
    )
    parser.add_argument(
        '--profile_output_json',
        type=str,
        default=None,
        help='Optional JSON path for runtime profiling results. Defaults to <output_dir>/runtime_profile.json when profiling is enabled.',
    )

    args = parser.parse_args()
    if not args.model_path:
        parser.error("Pass --model_path or set WAN_MODEL to a local Diffusers-format model directory.")
    model_path = require_model_path(args.model_path, "WAN_MODEL")

    device = resolve_device(args.gpu)
    if device == "cpu":
        sys.stderr.write("CUDA is not available, using CPU.\n")

    resolved_num_inference_steps = int(args.num_inference_steps)
    resolved_guidance_scale = float(args.guidance_scale)
    resolved_sink_mask_path = (
        resolve_sink_mask_path(args.sink_mask_path)
        if args.attn_mode == 'sink'
        else None
    )

    pipe, _ = build_wan_pipeline(
        model_path=model_path,
        device=device,
        attn_mode=args.attn_mode,
        enable_attention_timing=args.profile_runtime,
        enable_wan_fast_misc_fusion=args.enable_wan_fast_misc_fusion,
        enable_wan_fast_rotary=args.enable_wan_fast_rotary,
        enable_wan_fast_kernels=args.enable_wan_fast_kernels,
        disable_wan_fast_kernels=args.disable_wan_fast_kernels,
        sink_mask_path=resolved_sink_mask_path,
        sink_sample_gap=args.sink_sample_gap,
        sink_dynamic_mode=args.sink_dynamic_mode,
        sink_mixing_mode=args.sink_mixing_mode,
        sink_dynamic_kernel=args.sink_dynamic_kernel,
        sink_disable_dynamic_branch=args.sink_disable_dynamic_branch,
    )

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")

    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default negative prompt
    negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Batch inference
    profile_samples = []
    failed_prompts = []
    for i, prompt in enumerate(prompts):
        try:
            # Use a different seed for each prompt
            generator = torch.manual_seed(args.seed_base + i)
            prepare_generation(pipe, resolved_num_inference_steps)
            if torch.cuda.is_available():
                if args.profile_runtime:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter() if args.profile_runtime else None
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=resolved_num_inference_steps,
                guidance_scale=resolved_guidance_scale,
                generator=generator,
            ).frames[0]
            if args.profile_runtime:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak_memory_gb = None
                if torch.cuda.is_available():
                    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                runtime_profile = collect_generation_runtime_profile(
                    pipe,
                    pipe_time_s=elapsed,
                    peak_memory_gb=peak_memory_gb,
                )
            else:
                runtime_profile = None

            from diffusers.utils import export_to_video

            output_path = output_dir / f"{i+1:02d}.mp4"
            export_to_video(output, str(output_path), fps=16)
            if runtime_profile is not None:
                profile_samples.append(
                    {
                        "prompt_index": i,
                        "seed": args.seed_base + i,
                        "output_path": str(output_path),
                        "runtime_profile": runtime_profile,
                    }
                )
                attention_time_s = runtime_profile.get("attention_time_s")
                attention_share_of_pipe = runtime_profile.get("attention_share_of_pipe")
                peak_memory_gb = runtime_profile.get("peak_memory_gb")
                attention_text = (
                    f" attn={attention_time_s:.3f}s ({attention_share_of_pipe * 100.0:.1f}%)"
                    if attention_time_s is not None and attention_share_of_pipe is not None
                    else ""
                )
                peak_text = f" peak_mem={peak_memory_gb:.2f}GB" if peak_memory_gb is not None else ""
                sys.stderr.write(f"Runtime profile: total={runtime_profile['pipe_time_s']:.3f}s{attention_text}{peak_text}\n")

        except Exception as e:
            failed_prompts.append({"prompt_index": i, "error": str(e)})
            sys.stderr.write(f"Prompt {i + 1} failed: {e}\n")
            continue

    if args.profile_runtime:
        profile_summary = summarize_runtime_profiles([sample["runtime_profile"] for sample in profile_samples])
        profile_report = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attn_mode": args.attn_mode,
            "model_path": model_path,
            "output_dir": str(output_dir),
            "resolved_num_inference_steps": resolved_num_inference_steps,
            "resolved_guidance_scale": resolved_guidance_scale,
            "num_prompts": len(profile_samples),
            "samples": profile_samples,
            "summary": profile_summary,
        }
        profile_output_path = (
            Path(args.profile_output_json)
            if args.profile_output_json is not None
            else (output_dir / "runtime_profile.json")
        )
        profile_output_path.parent.mkdir(parents=True, exist_ok=True)
        profile_output_path.write_text(json.dumps(profile_report, indent=2), encoding='utf-8')
        sys.stderr.write(f"Runtime profile JSON saved to: {profile_output_path}\n")

    succeeded = len(prompts) - len(failed_prompts)
    sys.stdout.write(f"Saved {succeeded}/{len(prompts)} video(s) to {output_dir}\n")
    if failed_prompts:
        sys.stderr.write("Failed prompts:\n")
        for failure in failed_prompts:
            sys.stderr.write(f"  prompt {failure['prompt_index'] + 1}: {failure['error']}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
