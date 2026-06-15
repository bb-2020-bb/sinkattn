from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Optional

DEFAULT_HF_CACHE_ROOT = Path.home() / ".cache" / "sinkattention" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DIFFUSERS_CACHE", str(DEFAULT_HF_CACHE_ROOT / "diffusers"))

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sinkattention.sparse_attention import iter_seed_list, parse_int_list
from zimg.model_spec import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_LOCAL_MODEL_PATH,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    require_model_path,
    resolve_generation_defaults,
)
from zimg.runtime.sink_profiles import DEFAULT_SINK_PROFILE, list_sink_profiles, merge_sink_profile_overrides
from zimg.runtime.sink_sparse_zimage import (
    infer_zimg_token_layout,
    reset_sink_runtime_state_zimage,
    set_sink_sparse_attn_zimage,
)


RUNTIME_DIR = Path(__file__).resolve().parent
ZIMG_ROOT = RUNTIME_DIR.parent

DEFAULT_MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH
DEFAULT_PROMPT_FILE = ZIMG_ROOT / "prompts" / "inference_prompts.txt"
DEFAULT_OUTPUT_DIR = ZIMG_ROOT / "outputs" / "inference" / "zimage_batch"


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
    if not hasattr(instance, method_name):
        return

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


def set_attention_timing_enabled(enabled: bool) -> None:
    value = "1" if enabled else "0"
    os.environ["SINK_ATTN_ENABLE_TIMING"] = value
    os.environ["WAN_ATTN_ENABLE_TIMING"] = value


def enable_pipeline_runtime_profiling(pipe) -> PipelineWallClockCollector:
    collector = PipelineWallClockCollector(label="zimg_pipeline")
    pipe._pipeline_runtime_collector = collector

    _wrap_instance_method_with_collector(pipe, "encode_prompt", collector, "encode_prompt")
    _wrap_instance_method_with_collector(pipe.transformer, "forward", collector, "transformer_total")
    _wrap_instance_method_with_collector(pipe.scheduler, "step", collector, "scheduler_step")
    _wrap_instance_method_with_collector(pipe.vae, "decode", collector, "vae_decode")
    if hasattr(pipe, "image_processor"):
        _wrap_instance_method_with_collector(pipe.image_processor, "postprocess", collector, "postprocess")

    _wrap_instance_method_with_collector(pipe.transformer.t_embedder, "forward", collector, "transformer_t_embedder")
    _wrap_instance_method_with_collector(pipe.transformer.cap_embedder, "forward", collector, "transformer_cap_embedder")
    for module in pipe.transformer.all_x_embedder.values():
        _wrap_instance_method_with_collector(module, "forward", collector, "transformer_x_embedder")
    for module in pipe.transformer.all_final_layer.values():
        _wrap_instance_method_with_collector(module, "forward", collector, "transformer_final_layer")

    for block in pipe.transformer.noise_refiner:
        _wrap_instance_method_with_collector(block.attention, "forward", collector, "transformer_noise_refiner_attention_total")
        _wrap_instance_method_with_collector(block.feed_forward, "forward", collector, "transformer_noise_refiner_ffn_total")
    for block in pipe.transformer.context_refiner:
        _wrap_instance_method_with_collector(block.attention, "forward", collector, "transformer_context_refiner_attention_total")
        _wrap_instance_method_with_collector(block.feed_forward, "forward", collector, "transformer_context_refiner_ffn_total")
    siglip_refiner = getattr(pipe.transformer, "siglip_refiner", None)
    if siglip_refiner is not None:
        for block in siglip_refiner:
            _wrap_instance_method_with_collector(block.attention, "forward", collector, "transformer_siglip_refiner_attention_total")
            _wrap_instance_method_with_collector(block.feed_forward, "forward", collector, "transformer_siglip_refiner_ffn_total")
    for block in pipe.transformer.layers:
        _wrap_instance_method_with_collector(block.attention, "forward", collector, "transformer_main_attention_total")
        _wrap_instance_method_with_collector(block.feed_forward, "forward", collector, "transformer_main_ffn_total")

    return collector


def _sum_profile_sections(profile: Optional[dict], section_names: list[str]) -> float:
    if not profile or not profile.get("enabled", False):
        return 0.0
    by_section = profile.get("by_section") or {}
    total = 0.0
    for name in section_names:
        section = by_section.get(name)
        if section is None:
            continue
        total += float(section.get("total_s", 0.0))
    return float(total)


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


def _iter_sparse_runtime_states(pipe):
    for attr_name in ("_sink_runtime_state",):
        runtime_state = getattr(pipe.transformer, attr_name, None)
        if runtime_state is None:
            continue
        yield runtime_state


def get_sparse_runtime_summary(pipe) -> Optional[dict]:
    for runtime_state in _iter_sparse_runtime_states(pipe):
        if not hasattr(runtime_state, "summarize_sparsity"):
            continue
        summary = runtime_state.summarize_sparsity()
        if not summary:
            continue
        if summary.get("num_events", 0) <= 0:
            continue
        return summary
    return None


def get_attention_runtime_summary(pipe) -> Optional[dict]:
    for runtime_state in _iter_sparse_runtime_states(pipe):
        if not hasattr(runtime_state, "summarize_timing"):
            continue
        summary = runtime_state.summarize_timing()
        if not summary or not summary.get("enabled", False):
            continue
        if summary.get("num_records", 0) <= 0:
            continue
        return summary
    return None


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


def get_attention_section_total_s(summary: Optional[dict], section_name: str) -> Optional[float]:
    if not summary:
        return None
    by_section = summary.get("by_section") or {}
    section_summary = by_section.get(section_name)
    if section_summary is None:
        return None
    total_ms = section_summary.get("total_ms")
    if total_ms is None:
        return None
    return float(total_ms) / 1000.0


def collect_generation_runtime_profile(
    pipe,
    *,
    pipe_time_s: float,
    peak_memory_gb: Optional[float] = None,
) -> Optional[dict]:
    attention_timing = get_attention_runtime_summary(pipe)
    pipeline_timing = get_pipeline_runtime_summary(pipe)
    sparse_runtime_sparsity = get_sparse_runtime_summary(pipe)

    attention_time_s = get_attention_section_total_s(attention_timing, "inner_total")
    attention_time_ms = None if attention_time_s is None else float(attention_time_s) * 1000.0
    attention_share_of_pipe = None
    if attention_time_s is not None and pipe_time_s > 0:
        attention_share_of_pipe = float(attention_time_s) / float(pipe_time_s)

    encode_prompt_s = get_section_total_s(pipeline_timing, "encode_prompt")
    transformer_total_s = get_section_total_s(pipeline_timing, "transformer_total")
    scheduler_step_s = get_section_total_s(pipeline_timing, "scheduler_step")
    vae_decode_s = get_section_total_s(pipeline_timing, "vae_decode")
    postprocess_video_s = get_section_total_s(pipeline_timing, "postprocess")

    transformer_t_embedder_s = get_section_total_s(pipeline_timing, "transformer_t_embedder")
    transformer_cap_embedder_s = get_section_total_s(pipeline_timing, "transformer_cap_embedder")
    transformer_x_embedder_s = get_section_total_s(pipeline_timing, "transformer_x_embedder")
    transformer_input_prep_s = None
    if any(
        value is not None
        for value in (
            transformer_t_embedder_s,
            transformer_cap_embedder_s,
            transformer_x_embedder_s,
        )
    ):
        transformer_input_prep_s = float(
            sum(
                value
                for value in (
                    transformer_t_embedder_s,
                    transformer_cap_embedder_s,
                    transformer_x_embedder_s,
                )
                if value is not None
            )
        )

    transformer_noise_refiner_attention_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_noise_refiner_attention_total",
    )
    transformer_context_refiner_attention_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_context_refiner_attention_total",
    )
    transformer_siglip_refiner_attention_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_siglip_refiner_attention_total",
    )
    transformer_main_attention_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_main_attention_total",
    )
    transformer_self_attn_total_s = _sum_profile_sections(
        pipeline_timing,
        [
            "transformer_noise_refiner_attention_total",
            "transformer_context_refiner_attention_total",
            "transformer_siglip_refiner_attention_total",
            "transformer_main_attention_total",
        ],
    )

    transformer_noise_refiner_ffn_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_noise_refiner_ffn_total",
    )
    transformer_context_refiner_ffn_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_context_refiner_ffn_total",
    )
    transformer_siglip_refiner_ffn_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_siglip_refiner_ffn_total",
    )
    transformer_main_ffn_total_s = get_section_total_s(
        pipeline_timing,
        "transformer_main_ffn_total",
    )
    transformer_ffn_total_s = _sum_profile_sections(
        pipeline_timing,
        [
            "transformer_noise_refiner_ffn_total",
            "transformer_context_refiner_ffn_total",
            "transformer_siglip_refiner_ffn_total",
            "transformer_main_ffn_total",
        ],
    )

    transformer_final_layer_s = get_section_total_s(pipeline_timing, "transformer_final_layer")
    transformer_output_head_s = transformer_final_layer_s
    transformer_cross_attn_total_s = None

    transformer_non_attention_s = None
    if transformer_total_s is not None and attention_time_s is not None:
        transformer_non_attention_s = max(float(transformer_total_s) - float(attention_time_s), 0.0)

    transformer_self_attn_overhead_s = None
    if attention_time_s is not None:
        transformer_self_attn_overhead_s = max(
            float(transformer_self_attn_total_s) - float(attention_time_s),
            0.0,
        )

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
        and sparse_runtime_sparsity is None
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
        "transformer_t_embedder_s": transformer_t_embedder_s,
        "transformer_cap_embedder_s": transformer_cap_embedder_s,
        "transformer_x_embedder_s": transformer_x_embedder_s,
        "transformer_input_prep_s": transformer_input_prep_s,
        "transformer_self_attn_total_s": transformer_self_attn_total_s,
        "transformer_self_attn_overhead_s": transformer_self_attn_overhead_s,
        "transformer_cross_attn_total_s": transformer_cross_attn_total_s,
        "transformer_noise_refiner_attention_total_s": transformer_noise_refiner_attention_total_s,
        "transformer_context_refiner_attention_total_s": transformer_context_refiner_attention_total_s,
        "transformer_siglip_refiner_attention_total_s": transformer_siglip_refiner_attention_total_s,
        "transformer_main_attention_total_s": transformer_main_attention_total_s,
        "transformer_ffn_total_s": transformer_ffn_total_s,
        "transformer_noise_refiner_ffn_total_s": transformer_noise_refiner_ffn_total_s,
        "transformer_context_refiner_ffn_total_s": transformer_context_refiner_ffn_total_s,
        "transformer_siglip_refiner_ffn_total_s": transformer_siglip_refiner_ffn_total_s,
        "transformer_main_ffn_total_s": transformer_main_ffn_total_s,
        "transformer_final_layer_s": transformer_final_layer_s,
        "transformer_output_head_s": transformer_output_head_s,
        "transformer_other_s": transformer_other_s,
        "peak_memory_gb": None if peak_memory_gb is None else float(peak_memory_gb),
        "attention_timing": attention_timing,
        "pipeline_timing": pipeline_timing,
        "sparse_runtime_sparsity": sparse_runtime_sparsity,
        "sink_runtime_sparsity": sparse_runtime_sparsity,
    }


def build_runtime_profile_table(runtime_profile: Optional[dict]) -> list[dict]:
    if not runtime_profile:
        return []

    pipe_time_s = runtime_profile.get("pipe_time_s")
    transformer_total_s = runtime_profile.get("transformer_total_s")
    rows = [
        ("pipeline_total", runtime_profile.get("pipe_time_s"), False),
        ("transformer_total", runtime_profile.get("transformer_total_s"), True),
        ("attention_inner", runtime_profile.get("attention_time_s"), True),
        ("self_attn_overhead", runtime_profile.get("transformer_self_attn_overhead_s"), True),
        ("cross_attn_total", runtime_profile.get("transformer_cross_attn_total_s"), True),
        ("ffn_total", runtime_profile.get("transformer_ffn_total_s"), True),
        ("input_prep", runtime_profile.get("transformer_input_prep_s"), True),
        ("output_head", runtime_profile.get("transformer_output_head_s"), True),
        ("transformer_other", runtime_profile.get("transformer_other_s"), True),
        ("vae_decode", runtime_profile.get("vae_decode_s"), False),
        ("encode_prompt", runtime_profile.get("encode_prompt_s"), False),
        ("scheduler_step", runtime_profile.get("scheduler_step_s"), False),
        ("postprocess_video", runtime_profile.get("postprocess_video_s"), False),
    ]
    table = []
    for item, value, is_transformer_item in rows:
        share_of_pipe = None
        share_of_transformer = None
        if value is not None and pipe_time_s is not None and float(pipe_time_s) > 0:
            share_of_pipe = float(value) / float(pipe_time_s)
        if (
            is_transformer_item
            and value is not None
            and transformer_total_s is not None
            and float(transformer_total_s) > 0
        ):
            share_of_transformer = float(value) / float(transformer_total_s)
        table.append(
            {
                "item": item,
                "time_s": value,
                "share_of_pipe": share_of_pipe,
                "share_of_transformer": share_of_transformer,
            }
        )
    return table


def require_zimage_pipeline():
    try:
        import transformers

        getattr(transformers, "Qwen3Model")
    except Exception as exc:
        raise ImportError(
            "Z-Image requires a transformers build that exposes Qwen3Model. "
            "Upgrade transformers before running zimg inference."
        ) from exc
    try:
        import diffusers

        ZImagePipeline = getattr(diffusers, "ZImagePipeline")
    except Exception as exc:
        raise ImportError(
            "Z-Image requires a diffusers build with ZImagePipeline support. "
            "Install diffusers from source or a version >= 0.36.0."
        ) from exc
    return ZImagePipeline


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported --dtype value: {dtype_name}")


def load_prompts(prompt_file: str) -> list[str]:
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def resolve_prompts(prompt: Optional[str], prompt_file: str, num_prompts: Optional[int]) -> list[str]:
    if prompt is not None and prompt.strip():
        return [prompt.strip()]
    prompts = load_prompts(prompt_file)
    if num_prompts is not None and int(num_prompts) > 0:
        prompts = prompts[: int(num_prompts)]
    return prompts


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense/Sink inference for Z-Image-Turbo.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Local Z-Image model path.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype: bfloat16, float16 or float32.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt override.")
    parser.add_argument("--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE), help="Prompt file.")
    parser.add_argument("--num_prompts", type=int, default=None, help="Optional number of prompts to use from prompt_file.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output image directory.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id.")
    parser.add_argument("--seeds", type=str, default="8888", help="Comma-separated seeds.")
    parser.add_argument("--height", type=int, default=None, help="Image height.")
    parser.add_argument("--width", type=int, default=None, help="Image width.")
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS, help="Inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Guidance scale.")
    parser.add_argument("--max_sequence_length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH, help="Max prompt length.")
    parser.add_argument("--attn_mode", choices=["dense", "sink"], default="dense", help="Attention mode.")
    parser.add_argument("--sink_mask_path", type=str, default=None, help="Static sink-mask package path.")
    parser.add_argument("--sink_profile", type=str, default=DEFAULT_SINK_PROFILE, help=f"Sink profile: {', '.join(list_sink_profiles())}")
    parser.add_argument(
        "--sink_force_dense_steps",
        type=str,
        default=None,
        help="Comma-separated denoising steps that should run in dense mode. Supports negative indices like -1.",
    )
    parser.add_argument(
        "--sink_force_dense_layers",
        type=str,
        default=None,
        help="Comma-separated transformer layer indices that should run in dense mode. Supports negative indices like -1.",
    )
    parser.add_argument("--save_latent", action="store_true", help="Also save latent tensors.")
    args = parser.parse_args()
    if not args.model_path:
        parser.error("Pass --model_path or set ZIMG_MODEL to a local Diffusers-format model directory.")
    return args


def main() -> None:
    args = build_args()
    ZImagePipeline = require_zimage_pipeline()
    model_path = require_model_path(args.model_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Z-Image inference.")
    if int(args.gpu) >= torch.cuda.device_count():
        raise ValueError(f"Requested GPU {args.gpu}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")

    prompts = resolve_prompts(args.prompt, args.prompt_file, args.num_prompts)
    if not prompts:
        raise ValueError("No prompts resolved for inference.")
    seeds = iter_seed_list(parse_int_list(args.seeds) or [8888])
    dtype = resolve_torch_dtype(args.dtype)
    (
        model_spec,
        resolved_height,
        resolved_width,
        resolved_num_inference_steps,
        resolved_guidance_scale,
        resolved_text_length,
    ) = resolve_generation_defaults(
        model_path=model_path,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
    )
    token_width, token_height, token_depth = infer_zimg_token_layout(
        height=resolved_height,
        width=resolved_width,
    )

    device = f"cuda:{int(args.gpu)}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    runtime_collector = enable_pipeline_runtime_profiling(pipe)

    sink_setup = None
    if args.attn_mode == "sink":
        if not args.sink_mask_path:
            raise ValueError("--sink_mask_path is required when --attn_mode=sink")
        set_attention_timing_enabled(True)
        _, profile_overrides = merge_sink_profile_overrides(
            args.sink_profile,
            {
                "token_width": token_width,
                "token_height": token_height,
                "token_depth": token_depth,
                "text_length": resolved_text_length,
                "force_dense_steps": parse_int_list(args.sink_force_dense_steps),
                "force_dense_layers": parse_int_list(args.sink_force_dense_layers),
            },
        )
        sink_setup = set_sink_sparse_attn_zimage(
            pipe.transformer,
            args.sink_mask_path,
            config_overrides=profile_overrides,
        )
    else:
        set_attention_timing_enabled(False)

    run_summaries = []
    for prompt_idx, prompt in enumerate(prompts):
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            runtime_collector.reset()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)
            if args.attn_mode == "sink":
                reset_sink_runtime_state_zimage(pipe.transformer, num_inference_steps=resolved_num_inference_steps)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_t = time.perf_counter()
            output = pipe(
                prompt=prompt,
                height=resolved_height,
                width=resolved_width,
                num_inference_steps=resolved_num_inference_steps,
                guidance_scale=resolved_guidance_scale,
                generator=generator,
                max_sequence_length=resolved_text_length,
                output_type="pil",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - start_t

            image = output.images[0]
            tag = f"p{prompt_idx + 1:02d}_seed{seed}"
            image_path = output_dir / f"{tag}.png"
            image.save(image_path)

            latent_path = None
            if args.save_latent:
                latent_output = pipe(
                    prompt=prompt,
                    height=resolved_height,
                    width=resolved_width,
                    num_inference_steps=resolved_num_inference_steps,
                    guidance_scale=resolved_guidance_scale,
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                    max_sequence_length=resolved_text_length,
                    output_type="latent",
                )
                latent_path = output_dir / f"{tag}.pt"
                torch.save(latent_output.images, latent_path)

            peak_memory_gb = None
            if torch.cuda.is_available():
                peak_memory_gb = float(torch.cuda.max_memory_allocated(device=device)) / (1024.0 ** 3)
            runtime_profile = collect_generation_runtime_profile(
                pipe,
                pipe_time_s=float(elapsed_s),
                peak_memory_gb=peak_memory_gb,
            )
            runtime_profile_table = build_runtime_profile_table(runtime_profile)
            attention_timing = None if runtime_profile is None else runtime_profile.get("attention_timing")
            pipeline_timing = None if runtime_profile is None else runtime_profile.get("pipeline_timing")
            sparse_runtime_sparsity = None if runtime_profile is None else runtime_profile.get("sparse_runtime_sparsity")

            summary = {
                "prompt_index": prompt_idx,
                "seed": int(seed),
                "prompt": prompt,
                "image_path": str(image_path),
                "latent_path": None if latent_path is None else str(latent_path),
                "elapsed_s": float(elapsed_s),
                "attn_mode": args.attn_mode,
                "runtime_profile": runtime_profile,
                "runtime_profile_table": runtime_profile_table,
                "attention_timing": attention_timing,
                "pipeline_timing": pipeline_timing,
                "sparse_runtime_sparsity": sparse_runtime_sparsity,
                "sink_runtime_sparsity": sparse_runtime_sparsity,
            }
            run_summaries.append(summary)

    sink_setup_summary = None
    if sink_setup is not None:
        sink_setup_summary = dict(sink_setup)
        if "config" in sink_setup_summary:
            config_value = sink_setup_summary["config"]
            sink_setup_summary["config"] = asdict(config_value) if hasattr(config_value, "__dataclass_fields__") else config_value

    summary = {
        "model_spec": model_spec.describe(),
        "height": resolved_height,
        "width": resolved_width,
        "num_inference_steps": resolved_num_inference_steps,
        "guidance_scale": resolved_guidance_scale,
        "max_sequence_length": resolved_text_length,
        "attn_mode": args.attn_mode,
        "sink_setup": sink_setup_summary,
        "runs": run_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.stdout.write(f"Saved {len(run_summaries)} image(s) to {output_dir}; summary: {summary_path}\n")


if __name__ == "__main__":
    main()
