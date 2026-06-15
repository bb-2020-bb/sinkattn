from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

DEFAULT_HF_CACHE_ROOT = Path.home() / ".cache" / "sinkattention" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DIFFUSERS_CACHE", str(DEFAULT_HF_CACHE_ROOT / "diffusers"))

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sinkattention.sparse_attention import iter_seed_list, parse_int_list, summarize_sink_package
from zimg.model_spec import (
    DEFAULT_LOCAL_MODEL_PATH,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    require_model_path,
    resolve_generation_defaults,
)
from zimg.runtime.sink_sparse_zimage import (
    ZImageSinkMaskCalibrator,
    build_zimg_sink_config,
    set_sink_calibration_attn_zimage,
)


OFFLINE_DIR = Path(__file__).resolve().parent
ZIMG_ROOT = OFFLINE_DIR.parent

DEFAULT_MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH
DEFAULT_PROMPT_FILE = ZIMG_ROOT / "prompts" / "calibration_4p.txt"
DEFAULT_OUTPUT_MASK_PATH = ZIMG_ROOT / "outputs" / "sink_sparse" / "zimage_turbo_sink_mask_cov85.pt"


def require_zimage_pipeline():
    try:
        import transformers

        getattr(transformers, "Qwen3Model")
    except Exception as exc:
        raise ImportError(
            "Z-Image requires a transformers build that exposes Qwen3Model. "
            "Upgrade transformers before running zimg calibration."
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


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Sink-Sparse calibration for Z-Image-Turbo.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Local Z-Image model path.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype: bfloat16, float16 or float32.")
    parser.add_argument("--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE), help="Calibration prompt file.")
    parser.add_argument("--output_mask_path", type=str, default=str(DEFAULT_OUTPUT_MASK_PATH), help="Output sink-mask package path.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id.")
    parser.add_argument("--num_prompts", type=int, default=4, help="Number of prompts used for calibration.")
    parser.add_argument("--seeds", type=str, default="8888", help="Comma-separated calibration seeds.")
    parser.add_argument("--height", type=int, default=None, help="Image height.")
    parser.add_argument("--width", type=int, default=None, help="Image width.")
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS, help="Inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum prompt sequence length passed to the tokenizer/text encoder.",
    )
    parser.add_argument("--save_mean_scores", action="store_true", help="Also save raw mean/std tensors.")
    parser.add_argument("--offline_direct_coverage", type=float, default=0.85, help="Direct-coverage threshold.")
    parser.add_argument("--block_size", type=int, default=128, help="Block size.")
    parser.add_argument("--pooling_mode", choices=["exact", "triton", "mean"], default="exact", help="Block score extractor.")
    args = parser.parse_args()
    if not args.model_path:
        parser.error("Pass --model_path or set ZIMG_MODEL to a local Diffusers-format model directory.")
    return args


def main() -> None:
    args = build_args()
    ZImagePipeline = require_zimage_pipeline()
    model_path = require_model_path(args.model_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Z-Image Sink-Sparse calibration.")
    if int(args.gpu) >= torch.cuda.device_count():
        raise ValueError(f"Requested GPU {args.gpu}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")

    prompts = load_prompts(args.prompt_file)[: int(args.num_prompts)]
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")
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

    token_width, token_height, token_depth = model_spec.infer_token_layout(
        height=resolved_height,
        width=resolved_width,
    )
    config = build_zimg_sink_config(
        config_overrides={
            "block_size": args.block_size,
            "token_width": token_width,
            "token_height": token_height,
            "token_depth": token_depth,
            "text_length": resolved_text_length,
            "pooling_mode": args.pooling_mode,
            "offline_direct_coverage": float(args.offline_direct_coverage),
        }
    )
    calibrator = ZImageSinkMaskCalibrator(config)

    device = f"cuda:{int(args.gpu)}"
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)

    set_sink_calibration_attn_zimage(pipe.transformer, calibrator)

    for prompt in prompts:
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            _ = pipe(
                prompt=prompt,
                height=resolved_height,
                width=resolved_width,
                num_inference_steps=resolved_num_inference_steps,
                guidance_scale=resolved_guidance_scale,
                generator=generator,
                max_sequence_length=resolved_text_length,
                output_type="latent",
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    package = calibrator.save(args.output_mask_path, include_scores=args.save_mean_scores)

    summary = {
        "model_spec": model_spec.describe(),
        "height": resolved_height,
        "width": resolved_width,
        "num_inference_steps": resolved_num_inference_steps,
        "guidance_scale": resolved_guidance_scale,
        "max_sequence_length": resolved_text_length,
        "num_prompts": len(prompts),
        "seeds": seeds,
        "output_mask_path": str(args.output_mask_path),
        "config": asdict(config),
    }
    summary_path = Path(args.output_mask_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.stdout.write(
        f"Saved Sink mask package to {args.output_mask_path}: "
        f"{summarize_sink_package(package)}; summary: {summary_path}\n"
    )


if __name__ == "__main__":
    main()
