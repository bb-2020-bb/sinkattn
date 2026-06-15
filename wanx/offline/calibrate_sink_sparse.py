import argparse
import gc
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_HF_CACHE_ROOT = Path.home() / ".cache" / "sinkattention" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DIFFUSERS_CACHE", str(DEFAULT_HF_CACHE_ROOT / "diffusers"))

import torch
from sinkattention.sparse_attention import (
    SinkMaskCalibrator,
    SinkSparseConfig,
    iter_seed_list,
    parse_int_list,
    summarize_sink_package,
)

from wanx.runtime.sink_sparse_wan import (
    set_sink_calibration_attn_wanx,
)

WAN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = os.environ.get("WAN_MODEL")
DEFAULT_PROMPT_FILE = WAN_ROOT / "prompts" / "calibration" / "sink_calibration_wanx_v1.txt"
DEFAULT_OUTPUT_MASK_PATH = WAN_ROOT / "outputs" / "sink_sparse" / "wan_sink_mask.pt"

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def load_prompts(prompt_file: str) -> list[str]:
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def require_model_path(model_path: str | None, env_var: str) -> str:
    if model_path is not None and str(model_path).strip():
        return str(model_path)
    raise ValueError(f"Pass --model_path or set {env_var} to a local Diffusers-format model directory.")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Sink-Sparse calibration for Wan.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the Wan model.")
    parser.add_argument("--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE), help="Calibration prompt file.")
    parser.add_argument("--output_mask_path", type=str, default=str(DEFAULT_OUTPUT_MASK_PATH), help="Output path for the sink mask package.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id.")
    parser.add_argument("--num_prompts", type=int, default=4, help="Number of prompts sampled from the prompt file.")
    parser.add_argument("--seeds", type=str, default="8888,9999", help="Comma-separated calibration seeds.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps used during calibration.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale used during calibration.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--flow_shift", type=float, default=3.0, help="Wan flow shift.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument(
        "--output_type",
        type=str,
        default="latent",
        choices=["np", "pt", "pil", "latent"],
        help=(
            "Pipeline output type used during calibration. Use 'latent' when only attention "
            "statistics are needed, which skips VAE decode and reduces calibration memory."
        ),
    )
    parser.add_argument(
        "--save_mean_scores",
        action="store_true",
        help="Also save raw mean attention-score tensors. Disabled by default to keep sink-mask packages compact.",
    )
    parser.add_argument(
        "--offline_direct_coverage",
        type=float,
        default=0.85,
        help=(
            "For each head/query-row, keep the top key blocks until the cumulative "
            "mean block-attention mass reaches this threshold."
        ),
    )
    parser.add_argument("--block_size", type=int, default=128, help="Block size for sink scoring and sparse attention.")
    parser.add_argument("--sample_gap", type=int, default=30, help="Compression ratio for the dynamic branch.")
    parser.add_argument(
        "--pooling_mode",
        choices=["exact", "triton", "mean"],
        default="exact",
        help=(
            "Block score extractor. 'exact' aggregates real dense token attention into block-mass scores; "
            "'triton' and 'mean' keep the previous approximations."
        ),
    )
    parser.add_argument("--token_width", type=int, default=52, help="Wan latent token width.")
    parser.add_argument("--token_height", type=int, default=30, help="Wan latent token height.")
    parser.add_argument("--token_depth", type=int, default=21, help="Wan latent token depth.")
    args = parser.parse_args()
    if not args.model_path:
        parser.error("Pass --model_path or set WAN_MODEL to a local Diffusers-format model directory.")
    return args


def main() -> None:
    args = build_args()
    model_path = require_model_path(args.model_path, "WAN_MODEL")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan Sink-Sparse calibration.")

    device = "cuda:0"
    prompts = load_prompts(args.prompt_file)[: args.num_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")

    seeds = iter_seed_list(parse_int_list(args.seeds) or [8888])
    config = SinkSparseConfig(
        block_size=args.block_size,
        sample_gap=args.sample_gap,
        token_width=args.token_width,
        token_height=args.token_height,
        token_depth=args.token_depth,
        pooling_mode=args.pooling_mode,
        offline_direct_coverage=args.offline_direct_coverage,
    )

    from wanx.runtime.wan_model_loader import load_wan_pipeline

    pipe, _model_info = load_wan_pipeline(model_path, flow_shift=args.flow_shift)
    pipe.to(device)

    calibrator = SinkMaskCalibrator(config)
    set_sink_calibration_attn_wanx(pipe.transformer, calibrator)

    for prompt in prompts:
        for seed in seeds:
            calibrator.reset_generation()
            generator = torch.manual_seed(seed)
            with torch.no_grad():
                pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    output_type=args.output_type,
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    package = calibrator.save(args.output_mask_path, include_scores=args.save_mean_scores)
    sys.stdout.write(f"Saved Sink mask package to {args.output_mask_path}: {summarize_sink_package(package)}\n")


if __name__ == "__main__":
    main()
