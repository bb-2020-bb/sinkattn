from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

DIFFUSERS_REQUIRED_PATHS = [
    "model_index.json",
    "transformer/config.json",
    "vae/config.json",
    "tokenizer/tokenizer.json",
]
RAW_WAN_MARKER_PATHS = [
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
]


def _looks_like_local_path(model_path: str, path: Path) -> bool:
    expanded = str(model_path).strip()
    return path.is_absolute() or expanded.startswith(("~", ".", "..")) or "/" in expanded


def _missing_paths(root: Path, required_paths: list[str]) -> list[str]:
    return [rel_path for rel_path in required_paths if not (root / rel_path).exists()]


def is_diffusers_wan_dir(path: Path) -> bool:
    return path.is_dir() and not _missing_paths(path, DIFFUSERS_REQUIRED_PATHS)


def _looks_like_raw_wan_release(path: Path) -> bool:
    return path.is_dir() and any((path / rel_path).exists() for rel_path in RAW_WAN_MARKER_PATHS)


def inspect_wan_model_path(model_path: str) -> dict[str, Any]:
    requested_path = str(model_path)
    path = Path(model_path).expanduser()
    info: dict[str, Any] = {
        "requested_model_path": requested_path,
        "resolved_model_path": requested_path,
        "loader": "diffusers",
        "layout": "unresolved_or_repo_id",
    }

    if not path.exists():
        if _looks_like_local_path(requested_path, path):
            raise FileNotFoundError(
                "Wan model path does not exist locally: "
                f"{path}. Pass a local Diffusers-format Wan directory via --model_path "
                "or set WAN_MODEL."
            )
        info["message"] = (
            "Model path does not exist locally; treating it as a Diffusers repo id or unresolved path."
        )
        return info

    resolved_path = str(path.resolve())
    info["resolved_model_path"] = resolved_path

    if is_diffusers_wan_dir(path):
        info["layout"] = "diffusers_local"
        info["message"] = "Detected a local Diffusers-format Wan model directory."
        return info

    missing_diffusers = _missing_paths(path, DIFFUSERS_REQUIRED_PATHS)
    if _looks_like_raw_wan_release(path):
        raise FileNotFoundError(
            "The public Wan route expects a Diffusers-format model directory. "
            f"Path {path} looks like an original Wan release. Convert it to Diffusers format "
            "or pass an existing Diffusers snapshot with these files: "
            f"{DIFFUSERS_REQUIRED_PATHS}."
        )

    raise FileNotFoundError(
        f"Unsupported Wan model directory: {path}. Missing Diffusers files: {missing_diffusers}."
    )


def _build_unipc_scheduler(flow_shift: float) -> UniPCMultistepScheduler:
    return UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift,
    )


def _build_diffusers_pipeline(model_root: str, flow_shift: float) -> WanPipeline:
    vae = AutoencoderKLWan.from_pretrained(model_root, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_root, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = _build_unipc_scheduler(flow_shift)
    return pipe


def load_wan_pipeline(model_path: str, flow_shift: float = 3.0) -> tuple[WanPipeline, dict[str, Any]]:
    model_info = inspect_wan_model_path(model_path)
    pipe = _build_diffusers_pipeline(model_info["resolved_model_path"], flow_shift=flow_shift)
    pipe._wan_model_source_info = model_info
    return pipe, model_info
