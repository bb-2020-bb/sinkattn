from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_LOCAL_MODEL_PATH = os.environ.get("ZIMG_MODEL")
DEFAULT_IMAGE_HEIGHT = 2048
DEFAULT_IMAGE_WIDTH = 2048
DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 0.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512
_VAE_SCALE = 16


def _looks_like_local_path(model_path: str, path: Path) -> bool:
    expanded = str(model_path).strip()
    return path.is_absolute() or expanded.startswith(("~", ".", "..")) or "/" in expanded


@dataclass(frozen=True)
class ZImageModelSpec:
    model_path: str
    source: str
    dim: Optional[int]
    n_layers: Optional[int]
    n_heads: Optional[int]
    n_kv_heads: Optional[int]
    in_channels: Optional[int]
    patch_size: int
    f_patch_size: int
    max_sequence_length: int

    @property
    def default_height(self) -> int:
        return DEFAULT_IMAGE_HEIGHT

    @property
    def default_width(self) -> int:
        return DEFAULT_IMAGE_WIDTH

    @property
    def default_num_inference_steps(self) -> int:
        return DEFAULT_NUM_INFERENCE_STEPS

    @property
    def default_guidance_scale(self) -> float:
        return DEFAULT_GUIDANCE_SCALE

    @property
    def default_text_length(self) -> int:
        return int(self.max_sequence_length or DEFAULT_MAX_SEQUENCE_LENGTH)

    def infer_token_layout(self, *, height: int, width: int) -> tuple[int, int, int]:
        token_width = max(int(width) // _VAE_SCALE, 1)
        token_height = max(int(height) // _VAE_SCALE, 1)
        token_depth = 1
        return token_width, token_height, token_depth

    def describe(self) -> dict:
        return {
            "model_path": self.model_path,
            "source": self.source,
            "default_height": self.default_height,
            "default_width": self.default_width,
            "default_num_inference_steps": self.default_num_inference_steps,
            "default_guidance_scale": self.default_guidance_scale,
            "default_text_length": self.default_text_length,
            "patch_size": self.patch_size,
            "f_patch_size": self.f_patch_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "in_channels": self.in_channels,
        }


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_zimage_model_spec(model_path: str) -> ZImageModelSpec:
    model_dir = Path(model_path)
    transformer_config = _read_json(model_dir / "transformer" / "config.json")
    if transformer_config is None:
        return ZImageModelSpec(
            model_path=str(model_path),
            source="fallback",
            dim=None,
            n_layers=None,
            n_heads=None,
            n_kv_heads=None,
            in_channels=None,
            patch_size=2,
            f_patch_size=1,
            max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        )

    all_patch_size = transformer_config.get("all_patch_size", [2])
    all_f_patch_size = transformer_config.get("all_f_patch_size", [1])
    return ZImageModelSpec(
        model_path=str(model_path),
        source="local_transformer_config",
        dim=transformer_config.get("dim"),
        n_layers=transformer_config.get("n_layers"),
        n_heads=transformer_config.get("n_heads"),
        n_kv_heads=transformer_config.get("n_kv_heads"),
        in_channels=transformer_config.get("in_channels"),
        patch_size=int(all_patch_size[0]) if all_patch_size else 2,
        f_patch_size=int(all_f_patch_size[0]) if all_f_patch_size else 1,
        max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
    )


def require_model_path(model_path: Optional[str], env_var: str = "ZIMG_MODEL") -> str:
    if model_path is not None and str(model_path).strip():
        requested_path = str(model_path)
        path = Path(requested_path).expanduser()
        if _looks_like_local_path(requested_path, path) and not path.exists():
            raise FileNotFoundError(
                "Z-Image model path does not exist locally: "
                f"{path}. Pass a local Diffusers-format Z-Image directory via --model_path "
                f"or set {env_var}."
            )
        return requested_path
    raise ValueError(f"Pass --model_path or set {env_var} to a local Diffusers-format model directory.")


def resolve_generation_defaults(
    *,
    model_path: str,
    height: Optional[int],
    width: Optional[int],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    max_sequence_length: Optional[int],
) -> tuple[ZImageModelSpec, int, int, int, float, int]:
    spec = load_zimage_model_spec(model_path)
    resolved_height = int(height) if height is not None else spec.default_height
    resolved_width = int(width) if width is not None else spec.default_width
    resolved_num_inference_steps = (
        int(num_inference_steps) if num_inference_steps is not None else spec.default_num_inference_steps
    )
    resolved_guidance_scale = float(guidance_scale) if guidance_scale is not None else spec.default_guidance_scale
    resolved_text_length = int(max_sequence_length) if max_sequence_length is not None else spec.default_text_length
    return (
        spec,
        resolved_height,
        resolved_width,
        resolved_num_inference_steps,
        resolved_guidance_scale,
        resolved_text_length,
    )


__all__ = [
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_LOCAL_MODEL_PATH",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "ZImageModelSpec",
    "load_zimage_model_spec",
    "require_model_path",
    "resolve_generation_defaults",
]
