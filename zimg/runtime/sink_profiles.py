from __future__ import annotations

from copy import deepcopy
from typing import Optional


SINK_RUNTIME_PROFILES: dict[str, dict] = {
    "static_only": {
        "dynamic_mode": "none",
        "runtime_diagonal_band_width": 0,
        "mixing_mode": "adaptive",
        "sample_gap": 30,
        "force_text_global_attention": True,
    },
    "static_comp": {
        "dynamic_mode": "pooled_kv",
        "dynamic_kernel": "auto",
        "runtime_diagonal_band_width": 0,
        "mixing_mode": "adaptive",
        "sample_gap": 30,
        "force_text_global_attention": True,
    },
}

DEFAULT_SINK_PROFILE = "static_only"


def list_sink_profiles() -> list[str]:
    return sorted(SINK_RUNTIME_PROFILES.keys())


def resolve_default_sink_profile(profile_name: Optional[str]) -> str:
    normalized = None if profile_name is None else profile_name.strip().lower()
    if normalized:
        if normalized not in SINK_RUNTIME_PROFILES:
            raise ValueError(
                f"Unsupported sink profile: {profile_name}. "
                f"Available profiles: {', '.join(list_sink_profiles())}"
            )
        return normalized
    return DEFAULT_SINK_PROFILE


def resolve_sink_profile(profile_name: str | None) -> dict:
    name = resolve_default_sink_profile(profile_name)
    return deepcopy(SINK_RUNTIME_PROFILES[name])


def merge_sink_profile_overrides(profile_name: str | None, overrides: dict) -> tuple[str, dict]:
    resolved_name = resolve_default_sink_profile(profile_name)
    merged = resolve_sink_profile(resolved_name)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return resolved_name, merged


__all__ = [
    "DEFAULT_SINK_PROFILE",
    "list_sink_profiles",
    "merge_sink_profile_overrides",
    "resolve_default_sink_profile",
    "resolve_sink_profile",
]
