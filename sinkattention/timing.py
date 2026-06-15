from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import torch


class AttentionTimingCollector:
    def __init__(self, *, label: str, enabled: Optional[bool] = None):
        if enabled is None:
            raw_enabled = os.environ.get(
                "SINK_ATTN_ENABLE_TIMING",
                os.environ.get("WAN_ATTN_ENABLE_TIMING", "0"),
            )
            enabled = raw_enabled.strip().lower() in {"1", "true", "yes", "on"}
        self.label = label
        self.enabled = bool(enabled)
        self._cuda_records: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_records: List[Tuple[str, float]] = []

    def reset(self) -> None:
        self._cuda_records.clear()
        self._cpu_records.clear()

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_records.append((name, start, end))
            return

        start_t = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            self._cpu_records.append((name, elapsed_ms))

    def summarize(self) -> dict:
        if not self.enabled:
            return {
                "enabled": False,
                "label": self.label,
                "num_records": 0,
                "by_section": {},
                "total_ms": 0.0,
            }

        if self._cuda_records and torch.cuda.is_available():
            torch.cuda.synchronize()

        total_by_section: Dict[str, float] = defaultdict(float)
        count_by_section: Dict[str, int] = defaultdict(int)

        for name, start, end in self._cuda_records:
            total_by_section[name] += float(start.elapsed_time(end))
            count_by_section[name] += 1

        for name, elapsed_ms in self._cpu_records:
            total_by_section[name] += float(elapsed_ms)
            count_by_section[name] += 1

        ordered_names = [
            name
            for name, _ in sorted(total_by_section.items(), key=lambda item: (-item[1], item[0]))
        ]
        by_section = {
            name: {
                "total_ms": total_by_section[name],
                "mean_ms": total_by_section[name] / max(count_by_section[name], 1),
                "count": count_by_section[name],
            }
            for name in ordered_names
        }
        total_ms = float(sum(total_by_section.values()))
        return {
            "enabled": True,
            "label": self.label,
            "num_records": int(sum(count_by_section.values())),
            "by_section": by_section,
            "total_ms": total_ms,
        }


def extract_attention_time_ms(
    summary: Optional[dict],
    *,
    preferred_sections: Iterable[str] = ("inner_total",),
) -> Optional[float]:
    if not summary or not summary.get("enabled", False):
        return None

    by_section = summary.get("by_section") or {}
    for section_name in preferred_sections:
        section_summary = by_section.get(section_name)
        if section_summary is None:
            continue
        total_ms = section_summary.get("total_ms")
        if total_ms is not None:
            return float(total_ms)

    total_ms = summary.get("total_ms")
    if total_ms is None:
        return None
    return float(total_ms)


def aggregate_timing_summaries(summaries: Iterable[Optional[dict]]) -> Optional[dict]:
    valid_summaries = [summary for summary in summaries if summary and summary.get("enabled", False)]
    if not valid_summaries:
        return None

    values_by_section: Dict[str, List[float]] = defaultdict(list)
    total_ms_values: List[float] = []

    for summary in valid_summaries:
        total_ms = summary.get("total_ms")
        if total_ms is not None:
            total_ms_values.append(float(total_ms))
        by_section = summary.get("by_section") or {}
        for section_name, section_summary in by_section.items():
            total_ms = section_summary.get("total_ms")
            if total_ms is None:
                continue
            values_by_section[str(section_name)].append(float(total_ms))

    mean_by_section_ms = {
        name: float(sum(values) / len(values))
        for name, values in sorted(values_by_section.items(), key=lambda item: item[0])
        if values
    }
    max_by_section_ms = {
        name: float(max(values))
        for name, values in sorted(values_by_section.items(), key=lambda item: item[0])
        if values
    }
    total_by_section_ms = {
        name: float(sum(values))
        for name, values in sorted(values_by_section.items(), key=lambda item: item[0])
        if values
    }

    return {
        "count": len(valid_summaries),
        "labels": sorted({str(summary.get("label", "")) for summary in valid_summaries if summary.get("label")}),
        "mean_total_ms": float(sum(total_ms_values) / len(total_ms_values)) if total_ms_values else None,
        "max_total_ms": float(max(total_ms_values)) if total_ms_values else None,
        "mean_by_section_ms": mean_by_section_ms,
        "max_by_section_ms": max_by_section_ms,
        "total_by_section_ms": total_by_section_ms,
    }
