"""Burst detection: split a camera's frames wherever the time gap exceeds `burst_gap_s`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeItem:
    """The timing identity of one image."""

    camera: str
    timestamp: float
    ts_source: str = "exif"  # "exif" | "mtime"


def time_order(items: list[TimeItem]) -> list[int]:
    """Indices sorted by (camera, timestamp, original index)."""
    return sorted(range(len(items)), key=lambda i: (items[i].camera, items[i].timestamp, i))


def assign_bursts(items: list[TimeItem], gap_s: float = 2.0) -> list[int]:
    """Burst id per item (aligned with `items`), numbered in (camera, time) order.

    A new burst starts when the camera changes or the gap to the previous frame exceeds
    `gap_s`. Frames whose timestamp came from the file mtime get their own burst (the
    mtime of copied files says nothing about when the shot was taken).
    """
    ids = [0] * len(items)
    prev: TimeItem | None = None
    burst = -1
    for i in time_order(items):
        it = items[i]
        new = (
            prev is None
            or it.ts_source == "mtime"
            or prev.ts_source == "mtime"
            or it.camera != prev.camera
            or it.timestamp - prev.timestamp > gap_s
        )
        if new:
            burst += 1
        ids[i] = burst
        prev = it
    return ids
