"""asciinema v2 (`.cast`) export.

The format is one JSON header line followed by one JSON array per event:

    {"version": 2, "width": 80, "height": 30, ...}
    [0.0, "o", "[H..."]
    [0.041667, "o", "[3;7H..."]

Timestamps are absolute seconds from the start of the recording, not gaps. Writing gaps
into a field that means absolute time is the single easiest way to produce a file that
plays at the wrong speed and still parses, so `write` computes every timestamp from the
frame index and never accumulates.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator


def header(cols: int, rows: int, *, title: str = "", fps: float = 24.0) -> dict:
    return {
        "version": 2,
        "width": int(cols),
        "height": int(rows),
        "timestamp": 0,
        "idle_time_limit": 2.0,
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/sh"},
        "title": title or f"ascii-video {cols}x{rows} @ {fps:g}fps",
    }


def write(fh, cols: int, rows: int, chunks: Iterable[str], *, fps: float = 24.0,
          title: str = "", hide_cursor: bool = True) -> int:
    """Write a cast. Returns the number of events written."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    fh.write(json.dumps(header(cols, rows, title=title, fps=fps), sort_keys=True) + "\n")
    n = 0
    for i, data in enumerate(chunks):
        if i == 0 and hide_cursor:
            data = "\x1b[?25l\x1b[2J" + data
        # From the index every time. An accumulator here drifts by one float rounding per
        # frame, which is invisible for ten frames and is a visible lag over ten thousand.
        fh.write(json.dumps([round(i / fps, 6), "o", data]) + "\n")
        n += 1
    if hide_cursor and n:
        fh.write(json.dumps([round(n / fps, 6), "o", "\x1b[?25h\x1b[0m"]) + "\n")
        n += 1
    return n


def read(fh) -> tuple[dict, list[tuple[float, str, str]]]:
    """Parse a cast back. Raises on anything malformed rather than skipping the line."""
    lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty cast")
    head = json.loads(lines[0])
    if head.get("version") != 2:
        raise ValueError(f"expected a v2 cast, got version {head.get('version')!r}")
    for k in ("width", "height"):
        if not isinstance(head.get(k), int) or head[k] < 1:
            raise ValueError(f"cast header has a bad {k}: {head.get(k)!r}")
    events: list[tuple[float, str, str]] = []
    for i, ln in enumerate(lines[1:], start=2):
        ev = json.loads(ln)
        if not (isinstance(ev, list) and len(ev) == 3):
            raise ValueError(f"line {i} is not a 3-element event")
        t, kind, data = ev
        if not isinstance(t, (int, float)) or kind not in ("o", "i", "m", "r"):
            raise ValueError(f"line {i} has a bad timestamp or type")
        if events and t < events[-1][0]:
            raise ValueError(f"line {i} goes backwards in time: {t} after {events[-1][0]}")
        events.append((float(t), kind, str(data)))
    return head, events


def durations(events: list[tuple[float, str, str]]) -> Iterator[float]:
    for a, b in zip(events, events[1:]):
        yield b[0] - a[0]
