"""Terminal playback, with a clock that does not drift.

The obvious player sleeps `1/fps` after writing each frame. It is wrong, and the error is
cumulative: every frame pays for its own rendering time plus the operating system's rounding
on the sleep, and none of it is ever repaid. At 24 fps with half a millisecond of overhead
per frame, a ten minute clip finishes seven seconds late, and any audio played alongside it
has separated long before that.

The fix is to schedule against absolute deadlines derived from a single origin. Frame `i` is
due at `t0 + i / fps`, computed from `i` rather than accumulated, so a slow frame steals from
its own budget and not from every frame after it. When a frame is already overdue by more
than one frame interval it is dropped instead of shown late, which is what keeps the picture
attached to the sound.

Both policies live here so the difference can be measured rather than asserted:
`schedule` is the correct one and `naive_schedule` is the drifting one, and the test suite
runs both over a long simulated clip.
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from .render import CLEAR, HIDE, HOME, RESET, SHOW, DeltaEncoder, to_ansi


@dataclass(frozen=True)
class Tick:
    index: int
    due: float       # when this frame should be shown, seconds from the origin
    shown: float     # when it actually was
    dropped: bool


def schedule(n: int, fps: float, now: Callable[[], float], sleep: Callable[[float], None],
             *, origin: float | None = None, drop: bool = True) -> Iterator[Tick]:
    """Yield one Tick per frame, sleeping until each absolute deadline.

    `now` and `sleep` are injected so a test can run ten thousand frames of a clip in
    milliseconds and inspect the drift, which is not something you can measure by watching.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    t0 = now() if origin is None else origin
    period = 1.0 / fps
    for i in range(n):
        due = t0 + i * period          # from i, never `due += period`
        t = now()
        if t < due:
            sleep(due - t)
            t = now()
        if drop and t > due + period:
            yield Tick(i, due, t, True)
            continue
        yield Tick(i, due, t, False)


def naive_schedule(n: int, fps: float, now: Callable[[], float],
                   sleep: Callable[[float], None]) -> Iterator[Tick]:
    """The drifting player, kept so the drift is a measurement and not a story."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    t0 = now()
    period = 1.0 / fps
    for i in range(n):
        yield Tick(i, t0 + i * period, now(), False)
        sleep(period)


def audio_command(path: str) -> list[str] | None:
    """The command that would play `path`, or None if no player is installed.

    Returning None rather than raising, and rather than silently doing nothing, is the
    point: the caller has to decide out loud whether missing audio is acceptable.
    """
    for exe, args in (("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
                      ("aplay", ["-q"]),
                      ("paplay", [])):
        found = shutil.which(exe)
        if found:
            return [found, *args, path]
    return None


def play(frames: Iterable, fps: float = 24.0, *, out=None, delta: bool = True,
         loop: int = 1, audio: str | None = None, quiet: bool = False,
         delta_tol: int = 0) -> dict:
    """Play rendered frames to a terminal. Returns playback statistics.

    `frames` is a sequence of `render.Cells`. Delta encoding sends only the cells that
    changed, which is the difference between a clip that plays over SSH and one that stalls.
    """
    out = out or sys.stdout
    seq = list(frames)
    if not seq:
        raise ValueError("nothing to play")

    proc = None
    if audio:
        cmd = audio_command(audio)
        if cmd is None:
            raise RuntimeError(
                "audio was requested but no player was found. Install one of ffplay, aplay "
                "or paplay, or drop --audio. Video timing is unaffected either way."
            )
        import subprocess
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total = len(seq) * max(1, loop)
    written = dropped = 0
    bytes_out = 0
    worst = 0.0
    out.write(HIDE + CLEAR + HOME)
    try:
        enc = DeltaEncoder(tol=delta_tol) if delta else None
        for tick in schedule(total, fps, time.monotonic, _sleep_until_ready):
            cells = seq[tick.index % len(seq)]
            if tick.dropped:
                # A dropped frame is never sent, so the screen state is untouched and the
                # next delta is still correct. Nothing needs resetting.
                dropped += 1
                continue
            chunk = enc.encode(cells) if enc is not None else to_ansi(cells)
            out.write(chunk)
            out.flush()
            bytes_out += len(chunk)
            written += 1
            worst = max(worst, abs(tick.shown - tick.due))
    finally:
        out.write(RESET + SHOW + "\n")
        out.flush()
        if proc is not None:
            proc.terminate()

    stats = {
        "frames": total, "written": written, "dropped": dropped,
        "bytes": bytes_out, "bytes_per_frame": round(bytes_out / max(1, written), 1),
        "worst_offset_ms": round(worst * 1000.0, 3),
        "audio": bool(proc),
        "full_frames": enc.full_frames if enc else written,
        "delta_frames": enc.delta_frames if enc else 0,
    }
    if not quiet:
        print(f"{written} frames, {dropped} dropped, {stats['bytes_per_frame']} bytes/frame, "
              f"worst timing offset {stats['worst_offset_ms']} ms", file=sys.stderr)
    return stats


def _sleep_until_ready(d: float) -> None:
    if d > 0:
        time.sleep(d)
