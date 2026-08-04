#!/usr/bin/env python3
"""A deterministic fingerprint of the parts of the pipeline the study never touches.

`scripts/verify.sh` proves a sabotage changed something before asking whether anything
noticed, and it does that by comparing output against a baseline. The study alone is not
enough of a comparison: it never runs the player and never writes a `.cast`, so a sabotage
of the frame clock produces byte-identical study output and gets correctly reported as
proving nothing. That is a hole in the comparison, not in the checks.

So this prints numbers covering the clock, the exports and the delta encoder, from a
simulated clock so the output is reproducible. verify.sh folds it into the fingerprint.

    python3 scripts/probe.py
"""

from __future__ import annotations

import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from asciivid import cast, font, player, render, scenes  # noqa: E402

FPS = 24.0
FRAMES = 24 * 120          # two minutes, long enough for drift to show


def main() -> int:
    # 1. The clock. A fake clock with a fixed per-frame cost, so the drift is exact.
    state = {"t": 1000.0}
    overhead = 0.0004
    worst = 0.0
    for tick in player.schedule(FRAMES, FPS, lambda: state["t"],
                                lambda d: state.__setitem__("t", state["t"] + max(0.0, d))):
        state["t"] += overhead
        worst = max(worst, abs(tick.shown - tick.due))
    print(f"clock worst_offset_us {worst * 1e6:.0f}")
    print(f"clock elapsed_ms {(state['t'] - 1000.0) * 1000:.3f}")

    # 2. The export. Absolute timestamps, checked against the source frame rate.
    buf = io.StringIO()
    n = cast.write(buf, 40, 15, [f"f{i}" for i in range(200)], fps=FPS)
    buf.seek(0)
    head, events = cast.read(buf)
    gaps = list(cast.durations(events))
    print(f"cast events {n} width {head['width']} height {head['height']}")
    print(f"cast last_ts {events[-1][0]:.6f} gap_min {min(gaps):.6f} gap_max {max(gaps):.6f}")

    # 3. The delta encoder, on content where it helps and content where it does not.
    fnt = font.load()
    for scene, mode, tol in (("glyphs", "mono", 0), ("tunnel", "truecolor", 0),
                             ("plasma", "truecolor", 8)):
        spec = render.Spec("blocks", mode, "structure")
        frames = [render.render(scenes.frame(scene, i / FPS, 320, 240), fnt, spec, 40)
                  for i in range(12)]
        enc = render.DeltaEncoder(tol=tol)
        stream = sum(len(enc.encode(f)) for f in frames)
        full = sum(len(render.to_ansi(f)) for f in frames)
        print(f"delta {scene}/{mode}/tol{tol} stream {stream} full {full} "
              f"whole {enc.full_frames} delta {enc.delta_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
