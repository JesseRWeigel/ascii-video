"""Command line entry point.

    python3 -m asciivid play    --scene tunnel --ramp blocks --mode truecolor
    python3 -m asciivid render  --scene plasma --out clip.cast --seconds 4
    python3 -m asciivid study   --out out/
    python3 -m asciivid table   --out out/
    python3 -m asciivid video   --input clip.mp4 --out clip.cast     (needs ffmpeg)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

from . import cast, font as fontmod, player, ramps, render, scenes, study

DEFAULT_FPS = 24.0


def _spec(a) -> render.Spec:
    return render.Spec(ramp=a.ramp, mode=a.mode, match=a.match, dither=a.dither,
                       normalize=a.normalize)


def _clip(name: str, seconds: float, fps: float, cols: int, aspect: float,
          fnt, spec: render.Spec, width: int, height: int) -> list:
    n = max(1, int(round(seconds * fps)))
    return [render.render(scenes.frame(name, i / fps, width, height), fnt, spec, cols,
                          aspect=aspect)
            for i in range(n)]


def _ffmpeg_frames(path: str, fps: float, width: int, height: int):
    """Decode a real video file to raw RGB frames. Requires ffmpeg on PATH."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg is not on PATH, so a video file cannot be decoded. Install ffmpeg, or "
            "use --scene for the built-in procedural sources, which need nothing."
        )
    cmd = [exe, "-v", "error", "-i", path, "-vf", f"fps={fps},scale={width}:{height}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:300]}")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    per = width * height * 3
    if len(buf) < per:
        raise RuntimeError(f"ffmpeg produced {len(buf)} bytes, less than one {width}x{height} frame")
    n = len(buf) // per
    return buf[: n * per].reshape(n, height, width, 3)


def _add_render_args(p) -> None:
    p.add_argument("--ramp", default="ascii10", choices=sorted(ramps.RAMPS))
    p.add_argument("--mode", default="truecolor", choices=list(ramps.MODES))
    p.add_argument("--match", default="structure", choices=list(ramps.MATCHERS))
    p.add_argument("--dither", action="store_true",
                   help="ordered dither against the palette (ansi16 and ansi256 only)")
    p.add_argument("--normalize", action="store_true",
                   help="stretch source contrast onto the range the ramp can express "
                        "(--mode mono only, where ink is the only brightness there is)")
    p.add_argument("--cols", type=int, default=80)
    p.add_argument("--aspect", type=float, default=2.0,
                   help="terminal cell height divided by width; 2.0 is right, 1.0 is the "
                        "square-cell mistake and stretches the picture vertically")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="asciivid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("play", help="play a clip in this terminal")
    _add_render_args(pl)
    pl.add_argument("--scene", default="tunnel", choices=list(scenes.SCENES))
    pl.add_argument("--input", default=None, help="a video file instead of a scene (needs ffmpeg)")
    pl.add_argument("--seconds", type=float, default=6.0)
    pl.add_argument("--fps", type=float, default=DEFAULT_FPS)
    pl.add_argument("--loop", type=int, default=1)
    pl.add_argument("--audio", default=None, help="a sound file to start on the same clock")
    pl.add_argument("--no-delta", dest="delta", action="store_false",
                    help="send whole frames instead of only what changed")
    pl.add_argument("--delta-tol", type=int, default=0,
                    help="largest per-channel colour difference treated as no change; 0 is "
                         "exact and on smooth truecolor content saves nothing")

    rn = sub.add_parser("render", help="write a .cast recording or plain ANSI")
    _add_render_args(rn)
    rn.add_argument("--scene", default="tunnel", choices=list(scenes.SCENES))
    rn.add_argument("--input", default=None, help="a video file instead of a scene (needs ffmpeg)")
    rn.add_argument("--seconds", type=float, default=6.0)
    rn.add_argument("--fps", type=float, default=DEFAULT_FPS)
    rn.add_argument("--out", required=True)
    rn.add_argument("--no-delta", dest="delta", action="store_false")
    rn.add_argument("--delta-tol", type=int, default=0)

    st = sub.add_parser("study", help="measure fidelity for every ramp and colour mode")
    st.add_argument("--out", default="out")
    st.add_argument("--no-dump", dest="dump", action="store_false",
                    help="skip writing the per-frame source and ANSI the checker reads")

    tb = sub.add_parser("table", help="print the fidelity table from a finished study")
    tb.add_argument("--out", default="out")
    tb.add_argument("--where", default="",
                    help="filter, e.g. mode=mono,ramp=ascii67; applied before grouping")
    tb.add_argument("--by", default="ramp,mode",
                    help="comma separated grouping keys: ramp, mode, match, dither, normalize, scene")

    a = ap.parse_args(argv)
    fnt = fontmod.load()

    if a.cmd in ("play", "render"):
        spec = _spec(a)
        spec.validate()
        rows, cols = render.grid_for(640, 480, a.cols, a.aspect)
        w, h = cols * 8, rows * 16
        if a.input:
            raw = _ffmpeg_frames(a.input, a.fps, w, h)
            frames = [render.render(f, fnt, spec, a.cols, aspect=a.aspect) for f in raw]
        else:
            frames = _clip(a.scene, a.seconds, a.fps, a.cols, a.aspect, fnt, spec, w, h)

        if a.cmd == "play":
            player.play(frames, a.fps, delta=a.delta, loop=a.loop, audio=a.audio,
                        delta_tol=a.delta_tol)
            return 0

        out = pathlib.Path(a.out)
        enc = render.DeltaEncoder(tol=a.delta_tol) if a.delta else None
        chunks = [enc.encode(f) if enc else render.to_ansi(f) for f in frames]
        if out.suffix == ".cast":
            with out.open("w", encoding="utf-8") as fh:
                n = cast.write(fh, frames[0].cols, frames[0].rows, chunks,
                               fps=a.fps, title=f"{a.scene} {spec.key()}")
            print(f"{out}: {n} events, {len(frames)} frames, "
                  f"{out.stat().st_size} bytes", file=sys.stderr)
        else:
            out.write_text("".join(chunks), encoding="utf-8")
            print(f"{out}: {out.stat().st_size} bytes", file=sys.stderr)
        return 0

    if a.cmd == "study":
        def prog(i, n, key):
            print(f"  [{i + 1:2d}/{n}] {key}", file=sys.stderr)
        rep = study.run(fnt, pathlib.Path(a.out), dump=a.dump, progress=prog)
        print(f"{len(rep['results'])} measurements written to {a.out}/fidelity.json")
        return 0

    if a.cmd == "table":
        rep = json.loads((pathlib.Path(a.out) / "fidelity.json").read_text())
        by = tuple(k.strip() for k in a.by.split(",") if k.strip())
        results = rep["results"]
        for clause in (c for c in a.where.split(",") if c.strip()):
            k, _, v = clause.partition("=")
            k, v = k.strip(), v.strip()
            if not results or k not in results[0]:
                raise SystemExit(f"cannot filter on {k!r}; keys are "
                                 f"{sorted(results[0]) if results else 'none'}")
            want = {"true": True, "false": False}.get(v.lower(), v)
            results = [r for r in results if r[k] == want or str(r[k]) == v]
        if not results:
            raise SystemExit(f"filter {a.where!r} matched no results, so there is nothing "
                             f"to average; nothing is printed rather than an empty table")
        rows = study.aggregate(results, by)
        head = list(by) + ["ssim", "ssim_cs", "cell_r", "rgb_rmse", "bytes", "n"]
        widths = [max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in head]
        widths = [max(w, 8) for w in widths]
        print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
        for r in rows:
            cells = []
            for h, w in zip(head, widths):
                v = r[h]
                s = f"{v:.4f}" if isinstance(v, float) and h in ("ssim", "ssim_cs", "cell_r") else \
                    f"{v:.2f}" if isinstance(v, float) else str(v)
                cells.append(s.ljust(w))
            print("  ".join(cells))
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
