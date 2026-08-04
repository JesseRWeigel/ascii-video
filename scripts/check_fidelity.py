#!/usr/bin/env python3
"""Re-derive every fidelity number from the emitted ANSI, sharing no code with the renderer.

`asciivid/study.py` measures the frame it holds in memory. That is one derivation, and a
validator that only compares two derivations passes when both are wrong together. In
particular a bug in ANSI emission would leave the in-memory measurement perfect while the
bytes a terminal receives say something else entirely, and nothing that reads the cell
array can see it.

So this script takes a different route to the same numbers:

  1. It reads `out/source/*.ppm` with its own PPM parser.
  2. It reads `out/render/*.ansi` with its own escape-sequence parser, exactly as a terminal
     would: characters, SGR colour state, default white on black when nothing is set.
  3. It reads `data/font8x16.txt` with its own parser and composites the glyphs itself.
  4. It computes SSIM with integral images rather than a reshape, per-cell correlation from
     the same integral images, and RMSE directly.
  5. It re-derives each source frame from its own pure-Python implementation of the scene
     maths, which is what stops `out/source/` from being a copy of the render.

Then it compares against `out/fidelity.json` and fails on any disagreement. It imports
nothing from `asciivid/`, which `scripts/verify.sh` proves by walking the import graph with
`ast` rather than taking this docstring's word for it.

    python3 scripts/check_fidelity.py --out out [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

GW, GH = 8, 16                 # glyph cell, width by height
TOL = 1e-5                     # fidelity.json is rounded to 6 decimals
C1 = (0.01 * 255.0) ** 2
C2 = (0.03 * 255.0) ** 2
WIN = 8


# ---------------------------------------------------------------------------------------
# Readers, all written here rather than imported.

def read_ppm(path: pathlib.Path) -> np.ndarray:
    raw = path.read_bytes()
    if not raw.startswith(b"P6"):
        raise ValueError(f"{path} is not a P6 PPM")
    fields: list[bytes] = []
    i = 2
    while len(fields) < 3:
        while i < len(raw) and raw[i : i + 1].isspace():
            i += 1
        if raw[i : i + 1] == b"#":
            while raw[i : i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while j < len(raw) and not raw[j : j + 1].isspace():
            j += 1
        fields.append(raw[i:j])
        i = j
    w, h, maxv = (int(f) for f in fields)
    if maxv != 255:
        raise ValueError(f"{path} has maxval {maxv}, only 255 is handled")
    body = raw[i + 1 : i + 1 + w * h * 3]
    if len(body) != w * h * 3:
        raise ValueError(f"{path} is truncated: {len(body)} bytes for {w}x{h}")
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3)


def read_font(path: pathlib.Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        cp, hexdata = line.split()
        vals = np.array([int(hexdata[k : k + 2], 16) for k in range(0, len(hexdata), 2)],
                        dtype=np.float64)
        if vals.size != GW * GH:
            raise ValueError(f"glyph U+{cp} has {vals.size} bytes, expected {GW * GH}")
        out[chr(int(cp, 16))] = (vals / 255.0).reshape(GH, GW)
    if not out:
        raise ValueError(f"{path} contains no glyphs")
    return out


def build_palette() -> np.ndarray:
    base = [(0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205),
            (0, 205, 205), (229, 229, 229), (127, 127, 127), (255, 0, 0), (0, 255, 0),
            (255, 255, 0), (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
    steps = [0, 95, 135, 175, 215, 255]
    cube = [(r, g, b) for r in steps for g in steps for b in steps]
    grays = [(8 + 10 * i,) * 3 for i in range(24)]
    return np.array(base + cube + grays, dtype=np.int64)


PAL256 = build_palette()
_CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")


def parse_ansi(text: str) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """Interpret an ANSI frame the way a terminal would.

    Returns the character grid and two (rows, cols, 3) colour arrays. Colour state carries
    across cells and rows exactly as a real terminal carries it, which is the whole reason
    an emitter that forgets a reset produces a visibly wrong picture and an in-memory
    measurement that never notices.
    """
    chars: list[list[str]] = [[]]
    fgs: list[list[tuple[int, int, int]]] = [[]]
    bgs: list[list[tuple[int, int, int]]] = [[]]
    fg, bg = (255, 255, 255), (0, 0, 0)
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\x1b":
            m = _CSI.match(text, i)
            if not m:
                raise ValueError(f"unparseable escape at offset {i}: {text[i:i + 12]!r}")
            params, final = m.group(1), m.group(2)
            if final == "m":
                fg, bg = _sgr(params, fg, bg)
            elif final not in ("H", "J", "h", "l"):
                raise ValueError(f"unexpected final byte {final!r} at offset {i}")
            i = m.end()
            continue
        if ch == "\r":
            i += 1
            continue
        if ch == "\n":
            chars.append([]); fgs.append([]); bgs.append([])
            i += 1
            continue
        chars[-1].append(ch); fgs[-1].append(fg); bgs[-1].append(bg)
        i += 1
    while chars and not chars[-1]:
        chars.pop(); fgs.pop(); bgs.pop()
    widths = {len(r) for r in chars}
    if len(widths) != 1:
        raise ValueError(f"ragged frame, row widths {sorted(widths)}")
    return chars, np.array(fgs, dtype=np.int64), np.array(bgs, dtype=np.int64)


def _sgr(params: str, fg, bg):
    parts = [int(p) if p else 0 for p in params.split(";")] if params else [0]
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == 0:
            fg, bg = (255, 255, 255), (0, 0, 0)
            i += 1
        elif p in (38, 48):
            if i + 1 >= len(parts):
                raise ValueError(f"truncated SGR {params!r}")
            if parts[i + 1] == 2:
                col = tuple(parts[i + 2 : i + 5])
                if len(col) != 3:
                    raise ValueError(f"truncated truecolor SGR {params!r}")
                i += 5
            elif parts[i + 1] == 5:
                col = tuple(int(v) for v in PAL256[parts[i + 2]])
                i += 3
            else:
                raise ValueError(f"unknown colour form in SGR {params!r}")
            fg, bg = (col, bg) if p == 38 else (fg, col)
        elif 30 <= p <= 37:
            fg = tuple(int(v) for v in PAL256[p - 30]); i += 1
        elif 90 <= p <= 97:
            fg = tuple(int(v) for v in PAL256[p - 90 + 8]); i += 1
        elif 40 <= p <= 47:
            bg = tuple(int(v) for v in PAL256[p - 40]); i += 1
        elif 100 <= p <= 107:
            bg = tuple(int(v) for v in PAL256[p - 100 + 8]); i += 1
        else:
            raise ValueError(f"unhandled SGR parameter {p} in {params!r}")
    return fg, bg


def composite(chars, fg, bg, glyphs) -> np.ndarray:
    rows, cols = len(chars), len(chars[0])
    out = np.empty((rows * GH, cols * GW, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            a = glyphs[chars[r][c]][:, :, None]
            f = fg[r, c].astype(np.float64)[None, None, :]
            b = bg[r, c].astype(np.float64)[None, None, :]
            out[r * GH:(r + 1) * GH, c * GW:(c + 1) * GW] = b + a * (f - b)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------
# Metrics, by integral image rather than by reshape.

def integral(a: np.ndarray) -> np.ndarray:
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return s


def block_means(a: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """Mean over every non-overlapping kh by kw block, from a summed-area table."""
    s = integral(a)
    ys = np.arange(0, (a.shape[0] // kh) * kh + 1, kh)
    xs = np.arange(0, (a.shape[1] // kw) * kw + 1, kw)
    tot = (s[np.ix_(ys[1:], xs[1:])] - s[np.ix_(ys[:-1], xs[1:])]
           - s[np.ix_(ys[1:], xs[:-1])] + s[np.ix_(ys[:-1], xs[:-1])])
    return tot / float(kh * kw)


def luma(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float64)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def measure(src: np.ndarray, out: np.ndarray) -> dict:
    a, b = luma(src), luma(out)
    ma, mb = block_means(a, WIN, WIN), block_means(b, WIN, WIN)
    va = np.maximum(block_means(a * a, WIN, WIN) - ma * ma, 0.0)
    vb = np.maximum(block_means(b * b, WIN, WIN) - mb * mb, 0.0)
    cov = block_means(a * b, WIN, WIN) - ma * mb
    lum = (2 * ma * mb + C1) / (ma * ma + mb * mb + C1)
    cs = (2 * cov + C2) / (va + vb + C2)

    ca = block_means(a, GH, GW).reshape(-1)
    cb = block_means(b, GH, GW).reshape(-1)
    if ca.std() < 1e-9 or cb.std() < 1e-9:
        r = float("nan")
    else:
        r = float(((ca - ca.mean()) * (cb - cb.mean())).mean() / (ca.std() * cb.std()))
    d = src.astype(np.float64) - out.astype(np.float64)
    return {"ssim": float((lum * cs).mean()), "ssim_cs": float(cs.mean()),
            "cell_r": r, "rgb_rmse": float(math.sqrt(float((d * d).mean())))}


# ---------------------------------------------------------------------------------------
# The scenes, re-derived in pure Python so out/source/ cannot be a copy of the render.

def _fnv(*keys: int) -> float:
    h = 2166136261
    for k in keys:
        h = (h ^ (k & 0xFFFFFFFF)) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 2654435761) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 4294967296.0


def _q(v: float) -> int:
    return min(255, max(0, int(v * 255.0 + 0.5)))


def scene_pixel(name: str, t: float, w: int, h: int, px: int, py: int) -> tuple:
    """One source pixel, from the scene formula, without numpy."""
    x = (px + 0.5) / w * 2.0 - 1.0
    y = ((py + 0.5) / h * 2.0 - 1.0) * (h / w)
    if name == "plasma":
        v = (math.sin(3.0 * x + t * 1.7) + math.sin(4.0 * y - t * 1.1)
             + math.sin(2.5 * (x + y) + t * 0.9)
             + math.sin(5.0 * math.sqrt(x * x + y * y + 0.05) - t * 2.3)) / 4.0
        return tuple(_q(0.5 + 0.5 * math.sin(math.pi * v + p))
                     for p in (0.0, 2.094, 4.189))
    if name == "tunnel":
        cx, cy = 0.42 * math.cos(t * 0.8), 0.42 * math.sin(t * 1.3) * (h / w)
        d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius = 0.30
        if d < radius:
            z = math.sqrt(max(radius * radius - d * d, 0.0))
            nx, ny, nz = (x - cx) / radius, (y - cy) / radius, z / radius
            lam = min(1.0, max(0.0, nx * -0.4 + ny * -0.5 + nz * 0.77))
            sh = 0.08 + 0.92 * lam ** 1.4
            return _q(sh * 1.00), _q(sh * 0.82), _q(sh * 0.45)
        rad = math.sqrt(x * x + y * y) + 1e-6
        ang = math.atan2(y, x)
        stripe = 0.5 + 0.5 * math.sin(12.0 * (0.35 / rad) + 8.0 * ang + t * 3.0)
        br = stripe * min(1.0, max(0.0, rad * 1.5))
        return _q(br * 0.35), _q(br * 0.55), _q(br * 0.95)
    if name == "bars":
        pal = [(1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0),
               (1.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.05, 0.05, 0.05)]
        col = list(pal[int(math.floor((x + 1.0) * 4.0 + t * 1.5)) % 8])
        ca, sa = math.cos(t * 0.6), math.sin(t * 0.6)
        u, v = x * ca - y * sa, x * sa + y * ca
        if abs(y) < 0.30 and (math.floor(u * 9.0) + math.floor(v * 9.0)) % 2 == 0:
            col = [1.0 - c for c in col]
        return tuple(_q(c) for c in col)
    if name == "glyphs":
        cell = 8
        gx, gy = px // cell, py // cell
        step = int(t * 6.0)
        key = _fnv(gx + 7 * step, gy)
        ix, iy = px % cell, py % cell
        kind = int(key * 4)
        on = ((kind == 0 and abs(ix - iy) <= 1)
              or (kind == 1 and abs(ix + iy - (cell - 1)) <= 1)
              or (kind == 2 and 2 <= ix <= 5 and 2 <= iy <= 5)
              or (kind == 3 and iy in (3, 4)))
        hue = _fnv(gx, gy + 3 * step)
        if on:
            return _q(0.35 + 0.65 * hue), _q(0.85 - 0.45 * hue), _q(0.30 + 0.55 * (1.0 - hue))
        return _q(0.03), _q(0.03), _q(0.06)
    raise KeyError(name)


def check_source(name: str, t: float, img: np.ndarray, samples: int = 4000) -> str:
    """Confirm the committed source frame really is the scene, on a sampled grid."""
    if name == "starfield":
        # A point scene has no closed form per pixel worth re-deriving. What is checkable
        # is the property that makes it hard for a renderer: nearly everything is dark and
        # a small number of pixels are not.
        L = luma(img)
        frac = float((L > 60).mean())
        if not (0.002 < frac < 0.06):
            raise AssertionError(f"starfield has {frac:.4%} bright pixels, expected a sparse "
                                 f"field between 0.2% and 6%")
        if float(L.mean()) > 30.0:
            raise AssertionError(f"starfield mean luminance {L.mean():.1f} is not near-black")
        return f"sparse ({frac:.3%} bright, mean {L.mean():.1f})"
    h, w, _ = img.shape
    step = max(1, int(math.sqrt(w * h / samples)))
    checked = wrong = worst = 0
    for py in range(0, h, step):
        for px in range(0, w, step):
            want = scene_pixel(name, t, w, h, px, py)
            got = tuple(int(v) for v in img[py, px])
            e = max(abs(a - b) for a, b in zip(want, got))
            worst = max(worst, e)
            checked += 1
            if e > 1:
                wrong += 1
    if wrong:
        raise AssertionError(f"{name}@{t}: {wrong}/{checked} sampled pixels differ from the "
                             f"scene formula by more than 1 (worst {worst})")
    return f"{checked} sampled pixels match the formula (worst delta {worst})"


# ---------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--font", default=None)
    ap.add_argument("--limit", type=int, default=0, help="check only the first N renders")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    report_path = out / "fidelity.json"
    if not report_path.exists():
        print(f"CANNOT CHECK: {report_path} does not exist. Run `python3 -m asciivid study` "
              f"first; nothing below has been verified.", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text())
    results = report["results"]
    if not results:
        print("CANNOT CHECK: fidelity.json has no results", file=sys.stderr)
        return 2

    font_path = pathlib.Path(a.font) if a.font else \
        pathlib.Path(__file__).resolve().parent.parent / "data" / "font8x16.txt"
    glyphs = read_font(font_path)

    # 1. the source frames really are the scenes
    seen: set[tuple[str, float]] = set()
    for r in results:
        seen.add((r["scene"], r["t"]))
    for name, t in sorted(seen):
        img = read_ppm(out / "source" / f"{name}@{t}.ppm")
        note = check_source(name, t, img)
        print(f"  source {name}@{t}: {note}")

    # 2. every reported number, re-derived from the emitted ANSI
    sources: dict[tuple[str, float], np.ndarray] = {}
    todo = results[: a.limit] if a.limit else results
    bad: list[str] = []
    worst = {"ssim": 0.0, "ssim_cs": 0.0, "cell_r": 0.0, "rgb_rmse": 0.0}
    for i, r in enumerate(todo):
        key = (r["scene"], r["t"])
        if key not in sources:
            sources[key] = read_ppm(out / "source" / f"{r['scene']}@{r['t']}.ppm")
        src = sources[key]
        safe = r["spec"].replace("/", "-").replace("+", "-plus-")
        path = out / "render" / f"{r['scene']}@{r['t']}__{safe}.ansi"
        if not path.exists():
            bad.append(f"{path.name} is missing, so its numbers are unchecked")
            continue
        text = path.read_text(encoding="utf-8")
        chars, fg, bg = parse_ansi(text)
        if (len(chars), len(chars[0])) != (r["rows"], r["cols"]):
            bad.append(f"{path.name}: parsed {len(chars)}x{len(chars[0])} cells, "
                       f"the report claims {r['rows']}x{r['cols']}")
            continue
        got = measure(src, composite(chars, fg, bg, glyphs))
        for k, v in got.items():
            claimed = r[k]
            if isinstance(claimed, float) and math.isnan(claimed):
                if not math.isnan(v):
                    bad.append(f"{path.name}: {k} claimed NaN, re-derived {v:.6f}")
                continue
            if math.isnan(v):
                bad.append(f"{path.name}: {k} re-derived as NaN, claimed {claimed:.6f}")
                continue
            d = abs(v - claimed)
            worst[k] = max(worst[k], d)
            if d > TOL:
                bad.append(f"{path.name}: {k} claimed {claimed:.6f}, re-derived {v:.6f} "
                           f"(delta {d:.2e})")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} renders re-derived", flush=True)

    print(f"  {len(todo)} renders re-derived from the emitted ANSI")
    print("  worst disagreement: " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
    if bad:
        print(f"FIDELITY MISMATCH: {len(bad)} problems", file=sys.stderr)
        for line in bad[:10]:
            print(f"    {line}", file=sys.stderr)
        return 1
    print(f"  every number in {report_path} matches an independent re-derivation "
          f"within {TOL:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
