#!/usr/bin/env python3
"""Regenerate data/font8x16.txt from a system TrueType monospace font.

This is a development tool, not part of the pipeline. The renderer and the independent
checker both read the committed text file, never a font on the machine, so the whole
project reproduces on a box with no fonts installed and no Pillow.

Why a committed raster at all: measuring how much of an image survives the mapping to
characters requires knowing what the characters actually look like. Ink coverage alone
says "@" is dark and "." is light, and says nothing about "/" carrying a diagonal.

Each glyph is rendered at 8x supersample (64x128) and box-averaged down to an 8x16
grayscale cell. 8x16 is chosen because it is the classic terminal cell and its 1:2 shape
is the aspect ratio the renderer has to respect.

    python3 scripts/make_font.py [--font /path/to/Mono.ttf]

Output format, one glyph per line:

    <codepoint-hex> <128 hex bytes, row-major, 8 wide by 16 tall>

Pillow and FreeType versions can hint differently, so regenerating on another machine may
produce a slightly different file. That is exactly why the raster is committed.
"""

import argparse
import pathlib
import sys

CELL_W, CELL_H = 8, 16
SUPER = 8

# Printable ASCII plus the block-drawing characters the ANSI ramps and half-block mode need.
CHARS = [chr(c) for c in range(0x20, 0x7F)] + list("█▓▒░▀▄▌▐")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow is required to REGENERATE the font raster:  pip install Pillow", file=sys.stderr)
        print("The committed data/font8x16.txt is what the pipeline reads; nothing else needs it.",
              file=sys.stderr)
        return 2

    path = args.font
    if path is None:
        for c in FONT_CANDIDATES:
            if pathlib.Path(c).exists():
                path = c
                break
    if path is None:
        print("no monospace TTF found; pass --font", file=sys.stderr)
        return 2

    w, h = CELL_W * SUPER, CELL_H * SUPER
    # Pick the largest size whose advance width still fits the supersampled cell width.
    size = 1
    while True:
        f = ImageFont.truetype(path, size + 1)
        adv = f.getlength("M")
        asc, desc = f.getmetrics()
        if adv > w or asc + desc > h:
            break
        size += 1
    font = ImageFont.truetype(path, size)
    ascent, descent = font.getmetrics()
    # Centre the glyph box inside the supersampled cell.
    x0 = (w - font.getlength("M")) / 2.0
    y0 = (h - (ascent + descent)) / 2.0

    out = pathlib.Path(args.out or (pathlib.Path(__file__).resolve().parent.parent
                                    / "data" / "font8x16.txt"))
    lines = [
        f"# 8x16 grayscale glyph raster, {len(CHARS)} glyphs",
        f"# source: {pathlib.Path(path).name} at {size}px, {SUPER}x supersampled and box-averaged",
        "# format: <codepoint-hex> <128 hex bytes, row-major, 8 wide by 16 tall>",
    ]
    missing = []
    for ch in CHARS:
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        d.text((x0, y0), ch, font=font, fill=255)
        px = img.load()
        cell = []
        for cy in range(CELL_H):
            for cx in range(CELL_W):
                tot = 0
                for sy in range(SUPER):
                    for sx in range(SUPER):
                        tot += px[cx * SUPER + sx, cy * SUPER + sy]
                cell.append(tot // (SUPER * SUPER))
        if ch != " " and max(cell) == 0:
            missing.append(ch)
        lines.append(f"{ord(ch):04x} " + "".join(f"{v:02x}" for v in cell))

    if missing:
        print(f"font {path} has no glyph for: {missing!r}", file=sys.stderr)
        return 2

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(CHARS)} glyphs, {size}px, cell {CELL_W}x{CELL_H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
