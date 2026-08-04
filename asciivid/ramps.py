"""Character ramps and the terminal colour palettes, with no order assumed.

The usual ramp is a string someone typed in what they believed was brightness order. This
module never trusts that. Every ramp is sorted by the glyph's measured ink coverage against
the committed raster, so a ramp with two characters in the wrong place still behaves, and
the ordering claim becomes a fact about the font rather than a claim in a comment.
"""

from __future__ import annotations

import numpy as np

RAMPS: dict[str, str] = {
    # Two characters. The floor: everything is on or off.
    "binary": " #",
    # The ramp almost every ASCII art tool ships with.
    "ascii10": " .:-=+*#%@",
    # A long ramp, so quantisation of luminance stops being the limiting factor.
    "ascii67": (" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"),
    # Unicode shade blocks. Coverage is even, and every glyph fills the cell uniformly.
    "blocks": " ░▒▓█",
    # One glyph, upper half block. Useless without colour and excellent with it, because
    # fg and bg give the cell two independent colours and double the vertical resolution.
    "halfblock": "▀",
}

MODES = ("mono", "ansi16", "ansi256", "truecolor")
MATCHERS = ("mean", "structure")


def ordered(name: str, font) -> str:
    """The ramp's characters, deduplicated and sorted by measured ink coverage."""
    try:
        raw = RAMPS[name]
    except KeyError:
        raise KeyError(f"unknown ramp {name!r}; known: {', '.join(RAMPS)}") from None
    seen: dict[str, None] = {}
    for ch in raw:
        if not font.has(ch):
            raise KeyError(f"ramp {name!r} needs a glyph for {ch!r} (U+{ord(ch):04X})")
        seen.setdefault(ch, None)
    return "".join(sorted(seen, key=lambda c: (font.coverage(c), ord(c))))


# ---------------------------------------------------------------------------------------
# xterm palettes.

# The first 16 entries as xterm actually defines them. Terminals with a theme will draw
# these differently, which is a real limit on any 16-colour fidelity number and is stated
# in the README rather than hidden.
ANSI16 = np.array([
    (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
    (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
    (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
], dtype=np.float64)

_CUBE = np.array([0, 95, 135, 175, 215, 255], dtype=np.float64)


def _build256() -> np.ndarray:
    out = [ANSI16]
    r, g, b = np.meshgrid(_CUBE, _CUBE, _CUBE, indexing="ij")
    out.append(np.stack([r.reshape(-1), g.reshape(-1), b.reshape(-1)], axis=-1))
    gray = 8.0 + 10.0 * np.arange(24, dtype=np.float64)
    out.append(np.stack([gray, gray, gray], axis=-1))
    return np.concatenate(out, axis=0)


ANSI256 = _build256()

PALETTES = {"ansi16": ANSI16, "ansi256": ANSI256}

# 4x4 ordered dither, values centred on zero and scaled to +/- half a palette step later.
BAYER4 = (np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float64) + 0.5) / 16.0 - 0.5


def quantize(rgb: np.ndarray, palette: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """Nearest palette entry per pixel. Returns integer indices shaped like rgb[..., 0].

    `offset` is an additive per-pixel perturbation in 0-255 units, which is how ordered
    dithering works: nudge the colour before rounding so the rounding error alternates in
    space instead of forming a flat band.
    """
    v = rgb.astype(np.float64)
    if offset is not None:
        v = v + offset
    flat = v.reshape(-1, 3)
    # Squared distance to every palette entry. 256 entries keeps this small enough to be
    # exact rather than approximate, and an approximate nearest colour would be one more
    # thing the fidelity number could be blaming.
    d = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1)
    return d.argmin(axis=1).reshape(rgb.shape[:-1])
