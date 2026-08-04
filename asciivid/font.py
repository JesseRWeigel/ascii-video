"""The glyph raster, which is what makes fidelity measurable at all.

A character ramp is usually justified by "these characters get darker", and that is a claim
about ink coverage. It is not enough to predict what a terminal actually shows, because "/"
and "|" have nearly the same coverage and put it in completely different places. To measure
how much of an image survives the mapping to characters, you need the shapes.

So the project carries an 8x16 grayscale raster of every glyph it can emit, in
`data/font8x16.txt`, generated once by `scripts/make_font.py` and committed. Nothing at
runtime reads a font off the machine. The cell is 8 wide and 16 tall because that is the
classic terminal cell, and its 1:2 shape is the aspect ratio the renderer has to respect.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

CELL_W = 8
CELL_H = 16
CELL_PX = CELL_W * CELL_H

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "font8x16.txt"

# The raster is an input to every number this project reports, so its identity is pinned.
# scripts/make_font.py may produce a different file on a different FreeType, which is the
# reason the raster is committed rather than regenerated.
FONT_SHA256 = "4135bf5e24e4185dc90f035d86efafe21bba3a788db36a52ad1ccba2dd2531e0"


class Font:
    """8x16 grayscale glyphs, indexed by character."""

    def __init__(self, glyphs: dict[str, np.ndarray], digest: str):
        if not glyphs:
            raise ValueError("font contains no glyphs")
        self.glyphs = glyphs
        self.digest = digest

    def has(self, ch: str) -> bool:
        return ch in self.glyphs

    def alpha(self, ch: str) -> np.ndarray:
        """(16, 8) float array in [0, 1]. 1 means the pixel is fully inked."""
        try:
            return self.glyphs[ch]
        except KeyError:
            raise KeyError(
                f"no glyph for {ch!r} (U+{ord(ch):04X}) in the committed raster; "
                f"add it to CHARS in scripts/make_font.py and regenerate"
            ) from None

    def coverage(self, ch: str) -> float:
        """Fraction of the cell the glyph inks, between 0 and 1."""
        return float(self.alpha(ch).mean())

    def stack(self, chars: str) -> np.ndarray:
        """(n, 128) float array of flattened alphas, in the order given."""
        return np.stack([self.alpha(c).reshape(-1) for c in chars])


def load(path: pathlib.Path | str | None = None, *, check_digest: bool = True) -> Font:
    p = pathlib.Path(path) if path is not None else DATA
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if check_digest and digest != FONT_SHA256:
        raise ValueError(
            f"{p} has sha256 {digest}, expected {FONT_SHA256}. Every fidelity number in this "
            f"project is measured against this raster, so a changed raster invalidates them. "
            f"If the change is intentional, update FONT_SHA256 and re-run the fidelity study."
        )
    glyphs: dict[str, np.ndarray] = {}
    for line in raw.decode("ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        cp, hexdata = line.split()
        if len(hexdata) != CELL_PX * 2:
            raise ValueError(f"glyph U+{cp} has {len(hexdata) // 2} bytes, expected {CELL_PX}")
        arr = np.frombuffer(bytes.fromhex(hexdata), dtype=np.uint8)
        glyphs[chr(int(cp, 16))] = (arr.astype(np.float64) / 255.0).reshape(CELL_H, CELL_W)
    return Font(glyphs, digest)
