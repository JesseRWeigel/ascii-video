"""Cells back to pixels: what the terminal would actually put on the screen.

Fidelity cannot be measured against the character grid, because the grid is not what anyone
sees. It has to be measured against the image the terminal draws, which is the glyph raster
composited between the two chosen colours. That image is the same size as the source, so
the comparison needs no resampling and no excuse.
"""

from __future__ import annotations

import numpy as np

from .font import CELL_H, CELL_W


def rasterize(cells, font) -> np.ndarray:
    """(rows*16, cols*8, 3) uint8: the frame as a terminal would display it."""
    rows, cols = cells.chars.shape
    out = np.empty((rows, cols, CELL_H, CELL_W, 3), dtype=np.float64)
    # One alpha lookup per distinct character, not per cell. A 80x30 grid of a 67-glyph
    # ramp is 2400 cells and at most 67 distinct glyphs.
    for ch in np.unique(cells.chars):
        m = cells.chars == ch
        a = font.alpha(str(ch))[None, :, :, None]
        fg = cells.fg[m].astype(np.float64)[:, None, None, :]
        bg = cells.bg[m].astype(np.float64)[:, None, None, :]
        out[m] = bg + a * (fg - bg)
    img = out.transpose(0, 2, 1, 3, 4).reshape(rows * CELL_H, cols * CELL_W, 3)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def to_ppm_bytes(img: np.ndarray) -> bytes:
    h, w, _ = img.shape
    return b"P6\n%d %d\n255\n" % (w, h) + img.tobytes()


def to_png_bytes(img: np.ndarray) -> bytes:
    """Minimal PNG writer, so the docs page can show source and render side by side.

    Filter type 0 on every scanline. Real encoders choose per-line filters for size; this
    one only has to be correct and deterministic.
    """
    import struct
    import zlib

    h, w, _ = img.shape
    raw = np.concatenate(
        [np.zeros((h, 1), dtype=np.uint8), img.reshape(h, w * 3)], axis=1
    ).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))
