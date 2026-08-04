"""How much of the image survives the mapping to characters, as numbers.

Three measures, because each one alone can be fooled.

**SSIM** on luminance, over 8x8 windows. Structural similarity is the right primary measure
because it asks whether local contrast and local structure survived, and a renderer that
gets average brightness right while destroying every edge scores badly on it. Windows are
8x8, matching the cell width, so a window sits inside roughly one cell rather than
averaging across several and hiding the cell boundary.

**SSIM without the luminance term** (`ssim_cs`), the contrast and structure factors alone.
This one exists because of a hard physical limit that measuring found and reading would
not have. White-on-black ASCII cannot be bright: the heaviest ASCII glyph in this raster,
`@`, inks 26.2% of its cell, so a mono render of an image whose mean luminance is 150 tops
out around 67 and every bright cell clips to `@`. Full SSIM reports that correctly as a
failure, and it reports it so loudly that the difference between a 10-character ramp and a
67-character ramp disappears underneath it. Dropping the luminance factor asks the separate
question of whether the structure survived, given that the brightness could not.

**Per-cell luminance correlation.** Pearson r between the source cell's mean luminance and
the rendered cell's mean luminance. This is the weakest claim a renderer can make, and it
is worth reporting separately because a mapping can pass it while failing SSIM: it only
asks whether the cells are in the right brightness order, not whether anything inside them
survived. A renderer that scores 0.99 here and 0.4 on SSIM is doing exactly that.

**Colour RMSE**, root mean square error per channel in 0-255 units, over every pixel. This
is the one that moves when the palette changes, and it is why the 16-colour numbers look
the way they do.

All three compare the source against `raster.rasterize`, the image a terminal would draw.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .font import CELL_H, CELL_W
from .render import LUMA

WIN = 8
# Standard SSIM stabilisers for an 8-bit dynamic range: (0.01*L)^2 and (0.03*L)^2.
C1 = (0.01 * 255.0) ** 2
C2 = (0.03 * 255.0) ** 2


@dataclass(frozen=True)
class Scores:
    ssim: float
    ssim_cs: float
    cell_r: float
    rgb_rmse: float

    def as_dict(self) -> dict:
        return {k: round(v, 6) for k, v in asdict(self).items()}


def luminance(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) @ LUMA


def _box(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over every non-overlapping k by k window. Trailing partial windows are dropped.

    Non-overlapping rather than sliding: a sliding window over a 640x480 frame is 300k
    windows of mostly redundant information, and the score barely moves. Dropping the
    remainder is safe here because every frame is an exact multiple of the cell size.
    """
    h, w = a.shape
    hh, ww = (h // k) * k, (w // k) * k
    return a[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))


def ssim_parts(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(full SSIM, contrast-structure only) between two luminance planes.

    SSIM factorises into a luminance term and a contrast-structure term. Both are returned
    because for character rendering they answer genuinely different questions, and one of
    them is dominated by a limit of the alphabet rather than by the quality of the mapping.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mu_a, mu_b = _box(a, WIN), _box(b, WIN)
    va = np.maximum(_box(a * a, WIN) - mu_a * mu_a, 0.0)
    vb = np.maximum(_box(b * b, WIN) - mu_b * mu_b, 0.0)
    cov = _box(a * b, WIN) - mu_a * mu_b
    lum = (2 * mu_a * mu_b + C1) / (mu_a * mu_a + mu_b * mu_b + C1)
    cs = (2 * cov + C2) / (va + vb + C2)
    return float((lum * cs).mean()), float(cs.mean())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM between two luminance planes, over non-overlapping 8x8 windows."""
    return ssim_parts(a, b)[0]


def cell_correlation(src: np.ndarray, out: np.ndarray) -> float:
    """Pearson r between source and rendered mean luminance, one point per character cell."""
    a = luminance(src)
    b = luminance(out)
    rows, cols = a.shape[0] // CELL_H, a.shape[1] // CELL_W
    ca = a[:rows * CELL_H, :cols * CELL_W].reshape(rows, CELL_H, cols, CELL_W).mean(axis=(1, 3))
    cb = b[:rows * CELL_H, :cols * CELL_W].reshape(rows, CELL_H, cols, CELL_W).mean(axis=(1, 3))
    x, y = ca.reshape(-1), cb.reshape(-1)
    sx, sy = x.std(), y.std()
    if sx < 1e-9 or sy < 1e-9:
        # A constant plane has no correlation, defined or otherwise. Returning 0.0 here
        # would read as "measured, and bad"; NaN says "this frame cannot answer".
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def score(src: np.ndarray, out: np.ndarray) -> Scores:
    if src.shape != out.shape:
        raise ValueError(f"source is {src.shape} and the render is {out.shape}")
    d = src.astype(np.float64) - out.astype(np.float64)
    full, cs = ssim_parts(luminance(src), luminance(out))
    return Scores(
        ssim=full,
        ssim_cs=cs,
        cell_r=cell_correlation(src, out),
        rgb_rmse=float(np.sqrt((d * d).mean())),
    )
