"""Image to character cells, and cells to an ANSI byte stream.

Two things here are easy to get wrong and are handled explicitly.

**Aspect ratio.** A terminal cell is about twice as tall as it is wide. Choosing the grid as
if cells were square stretches everything vertically by a factor of two, and it is the most
common defect in ASCII art tools. The grid is derived from the font cell, 8 by 16, so it is
correct by construction. `--aspect` exists so the wrong choice can be measured too, and it
is: the fidelity table in the README carries the number.

**Colour depth.** A terminal that does not do truecolor gets 256 or 16, and the cost of that
fallback is a measured number rather than an apology. Ordered dithering against the palette
is available for the fallback modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ramps
from .font import CELL_H, CELL_PX, CELL_W

# Rec. 709 luma on the gamma-encoded values, which is what a terminal displays and what a
# viewer's eye integrates over a cell. Doing this in linear light would be more correct
# physically and would not match what the character actually has to imitate.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


@dataclass(frozen=True)
class Spec:
    ramp: str = "ascii10"
    mode: str = "truecolor"
    match: str = "structure"
    dither: bool = False
    normalize: bool = False

    def key(self) -> str:
        return (f"{self.ramp}/{self.mode}/{self.match}"
                + ("+dither" if self.dither else "")
                + ("+norm" if self.normalize else ""))

    def validate(self) -> None:
        if self.ramp not in ramps.RAMPS:
            raise ValueError(f"unknown ramp {self.ramp!r}")
        if self.mode not in ramps.MODES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.match not in ramps.MATCHERS:
            raise ValueError(f"unknown matcher {self.match!r}")
        if self.dither and self.mode not in ramps.PALETTES:
            raise ValueError(f"dithering is meaningless in {self.mode!r}, there is no palette")
        if self.normalize and self.mode != "mono":
            # In a colour mode the two fitted colours already carry the cell's brightness,
            # so rescaling the luminance before choosing a glyph changes nothing. A flag
            # that silently does nothing is worse than one that refuses.
            raise ValueError("normalize only applies to --mode mono, where ink is the only "
                             "brightness available")


@dataclass(frozen=True)
class Cells:
    """One rendered frame: a character grid plus a foreground and background colour each."""

    chars: np.ndarray  # (rows, cols) unicode
    fg: np.ndarray     # (rows, cols, 3) uint8
    bg: np.ndarray     # (rows, cols, 3) uint8
    spec: Spec

    @property
    def rows(self) -> int:
        return self.chars.shape[0]

    @property
    def cols(self) -> int:
        return self.chars.shape[1]

    def text(self) -> str:
        """The characters alone, no colour. What `--mode mono` would print."""
        return "\n".join("".join(row) for row in self.chars)


def grid_for(width: int, height: int, cols: int, aspect: float = 2.0) -> tuple[int, int]:
    """Rows and columns for an image, given a target column count.

    `aspect` is cell height divided by cell width. The real terminal value is 2.0. Passing
    1.0 is the square-cell mistake, and the resulting grid stretches the image vertically.
    """
    if cols < 1:
        raise ValueError("cols must be at least 1")
    rows = max(1, int(round(height / width * cols / aspect)))
    return rows, cols


# Subcell grid used by the mono matcher: 4x4 pixel blocks, so a cell is 2 across and 4 down.
SUB_X, SUB_Y = CELL_W // 4, CELL_H // 4


def _subcells(v: np.ndarray) -> np.ndarray:
    """(n, 128) flattened cells to (n, 8) means over 4x4 subcells."""
    n = v.shape[0]
    return (v.reshape(n, SUB_Y, CELL_H // SUB_Y, SUB_X, CELL_W // SUB_X)
             .mean(axis=(2, 4))
             .reshape(n, SUB_Y * SUB_X))


def _cellwise(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Resample to rows*CELL_H by cols*CELL_W, then view as (rows, cols, CELL_PX, 3).

    Nearest neighbour on purpose. Any smoothing here would quietly do part of the job the
    character mapping is supposed to be measured on.
    """
    h, w, _ = img.shape
    th, tw = rows * CELL_H, cols * CELL_W
    if (h, w) != (th, tw):
        yi = np.minimum((np.arange(th) * h) // th, h - 1)
        xi = np.minimum((np.arange(tw) * w) // tw, w - 1)
        img = img[yi][:, xi]
    return (img.reshape(rows, CELL_H, cols, CELL_W, 3)
               .transpose(0, 2, 1, 3, 4)
               .reshape(rows, cols, CELL_PX, 3))


def render(img: np.ndarray, font, spec: Spec, cols: int, *, aspect: float = 2.0) -> Cells:
    """Render one RGB frame to cells."""
    spec.validate()
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) image, got {img.shape}")
    h, w, _ = img.shape
    rows, cols = grid_for(w, h, cols, aspect)

    blocks = _cellwise(img, rows, cols).astype(np.float64)     # (rows, cols, 128, 3)
    lum = blocks @ LUMA                                        # (rows, cols, 128)
    n = rows * cols
    x = lum.reshape(n, CELL_PX)

    charset = ramps.ordered(spec.ramp, font)
    A = font.stack(charset)                                    # (g, 128) alpha in [0,1]
    xa = x @ A.T                                               # (n, g)
    x1 = x.sum(axis=1)[:, None]
    xx = (x * x).sum(axis=1)[:, None]

    if spec.match == "mean" or spec.mode == "mono":
        # Both descriptors are (rows*cols, S) for the cell and (glyphs, S) for the
        # alphabet, so the same nearest-neighbour cost serves both.
        if spec.match == "mean":
            # The classic mapping: cell mean brightness against glyph ink coverage.
            D = (A.mean(axis=1) * 255.0)[:, None]
            xs = (x1 / CELL_PX)
        else:
            # Mono has no free colours, so the glyph has to carry the shape itself. The
            # comparison is at 4x4 subcell resolution rather than per pixel, because ink
            # in a mono terminal reads as local density and not as individual lit pixels.
            D = _subcells(A) * 255.0
            xs = _subcells(x)
        if spec.normalize:
            # White-on-black ASCII has a hard ceiling: the heaviest glyph is all the light
            # there is. `@` inks 26.2% of its cell, so on an image of mean luminance 150
            # every bright cell clips to `@` and a 67-character ramp becomes a 2-character
            # ramp. Measured, not assumed; the README carries the before and after.
            #
            # The stretch targets the range the alphabet can actually express, in whatever
            # units the matcher compares in, which is why it is applied here and not to the
            # source. The cost is real and stated: the range is per frame, so a clip whose
            # brightness changes will pump.
            lo, hi = np.percentile(xs, 1.0), np.percentile(xs, 99.0)
            if hi - lo > 1e-6:
                xs = np.clip((xs - lo) / (hi - lo), 0.0, 1.0) * float(D.max())
        cost = (xs * xs).sum(1)[:, None] - 2.0 * xs @ D.T + (D * D).sum(1)[None, :]
    else:
        # Colour is free, so the glyph is a partition and fg/bg are fitted to it. Cost is
        # the residual of least squares on the basis [1 - alpha, alpha], which is what the
        # cell will actually look like once the two colours are chosen.
        s0 = ((1 - A) ** 2).sum(axis=1)[None, :]
        s1 = ((1 - A) * A).sum(axis=1)[None, :]
        s2 = (A * A).sum(axis=1)[None, :]
        b0 = x1 - xa
        b1 = xa
        det = s0 * s2 - s1 * s1
        # A glyph that is entirely on or entirely off makes the 2x2 singular: one of the two
        # colours is never shown, so only the other is fitted.
        safe = det > 1e-9
        explained = np.where(
            safe,
            (b0 * b0 * s2 - 2.0 * b0 * b1 * s1 + b1 * b1 * s0) / np.where(safe, det, 1.0),
            np.where(s2 > 1e-9, b1 * b1 / np.where(s2 > 1e-9, s2, 1.0), 0.0)
            + np.where(s0 > 1e-9, b0 * b0 / np.where(s0 > 1e-9, s0, 1.0), 0.0),
        )
        cost = xx - explained

    pick = cost.argmin(axis=1)
    chars = np.array(list(charset), dtype="<U1")[pick].reshape(rows, cols)

    if spec.mode == "mono":
        fg = np.full((rows, cols, 3), 255, dtype=np.uint8)
        bg = np.zeros((rows, cols, 3), dtype=np.uint8)
        return Cells(chars, fg, bg, spec)

    # Foreground is the mean colour under the ink, background the mean colour outside it.
    a = A[pick]                                                # (n, 128)
    px = blocks.reshape(n, CELL_PX, 3)
    wa = a.sum(axis=1)[:, None]
    wb = (1.0 - a).sum(axis=1)[:, None]
    mean_all = px.mean(axis=1)
    fg = np.where(wa > 1e-6, np.einsum("np,npc->nc", a, px) / np.maximum(wa, 1e-6), mean_all)
    bg = np.where(wb > 1e-6, np.einsum("np,npc->nc", 1.0 - a, px) / np.maximum(wb, 1e-6), mean_all)
    fg = fg.reshape(rows, cols, 3)
    bg = bg.reshape(rows, cols, 3)

    if spec.mode in ramps.PALETTES:
        pal = ramps.PALETTES[spec.mode]
        off = None
        if spec.dither:
            # One dither offset per cell, tiled over the grid. The amplitude is the mean
            # gap between neighbouring palette entries, which is what the rounding error
            # is worth spreading around.
            step = _palette_step(pal)
            tile = ramps.BAYER4[np.arange(rows)[:, None] % 4, np.arange(cols)[None, :] % 4]
            off = (tile * step)[..., None]
        fg = pal[ramps.quantize(fg, pal, off)]
        bg = pal[ramps.quantize(bg, pal, off)]

    fg = np.clip(fg + 0.5, 0, 255).astype(np.uint8)
    bg = np.clip(bg + 0.5, 0, 255).astype(np.uint8)
    return Cells(chars, fg, bg, spec)


def _palette_step(pal: np.ndarray) -> float:
    d = np.sqrt(((pal[:, None, :] - pal[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean() / np.sqrt(3.0))


# ---------------------------------------------------------------------------------------
# ANSI emission.

RESET = "\x1b[0m"
HOME = "\x1b[H"
CLEAR = "\x1b[2J"
HIDE = "\x1b[?25l"
SHOW = "\x1b[?25h"


def _sgr(mode: str, fg: np.ndarray, bg: np.ndarray, pal: np.ndarray | None) -> str:
    if mode == "truecolor":
        return f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m"
    assert pal is not None
    fi = int(np.abs(pal - fg).sum(axis=1).argmin())
    bi = int(np.abs(pal - bg).sum(axis=1).argmin())
    if mode == "ansi256":
        return f"\x1b[38;5;{fi};48;5;{bi}m"
    return f"\x1b[{30 + fi if fi < 8 else 90 + fi - 8};{40 + bi if bi < 8 else 100 + bi - 8}m"


def to_ansi(cells: Cells, *, home: bool = True, newline: str = "\r\n") -> str:
    """A full frame as an ANSI string, one SGR only where the colour actually changes."""
    mode = cells.spec.mode
    pal = ramps.PALETTES.get(mode)
    out: list[str] = []
    if home:
        out.append(HOME)
    last: str | None = None
    for r in range(cells.rows):
        if r:
            out.append(newline)
        for c in range(cells.cols):
            if mode != "mono":
                s = _sgr(mode, cells.fg[r, c], cells.bg[r, c], pal)
                if s != last:
                    out.append(s)
                    last = s
            out.append(str(cells.chars[r, c]))
        if mode != "mono":
            out.append(RESET)
            last = RESET
    return "".join(out)


def to_delta(cells: Cells, prev: Cells | None, *, newline: str = "\r\n",
             tol: int = 0) -> str:
    """Only the cells that changed, addressed with cursor moves.

    `tol` is the largest per-channel colour difference treated as no change. It exists
    because measuring showed that exact delta encoding saves nothing at all on smooth
    truecolor content: a slowly moving gradient changes 9% of the characters and very
    nearly 100% of the colours, by one or two units each. Callers that want the screen
    state tracked correctly under a non-zero tolerance should use `DeltaEncoder` rather
    than calling this directly, because with `tol` the screen stops being the previous
    frame.
    """
    if prev is None:
        return to_ansi(cells, newline=newline)
    if prev.rows != cells.rows or prev.cols != cells.cols:
        raise ValueError("frame size changed mid-stream")
    mode = cells.spec.mode
    pal = ramps.PALETTES.get(mode)
    changed = (cells.chars != prev.chars)
    if mode != "mono":
        df = np.abs(cells.fg.astype(np.int16) - prev.fg.astype(np.int16)).max(axis=-1)
        db = np.abs(cells.bg.astype(np.int16) - prev.bg.astype(np.int16)).max(axis=-1)
        changed |= (df > tol) | (db > tol)

    GAP = 8  # a cursor move costs about this many characters, so shorter runs are reprinted
    out: list[str] = []
    last: str | None = None
    for r in range(cells.rows):
        cs = np.nonzero(changed[r])[0]
        if not len(cs):
            continue
        segs: list[tuple[int, int]] = []
        start = prev_c = int(cs[0])
        for c in cs[1:]:
            c = int(c)
            if c - prev_c > GAP:
                segs.append((start, prev_c))
                start = c
            prev_c = c
        segs.append((start, prev_c))
        for a, b in segs:
            out.append(f"\x1b[{r + 1};{a + 1}H")
            last = None
            for c in range(a, b + 1):
                if mode != "mono":
                    s = _sgr(mode, cells.fg[r, c], cells.bg[r, c], pal)
                    if s != last:
                        out.append(s)
                        last = s
                out.append(str(cells.chars[r, c]))
            if mode != "mono":
                out.append(RESET)
                last = RESET
    return "".join(out)


class DeltaEncoder:
    """Delta encoding that tracks what is actually on the screen.

    With a colour tolerance the screen stops being the previous frame: a cell held back
    because its colour moved by one unit still shows the old colour, and the next frame has
    to be compared against that, not against the frame it was held back from. Encoding
    against the previous frame instead lets those one-unit differences accumulate silently
    until the picture is visibly stale. This class keeps the screen state, so what a viewer
    sees is exactly `self.screen` and the fidelity of that is measurable.

    It also falls back to a whole frame whenever the delta would be larger, which on
    smooth truecolor content at `tol=0` is most frames.
    """

    def __init__(self, *, tol: int = 0, newline: str = "\r\n"):
        if tol < 0:
            raise ValueError("tol must not be negative")
        self.tol = int(tol)
        self.newline = newline
        self.screen: Cells | None = None
        self.full_frames = 0
        self.delta_frames = 0

    def encode(self, cells: Cells) -> str:
        whole = to_ansi(cells, newline=self.newline)
        if self.screen is None:
            self.screen = cells
            self.full_frames += 1
            return whole
        if self.tol == 0:
            shown = cells
        else:
            keep = (cells.chars == self.screen.chars)
            df = np.abs(cells.fg.astype(np.int16) - self.screen.fg.astype(np.int16)).max(-1)
            db = np.abs(cells.bg.astype(np.int16) - self.screen.bg.astype(np.int16)).max(-1)
            keep &= (df <= self.tol) & (db <= self.tol)
            k3 = keep[..., None]
            shown = Cells(np.where(keep, self.screen.chars, cells.chars),
                          np.where(k3, self.screen.fg, cells.fg),
                          np.where(k3, self.screen.bg, cells.bg),
                          cells.spec)
        chunk = to_delta(shown, self.screen, newline=self.newline)
        if len(chunk) >= len(whole):
            self.screen = cells
            self.full_frames += 1
            return whole
        self.screen = shown
        self.delta_frames += 1
        return chunk
