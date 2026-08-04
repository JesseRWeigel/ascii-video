"""Procedurally generated source video, so the repository carries no binary.

Five scenes, picked to disagree with each other. A renderer that looks fine on one test
image and falls apart on another is the normal outcome, and a single smooth gradient will
flatter any character ramp ever written.

    plasma      smooth low-frequency colour, nothing near the cell Nyquist
    tunnel      shaded solid with a hard silhouette edge, mid-frequency
    bars        saturated hard-edged geometry, maximum contrast
    starfield   sparse bright points on near-black, almost no ink to spend
    glyphs      dense grid of marks at roughly one cell per feature, worst case

Every scene is a closed-form function of (x, y, t). No random number generator state
crosses a frame boundary, so any frame can be regenerated on its own and two runs produce
identical bytes. Pseudo-random values come from an integer hash of the coordinates.
"""

from __future__ import annotations

import numpy as np

SCENES = ("plasma", "tunnel", "bars", "starfield", "glyphs")


def _hash01(*keys: np.ndarray) -> np.ndarray:
    """Deterministic uint32 hash of integer coordinate arrays, returned in [0, 1).

    Integer arithmetic on purpose. `np.random` with a seed would also be deterministic,
    but only if every caller draws in the same order, which couples frames together.
    """
    h = np.uint32(2166136261)
    for k in keys:
        h = np.bitwise_xor(h, k.astype(np.uint32))
        h = (h * np.uint32(16777619)).astype(np.uint32)
        h = np.bitwise_xor(h, h >> np.uint32(13))
        h = (h * np.uint32(2654435761)).astype(np.uint32)
    h = np.bitwise_xor(h, h >> np.uint32(16))
    return h.astype(np.float64) / 4294967296.0


def _grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised coordinates. x spans [-1, 1]; y is scaled so a pixel stays square."""
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w * 2.0 - 1.0
    ys = ((np.arange(h, dtype=np.float64) + 0.5) / h * 2.0 - 1.0) * (h / w)
    return np.meshgrid(xs, ys)


def _pack(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def plasma(t: float, w: int, h: int) -> np.ndarray:
    x, y = _grid(w, h)
    v = (
        np.sin(3.0 * x + t * 1.7)
        + np.sin(4.0 * y - t * 1.1)
        + np.sin(2.5 * (x + y) + t * 0.9)
        + np.sin(5.0 * np.sqrt(x * x + y * y + 0.05) - t * 2.3)
    ) / 4.0
    r = 0.5 + 0.5 * np.sin(np.pi * v + 0.0)
    g = 0.5 + 0.5 * np.sin(np.pi * v + 2.094)
    b = 0.5 + 0.5 * np.sin(np.pi * v + 4.189)
    return _pack(r, g, b)


def tunnel(t: float, w: int, h: int) -> np.ndarray:
    """A shaded sphere orbiting inside a receding tunnel: smooth shading plus a hard edge."""
    x, y = _grid(w, h)
    rad = np.sqrt(x * x + y * y) + 1e-6
    ang = np.arctan2(y, x)
    depth = 0.35 / rad
    stripe = 0.5 + 0.5 * np.sin(12.0 * depth + 8.0 * ang + t * 3.0)
    fog = np.clip(rad * 1.5, 0.0, 1.0)
    br = stripe * fog
    r, g, b = br * 0.35, br * 0.55, br * 0.95

    cx, cy = 0.42 * np.cos(t * 0.8), 0.42 * np.sin(t * 1.3) * (h / w)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius = 0.30
    inside = d < radius
    z = np.sqrt(np.clip(radius * radius - d * d, 0.0, None))
    # Lambert shading from a fixed light, so the sphere is a smooth ramp with a hard rim.
    nx, ny, nz = (x - cx) / radius, (y - cy) / radius, z / radius
    lam = np.clip(nx * -0.4 + ny * -0.5 + nz * 0.77, 0.0, 1.0)
    shade = 0.08 + 0.92 * lam ** 1.4
    r = np.where(inside, shade * 1.00, r)
    g = np.where(inside, shade * 0.82, g)
    b = np.where(inside, shade * 0.45, b)
    return _pack(r, g, b)


def bars(t: float, w: int, h: int) -> np.ndarray:
    """Saturated colour bars sliding over a rotating checkerboard. Nothing is gradual."""
    x, y = _grid(w, h)
    palette = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.05, 0.05, 0.05]],
        dtype=np.float64,
    )
    idx = np.floor((x + 1.0) * 4.0 + t * 1.5).astype(np.int64) % 8
    col = palette[idx]

    ca, sa = np.cos(t * 0.6), np.sin(t * 0.6)
    u, v = x * ca - y * sa, x * sa + y * ca
    check = ((np.floor(u * 9.0) + np.floor(v * 9.0)) % 2 == 0)
    band = np.abs(y) < 0.30
    col = np.where(band[..., None] & check[..., None], 1.0 - col, col)
    return _pack(col[..., 0], col[..., 1], col[..., 2])


def starfield(t: float, w: int, h: int) -> np.ndarray:
    """Sparse points on near-black. Almost every cell is empty, and a few must not be."""
    n = 1500
    i = np.arange(n)
    # Each star has a fixed direction and speed; z wraps, which is the only time dependence.
    ax = _hash01(i, np.full(n, 11)) * 2.0 - 1.0
    ay = _hash01(i, np.full(n, 22)) * 2.0 - 1.0
    speed = 0.25 + 0.75 * _hash01(i, np.full(n, 33))
    z = (_hash01(i, np.full(n, 44)) + t * speed * 0.35) % 1.0
    z = 0.20 + 0.80 * z
    # The 0.35 keeps most of the field on screen. Without it the projection throws almost
    # everything past the edge and 98% of cells are identical black, which makes per-cell
    # correlation meaningless rather than hard.
    px = ax * 0.35 / z
    py = ay * 0.35 / z
    bright = np.clip((1.0 - z) ** 2.2 * 2.4, 0.0, 1.0)

    img = np.zeros((h, w, 3), dtype=np.float64)
    sx = ((px * 0.5 + 0.5) * w).astype(np.int64)
    sy = ((py * 0.5 * (w / h) + 0.5) * h).astype(np.int64)
    keep = (sx >= 1) & (sx < w - 1) & (sy >= 1) & (sy < h - 1)
    tint = np.stack([0.75 + 0.25 * _hash01(i, np.full(n, 55)),
                     0.80 + 0.20 * _hash01(i, np.full(n, 66)),
                     np.ones(n)], axis=-1)
    # A 2x2 dot, so a star is not a single pixel that any downsample would erase.
    for dy in (0, 1):
        for dx in (0, 1):
            np.add.at(img, (sy[keep] + dy, sx[keep] + dx),
                      (bright[keep, None] * tint[keep]))
    img += 0.02
    return _pack(img[..., 0], img[..., 1], img[..., 2])


def glyphs(t: float, w: int, h: int) -> np.ndarray:
    """A dense grid of marks about one cell across: detail right at the sampling limit."""
    cell = 8
    gx = (np.arange(w) // cell)[None, :] + np.zeros((h, 1), dtype=np.int64)
    gy = (np.arange(h) // cell)[:, None] + np.zeros((1, w), dtype=np.int64)
    step = int(t * 6.0)
    key = _hash01(gx + 7 * step, gy)
    ix = np.arange(w)[None, :] % cell + np.zeros((h, 1), dtype=np.int64)
    iy = np.arange(h)[:, None] % cell + np.zeros((1, w), dtype=np.int64)

    kind = (key * 4).astype(np.int64)
    on = np.zeros((h, w), dtype=bool)
    on |= (kind == 0) & (np.abs(ix - iy) <= 1)                  # slash
    on |= (kind == 1) & (np.abs(ix + iy - (cell - 1)) <= 1)     # backslash
    on |= (kind == 2) & ((ix >= 2) & (ix <= 5) & (iy >= 2) & (iy <= 5))  # box
    on |= (kind == 3) & ((iy == 3) | (iy == 4))                 # bar

    hue = _hash01(gx, gy + 3 * step)
    r = np.where(on, 0.35 + 0.65 * hue, 0.03)
    g = np.where(on, 0.85 - 0.45 * hue, 0.03)
    b = np.where(on, 0.30 + 0.55 * (1.0 - hue), 0.06)
    return _pack(r, g, b)


_TABLE = {
    "plasma": plasma,
    "tunnel": tunnel,
    "bars": bars,
    "starfield": starfield,
    "glyphs": glyphs,
}


def frame(name: str, t: float, w: int, h: int) -> np.ndarray:
    """One RGB frame, (h, w, 3) uint8. Same arguments always give the same bytes."""
    try:
        fn = _TABLE[name]
    except KeyError:
        raise KeyError(f"unknown scene {name!r}; known: {', '.join(SCENES)}") from None
    img = fn(float(t), int(w), int(h))
    if img.shape != (h, w, 3) or img.dtype != np.uint8:
        raise AssertionError(f"scene {name} returned {img.shape} {img.dtype}")
    return img


def write_ppm(path, img: np.ndarray) -> None:
    """Binary PPM (P6). Chosen so an independent checker can read it in ten lines."""
    h, w, _ = img.shape
    with open(path, "wb") as fh:
        fh.write(b"P6\n%d %d\n255\n" % (w, h))
        fh.write(img.tobytes())
