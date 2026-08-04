"""The fidelity study: every ramp against every colour mode against every scene.

The output of this module is the only reason to believe anything in the README. It writes,
for each combination:

  * `out/source/<scene>@<t>.ppm`   the source frame, as plain bytes
  * `out/render/<scene>@<t>__<key>.ansi`  the exact ANSI a terminal would receive
  * `out/fidelity.json`            the reported numbers

The source frame and the ANSI stream are both written so an independent checker can
re-derive the numbers from the emitted text rather than from anything in memory here. That
distinction matters: a bug in ANSI emission would leave the in-memory measurement perfect
and the actual terminal output wrong, and only a checker that reads the text can see it.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from . import fidelity, raster, ramps, render, scenes

# Two moments per scene. One frame per scene would let a lucky still stand in for a clip,
# and a hundred would say nothing the second one did not.
TIMES = (0.6, 2.4)
COLS = 80
SCENE_W, SCENE_H = 640, 480


def combinations() -> list[render.Spec]:
    out: list[render.Spec] = []
    for ramp in ramps.RAMPS:
        for mode in ramps.MODES:
            for match in ramps.MATCHERS:
                out.append(render.Spec(ramp, mode, match))
                # Dithering only has meaning where there is a palette to dither against,
                # and only the structural matcher is carried through that variant, because
                # crossing every option with every other one triples the runtime and adds
                # no combination anyone would ship.
                if mode in ramps.PALETTES and match == "structure":
                    out.append(render.Spec(ramp, mode, match, dither=True))
                # Contrast normalisation only means anything where ink is the only
                # brightness available, which is mono.
                if mode == "mono":
                    out.append(render.Spec(ramp, mode, match, normalize=True))
    return out


def safe(name: str) -> str:
    return name.replace("/", "-").replace("+", "-plus-")


def spec_from_key(key: str) -> render.Spec:
    """Parse `ramp/mode/matcher[+dither][+norm]` back into a Spec."""
    head, _, flags = key.partition("+")
    parts = head.split("/")
    if len(parts) != 3:
        raise ValueError(f"bad spec key {key!r}, expected ramp/mode/matcher[+dither][+norm]")
    tags = set(f for f in flags.split("+") if f)
    unknown = tags - {"dither", "norm"}
    if unknown:
        raise ValueError(f"bad spec key {key!r}: unknown flag(s) {sorted(unknown)}")
    spec = render.Spec(parts[0], parts[1], parts[2], "dither" in tags, "norm" in tags)
    spec.validate()
    return spec


def run(font, outdir: pathlib.Path, *, dump: bool = True, progress=None,
        scene_names=None, times=None, specs=None) -> dict:
    outdir = pathlib.Path(outdir)
    (outdir / "source").mkdir(parents=True, exist_ok=True)
    (outdir / "render").mkdir(parents=True, exist_ok=True)

    scene_names = list(scene_names) if scene_names else list(scenes.SCENES)
    times = list(times) if times else list(TIMES)
    specs = list(specs) if specs else combinations()
    frames: list[tuple[str, float, np.ndarray]] = []
    for name in scene_names:
        for t in times:
            img = scenes.frame(name, t, SCENE_W, SCENE_H)
            frames.append((name, t, img))
            if dump:
                scenes.write_ppm(outdir / "source" / f"{name}@{t}.ppm", img)

    results: list[dict] = []
    for si, spec in enumerate(specs):
        if progress:
            progress(si, len(specs), spec.key())
        for name, t, img in frames:
            cells = render.render(img, font, spec, COLS)
            back = raster.rasterize(cells, font)
            sc = fidelity.score(img, back)
            text = render.to_ansi(cells)
            if dump:
                fn = f"{name}@{t}__{safe(spec.key())}.ansi"
                (outdir / "render" / fn).write_text(text, encoding="utf-8")
            results.append({
                "scene": name, "t": t, "spec": spec.key(),
                "ramp": spec.ramp, "mode": spec.mode, "match": spec.match,
                "dither": spec.dither, "normalize": spec.normalize,
                "rows": cells.rows, "cols": cells.cols,
                "bytes": len(text.encode("utf-8")),
                **sc.as_dict(),
            })

    report = {
        "font_sha256": font.digest,
        "grid": {"cols": COLS, "rows": SCENE_H // 16},
        "source": {"width": SCENE_W, "height": SCENE_H, "scenes": scene_names,
                   "times": times},
        "metrics": {
            "ssim": "mean SSIM over non-overlapping 8x8 luminance windows, 1.0 is identical",
            "ssim_cs": "the same windows with the luminance term dropped: did the structure "
                       "survive, setting aside a brightness ceiling the alphabet cannot beat",
            "cell_r": "Pearson r of per-cell mean luminance, source against render",
            "rgb_rmse": "root mean square error per channel in 0-255 units, lower is better",
        },
        "results": results,
    }
    (outdir / "fidelity.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def aggregate(results: list[dict], by: tuple[str, ...]) -> list[dict]:
    """Mean of each metric, grouped by the given result keys, in a stable order."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    out = []
    for key, rs in groups.items():
        row = dict(zip(by, key))
        for m in ("ssim", "ssim_cs", "cell_r", "rgb_rmse", "bytes"):
            vals = [r[m] for r in rs
                    if r[m] is not None and not (isinstance(r[m], float) and np.isnan(r[m]))]
            row[m] = float(np.mean(vals)) if vals else float("nan")
        row["n"] = len(rs)
        out.append(row)
    out.sort(key=lambda r: -r["ssim"])
    return out
