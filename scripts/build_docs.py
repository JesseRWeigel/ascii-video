#!/usr/bin/env python3
"""Build docs/index.html from a finished study, with the demo clip embedded.

The page draws the clip the same way the fidelity measurement does: it composites the
committed glyph raster between each cell's two colours. So what a visitor sees is the same
image the SSIM number was computed against, and not a screenshot of something else.

Everything is inlined. The frame stream is delta encoded, gzipped and base64'd, which is
what keeps a 40 frame clip in four render modes inside a page under a megabyte.

    python3 scripts/build_docs.py [--out out] [--docs docs]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asciivid import fidelity, font as fontmod, ramps, raster, render, scenes, study  # noqa: E402

DEMO_SCENE = "tunnel"
DEMO_COLS = 56
DEMO_ROWS = 21
DEMO_FRAMES = 40
# The same colour tolerance the terminal player uses. Without it the packed clip is 1.2 MB
# of one-unit colour changes, and the page reports what the tolerance actually costs.
DEMO_TOL = 8
DEMO_FPS = 24.0
DEMO_SPECS = [
    render.Spec("blocks", "truecolor", "structure"),
    render.Spec("ascii67", "truecolor", "structure"),
    render.Spec("ascii10", "ansi256", "structure", dither=True),
    render.Spec("ascii67", "mono", "structure", normalize=True),
]
STILL_SCENES = list(scenes.SCENES)
STILL_COLS = 32
STILL_SPEC = render.Spec("blocks", "truecolor", "structure")
TITLE = "Video, rendered to characters"


def pack_clip(frames: list, charset: str) -> bytes:
    """Delta encoded cell stream: 7 bytes per cell, only the cells that changed.

    A parallel format to the ANSI delta stream rather than the same one, because the page
    draws pixels and has no use for cursor addressing. Both are exercised: verify.sh checks
    the ANSI stream, and the browser check compares the frames this stream decodes to
    against the numbers the study measured.
    """
    rows, cols = frames[0].rows, frames[0].cols
    idx = {c: i for i, c in enumerate(charset)}
    out = bytearray()
    out += struct.pack("<HHH", rows, cols, len(frames))
    cs = charset.encode("utf-8")
    out += struct.pack("<H", len(cs)) + cs
    prev = None
    for f in frames:
        ci = np.vectorize(idx.__getitem__)(f.chars).astype(np.uint8)
        cells = np.concatenate([ci[..., None], f.fg, f.bg], axis=-1).astype(np.uint8)
        flat = cells.reshape(-1, 7)
        if prev is None:
            out += struct.pack("<I", rows * cols)
            out += np.arange(rows * cols, dtype="<u2").tobytes()
            out += flat.tobytes()
        else:
            changed = np.nonzero((flat != prev).any(axis=1))[0]
            out += struct.pack("<I", len(changed))
            out += changed.astype("<u2").tobytes()
            out += flat[changed].tobytes()
        prev = flat
    return bytes(out)


def _screen_at(srcs, fnt, spec, frame: int):
    """The screen state after `frame` frames of delta encoding, which is what the page shows."""
    enc = render.DeltaEncoder(tol=DEMO_TOL)
    for s in srcs[: frame + 1]:
        enc.encode(render.render(s, fnt, spec, DEMO_COLS))
    return enc.screen


def b64gz(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, 9, mtime=0)).decode("ascii")


def font_blob(fnt) -> tuple[str, str]:
    """Every glyph the demo can use, as one base64 alpha blob plus its codepoint index."""
    chars = sorted({c for spec in DEMO_SPECS + [STILL_SPEC]
                    for c in ramps.ordered(spec.ramp, fnt)})
    blob = b"".join((fnt.alpha(c) * 255.0 + 0.5).astype(np.uint8).tobytes() for c in chars)
    return "".join(chars), base64.b64encode(blob).decode("ascii")


def table_rows(report: dict, by: tuple[str, ...], where=None) -> list[dict]:
    rs = report["results"]
    if where:
        rs = [r for r in rs if all(r[k] == v for k, v in where.items())]
    return study.aggregate(rs, by)


def fmt(v, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "n/a" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def html_table(rows: list[dict], cols: list[tuple[str, str, int]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in cols)
    body = []
    for r in rows:
        tds = "".join(f"<td>{html.escape(fmt(r[k], d))}</td>" for k, _, d in cols)
        body.append(f"<tr>{tds}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--docs", default="docs")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    docs = pathlib.Path(a.docs)
    docs.mkdir(parents=True, exist_ok=True)
    report_path = out / "fidelity.json"
    if not report_path.exists():
        print(f"{report_path} is missing. Run `python3 -m asciivid study` first; the page "
              f"has no numbers to show without it.", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text())

    fnt = fontmod.load()
    chars, fontb64 = font_blob(fnt)

    clips = []
    w, h = DEMO_COLS * 8, DEMO_ROWS * 16
    srcs = [scenes.frame(DEMO_SCENE, i / DEMO_FPS, w, h) for i in range(DEMO_FRAMES)]
    for spec in DEMO_SPECS:
        ideal = [render.render(s, fnt, spec, DEMO_COLS) for s in srcs]
        # Pack what the delta encoder would leave on a screen, not the ideal frames, so the
        # numbers on the page describe the animation the page actually plays.
        enc = render.DeltaEncoder(tol=DEMO_TOL)
        frames = []
        for f in ideal:
            enc.encode(f)
            frames.append(enc.screen)
        cset = ramps.ordered(spec.ramp, fnt)
        packed = pack_clip(frames, cset)
        sc = [fidelity.score(s, raster.rasterize(f, fnt)) for s, f in zip(srcs, frames)]
        loss = float(np.mean([fidelity.score(s, raster.rasterize(f, fnt)).ssim
                              for s, f in zip(srcs, ideal)])
                     - np.mean([x.ssim for x in sc]))
        clips.append({
            "key": spec.key(),
            "bytes": len(packed),
            "ssim": round(float(np.mean([x.ssim for x in sc])), 4),
            "cell_r": round(float(np.mean([x.cell_r for x in sc])), 4),
            "ansi_bytes": int(np.mean([len(render.to_ansi(f)) for f in frames])),
            "tol_ssim_cost": round(loss, 5),
            "data": b64gz(packed),
        })

    stills = []
    for name in STILL_SCENES:
        src = scenes.frame(name, 1.0, STILL_COLS * 8, int(STILL_COLS * 8 * 0.75) // 16 * 16)
        cells = render.render(src, fnt, STILL_SPEC, STILL_COLS)
        back = raster.rasterize(cells, fnt)
        sc = fidelity.score(src, back)
        stills.append({
            "name": name,
            "src": base64.b64encode(raster.to_png_bytes(src)).decode("ascii"),
            "out": base64.b64encode(raster.to_png_bytes(back)).decode("ascii"),
            "ssim": sc.ssim, "cell_r": sc.cell_r,
        })

    by_mode = table_rows(report, ("mode",))
    by_ramp_mode = table_rows(report, ("ramp", "mode"))
    mono = table_rows(report, ("ramp", "match", "normalize"), {"mode": "mono"})
    by_scene = table_rows(report, ("scene",), {"mode": "truecolor"})

    payload = {
        "chars": chars, "font": fontb64, "clips": clips, "fps": DEMO_FPS,
        "scene": DEMO_SCENE,
    }

    page = TEMPLATE.format(
        title=html.escape(TITLE),
        n=len(report["results"]),
        scenes=len(report["source"]["scenes"]),
        combos=len({r["spec"] for r in report["results"]}),
        grid=f"{report['grid']['cols']}x{report['grid']['rows']}",
        source=f"{report['source']['width']}x{report['source']['height']}",
        font_sha=report["font_sha256"][:16],
        demo=f"{DEMO_COLS}x{DEMO_ROWS}, {DEMO_FRAMES} frames at {DEMO_FPS:g} fps",
        mode_table=html_table(by_mode, [("mode", "colour mode", 0), ("ssim", "SSIM", 4),
                                        ("ssim_cs", "SSIM (structure only)", 4),
                                        ("cell_r", "per-cell r", 4),
                                        ("rgb_rmse", "RGB RMSE", 2),
                                        ("bytes", "bytes/frame", 0)]),
        ramp_table=html_table(by_ramp_mode, [("ramp", "ramp", 0), ("mode", "mode", 0),
                                             ("ssim", "SSIM", 4),
                                             ("ssim_cs", "SSIM (structure only)", 4),
                                             ("cell_r", "per-cell r", 4),
                                             ("rgb_rmse", "RGB RMSE", 2)]),
        mono_table=html_table(mono, [("ramp", "ramp", 0), ("match", "matcher", 0),
                                     ("normalize", "normalised", 0),
                                     ("ssim", "SSIM", 4),
                                     ("ssim_cs", "SSIM (structure only)", 4),
                                     ("cell_r", "per-cell r", 4)]),
        scene_table=html_table(by_scene, [("scene", "source", 0), ("ssim", "SSIM", 4),
                                          ("ssim_cs", "SSIM (structure only)", 4),
                                          ("cell_r", "per-cell r", 4),
                                          ("rgb_rmse", "RGB RMSE", 2)]),
        stills="".join(
            f'<figure><figcaption>{html.escape(s["name"])} '
            f'<span class="num">SSIM {s["ssim"]:.3f}, per-cell r {fmt(s["cell_r"], 3)}'
            f'</span></figcaption>'
            f'<div class="pair"><img alt="{html.escape(s["name"])} source" '
            f'src="data:image/png;base64,{s["src"]}">'
            f'<img alt="{html.escape(s["name"])} rendered to characters" '
            f'src="data:image/png;base64,{s["out"]}"></div></figure>'
            for s in stills),
        payload=json.dumps(payload, separators=(",", ":")),
    )
    # One frame of the first clip, as Python composites it. The browser check pulls the
    # same frame off the canvas and compares, which is the only way to know that the page's
    # own decoder and compositor agree with the implementation the numbers came from.
    ref_clip, ref_frame = 0, 7
    ref = raster.rasterize(
        render.DeltaEncoder(tol=DEMO_TOL) and _screen_at(srcs, fnt, DEMO_SPECS[ref_clip],
                                                         ref_frame), fnt)
    (out / "browser_expect.json").write_text(json.dumps({
        "clip": ref_clip, "frame": ref_frame, "spec": DEMO_SPECS[ref_clip].key(),
        "width": ref.shape[1], "height": ref.shape[0],
        "rgb": base64.b64encode(ref.tobytes()).decode("ascii"),
    }))

    (docs / "index.html").write_text(page, encoding="utf-8")
    size = (docs / "index.html").stat().st_size
    print(f"docs/index.html: {size} bytes, {len(clips)} clips, "
          f"{sum(len(c['data']) for c in clips)} bytes of clip data")
    for c in clips:
        print(f"    {c['key']:38s} SSIM {c['ssim']:.4f}  {c['bytes']:7d} B packed  "
              f"{len(c['data']):7d} B base64+gzip")
    return 0


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: #0d1117; --fg: #e6edf3; --dim: #8b949e; --line: #30363d; --accent: #7ee787;
    --card: #161b22;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2rem 1rem 5rem; background: var(--bg); color: var(--fg);
    font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  main {{ max-width: 60rem; margin: 0 auto; }}
  h1 {{ font-size: clamp(1.6rem, 4vw, 2.4rem); margin: 0 0 .3rem; line-height: 1.2; }}
  h2 {{ font-size: 1.25rem; margin: 2.6rem 0 .6rem; border-bottom: 1px solid var(--line);
       padding-bottom: .4rem; }}
  p.lede {{ color: var(--dim); margin: 0 0 2rem; max-width: 46rem; }}
  code, .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .player {{ background: #000; border: 1px solid var(--line); border-radius: 8px;
             padding: .75rem; }}
  canvas {{ display: block; width: 100%; height: auto; image-rendering: pixelated;
            background: #000; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
               margin: .8rem 0 0; }}
  button {{ font: inherit; font-size: .85rem; color: var(--fg); background: var(--card);
            border: 1px solid var(--line); border-radius: 6px; padding: .35rem .7rem;
            cursor: pointer; }}
  button[aria-pressed="true"] {{ border-color: var(--accent); color: var(--accent); }}
  .stat {{ color: var(--dim); font-size: .85rem; margin-left: auto; }}
  .scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
  table {{ border-collapse: collapse; font-size: .85rem; min-width: 100%; }}
  th, td {{ padding: .35rem .7rem; text-align: right; white-space: nowrap;
            border-bottom: 1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--dim); font-weight: 600; }}
  td {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  figure {{ margin: 1.4rem 0; }}
  figcaption {{ color: var(--fg); font-size: .9rem; margin-bottom: .4rem; }}
  .num {{ color: var(--dim); font-size: .82rem; }}
  .pair {{ display: flex; gap: .5rem; overflow-x: auto; }}
  .pair img {{ width: 320px; height: auto; flex: none; border: 1px solid var(--line);
               image-rendering: pixelated; }}
  footer {{ color: var(--dim); font-size: .85rem; margin-top: 3rem;
            border-top: 1px solid var(--line); padding-top: 1rem; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="lede">
    Every frame below is drawn the way the fidelity numbers were measured: the committed
    glyph raster, composited between each cell's two colours. Nothing here is a screenshot.
    {n} measurements over {scenes} procedurally generated sources and {combos} render
    settings, at a {grid} character grid against a {source} source.
  </p>

  <h2>The clip</h2>
  <div class="player">
    <canvas id="screen" width="512" height="384"></canvas>
    <div class="controls" id="picker"></div>
    <div class="controls">
      <button id="toggle" type="button">Pause</button>
      <span class="stat" id="stat">loading</span>
    </div>
  </div>

  <h2>Colour depth costs this much</h2>
  {mode_table}

  <h2>Every ramp against every colour mode</h2>
  {ramp_table}

  <h2>Mono, where ink is the only brightness there is</h2>
  <p class="lede">
    The heaviest ASCII glyph in this raster inks 26.2% of its cell, so a bright image
    clips every cell to <code>@</code> and the ramp stops carrying anything. Normalising
    stretches the source onto the range the alphabet can actually express. SSIM stays low
    either way, because white ink on black cannot be as bright as the picture; the
    per-cell correlation is what moves.
  </p>
  {mono_table}

  <h2>Some sources are much harder than others</h2>
  {scene_table}
  {stills}

  <footer>
    Source raster <code class="mono">{font_sha}</code>. Demo clip: {demo}.
    <a href="https://github.com/JesseRWeigel/ascii-video">Repository and method</a>.
  </footer>
</main>
<script id="payload" type="application/json">{payload}</script>
<script>
(async function () {{
  const P = JSON.parse(document.getElementById('payload').textContent);
  const GW = 8, GH = 16;

  const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  async function gunzip(u8) {{
    const ds = new DecompressionStream('gzip');
    const buf = await new Response(new Blob([u8]).stream().pipeThrough(ds)).arrayBuffer();
    return new Uint8Array(buf);
  }}

  const fontBytes = bytes(P.font);
  const glyphs = new Map();
  [...P.chars].forEach((ch, i) => {{
    glyphs.set(ch, fontBytes.subarray(i * GW * GH, (i + 1) * GW * GH));
  }});

  function unpack(u8) {{
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let o = 0;
    const rows = dv.getUint16(o, true); o += 2;
    const cols = dv.getUint16(o, true); o += 2;
    const nframes = dv.getUint16(o, true); o += 2;
    const clen = dv.getUint16(o, true); o += 2;
    const charset = [...new TextDecoder().decode(u8.subarray(o, o + clen))]; o += clen;
    const n = rows * cols;
    const state = new Uint8Array(n * 7);
    const frames = [];
    for (let f = 0; f < nframes; f++) {{
      const count = dv.getUint32(o, true); o += 4;
      const idx = new Uint16Array(u8.slice(o, o + count * 2).buffer); o += count * 2;
      for (let k = 0; k < count; k++) {{
        const c = idx[k];
        for (let b = 0; b < 7; b++) state[c * 7 + b] = u8[o + k * 7 + b];
      }}
      o += count * 7;
      frames.push(state.slice());
    }}
    return {{ rows, cols, charset, frames }};
  }}

  const canvas = document.getElementById('screen');
  const ctx = canvas.getContext('2d', {{ willReadFrequently: false }});
  const picker = document.getElementById('picker');
  const stat = document.getElementById('stat');
  const toggle = document.getElementById('toggle');

  const clips = [];
  for (const c of P.clips) clips.push({{ meta: c, clip: unpack(await gunzip(bytes(c.data))) }});

  let current = 0, playing = true, frame = 0, origin = performance.now();
  let image = null;

  function draw(which, f) {{
    const {{ rows, cols, charset, frames }} = clips[which].clip;
    const W = cols * GW, H = rows * GH;
    if (canvas.width !== W || canvas.height !== H || !image ||
        image.width !== W || image.height !== H) {{
      canvas.width = W; canvas.height = H;
      image = ctx.createImageData(W, H);
    }}
    const px = image.data;
    const st = frames[((f % frames.length) + frames.length) % frames.length];
    for (let r = 0; r < rows; r++) {{
      for (let c = 0; c < cols; c++) {{
        const o = (r * cols + c) * 7;
        const g = glyphs.get(charset[st[o]]);
        const fr = st[o + 1], fg = st[o + 2], fb = st[o + 3];
        const br = st[o + 4], bg = st[o + 5], bb = st[o + 6];
        for (let y = 0; y < GH; y++) {{
          let d = ((r * GH + y) * W + c * GW) * 4;
          const gr = y * GW;
          for (let x = 0; x < GW; x++) {{
            const a = g[gr + x] / 255;
            px[d++] = br + a * (fr - br);
            px[d++] = bg + a * (fg - bg);
            px[d++] = bb + a * (fb - bb);
            px[d++] = 255;
          }}
        }}
      }}
    }}
    ctx.putImageData(image, 0, 0);
  }}

  function label(i) {{
    const m = clips[i].meta;
    return `${{m.key}}`;
  }}

  clips.forEach((_, i) => {{
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label(i);
    b.setAttribute('aria-pressed', String(i === 0));
    b.addEventListener('click', () => {{
      current = i;
      [...picker.children].forEach((el, j) =>
        el.setAttribute('aria-pressed', String(j === i)));
      origin = performance.now() - (frame / P.fps) * 1000;
      draw(current, frame);
      updateStat();
    }});
    picker.appendChild(b);
  }});

  function updateStat() {{
    const m = clips[current].meta;
    stat.textContent = `${{P.scene}} - SSIM ${{m.ssim.toFixed(4)}}, per-cell r `
      + `${{m.cell_r.toFixed(4)}}, ${{m.ansi_bytes}} bytes/frame of ANSI`;
  }}

  toggle.addEventListener('click', () => {{
    playing = !playing;
    toggle.textContent = playing ? 'Pause' : 'Play';
    origin = performance.now() - (frame / P.fps) * 1000;
    if (playing) requestAnimationFrame(loop);
  }});

  function loop(now) {{
    // Frame index from elapsed time, never incremented, so a slow tab does not fall behind.
    // The wrap has to be the positive modulo: a requestAnimationFrame timestamp can be
    // EARLIER than the performance.now() taken just before the frame was requested, and
    // JavaScript's % keeps the sign, so the plain form indexes frames[-1] and throws on the
    // very first frame. Found by running the page in a browser; every unit test passed.
    const n = clips[current].clip.frames.length;
    const i = Math.floor(((now - origin) / 1000) * P.fps);
    frame = ((i % n) + n) % n;
    draw(current, frame);
    if (playing) requestAnimationFrame(loop);
  }}

  draw(0, 0);
  updateStat();
  requestAnimationFrame(loop);
  document.body.dataset.pageReady = '1';
  window.__asciivid = {{
    clips: clips.map((c) => ({{
      key: c.meta.key, ssim: c.meta.ssim, cellR: c.meta.cell_r,
      frames: c.clip.frames.length, rows: c.clip.rows, cols: c.clip.cols,
    }})),
    frameAt: (i, f) => {{ draw(i, f); return true; }},
  }};
}})().catch((e) => {{
  document.body.dataset.pageError = String(e && e.message ? e.message : e);
  const s = document.getElementById('stat');
  if (s) s.textContent = 'failed: ' + e;
}});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
