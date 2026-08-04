"""Unit tests. Every claim here is paired with a negative control.

A test that only ever sees correct input proves that the code did not crash. The pattern
throughout this file is: assert the good case, then construct the bad case and assert the
same check rejects it. Where that is not possible in-process, the negative control lives in
`scripts/verify.sh` as a sabotage.
"""

from __future__ import annotations

import io
import json
import math
import pathlib
import subprocess
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from asciivid import cast, fidelity, font as fontmod, player, ramps, raster, render, scenes
import check_fidelity as chk

FONT = fontmod.load()


def disc(size: int = 480, width: int = 640) -> np.ndarray:
    """A centred circle on black. Round in the source, so any stretch is measurable."""
    y, x = np.mgrid[0:size, 0:width]
    r = np.sqrt((x - width / 2) ** 2 + (y - size / 2) ** 2)
    v = np.clip(1.0 - np.maximum(r - size * 0.3, 0.0), 0.0, 1.0)
    return (np.stack([v, v, v], axis=-1) * 255).astype(np.uint8)


def bbox_ratio(img: np.ndarray, thresh: int = 96) -> float:
    """Width divided by height of the bright region. 1.0 means it stayed round."""
    m = fidelity.luminance(img) > thresh
    ys, xs = np.nonzero(m)
    if not len(ys):
        raise AssertionError("nothing bright enough to measure")
    return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)


class TestFont(unittest.TestCase):
    def test_raster_is_pinned_and_sane(self):
        self.assertEqual(FONT.digest, fontmod.FONT_SHA256)
        self.assertGreaterEqual(len(FONT.glyphs), 100)
        self.assertEqual(FONT.coverage(" "), 0.0)
        self.assertGreater(FONT.coverage("█"), 0.95)   # full block
        self.assertAlmostEqual(FONT.coverage("▀"), 0.5, delta=0.03)  # upper half
        for ch in " .:-=+*#%@":
            self.assertEqual(FONT.alpha(ch).shape, (16, 8))

    def test_a_tampered_raster_is_rejected(self):
        """Negative control: the digest is what makes every fidelity number reproducible."""
        raw = fontmod.DATA.read_text().splitlines()
        for i, line in enumerate(raw):
            if not line.startswith("#"):
                cp, hexdata = line.split()
                raw[i] = f"{cp} " + ("ff" + hexdata[2:])
                break
        tmp = pathlib.Path(self.id().split(".")[-1] + ".font.tmp")
        tmp.write_text("\n".join(raw) + "\n")
        try:
            with self.assertRaises(ValueError) as e:
                fontmod.load(tmp)
            self.assertIn("expected", str(e.exception))
        finally:
            tmp.unlink()

    def test_missing_glyph_names_itself(self):
        with self.assertRaises(KeyError) as e:
            FONT.alpha("☃")
        self.assertIn("U+2603", str(e.exception))


class TestRamps(unittest.TestCase):
    def test_ramps_are_sorted_by_measured_coverage(self):
        for name in ramps.RAMPS:
            cs = ramps.ordered(name, FONT)
            cov = [FONT.coverage(c) for c in cs]
            self.assertEqual(cov, sorted(cov), f"{name} is not in coverage order")
            self.assertEqual(len(set(cs)), len(cs), f"{name} has a duplicate")

    def test_a_scrambled_ramp_still_comes_back_sorted(self):
        """Negative control: the ordering is measured, so a typo in the string cannot break it."""
        ramps.RAMPS["scrambled"] = "@ .%#"
        try:
            cs = ramps.ordered("scrambled", FONT)
            self.assertEqual(cs[0], " ")
            self.assertEqual(cs[-1], "@")
        finally:
            del ramps.RAMPS["scrambled"]

    def test_quantize_lands_on_the_palette(self):
        rgb = np.array([[[7, 200, 13], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
        idx = ramps.quantize(rgb, ramps.ANSI256)
        self.assertEqual(tuple(ramps.ANSI256[idx[0, 1]]), (255, 255, 255))
        self.assertEqual(tuple(ramps.ANSI256[idx[0, 2]]), (0, 0, 0))
        # Negative control: the 16-colour palette cannot land where the 256 one can.
        idx16 = ramps.quantize(rgb, ramps.ANSI16)
        self.assertNotEqual(tuple(ramps.ANSI16[idx16[0, 0]]), (7, 200, 13))


class TestAspect(unittest.TestCase):
    """Characters are twice as tall as they are wide, and ignoring it stretches everything."""

    def test_grid_geometry(self):
        self.assertEqual(render.grid_for(640, 480, 80, 2.0), (30, 80))
        self.assertEqual(render.grid_for(640, 480, 80, 1.0), (60, 80))

    def test_a_circle_stays_round_at_2_and_stretches_at_1(self):
        src = disc()
        spec = render.Spec("blocks", "truecolor", "structure")
        good = raster.rasterize(render.render(src, FONT, spec, 80, aspect=2.0), FONT)
        bad = raster.rasterize(render.render(src, FONT, spec, 80, aspect=1.0), FONT)
        self.assertAlmostEqual(bbox_ratio(src), 1.0, delta=0.02)
        self.assertAlmostEqual(bbox_ratio(good), 1.0, delta=0.05)
        # Negative control: the square-cell assumption is not a matter of taste.
        self.assertLess(bbox_ratio(bad), 0.60)


class TestCellOrientation(unittest.TestCase):
    """A cell is 8 wide and 16 tall, and the two axes are not interchangeable.

    Every metric in `fidelity.py` that aggregates over a cell is blind to the cell's
    internal orientation: transpose the pixels inside each cell and the cell mean does not
    move, so `cell_r` stays at 1.0000 and SSIM on smooth content moves by 0.0003. Measured
    on a real sabotage. These two cases look at the axes directly.
    """

    @staticmethod
    def _ink(cells):
        b = fidelity.luminance(raster.rasterize(cells, FONT)).reshape(30, 16, 80, 8)
        return {"tb": b[:, :8].mean() / max(b[:, 8:].mean(), 0.01),
                "lr": b[:, :, :, :4].mean() / max(b[:, :, :, 4:].mean(), 0.01)}

    @staticmethod
    def _source(axis):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        if axis == "horizontal":       # bright in the top half of every cell
            img[(np.arange(480) % 16) < 8] = 255
        else:                          # bright in the left half of every cell
            img[:, (np.arange(640) % 8) < 4] = 255
        return img

    def test_a_horizontal_edge_inside_each_cell_stays_horizontal(self):
        spec = render.Spec("ascii67", "mono", "structure")
        m = self._ink(render.render(self._source("horizontal"), FONT, spec, 80))
        self.assertGreater(m["tb"], 1.40, f"top/bottom ink ratio only {m['tb']:.2f}")
        self.assertLess(m["lr"], 1.03,
                        f"a source uniform across x rendered {m['lr']:.2f} left-heavy")

    def test_a_vertical_edge_inside_each_cell_does_not_become_horizontal(self):
        spec = render.Spec("ascii67", "mono", "structure")
        m = self._ink(render.render(self._source("vertical"), FONT, spec, 80))
        self.assertGreater(m["lr"], 1.05, f"left/right ink ratio only {m['lr']:.2f}")
        self.assertLess(m["tb"], 1.40,
                        f"a horizontal edge appeared from a vertical source ({m['tb']:.2f})")

    def test_the_two_axes_give_different_answers(self):
        """Negative control: a renderer ignoring its input would pass both tests above."""
        spec = render.Spec("ascii67", "mono", "structure")
        h = self._ink(render.render(self._source("horizontal"), FONT, spec, 80))
        v = self._ink(render.render(self._source("vertical"), FONT, spec, 80))
        self.assertGreater(abs(h["tb"] - v["tb"]), 0.2)
        self.assertGreater(abs(h["lr"] - v["lr"]), 0.05)


class TestGlyphDensity(unittest.TestCase):
    """The core claim of any ramp: a darker cell gets a heavier glyph."""

    def _density_r(self, spec, scene):
        src = scenes.frame(scene, 0.6, 640, 480)
        cells = render.render(src, FONT, spec, 80)
        lum = fidelity.luminance(src).reshape(30, 16, 80, 8).mean(axis=(1, 3)).ravel()
        cov = np.array([FONT.coverage(str(c)) for c in cells.chars.ravel()])
        if cov.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(lum, cov)[0, 1])

    def test_mean_matching_holds_above_0_95_on_every_source(self):
        """The stated threshold. Measured range is 0.9448 to 0.9994, so 0.94 is the floor."""
        for ramp in ("ascii67", "ascii10", "blocks"):
            spec = render.Spec(ramp, "mono", "mean", normalize=True)
            for scene in scenes.SCENES:
                r = self._density_r(spec, scene)
                self.assertGreater(r, 0.94,
                                   f"{spec.key()} on {scene}: density correlation {r:.4f}")

    def test_structure_matching_trades_density_for_shape_and_says_where(self):
        """The same measure on the other matcher, which is deliberately worse at it.

        `--match structure` optimises the glyph's ink *pattern* rather than its ink
        *amount*, so its density correlation is lower everywhere and collapses to 0.32 on
        the `glyphs` source, where the right glyph is a diagonal stroke and its coverage
        has nothing to do with the cell's brightness. That is the trade, and pinning it
        here keeps it from being quietly reversed.
        """
        spec = render.Spec("ascii67", "mono", "structure", normalize=True)
        for scene in ("plasma", "tunnel", "bars", "starfield"):
            r = self._density_r(spec, scene)
            self.assertGreater(r, 0.85, f"{scene}: {r:.4f}")
        hard = self._density_r(spec, "glyphs")
        self.assertLess(hard, 0.6, f"glyphs unexpectedly easy: {hard:.4f}")
        self.assertGreater(hard, 0.15, f"glyphs collapsed further than expected: {hard:.4f}")

    def test_the_correlation_can_fail(self):
        """Negative control: shuffle the chosen glyphs and the correlation collapses."""
        src = scenes.frame("tunnel", 0.6, 640, 480)
        cells = render.render(src, FONT, render.Spec("ascii67", "mono", "mean",
                                                     normalize=True), 80)
        lum = fidelity.luminance(src).reshape(30, 16, 80, 8).mean(axis=(1, 3)).ravel()
        cov = np.array([FONT.coverage(str(c)) for c in cells.chars.ravel()])
        rng = np.random.default_rng(7)
        shuffled = cov.copy()
        rng.shuffle(shuffled)
        self.assertLess(abs(float(np.corrcoef(lum, shuffled)[0, 1])), 0.1)


class TestRender(unittest.TestCase):
    def test_mono_uses_no_colour_and_colour_modes_do(self):
        src = scenes.frame("tunnel", 0.6, 640, 480)
        mono = render.to_ansi(render.render(src, FONT, render.Spec("ascii10", "mono",
                                                                   "structure"), 80))
        self.assertNotIn("\x1b[38", mono)
        tc = render.to_ansi(render.render(src, FONT, render.Spec("ascii10", "truecolor",
                                                                 "structure"), 80))
        self.assertIn("\x1b[38;2;", tc)

    def test_palette_modes_emit_only_palette_colours(self):
        src = scenes.frame("bars", 0.6, 640, 480)
        for mode, hi in (("ansi16", 15), ("ansi256", 255)):
            cells = render.render(src, FONT, render.Spec("blocks", mode, "structure"), 80)
            used = {tuple(c) for c in cells.fg.reshape(-1, 3)} | \
                   {tuple(c) for c in cells.bg.reshape(-1, 3)}
            allowed = {tuple(c) for c in ramps.PALETTES[mode].astype(np.uint8)}
            self.assertTrue(used <= allowed, f"{mode} emitted an off-palette colour")
            self.assertLessEqual(len(used), hi + 1)

    def test_normalize_spends_the_whole_ramp_and_naked_mono_does_not(self):
        """The measured reason --normalize exists."""
        src = scenes.frame("plasma", 0.6, 640, 480)
        plain = render.render(src, FONT, render.Spec("ascii10", "mono", "mean"), 80)
        norm = render.render(src, FONT, render.Spec("ascii10", "mono", "mean",
                                                    normalize=True), 80)
        self.assertLessEqual(len(np.unique(plain.chars)), 3,
                             "a bright image should clip a naked ASCII ramp")
        self.assertGreaterEqual(len(np.unique(norm.chars)), 8)
        rp = fidelity.cell_correlation(src, raster.rasterize(plain, FONT))
        rn = fidelity.cell_correlation(src, raster.rasterize(norm, FONT))
        self.assertGreater(rn, rp + 0.10)

    def test_specs_that_do_nothing_are_refused(self):
        with self.assertRaises(ValueError):
            render.Spec("blocks", "truecolor", "structure", dither=True).validate()
        with self.assertRaises(ValueError):
            render.Spec("blocks", "truecolor", "structure", normalize=True).validate()
        with self.assertRaises(ValueError):
            render.Spec("nope", "mono", "mean").validate()

    def test_dithering_changes_the_output(self):
        src = scenes.frame("plasma", 0.6, 640, 480)
        a = render.render(src, FONT, render.Spec("blocks", "ansi16", "structure"), 80)
        b = render.render(src, FONT, render.Spec("blocks", "ansi16", "structure",
                                                 dither=True), 80)
        self.assertFalse(np.array_equal(a.fg, b.fg), "dithering had no effect at all")

    def test_structure_matching_beats_mean_matching_where_shape_matters(self):
        src = scenes.frame("glyphs", 0.6, 640, 480)
        out = {}
        for m in ("mean", "structure"):
            cells = render.render(src, FONT, render.Spec("ascii67", "mono", m,
                                                         normalize=True), 80)
            out[m] = fidelity.score(src, raster.rasterize(cells, FONT)).ssim
        self.assertGreater(out["structure"], out["mean"])


class TestFidelity(unittest.TestCase):
    def test_identical_images_score_one(self):
        src = scenes.frame("bars", 0.6, 640, 480)
        s = fidelity.score(src, src)
        self.assertAlmostEqual(s.ssim, 1.0, places=9)
        self.assertAlmostEqual(s.ssim_cs, 1.0, places=9)
        self.assertAlmostEqual(s.cell_r, 1.0, places=9)
        self.assertEqual(s.rgb_rmse, 0.0)

    def test_an_unrelated_image_does_not(self):
        """Negative control: the metric has to be able to say no."""
        a = scenes.frame("bars", 0.6, 640, 480)
        b = scenes.frame("glyphs", 0.6, 640, 480)
        s = fidelity.score(a, b)
        self.assertLess(s.ssim, 0.1)
        self.assertGreater(s.rgb_rmse, 40.0)

    def test_the_floor_is_not_zero_for_two_smooth_images(self):
        """SSIM has a floor, and pretending otherwise would misread the whole table.

        Windowed SSIM with the standard stabilisers scores two unrelated but locally flat
        images at 0.57 here, because inside an 8x8 window both are nearly constant and the
        C2 term dominates. So 0.57 is not a good score for a smooth scene, and a table read
        as if 0 were the floor would be read wrong.
        """
        a = scenes.frame("bars", 0.6, 640, 480)
        b = scenes.frame("plasma", 0.6, 640, 480)
        self.assertAlmostEqual(fidelity.score(a, b).ssim, 0.567, delta=0.01)
        # Where both images carry detail there is no such floor.
        c = scenes.frame("glyphs", 0.6, 640, 480)
        self.assertLess(fidelity.score(a, c).ssim, 0.05)

    def test_a_flat_render_reports_nan_rather_than_zero(self):
        src = scenes.frame("plasma", 0.6, 640, 480)
        flat = np.zeros_like(src)
        self.assertTrue(math.isnan(fidelity.cell_correlation(src, flat)),
                        "a constant render cannot have a correlation, and must not "
                        "report one")

    def test_ssim_notices_a_one_pixel_shift(self):
        src = scenes.frame("glyphs", 0.6, 640, 480)
        shifted = np.roll(src, 3, axis=1)
        self.assertLess(fidelity.score(src, shifted).ssim, 0.9)


class TestAnsiRoundTrip(unittest.TestCase):
    """What the renderer holds and what a terminal receives must be the same picture."""

    def test_every_mode_survives_the_wire(self):
        src = scenes.frame("tunnel", 0.6, 640, 480)
        glyphs = chk.read_font(fontmod.DATA)
        for mode in ramps.MODES:
            spec = render.Spec("ascii10", mode, "structure")
            cells = render.render(src, FONT, spec, 80)
            chars, fg, bg = chk.parse_ansi(render.to_ansi(cells))
            self.assertEqual(["".join(r) for r in chars], cells.text().split("\n"), mode)
            np.testing.assert_array_equal(fg, cells.fg, err_msg=f"{mode} foreground")
            np.testing.assert_array_equal(bg, cells.bg, err_msg=f"{mode} background")
            here = raster.rasterize(cells, FONT)
            there = chk.composite(chars, fg, bg, glyphs)
            np.testing.assert_array_equal(here, there, err_msg=f"{mode} raster")

    def test_a_changed_colour_on_the_wire_is_visible(self):
        """Negative control: the round trip above must not be able to agree with anything.

        A wrong colour in the byte stream has to come back out as a wrong colour, otherwise
        the parser is reconstructing the frame from something other than what was sent.
        """
        src = scenes.frame("bars", 0.6, 640, 480)
        cells = render.render(src, FONT, render.Spec("blocks", "truecolor", "structure"), 80)
        good = render.to_ansi(cells)
        first = good.index("\x1b[38;2;")
        end = good.index("m", first)
        tampered = good[:first] + "\x1b[38;2;3;5;7;48;2;9;11;13" + good[end:]
        _, fg, bg = chk.parse_ansi(tampered)
        self.assertEqual(tuple(fg[0, 0]), (3, 5, 7))
        self.assertEqual(tuple(bg[0, 0]), (9, 11, 13))
        self.assertFalse(np.array_equal(fg, cells.fg))


class TestDelta(unittest.TestCase):
    def _stream(self, scene: str, mode: str, tol: int, n: int = 12):
        spec = render.Spec("blocks", mode, "structure")
        frames = [render.render(scenes.frame(scene, i / 24.0, 320, 240), FONT, spec, 40)
                  for i in range(n)]
        enc = render.DeltaEncoder(tol=tol)
        glyphs = chk.read_font(fontmod.DATA)
        screen = None
        stream = full = 0
        for i, f in enumerate(frames):
            chunk = enc.encode(f)
            stream += len(chunk)
            full += len(render.to_ansi(f))
            screen = _apply(screen, chunk, f.rows, f.cols)
            # What the terminal now shows must equal what the encoder believes it shows.
            want = chk.parse_ansi(render.to_ansi(enc.screen))
            self.assertEqual(screen[0], want[0], f"characters diverged at frame {i}")
            np.testing.assert_array_equal(screen[1], want[1], f"foreground at frame {i}")
            np.testing.assert_array_equal(screen[2], want[2], f"background at frame {i}")
            # np.testing rather than assertEqual on .tolist(): on failure unittest runs
            # difflib over the two nested lists, and for a 336x448x3 frame that takes
            # minutes and prints nothing useful. A sabotage run found this by hanging.
            np.testing.assert_array_equal(chk.composite(*screen, glyphs),
                                          raster.rasterize(enc.screen, FONT),
                                          err_msg=f"raster at frame {i}")
        return stream, full, enc

    def test_the_stream_reconstructs_the_screen_exactly(self):
        for scene, mode, tol in (("tunnel", "truecolor", 0), ("plasma", "truecolor", 12),
                                 ("glyphs", "mono", 0), ("bars", "ansi256", 4)):
            with self.subTest(scene=scene, mode=mode, tol=tol):
                self._stream(scene, mode, tol)

    def test_exact_deltas_save_a_lot_on_mono_and_nothing_on_smooth_truecolor(self):
        """Both halves are measurements. The second one is the surprising one."""
        mono, mono_full, _ = self._stream("glyphs", "mono", 0)
        self.assertLess(mono, mono_full * 0.2, f"{mono} vs {mono_full}")
        smooth, smooth_full, enc = self._stream("tunnel", "truecolor", 0)
        # Every cell's colour moves by a unit or two, so an exact delta is never smaller
        # and the encoder correctly falls back to whole frames on all of them.
        self.assertEqual(smooth, smooth_full)
        self.assertEqual(enc.delta_frames, 0)

    def test_a_colour_tolerance_buys_bytes_back(self):
        exact, full, _ = self._stream("plasma", "truecolor", 0)
        loose, _, enc = self._stream("plasma", "truecolor", 12)
        self.assertLess(loose, exact * 0.9, f"tol=12 saved nothing: {loose} vs {exact}")
        self.assertGreater(enc.delta_frames, 0)
        # And the picture it leaves on the screen is still the picture.
        src = scenes.frame("plasma", 11 / 24.0, 320, 240)
        want = render.render(src, FONT, render.Spec("blocks", "truecolor", "structure"), 40)
        exact_ssim = fidelity.score(src, raster.rasterize(want, FONT)).ssim
        shown_ssim = fidelity.score(src, raster.rasterize(enc.screen, FONT)).ssim
        self.assertLess(exact_ssim - shown_ssim, 0.005,
                        f"the tolerance cost {exact_ssim - shown_ssim:.4f} of SSIM")

    def test_a_corrupted_delta_is_caught(self):
        """Negative control: the reconstruction check must be able to fail."""
        spec = render.Spec("ascii10", "mono", "structure")
        a = render.render(scenes.frame("tunnel", 0.0, 320, 240), FONT, spec, 40)
        b = render.render(scenes.frame("tunnel", 0.5, 320, 240), FONT, spec, 40)
        screen = _apply(None, render.to_ansi(a), a.rows, a.cols)
        d = render.to_delta(b, a)
        self.assertIn("\x1b[", d)
        corrupt = d.replace("H", "H ", 1)
        screen = _apply(screen, corrupt, b.rows, b.cols)
        self.assertNotEqual(screen[0], chk.parse_ansi(render.to_ansi(b))[0])

    def test_a_negative_tolerance_is_refused(self):
        with self.assertRaises(ValueError):
            render.DeltaEncoder(tol=-1)

    def test_frame_size_cannot_change_mid_stream(self):
        spec = render.Spec("ascii10", "mono", "structure")
        a = render.render(scenes.frame("tunnel", 0.0, 320, 240), FONT, spec, 40)
        b = render.render(scenes.frame("tunnel", 0.0, 320, 240), FONT, spec, 20)
        with self.assertRaises(ValueError):
            render.to_delta(b, a)


def _apply(screen, chunk: str, rows: int, cols: int):
    """A minimal terminal: absolute cursor moves, SGR colour state, printable cells.

    Written here rather than reused from the renderer, because a reconstruction that used
    the renderer's own idea of where the cursor is would agree with it by construction.
    """
    import re
    if screen is None:
        chars = [[" "] * cols for _ in range(rows)]
        fg = np.zeros((rows, cols, 3), dtype=np.int64)
        bg = np.zeros((rows, cols, 3), dtype=np.int64)
    else:
        chars = [row[:] for row in screen[0]]
        fg, bg = screen[1].copy(), screen[2].copy()
    csi = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
    r = c = 0
    cur_fg, cur_bg = (255, 255, 255), (0, 0, 0)
    i = 0
    while i < len(chunk):
        ch = chunk[i]
        if ch == "\x1b":
            m = csi.match(chunk, i)
            if not m:
                raise AssertionError(f"bad escape at {i}")
            p, final = m.group(1), m.group(2)
            if final == "H":
                nums = [int(x) if x else 1 for x in p.split(";")] if p else [1, 1]
                r, c = nums[0] - 1, (nums[1] - 1 if len(nums) > 1 else 0)
            elif final == "m":
                cur_fg, cur_bg = chk._sgr(p, cur_fg, cur_bg)
            i = m.end()
            continue
        if ch == "\r":
            c = 0; i += 1; continue
        if ch == "\n":
            r += 1; i += 1; continue
        if 0 <= r < rows and 0 <= c < cols:
            chars[r][c] = ch
            fg[r, c] = cur_fg
            bg[r, c] = cur_bg
        c += 1
        i += 1
    return chars, fg, bg


class TestTiming(unittest.TestCase):
    """A ten minute clip is where drift becomes visible, so that is what gets simulated."""

    FPS = 24.0
    N = 24 * 600          # ten minutes
    OVERHEAD = 0.0005     # half a millisecond of real work per frame

    def _clock(self):
        state = {"t": 1000.0}

        def now():
            return state["t"]

        def sleep(d):
            state["t"] += max(0.0, d)

        def work():
            state["t"] += self.OVERHEAD
        return state, now, sleep, work

    def test_absolute_deadlines_do_not_drift(self):
        state, now, sleep, work = self._clock()
        t0 = state["t"]
        worst = 0.0
        for tick in player.schedule(self.N, self.FPS, now, sleep):
            work()
            worst = max(worst, abs(tick.shown - tick.due))
        self.assertLess(worst, 0.002, f"worst offset {worst * 1000:.2f} ms")
        # The last frame is due at (N-1)/fps, and the work for it lands after that.
        elapsed = state["t"] - t0
        self.assertAlmostEqual(elapsed, (self.N - 1) / self.FPS + self.OVERHEAD, places=6)

    def test_the_naive_player_drifts_and_the_test_can_see_it(self):
        """Negative control. If this ever passes, the drift test above proves nothing."""
        state, now, sleep, work = self._clock()
        worst = 0.0
        for tick in player.naive_schedule(self.N, self.FPS, now, sleep):
            work()
            worst = max(worst, abs(tick.shown - tick.due))
        self.assertGreater(worst, 1.0,
                           f"the naive player only drifted {worst:.3f} s, so the "
                           f"comparison is not measuring anything")
        self.assertAlmostEqual(worst, self.N * self.OVERHEAD, delta=0.05)

    def test_a_stalled_frame_is_dropped_rather_than_shown_late(self):
        state, now, sleep, _ = self._clock()
        seen = []
        it = player.schedule(10, self.FPS, now, sleep)
        for tick in it:
            seen.append(tick)
            if tick.index == 2:
                state["t"] += 0.5     # something blocked for half a second
        dropped = [t.index for t in seen if t.dropped]
        self.assertTrue(dropped, "a half second stall must cost frames")
        self.assertLessEqual(max(t.shown - t.due for t in seen if not t.dropped), 0.5)

    def test_audio_and_video_share_one_origin(self):
        """Audio starts at the origin, so frame i is due at i/fps after the sound starts."""
        state, now, sleep, _ = self._clock()
        origin = 500.0
        for tick in player.schedule(2000, self.FPS, now, sleep, origin=origin):
            self.assertAlmostEqual(tick.due - origin, tick.index / self.FPS, places=9)

    def test_zero_fps_is_refused(self):
        with self.assertRaises(ValueError):
            list(player.schedule(1, 0.0, lambda: 0.0, lambda d: None))


class TestCast(unittest.TestCase):
    def test_round_trip_and_absolute_timestamps(self):
        buf = io.StringIO()
        n = cast.write(buf, 40, 12, [f"frame{i}" for i in range(50)], fps=30.0)
        buf.seek(0)
        head, events = cast.read(buf)
        self.assertEqual(head["width"], 40)
        self.assertEqual(head["height"], 12)
        self.assertEqual(n, len(events))
        for i in range(50):
            self.assertAlmostEqual(events[i][0], i / 30.0, places=5)
        self.assertAlmostEqual(events[-1][0], 50 / 30.0, places=5)
        gaps = list(cast.durations(events))
        self.assertLess(max(gaps) - min(gaps), 1e-5, "frame gaps are not even")

    def test_a_cast_that_goes_backwards_is_rejected(self):
        """Negative control: the reader has to be able to refuse a bad file."""
        bad = json.dumps({"version": 2, "width": 4, "height": 2}) + "\n" \
            + json.dumps([1.0, "o", "a"]) + "\n" + json.dumps([0.5, "o", "b"]) + "\n"
        with self.assertRaises(ValueError) as e:
            cast.read(io.StringIO(bad))
        self.assertIn("backwards", str(e.exception))

    def test_version_1_is_rejected(self):
        bad = json.dumps({"version": 1, "width": 4, "height": 2}) + "\n"
        with self.assertRaises(ValueError):
            cast.read(io.StringIO(bad))

    def test_timestamps_do_not_accumulate_error_over_a_long_clip(self):
        buf = io.StringIO()
        cast.write(buf, 10, 5, ["x"] * 20000, fps=23.976, hide_cursor=False)
        buf.seek(0)
        _, events = cast.read(buf)
        self.assertAlmostEqual(events[-1][0], 19999 / 23.976, places=4)


class TestScenes(unittest.TestCase):
    def test_every_scene_is_deterministic_and_distinct(self):
        rendered = {}
        for name in scenes.SCENES:
            a = scenes.frame(name, 0.7, 160, 120)
            b = scenes.frame(name, 0.7, 160, 120)
            np.testing.assert_array_equal(a, b, err_msg=f"{name} is not deterministic")
            self.assertEqual(a.shape, (120, 160, 3))
            self.assertGreater(a.std(), 5.0, f"{name} is nearly constant")
            rendered[name] = a
        keys = list(rendered)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                self.assertFalse(np.array_equal(rendered[a], rendered[b]), f"{a} == {b}")

    def test_scenes_move(self):
        for name in scenes.SCENES:
            a = scenes.frame(name, 0.0, 160, 120)
            b = scenes.frame(name, 1.3, 160, 120)
            self.assertGreater(np.abs(a.astype(int) - b.astype(int)).mean(), 1.0,
                               f"{name} does not animate")

    def test_an_unknown_scene_says_so(self):
        with self.assertRaises(KeyError) as e:
            scenes.frame("nope", 0.0, 8, 8)
        self.assertIn("plasma", str(e.exception))

    def test_ppm_round_trip(self):
        img = scenes.frame("bars", 0.3, 64, 48)
        p = pathlib.Path("test_roundtrip.ppm")
        try:
            scenes.write_ppm(p, img)
            np.testing.assert_array_equal(chk.read_ppm(p), img)
        finally:
            p.unlink()

    def test_png_writer_is_a_real_png(self):
        img = scenes.frame("plasma", 0.3, 32, 16)
        data = raster.to_png_bytes(img)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IEND", data)
        import zlib, struct
        w, h = struct.unpack(">II", data[16:24])
        self.assertEqual((w, h), (32, 16))
        start = data.index(b"IDAT") + 4
        length = struct.unpack(">I", data[start - 8:start - 4])[0]
        raw = zlib.decompress(data[start:start + length])
        self.assertEqual(len(raw), h * (1 + w * 3))
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, 1 + w * 3)
        self.assertTrue((rows[:, 0] == 0).all(), "only filter type 0 is written")
        np.testing.assert_array_equal(rows[:, 1:].reshape(h, w, 3), img)


class TestDeterminism(unittest.TestCase):
    def test_two_runs_in_one_process_are_identical(self):
        src = scenes.frame("plasma", 1.1, 640, 480)
        spec = render.Spec("ascii67", "ansi256", "structure", dither=True)
        a = render.to_ansi(render.render(src, FONT, spec, 80))
        b = render.to_ansi(render.render(src, FONT, spec, 80))
        self.assertEqual(a, b)

    def test_a_different_hash_seed_renders_the_same_bytes(self):
        """Anything keyed on Python's hash() looks deterministic inside one process."""
        code = (
            "import sys, hashlib; sys.path.insert(0, %r);"
            "from asciivid import font, render, scenes;"
            "f=font.load(); s=scenes.frame('glyphs', 1.1, 640, 480);"
            "sp=render.Spec('ascii67','ansi256','structure',dither=True);"
            "print(hashlib.sha256(render.to_ansi(render.render(s,f,sp,80))"
            ".encode()).hexdigest())" % str(ROOT)
        )
        outs = []
        for seed in ("0", "12345"):
            env = {"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout.strip())
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(len(outs[0]), 64)


class TestCheckerIndependence(unittest.TestCase):
    def test_the_checker_imports_nothing_from_the_package(self):
        """verify.sh proves this too; having it here means the unit suite alone catches it."""
        import ast
        tree = ast.parse((ROOT / "scripts" / "check_fidelity.py").read_text())
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits += [a.name for a in node.names if a.name.split(".")[0] == "asciivid"]
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "asciivid":
                    hits.append(node.module)
        self.assertEqual(hits, [], f"the checker imports {hits} from the thing it checks")

    def test_the_checker_agrees_with_the_renderer_on_a_fresh_frame(self):
        src = scenes.frame("bars", 0.9, 320, 240)
        spec = render.Spec("blocks", "ansi256", "structure")
        cells = render.render(src, FONT, spec, 40)
        mine = fidelity.score(src, raster.rasterize(cells, FONT))
        theirs = chk.measure(src, chk.composite(*chk.parse_ansi(render.to_ansi(cells)),
                                                chk.read_font(fontmod.DATA)))
        self.assertAlmostEqual(mine.ssim, theirs["ssim"], places=9)
        self.assertAlmostEqual(mine.ssim_cs, theirs["ssim_cs"], places=9)
        self.assertAlmostEqual(mine.cell_r, theirs["cell_r"], places=9)
        self.assertAlmostEqual(mine.rgb_rmse, theirs["rgb_rmse"], places=9)

    def test_the_checker_rejects_a_frame_that_is_not_what_was_measured(self):
        """Negative control: agreement above must not be automatic."""
        src = scenes.frame("bars", 0.9, 320, 240)
        other = scenes.frame("plasma", 0.9, 320, 240)
        cells = render.render(src, FONT, render.Spec("blocks", "ansi256", "structure"), 40)
        theirs = chk.measure(other, chk.composite(*chk.parse_ansi(render.to_ansi(cells)),
                                                  chk.read_font(fontmod.DATA)))
        mine = fidelity.score(src, raster.rasterize(cells, FONT))
        self.assertGreater(abs(mine.ssim - theirs["ssim"]), 0.05)

    def test_the_checker_refuses_malformed_ansi(self):
        with self.assertRaises(ValueError):
            chk.parse_ansi("\x1b[38;2;1;2m@")          # truncated truecolor
        with self.assertRaises(ValueError):
            chk.parse_ansi("abc\ndefg")                 # ragged rows
        with self.assertRaises(ValueError):
            chk.parse_ansi("\x1b[99m@")                 # SGR nobody emits


if __name__ == "__main__":
    unittest.main()
