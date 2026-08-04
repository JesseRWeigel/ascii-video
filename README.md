# ascii-video

Render video to ANSI characters, play it in a terminal, and **measure how much of the
picture survives**.

**[Watch it in a browser](https://jesserweigel.github.io/ascii-video/)**

```bash
python3 -m asciivid play   --scene tunnel --ramp blocks --mode truecolor
python3 -m asciivid render --scene plasma --mode ansi256 --seconds 6 --out clip.cast
python3 -m asciivid study  --out out/          # the whole fidelity table, from scratch
python3 -m asciivid table  --out out/ --by ramp,mode
bash scripts/verify.sh                         # everything below, checked
```

Python 3 and numpy. Nothing else is required. `ffmpeg` is optional and only needed to feed
it a real video file; the committed demo is generated from code, so the repository carries
no video binary and the whole pipeline reproduces from source.

Task ART-046 from [722 things to build](https://github.com/JesseRWeigel/722-things-to-build).

## The point is the measurement

Every ASCII art renderer claims a good ramp. Almost none of them say how good, because
saying so means comparing the characters against the image they replaced, and characters
are not pixels. This one does it by rendering the character grid **back into pixels**, the
way a terminal would: an 8x16 grayscale raster of every glyph it can emit, committed to
`data/font8x16.txt`, composited between each cell's two colours. That image is the same
size as the source, so the comparison needs no resampling and no excuse.

Four numbers, over five procedurally generated sources with deliberately different
characteristics, at 60 render settings. 600 measurements in total.

| metric | what it asks |
|---|---|
| `ssim` | structural similarity over non-overlapping 8x8 luminance windows |
| `ssim_cs` | the same, with the luminance term dropped: did the *structure* survive |
| `cell_r` | Pearson r between source and rendered mean luminance, one point per cell |
| `rgb_rmse` | root mean square colour error per channel, 0-255 units |

`ssim_cs` exists because of a hard limit, described below, that makes full SSIM say the
same thing about every mono setting. `cell_r` is the weakest claim a renderer can make, and
it is reported separately because a mapping can score 0.99 there while destroying every
edge inside every cell. Watching those two disagree is most of what the table is for.

## What colour depth costs

600 measurements, 5 sources, 60 render settings, 80x30 characters against a 640x480 source.

| colour mode | SSIM | SSIM (structure only) | per-cell r | RGB RMSE | bytes/frame |
|---|---|---|---|---|---|
| truecolor | **0.6706** | 0.6790 | 0.9997 | 29.45 | 61110 |
| ansi256 | 0.6310 | 0.6779 | 0.8797 | 34.28 | 31228 |
| ansi16 | 0.4611 | **0.6822** | 0.7113 | 47.95 | 8994 |
| mono | 0.0786 | 0.2751 | 0.8487 | 115.50 | 4117 |

Dropping from 16 million colours to 256 costs 0.04 of SSIM and halves the bytes. Dropping
to 16 costs 0.21 and divides the bytes by seven. Note the third column: **ansi16 has the
best structure score of any mode** and the worst full SSIM. Sixteen colours reproduce edges
perfectly well and put them in the wrong colour, which is exactly the shape the two numbers
describe when you keep them apart.

## In truecolor the character barely matters

| ramp | glyphs | SSIM (truecolor) | SSIM (mono, normalised) |
|---|---|---|---|
| `ascii67` | 67 | **0.6920** | **0.1221** |
| `ascii10` | 10 | 0.6685 | 0.0661 |
| `halfblock` (`▀`) | 1 | 0.6678 | 0.0965 |
| `blocks` (` ░▒▓█`) | 5 | 0.6656 | 0.1058 |
| `binary` (` #`) | 2 | 0.6591 | 0.0583 |

A 67-character ramp beats a **2-character** ramp by 0.033 of SSIM in truecolor. That is
almost nothing, and it is not a defect. In a colour mode the glyph is only a partition of
the cell into two regions, and the two fitted colours carry everything else, so ` #` with
good colours is nearly as good as a full ASCII alphabet with good colours. The ramp starts
to matter once the colours are taken away, and there the same comparison is a factor of two.

## Mono has a ceiling, and the ceiling is the alphabet

The heaviest ASCII glyph in this raster, `@`, inks **26.2%** of its cell. So a white-on-black
render of an image whose mean luminance is 150 cannot get past about 67, every bright cell
clips to `@`, and the ramp stops carrying information. Measured on the plasma source, a
10-character ramp used **two** of its characters and a 67-character ramp used two of its
sixty-seven, and the two rendered identically.

`--normalize` stretches the source onto the range the alphabet can actually express, in
whatever units the matcher compares in. Averaged over all five sources and all five ramps:

| matcher | normalised | SSIM | SSIM (structure only) | per-cell r |
|---|---|---|---|---|
| `structure` | yes | 0.0898 | **0.3145** | 0.8334 |
| `structure` | no | 0.0808 | 0.2659 | 0.7612 |
| `mean` | yes | 0.0760 | 0.2987 | **0.9512** |
| `mean` | no | 0.0678 | 0.2214 | 0.8246 |

Full SSIM stays near 0.08 whatever you do, because white ink on black cannot be as bright
as the picture and no ramp fixes that. The two columns that move are the ones that ignore
absolute brightness. `--match mean` maps cell average brightness to glyph ink coverage and
wins on per-cell correlation, reaching **0.9971** for `ascii67`. `--match structure`
compares the cell's luminance pattern to the glyph's ink pattern at 4x4 subcell resolution
and wins on structure, because it is the only one that can tell a `/` from a `\`.

**The SSIM floor is not zero.** Two unrelated but locally smooth images score 0.567 on this
metric, because inside an 8x8 window both are nearly flat and the stabiliser dominates. A
mono score of 0.08 is genuinely bad and a truecolor score of 0.67 is genuinely good, and
0.567 is not halfway between them. A unit test pins that number so the table cannot be
misread.

## Ordered dithering made it worse

The standard pitch for dithering against a limited palette is that it trades spatial
resolution for colour resolution. Measured on ansi256, over 150 renders:

| dithered | SSIM | SSIM (structure only) | per-cell r | bytes/frame |
|---|---|---|---|---|
| no | **0.6484** | 0.6757 | **0.9141** | 27481 |
| yes | 0.5964 | **0.6822** | 0.8109 | 38720 |

It costs 0.05 of SSIM, costs 0.10 of per-cell correlation, adds **41% to the byte count**,
and buys 0.0065 of structure score. The reason is that a cell has only two colours, so the
dither offset lands per character cell rather than per pixel, where it is plainly visible
instead of averaging out. The flag stays because the number is worth knowing. The default
is off.

## Some sources are 20 times harder than others

Truecolor, per source:

| source | SSIM | RGB RMSE | what it is |
|---|---|---|---|
| `plasma` | **0.9564** | 4.13 | smooth low-frequency colour |
| `starfield` | 0.8169 | 14.66 | sparse bright points on near-black |
| `bars` | 0.8143 | 48.09 | saturated hard-edged geometry |
| `tunnel` | 0.7163 | 10.32 | shaded solid with a hard silhouette |
| `glyphs` | **0.0492** | 70.06 | a dense grid of marks about one cell across |

This is the entire reason there are five sources. A renderer benchmarked on `plasma` alone
would report 0.96 and look excellent. The same renderer on `glyphs`, whose features sit
right at the size of a character cell, scores **0.049**, which is worse than the SSIM floor
between two unrelated images. Detail at the sampling limit does not survive at all, in any
colour mode, and the per-cell correlation for that source is still 0.9996: every cell has
the right average brightness and none of them has the right contents.

## Does a darker cell get a heavier glyph

That is the one claim every character ramp makes, so it is measured directly: Pearson r
between the source cell's mean luminance and the **ink coverage of the glyph actually
chosen** for that cell, one point per cell, per source.

| ramp | matcher | plasma | tunnel | bars | starfield | glyphs |
|---|---|---|---|---|---|---|
| `ascii67` | `mean` | 0.9994 | 0.9984 | 0.9992 | 0.9836 | 0.9990 |
| `ascii10` | `mean` | 0.9904 | 0.9874 | 0.9926 | 0.9794 | 0.9783 |
| `blocks` | `mean` | 0.9797 | 0.9603 | 0.9890 | 0.9645 | 0.9448 |
| `ascii67` | `structure` | 0.9145 | 0.9437 | 0.9297 | 0.8698 | **0.3171** |

The stated threshold for `--match mean` is **0.94 on every source**, asserted in the test
suite, against a measured range of 0.9448 to 0.9994. `--match structure` is deliberately
worse at this and the test pins that too: it optimises the glyph's ink *pattern* rather than
its ink *amount*, and on `glyphs`, where the right answer is a diagonal stroke whose coverage
says nothing about brightness, it drops to 0.317. The negative control shuffles the chosen
glyphs and the correlation falls below 0.1.

## Aspect ratio, and the other axis

A terminal cell is twice as tall as it is wide. The grid is derived from the 8x16 font cell,
so this is right by construction, and `--aspect` exists so the wrong answer can be measured:

```
source circle, width/height          1.000
--aspect 2.0 rendered, width/height  1.000   (640x480)
--aspect 1.0 rendered, width/height  0.500   (640x960)
```

The square-cell assumption stretches everything by exactly a factor of two. A test asserts
both numbers, so the correct behaviour cannot quietly regress into the common one.

Inside the cell, the two axes are also not interchangeable, and **every metric that
aggregates over a cell is blind to that**. A sabotage that transposes the pixels within each
cell leaves `cell_r` at 1.0000 and moves SSIM on smooth content by 0.0003. So the axes are
checked directly: a source that is bright in the top half of every cell and uniform across x
must render with 1.65 times more ink in the top half than the bottom and no left/right bias,
and the transposed renderer produces 1.20 and a 1.12 left/right bias instead. Both
assertions fail under the sabotage.

## Delta encoding, and where it does nothing

The player sends only the cells that changed, addressed with cursor moves. Measured over 24
frames at 40x15 characters, as a fraction of sending whole frames:

| source | mono | truecolor, exact | truecolor, `--delta-tol 8` |
|---|---|---|---|
| `glyphs` | **0.04** | **0.25** | 0.25 |
| `starfield` | 0.21 | 0.98 | 0.96 |
| `plasma` | 0.26 | 1.00 | **0.86** |
| `tunnel` | 0.49 | 1.00 | 0.95 |
| `bars` | 0.63 | 0.97 | 0.97 |

Exact delta encoding saves **nothing at all** on smooth truecolor content. A slowly moving
gradient changes 9% of the characters and very nearly 100% of the colours, by one or two
units each, so every cell has to be resent and the cursor moves are pure overhead. The
encoder detects this and falls back to whole frames, which is why the ratio is 1.00 and not
1.05. `--delta-tol` treats a small colour difference as no change and buys the bytes back:
at tolerance 8 the demo clip on the page costs 0.003 to 0.006 of SSIM, which is not visible,
and the page would otherwise be 1.2 MB of one-unit colour changes.

Because a held-back cell leaves the *old* colour on screen, the encoder tracks the screen
state rather than the previous frame. Encoding against the previous frame lets those
one-unit differences accumulate silently until the picture is stale. A test replays the byte
stream through a minimal terminal model and asserts the screen matches the encoder's idea of
it, frame by frame, pixel for pixel.

## Timing that does not drift

The obvious player sleeps `1/fps` after each frame. Every frame then pays for its own
rendering time plus the operating system's sleep rounding, and none of it is repaid.
Simulated over ten minutes at 24 fps with half a millisecond of overhead per frame:

```
absolute deadlines from a single origin   worst offset  < 2 ms
sleep(1/fps) after each frame             worst offset  7.2 s
```

Both policies are in `asciivid/player.py` and the test suite runs both, because the drifting
one is the negative control that proves the measurement can see drift at all. A frame more
than one interval overdue is dropped rather than shown late, which is what keeps the picture
attached to the sound. Audio is started from the same origin, so frame `i` is due exactly
`i/fps` after the sound begins.

**Not checked here:** no sound card is exercised by `scripts/verify.sh`. `--audio` shells out
to `ffplay`, `aplay` or `paplay` and raises with an actionable message if none is installed,
rather than playing silently. What is tested is that the frame schedule and the audio start
share one origin.

## Two ways to be wrong, so two checkers

`asciivid/study.py` measures the frame it holds in memory. That is one derivation, and a
validator that only compares two derivations passes when both are wrong together.

`scripts/check_fidelity.py` takes a different route to the same numbers. It reads the source
frames as PPM with its own parser, reads the emitted `.ansi` files with its own escape
parser exactly as a terminal would, reads the glyph raster with its own reader, composites
the frames itself, and computes SSIM from **integral images** rather than a reshape. It also
re-derives every source frame from its own pure-Python copy of the scene formulas, which is
what stops `out/source/` from quietly becoming a copy of the render.

It imports nothing from `asciivid/`. `scripts/verify.sh` proves that by walking the import
graph with `ast` rather than trusting the docstring.

The disagreement across all 600 numbers is at most **5.0e-07**, which is the rounding in
`fidelity.json`.

The split is not decoration. Two of the sabotages in `verify.sh`, dropping the background
colour from the escape sequence and shifting the palette index by one, leave the in-memory
cell array perfect and change what a terminal draws. Anything that measured only the cells
would report both as fine.

## Sabotage

Thirteen deliberate defects, each **proved to have applied** and **proved to have changed
the output** before its detection counts for anything. An attack that changes nothing proves
nothing, and treating one as evidence of a gap is how a correct check gets weakened.

`luma-weights-swapped`, `foreground-and-background-swapped`, `glyph-choice-ignored`,
`cells-transposed`, `aspect-ratio-ignored`, `background-colour-never-sent`,
`palette-index-off-by-one`, `raster-ignores-the-glyph`, `ssim-always-perfect`,
`undefined-correlation-reported-as-zero`, `source-dump-is-the-render`, `a-scene-stops-moving`,
`frame-deadlines-accumulate`.

`source-dump-is-the-render` is the one that only the independent checker can catch: it makes
`out/source/` disagree with the scene formula, and the checker's pure-Python re-derivation
is the only thing that ever looks.

Three of these initially reported "changed the output and nothing noticed" or "produced
identical output", and each was a real hole rather than a false alarm.

- `cells-transposed` changed the picture and every check agreed with itself, which is what
  produced the two orientation tests above.
- `undefined-correlation-reported-as-zero` changed nothing, because the settings the
  sabotage harness rendered never produce a flat frame. `halfblock/mono` is now in that list;
  it is the one setting that renders every cell identically.
- `frame-deadlines-accumulate` changed nothing, because the study never runs the player.
  `scripts/probe.py` now contributes the clock, the `.cast` timestamps and the delta encoder
  byte counts to the comparison.

The rule that a sabotage must be proved effectful before its detection counts is what turned
all three into fixes instead of into a conclusion that the checks were fine.

## The page is checked in a browser

`docs/index.html` decodes the clip and composites the glyphs in JavaScript, so it is a
second implementation of the renderer's back end. `scripts/browser_check.mjs` launches a
real browser, pulls one frame off the canvas, and compares it byte for byte against the same
frame composited by Python: 451584 bytes, zero differing. It also checks that a *different*
frame does not match, so the comparison cannot be vacuously true.

It needs `playwright-core` and a matching Chromium. `scripts/verify.sh` fails rather than
skips when it cannot find one, and prints the install line:

```bash
npm install --no-save playwright-core && npx playwright install chromium
```

That check earned its keep immediately. The page threw on its very first animation frame,
because a `requestAnimationFrame` timestamp can be earlier than the `performance.now()`
taken just before the frame was requested, and JavaScript's `%` keeps the sign, so the frame
index went to `-1`. Every unit test passed the whole time.

## What is here

```
asciivid/font.py      the committed 8x16 glyph raster, pinned by sha256
asciivid/scenes.py    five procedural sources, closed-form in (x, y, t)
asciivid/ramps.py     ramps sorted by measured ink coverage, xterm palettes
asciivid/render.py    image to cells, cells to ANSI, delta encoding
asciivid/raster.py    cells back to pixels, the way a terminal draws them
asciivid/fidelity.py  SSIM, structure-only SSIM, per-cell correlation, RMSE
asciivid/player.py    drift-free scheduling, frame dropping, audio origin
asciivid/cast.py      asciinema v2 export and a strict reader
asciivid/study.py     the whole table
scripts/check_fidelity.py   the independent checker, shares no code
scripts/make_font.py        regenerates the raster (needs Pillow; not part of the pipeline)
```

## Assumptions and limits

- **The 16-colour numbers assume xterm's palette.** A terminal with a theme draws those
  sixteen colours differently and the `ansi16` row would move. The 256-colour cube and the
  greys are far more stable across terminals.
- **The glyph raster is one font at one size.** It is DejaVu Sans Mono rendered at 8x
  supersample and box-averaged to 8x16, and the shade blocks ` ░▒▓` are that font's dot
  patterns rather than idealised percentages. A terminal using a different font would give
  different numbers, which is why the raster is committed and its hash is checked.
- **SSIM is computed at pixel scale.** That is the strictest reading. A viewer sitting back
  from the screen integrates over several pixels and would judge the mono renders more
  kindly than 0.08 suggests.
- **`--normalize` recomputes its range per frame**, so a clip whose overall brightness
  changes will pump. Computing it once per clip would trade the pumping for clipping.
- **`out/` is not committed.** It is 22 MB of source frames and ANSI dumps, regenerated by
  `python3 -m asciivid study` in about 35 seconds. `scripts/browser_check.mjs` takes that
  directory as `ASCIIVID_OUT` or as its first argument, and defaults to `./out`.

## Status

`bash scripts/verify.sh`, run in full. Temporary directory names and sub-millisecond
timings differ between runs; nothing else does.

```
$ bash scripts/verify.sh
0. environment
        Python 3.12.3
        numpy 2.5.1
        node v24.13.0
  ok    python and numpy present

1. unit suite
  ok    53 unit tests passed

2. the fidelity study over every source and every render setting
  ok    600 measurements over 5 sources and 60 render settings
        mode       ssim                ssim_cs              cell_r              rgb_rmse            bytes              n       
        truecolor  0.6706              0.6790               0.9997              29.45               61110.46           100     
        ansi256    0.6310              0.6779               0.8797              34.28               31227.52           150     
        ansi16     0.4611              0.6822               0.7113              47.95               8994.35            150     
        mono       0.0786              0.2751               0.8487              115.50              4117.00            200     

3. every reported number re-derived from the emitted ANSI, by code that shares none
          source bars@0.6: 4800 sampled pixels match the formula (worst delta 0)
          source bars@2.4: 4800 sampled pixels match the formula (worst delta 0)
          source glyphs@0.6: 4800 sampled pixels match the formula (worst delta 0)
          source glyphs@2.4: 4800 sampled pixels match the formula (worst delta 0)
          source plasma@0.6: 4800 sampled pixels match the formula (worst delta 0)
          source plasma@2.4: 4800 sampled pixels match the formula (worst delta 0)
          source starfield@0.6: sparse (0.909% bright, mean 6.5)
          source starfield@2.4: sparse (0.831% bright, mean 6.3)
          source tunnel@0.6: 4800 sampled pixels match the formula (worst delta 0)
          source tunnel@2.4: 4800 sampled pixels match the formula (worst delta 0)
          100/600 renders re-derived
          200/600 renders re-derived
          300/600 renders re-derived
          400/600 renders re-derived
          500/600 renders re-derived
          600/600 renders re-derived
          600 renders re-derived from the emitted ANSI
          worst disagreement, as a fraction of the tolerance allowed: ssim 0.249, ssim_cs 0.248, cell_r 0.237, rgb_rmse 0.012
          every number in /tmp/tmp.rzj61XB1od/out/fidelity.json matches an independent re-derivation within 1e-05 relative (floor 2e-06)
  ok    the independent checker agrees with all 600 numbers

4. the checker shares no code with what it checks
        check_fidelity.py imports nothing from asciivid/ and defines its own 15 functions, including read_ppm, read_font, parse_ansi, composite, integral, measure, scene_pixel
  ok    the checker is independent

5. the same input always renders the same bytes
  ok    identical ANSI across runs and across PYTHONHASHSEED (29c9b67dfef951d3)
  ok    the source frames and the measurements are byte-identical between two runs

6. the exports and the player
        /tmp/tmp.rzj61XB1od/clip.cast: 49 events, 48 frames, 4786302 bytes
        asciinema v2, 80x30, 49 events, 2.000s, gap spread 1.00e-06s
  ok    .cast export parses and its clock is even
        played 30 frames at 60fps, 0 dropped, 1 whole and 29 delta, 14120.4 bytes/frame, worst offset 0.101 ms
  ok    the player runs against a real clock
        /tmp/tmp.rzj61XB1od/real.cast: 25 events, 24 frames, 145852 bytes
  ok    a real video file decodes and renders through ffmpeg

7. the page rebuilds identically and runs in a real browser
        docs/index.html: 811625 bytes, 4 clips, 558112 bytes of clip data
            blocks/truecolor/structure             SSIM 0.5804   301015 B packed   211192 B base64+gzip
            ascii67/truecolor/structure            SSIM 0.6398   333472 B packed   235192 B base64+gzip
            ascii10/ansi256/structure+dither       SSIM 0.5251   265867 B packed    83192 B base64+gzip
            ascii67/mono/structure+norm            SSIM 0.0807    92470 B packed    28536 B base64+gzip
  ok    docs/index.html is exactly what the study produces
        playwright-core from ~/Projects/thousand/projects/a11y-sweep/node_modules/playwright-core
        ok    the served page is this project's ("Video, rendered to characters")
        ok    no uncaught errors
        ok    the page makes no external request
        ok    nothing overflows sideways at 1200px
        ok    4 clips decoded in the browser, 40 frames each
        ok    consecutive frames differ, so the clip really animates
        ok    canvas frame 7 of blocks/truecolor/structure is byte-identical to the Python composite (451584 bytes)
        ok    the comparison rejects a frame that is not the one measured
        ok    all 4 colour-mode SSIM figures match a re-aggregation of fidelity.json
        ok    10 inline source and render stills all decoded
        ok    nothing overflows sideways at 390px
  ok    11 passed, 0 failed in a real browser

8. sabotage
        baseline fingerprint a436128f090b3cd6
        luma-weights-swapped       output changed, unit 1, independent checker 1
          FIDELITY MISMATCH: 21 problems
AssertionError: 0.539039788799251 != 0.5760676235776435 within 9 places (0.037027834778392554 difference)
  ok    sabotage "luma-weights-swapped" is caught
        foreground-and-background-swapped output changed, unit 1, independent checker 0
          AssertionError: np.float64(0.9473684210526315) != 1.0 within 0.05 delta (np.float64(0.052631578947368474) diff
  ok    sabotage "foreground-and-background-swapped" is caught
        glyph-choice-ignored       output changed, unit 1, independent checker 0
          AssertionError: np.float64(0.0) not greater than 1.4 : top/bottom ink ratio only 0.00
  ok    sabotage "glyph-choice-ignored" is caught
        cells-transposed           output changed, unit 1, independent checker 0
          AssertionError: np.float64(1.2046706901075832) not greater than 1.4 : top/bottom ink ratio only 1.20
  ok    sabotage "cells-transposed" is caught
  ok    sabotage "aspect-ratio-ignored" is caught (the pipeline itself fails)
        background-colour-never-sent output changed, unit 1, independent checker 1
          FIDELITY MISMATCH: 8 problems
    raise AssertionError(msg)
  ok    sabotage "background-colour-never-sent" is caught
        palette-index-off-by-one   output changed, unit 1, independent checker 1
          FIDELITY MISMATCH: 8 problems
    raise AssertionError(msg)
  ok    sabotage "palette-index-off-by-one" is caught
        raster-ignores-the-glyph   output changed, unit 1, independent checker 1
          FIDELITY MISMATCH: 30 problems
    raise AssertionError(msg)
  ok    sabotage "raster-ignores-the-glyph" is caught
        ssim-always-perfect        output changed, unit 1, independent checker 1
          FIDELITY MISMATCH: 8 problems
AssertionError: 1.0 != 0.5723235258387962 within 9 places (0.4276764741612038 difference)
  ok    sabotage "ssim-always-perfect" is caught
        undefined-correlation-reported-as-zero output changed, unit 1, independent checker 1
          FIDELITY MISMATCH: 2 problems
AssertionError: False is not true : a constant render cannot have a correlation, and must not report one
  ok    sabotage "undefined-correlation-reported-as-zero" is caught
        source-dump-is-the-render  output changed, unit 0, independent checker 1
              raise AssertionError(f"{name}@{t}: {wrong}/{checked} sampled pixels differ from the "
  ok    sabotage "source-dump-is-the-render" is caught
        a-scene-stops-moving       output changed, unit 1, independent checker 1
              raise AssertionError(f"{name}@{t}: {wrong}/{checked} sampled pixels differ from the "
AssertionError: 20346 not less than 18311.4 : tol=12 saved nothing: 20346 vs 20346
  ok    sabotage "a-scene-stops-moving" is caught
        frame-deadlines-accumulate output changed, unit 1, independent checker 0
          AssertionError: 600.258812501087 != 599.9588333333334 within 6 places (0.2999791677536905 difference)
  ok    sabotage "frame-deadlines-accumulate" is caught

9. hygiene
  ok    this directory is its own git repository
  ok    no absolute home paths in tracked files
  ok    no credential-shaped strings in tracked files
        25 tracked files, none contain NUL
  ok    no tracked file is binary to the secret scan
  ok    no tracked file over 1 MB (largest: 811625 docs/index.html)
  ok    no video binary is tracked (the sources are generated from code)
  ok    README has a Status section whose numbers still match this run

32 passed, 0 failed
VERIFY OK
```

## Unfinished

- No audio is generated or synchronised in the committed demo. `--audio` takes a file and
  starts it on the video's clock, and nothing in `verify.sh` exercises a sound card.
- The `.cast` export writes video only. asciinema v2 has no audio track, so "syncs audio" is
  a property of the local player and not of the exported file.
- The ffmpeg path decodes to a fixed size and does not read the source's own aspect ratio,
  so a non-4:3 video is stretched. Scenes and `--cols` are unaffected.
- Frame timing is checked against a simulated clock at ten minutes. It has not been run
  against a real terminal over a real ten minute clip.
- The renderer picks each cell independently. Error diffusion across cells, which would let
  a run of cells share the error a single glyph cannot express, is not implemented.

## Licence

MIT.
