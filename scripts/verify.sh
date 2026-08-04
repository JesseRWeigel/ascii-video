#!/usr/bin/env bash
# Verification for ascii-video.
#
# The claim this project makes is a fidelity claim, so most of this script exists to make
# that claim falsifiable rather than to check that the code runs.
#
# There are two independent ways for a fidelity number to be wrong.
#
#   The MEASUREMENT can be wrong. SSIM that always returns 1, a correlation that reports 0
#   where it should report "undefined", a rasteriser that ignores the glyph. The unit suite
#   covers these, each with a negative control.
#
#   The THING MEASURED can be wrong. The study measures the frame it holds in memory, and a
#   bug in ANSI emission leaves that measurement perfect while the bytes a terminal receives
#   say something else. Nothing that reads the cell array can see it. So
#   scripts/check_fidelity.py re-derives every number from the emitted ANSI, with its own
#   escape parser, its own font reader, its own compositor and its own SSIM, and it
#   re-derives the source frames from its own pure-Python copy of the scene maths so
#   out/source/ cannot quietly become a copy of the render.
#
# Layers:
#   1  unit suite, every test paired with a negative control
#   2  the full study: 5 sources, 60 render settings, 600 measurements
#   3  every one of those 600 numbers re-derived by an independent checker
#   4  that checker proved independent by walking its import graph with ast
#   5  determinism: byte-identical across runs and across PYTHONHASHSEED
#   6  the .cast export, the terminal player, and the real-video path through ffmpeg
#   7  the page rebuilt identically and run in a real browser, where one canvas frame is
#      compared byte for byte against the Python composite
#   8  sabotages, each proved to have applied AND to have changed the output
#   9  hygiene, including the README's own numbers
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
PY=python3
export PYTHONDONTWRITEBYTECODE=1

pass=0; fail=0
ok()  { printf '  ok    %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Every reference to $HOME in printed output is folded back to a tilde. The escape on the
# tilde is required: bash tilde-expands the replacement otherwise and the substitution
# silently does nothing.
scrub() { sed "s|$HOME|\~|g"; }

echo "0. environment"
$PY --version 2>&1 | sed 's/^/        /'
$PY -c 'import numpy; print("numpy", numpy.__version__)' 2>&1 | sed 's/^/        /'
node --version 2>/dev/null | sed 's/^/        node /' || echo "        node missing"
if $PY -c 'import numpy' 2>/dev/null; then ok "python and numpy present"
else bad "numpy is required: pip install numpy"; fi

echo
echo "1. unit suite"
if $PY -m unittest discover -s tests -v >"$TMP/unit.log" 2>&1; then
  N=$(grep -oE 'Ran [0-9]+ tests' "$TMP/unit.log" | grep -oE '[0-9]+')
  ok "$N unit tests passed"
  # The count is a claim like any other, and the README quotes it.
  if grep -q "Ran $N tests" "$TMP/unit.log"; then :; fi
  echo "$N" > "$TMP/testcount"
else
  bad "unit suite"; grep -E 'FAIL:|ERROR:|AssertionError' "$TMP/unit.log" | head -12 \
    | sed 's/^/        /'
  echo 0 > "$TMP/testcount"
fi

echo
echo "2. the fidelity study over every source and every render setting"
if $PY -m asciivid study --out "$TMP/out" >"$TMP/study.log" 2>"$TMP/study.err"; then
  COUNT=$($PY -c "import json;print(len(json.load(open('$TMP/out/fidelity.json'))['results']))")
  SPECS=$($PY -c "import json;print(len({r['spec'] for r in json.load(open('$TMP/out/fidelity.json'))['results']}))")
  SRC=$($PY -c "import json;print(len(json.load(open('$TMP/out/fidelity.json'))['source']['scenes']))")
  ok "$COUNT measurements over $SRC sources and $SPECS render settings"
  $PY -m asciivid table --out "$TMP/out" --by mode | sed 's/^/        /'
else
  bad "the study failed"; tail -5 "$TMP/study.err" | sed 's/^/        /'
fi

echo
echo "3. every reported number re-derived from the emitted ANSI, by code that shares none"
if [ -f "$TMP/out/fidelity.json" ]; then
  if $PY scripts/check_fidelity.py --out "$TMP/out" >"$TMP/check.log" 2>&1; then
    grep -E 'source |re-derived|worst|matches' "$TMP/check.log" | sed 's/^/        /'
    ok "the independent checker agrees with all $COUNT numbers"
  else
    rc=$?
    bad "the independent checker disagrees (exit $rc)"
    tail -12 "$TMP/check.log" | sed 's/^/        /'
  fi
else
  bad "no study output, so nothing was re-derived"
fi

echo
echo "4. the checker shares no code with what it checks"
if $PY - <<'PY' >"$TMP/imports.log" 2>&1; then
import ast, pathlib, sys
name = "scripts/check_fidelity.py"
tree = ast.parse(pathlib.Path(name).read_text())
hits = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        hits += [a.name for a in node.names if a.name.split(".")[0] == "asciivid"]
    elif isinstance(node, ast.ImportFrom):
        if (node.module or "").split(".")[0] == "asciivid":
            hits.append(node.module)
# A sys.path insert plus a late import would defeat the check above, so look for that too.
src = pathlib.Path(name).read_text()
for needle in ("importlib", "__import__", "exec("):
    if needle in src:
        hits.append(f"dynamic import via {needle}")
if hits:
    sys.exit(f"the checker reaches into asciivid/: {hits}")
funcs = sorted(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
print(f"check_fidelity.py imports nothing from asciivid/ and defines its own "
      f"{len(funcs)} functions, including read_ppm, read_font, parse_ansi, composite, "
      f"integral, measure, scene_pixel")
PY
  sed 's/^/        /' "$TMP/imports.log"; ok "the checker is independent"
else
  bad "$(cat "$TMP/imports.log")"
fi

echo
echo "5. the same input always renders the same bytes"
D1="$TMP/d1.ansi"; D2="$TMP/d2.ansi"
$PY -m asciivid render --scene glyphs --ramp ascii67 --mode ansi256 --dither \
  --seconds 1 --out "$D1" >/dev/null 2>&1
PYTHONHASHSEED=987654 $PY -m asciivid render --scene glyphs --ramp ascii67 --mode ansi256 \
  --dither --seconds 1 --out "$D2" >/dev/null 2>&1
if [ -s "$D1" ] && cmp -s "$D1" "$D2"; then
  ok "identical ANSI across runs and across PYTHONHASHSEED ($(sha256sum "$D1" | cut -c1-16))"
else
  # A mapping keyed on Python's hash() looks perfectly deterministic inside one process.
  bad "the render is not reproducible"; sha256sum "$D1" "$D2" 2>&1 | sed 's/^/        /'
fi
$PY -m asciivid study --out "$TMP/det1" --scenes bars --times 0.6 \
  --specs "blocks/truecolor/structure" >/dev/null 2>&1
$PY -m asciivid study --out "$TMP/det2" --scenes bars --times 0.6 \
  --specs "blocks/truecolor/structure" >/dev/null 2>&1
if cmp -s "$TMP/det1/fidelity.json" "$TMP/det2/fidelity.json" &&
   cmp -s "$TMP/det1/source/bars@0.6.ppm" "$TMP/det2/source/bars@0.6.ppm"; then
  ok "the source frames and the measurements are byte-identical between two runs"
else
  bad "the study is not reproducible"
fi

echo
echo "6. the exports and the player"
if $PY -m asciivid render --scene tunnel --ramp blocks --mode truecolor --seconds 2 \
     --out "$TMP/clip.cast" >"$TMP/cast.log" 2>&1; then
  sed 's/^/        /' "$TMP/cast.log"
  if $PY - "$TMP/clip.cast" <<'PY' >"$TMP/castcheck.log" 2>&1; then
import json, sys
sys.path.insert(0, ".")
from asciivid import cast
with open(sys.argv[1]) as fh:
    head, events = cast.read(fh)
gaps = list(cast.durations(events))
assert head["version"] == 2, head
assert len(events) >= 48, len(events)
spread = max(gaps) - min(gaps)
assert spread < 1e-5, f"frame gaps vary by {spread}"
assert abs(events[-1][0] - len(events[:-1]) / 24.0) < 1e-4, events[-1][0]
print(f"asciinema v2, {head['width']}x{head['height']}, {len(events)} events, "
      f"{events[-1][0]:.3f}s, gap spread {spread:.2e}s")
PY
    sed 's/^/        /' "$TMP/castcheck.log"; ok ".cast export parses and its clock is even"
  else
    bad "the .cast export is malformed"; tail -3 "$TMP/castcheck.log" | sed 's/^/        /'
  fi
else
  bad "the .cast export failed"; tail -3 "$TMP/cast.log" | sed 's/^/        /'
fi

if $PY - <<'PY' >"$TMP/play.log" 2>&1; then
import io, sys
sys.path.insert(0, ".")
from asciivid import font, player, render, scenes
f = font.load()
spec = render.Spec("blocks", "truecolor", "structure")
frames = [render.render(scenes.frame("tunnel", i / 60.0, 320, 240), f, spec, 40)
          for i in range(30)]
buf = io.StringIO()
st = player.play(frames, 60.0, out=buf, delta=True, delta_tol=8, quiet=True)
assert st["written"] + st["dropped"] == 30, st
assert st["worst_offset_ms"] < 40.0, st
assert buf.tell() > 1000, st
print(f"played {st['written']} frames at 60fps, {st['dropped']} dropped, "
      f"{st['full_frames']} whole and {st['delta_frames']} delta, "
      f"{st['bytes_per_frame']} bytes/frame, worst offset {st['worst_offset_ms']} ms")
PY
  sed 's/^/        /' "$TMP/play.log"; ok "the player runs against a real clock"
else
  bad "the player failed"; tail -4 "$TMP/play.log" | sed 's/^/        /'
fi

# The real-video path. It is optional in the sense that the committed demo does not need it,
# and it is NOT optional in the sense of being skippable: a check that cannot run says so.
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -v error -f lavfi -i "testsrc=size=320x240:rate=12:duration=2" \
    -pix_fmt yuv420p "$TMP/real.mp4" 2>"$TMP/ff.log"
  if [ -s "$TMP/real.mp4" ] && $PY -m asciivid render --input "$TMP/real.mp4" --fps 12 \
       --ramp blocks --mode truecolor --cols 40 --out "$TMP/real.cast" \
       >"$TMP/real.log" 2>&1; then
    sed 's/^/        /' "$TMP/real.log"
    ok "a real video file decodes and renders through ffmpeg"
  else
    bad "the ffmpeg path failed"; tail -4 "$TMP/real.log" "$TMP/ff.log" 2>/dev/null \
      | sed 's/^/        /'
  fi
else
  bad "ffmpeg is not installed, so the real-video path is UNCHECKED. Install it with
        apt-get install ffmpeg. Everything else here uses the procedural sources, which
        need nothing, so this is the only part of the pipeline left unexercised."
fi

echo
echo "7. the page rebuilds identically and runs in a real browser"
cp docs/index.html "$TMP/page.html"
if $PY scripts/build_docs.py --out "$TMP/out" >"$TMP/docs.log" 2>&1; then
  sed 's/^/        /' "$TMP/docs.log"
  if cmp -s docs/index.html "$TMP/page.html"; then
    ok "docs/index.html is exactly what the study produces"
  else
    bad "docs/index.html differs from a fresh build; the committed page is stale"
    cp "$TMP/page.html" docs/index.html
  fi
else
  bad "the page could not be built"; tail -5 "$TMP/docs.log" | sed 's/^/        /'
fi

if command -v node >/dev/null 2>&1; then
  if ASCIIVID_OUT="$TMP/out" node scripts/browser_check.mjs >"$TMP/browser.log" 2>&1; then
    grep -E '^  (ok|FAIL|playwright)' "$TMP/browser.log" | scrub | sed 's/^/      /'
    ok "$(tail -1 "$TMP/browser.log") in a real browser"
  else
    rc=$?
    bad "the browser check failed (exit $rc); the page's own decoder is unverified"
    tail -12 "$TMP/browser.log" | scrub | sed 's/^/        /'
  fi
else
  bad "node is not installed, so nothing checks that the page draws the frames it claims.
        Install node, then: npm install --no-save playwright-core && npx playwright install chromium"
fi

echo
echo "8. sabotage"
# Each attack is proved to have APPLIED and proved to have CHANGED the output before its
# detection counts for anything. An attack that changes nothing proves nothing, and
# concluding "the verify has a gap" from one is how a correct check gets weakened.
MINI_SCENES="plasma,glyphs"
MINI_TIMES="0.6"
# halfblock/mono is in the list because it is the one setting that renders every cell
# identically, which is the only way the "undefined correlation" case ever arises. Without it
# a sabotage of that branch changes nothing and correctly proves nothing.
MINI_SPECS="blocks/truecolor/structure,ascii67/mono/structure+norm,ascii10/ansi256/structure+dither,halfblock/mono/structure"
mini() {
  ( cd "$1" && $PY -m asciivid study --out "$1/mini" --scenes "$MINI_SCENES" \
      --times "$MINI_TIMES" --specs "$MINI_SPECS" \
      && $PY scripts/probe.py > "$1/mini/probe.txt" ) >/dev/null 2>&1
}
# The study never runs the player and never writes a .cast, so probe.txt is folded in here.
# Otherwise a sabotage of the frame clock produces byte-identical study output and gets
# reported as proving nothing, which would be a hole in the comparison, not in the checks.
fingerprint() {
  ( cd "$1" && find mini -type f \( -name '*.ansi' -o -name '*.ppm' -o -name '*.json' \
      -o -name 'probe.txt' \) -print0 | sort -z | xargs -0 sha256sum ) 2>/dev/null \
      | sha256sum | cut -d' ' -f1
}

BASE="$TMP/base"
mkdir -p "$BASE"
tar -cf - --exclude=.git --exclude=__pycache__ --exclude=node_modules --exclude=out \
    -C "$ROOT" . | tar -xf - -C "$BASE"
if mini "$BASE"; then
  BASE_FP="$(fingerprint "$BASE")"
  echo "        baseline fingerprint ${BASE_FP:0:16}"
else
  bad "the baseline mini-study failed, so no sabotage below proves anything"
  BASE_FP=""
fi

attack() {
  local name="$1" file="$2" old="$3" new="$4"
  local dir="$TMP/attack-$name"
  [ -z "$BASE_FP" ] && { bad "sabotage \"$name\" skipped, no baseline"; return; }
  rm -rf "$dir"; mkdir -p "$dir"
  tar -cf - --exclude=.git --exclude=__pycache__ --exclude=node_modules --exclude=out \
      --exclude=mini -C "$BASE" . | tar -xf - -C "$dir"
  rm -rf "$dir/mini"
  if ! $PY - "$dir/$file" "$old" "$new" <<'PY'
import pathlib, sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
t = p.read_text()
if old not in t:
    sys.exit(f"SABOTAGE DID NOT APPLY: {old!r} is absent from {path}")
if old == new:
    sys.exit("SABOTAGE IS A NO-OP: old and new are the same text")
p.write_text(t.replace(old, new, 1))
PY
  then
    bad "sabotage \"$name\" did not apply, so it proves nothing"; return
  fi

  if ! mini "$dir"; then
    ok "sabotage \"$name\" is caught (the pipeline itself fails)"; return
  fi
  local fp; fp="$(fingerprint "$dir")"
  if [ "$fp" = "$BASE_FP" ]; then
    bad "sabotage \"$name\" produced byte-identical output, so it proves nothing"
    return
  fi

  local urc crc
  ( cd "$dir" && $PY -m unittest discover -s tests ) >"$TMP/$name-unit.log" 2>&1; urc=$?
  ( cd "$dir" && $PY scripts/check_fidelity.py --out "$dir/mini" ) \
      >"$TMP/$name-chk.log" 2>&1; crc=$?
  printf '        %-26s output changed, unit %s, independent checker %s\n' "$name" "$urc" "$crc"
  if [ "$urc" -ne 0 ] || [ "$crc" -ne 0 ]; then
    local why
    why=$(grep -m1 -hE 'FIDELITY MISMATCH|AssertionError|differ from the scene' \
          "$TMP/$name-chk.log" "$TMP/$name-unit.log" 2>/dev/null || true)
    [ -n "$why" ] && printf '          %s\n' "$(printf '%s' "$why" | cut -c1-110)"
    ok "sabotage \"$name\" is caught"
  else
    bad "sabotage \"$name\" changed the output and nothing noticed"
  fi
}

# Rendering faults: the picture changes and the measurement should follow it down.
attack "luma-weights-swapped" "asciivid/render.py" \
  "LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)" \
  "LUMA = np.array([0.7152, 0.0722, 0.2126], dtype=np.float64)"
attack "foreground-and-background-swapped" "asciivid/render.py" \
  "    fg = fg.reshape(rows, cols, 3)
    bg = bg.reshape(rows, cols, 3)" \
  "    fg, bg = bg.reshape(rows, cols, 3), fg.reshape(rows, cols, 3)"
attack "glyph-choice-ignored" "asciivid/render.py" \
  "    pick = cost.argmin(axis=1)" \
  "    pick = np.zeros(cost.shape[0], dtype=np.int64)"
attack "cells-transposed" "asciivid/render.py" \
  "               .transpose(0, 2, 1, 3, 4)" \
  "               .transpose(0, 2, 3, 1, 4)"
attack "aspect-ratio-ignored" "asciivid/render.py" \
  "    rows = max(1, int(round(height / width * cols / aspect)))" \
  "    rows = max(1, int(round(height / width * cols)))"

# Emission faults: the in-memory frame is perfect and the wire is wrong. Only a checker that
# reads the emitted bytes can see these, which is the reason that checker exists.
attack "background-colour-never-sent" "asciivid/render.py" \
  'return f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m"' \
  'return f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m"'
attack "palette-index-off-by-one" "asciivid/render.py" \
  '        return f"\x1b[38;5;{fi};48;5;{bi}m"' \
  '        return f"\x1b[38;5;{fi + 1};48;5;{bi}m"'

# Rasteriser faults: the glyph stops mattering, so every ramp scores the same.
attack "raster-ignores-the-glyph" "asciivid/raster.py" \
  "        out[m] = bg + a * (fg - bg)" \
  "        out[m] = bg + 0.5 * (fg - bg)"

# Measurement faults: the picture is fine and the number is a lie.
attack "ssim-always-perfect" "asciivid/fidelity.py" \
  "    return float((lum * cs).mean()), float(cs.mean())" \
  "    return 1.0, float(cs.mean())"
attack "undefined-correlation-reported-as-zero" "asciivid/fidelity.py" \
  '        return float("nan")' \
  "        return 0.0"

# Source faults: the thing being compared against is not the source at all.
attack "source-dump-is-the-render" "asciivid/study.py" \
  "            if dump:
                scenes.write_ppm(outdir / \"source\" / f\"{name}@{t}.ppm\", img)" \
  "            if dump:
                import numpy as _np
                scenes.write_ppm(outdir / \"source\" / f\"{name}@{t}.ppm\",
                                 _np.clip(img.astype(int) + 8, 0, 255).astype(_np.uint8))"
attack "a-scene-stops-moving" "asciivid/scenes.py" \
  "    img = fn(float(t), int(w), int(h))" \
  "    img = fn(0.0, int(w), int(h))"

# Timing faults: playback drifts away from the sound.
attack "frame-deadlines-accumulate" "asciivid/player.py" \
  "        due = t0 + i * period          # from i, never \`due += period\`" \
  "        due = t0 + i * period if i == 0 else due + period * 1.0005"

echo
echo "9. hygiene"
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  TOP="$(git -C "$ROOT" rev-parse --show-toplevel)"
  if [ "$TOP" = "$ROOT" ]; then ok "this directory is its own git repository"
  else bad "git root is $TOP, not $ROOT; publishing from here would push the parent repo"; fi

  hits=$(git -C "$ROOT" grep -In -e "/home/$(id -un)" -e "/Users/$(id -un)" -- . 2>/dev/null \
         | grep -v '^scripts/verify.sh' || true)
  if [ -z "$hits" ]; then ok "no absolute home paths in tracked files"
  else bad "absolute home paths found"; printf '%s\n' "$hits" | head -5 | sed 's/^/        /'; fi

  # A tilde path defeats the check above and still names every directory above this one.
  # This repository sits inside a private tree, and the names of that tree's directories are
  # not this repository's information to publish. Found in a pasted verify transcript, where
  # a scrubbed home-anchored path had replaced an absolute one and read as clean.
  above=""
  d="$(dirname "$ROOT")"
  while [ "$d" != "/" ] && [ "$d" != "$HOME" ]; do
    b="$(basename "$d")"
    case "$b" in
      projects|Projects|src|code|repos|work|dev|home|users|tmp|opt|var) ;;
      ?|??|???) ;;
      *) above="$above $b";;
    esac
    d="$(dirname "$d")"
  done
  if [ -z "$above" ]; then
    ok "no enclosing directory name is distinctive enough to leak"
  else
    named=""
    for b in $above; do
      if git -C "$ROOT" grep -qIn -e "$b" -- . ":!scripts/verify.sh" 2>/dev/null; then
        named="$named $b"
      fi
    done
    # The count, never the names. Printing them here would put them in the README, which is
    # the exact leak this check exists to prevent.
    n_above=$(printf '%s\n' $above | wc -l)
    if [ -z "$named" ]; then
      ok "no tracked file names any of the $n_above enclosing directories"
    else
      bad "tracked files name $(printf '%s\n' $named | wc -l) enclosing private directories"
      for b in $named; do
        git -C "$ROOT" grep -Ilne "$b" -- . ":!scripts/verify.sh" | head -3 | sed 's/^/        in /'
      done
    fi
  fi

  keys=$(git -C "$ROOT" grep -n -E 'sk-[A-Za-z0-9]{20}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10}' \
         -- . 2>/dev/null || true)
  if [ -z "$keys" ]; then ok "no credential-shaped strings in tracked files"
  else bad "possible credential"; printf '%s\n' "$keys" | head -3 | sed 's/^/        /'; fi

  # A file containing a NUL is binary to git and grep, so the two scans above skip it
  # entirely and report the same "clean" as a scan that actually read it.
  if $PY - "$ROOT" <<'PY' >"$TMP/nul.log" 2>&1; then
import os, subprocess, sys
root = sys.argv[1]
files = [f.decode() for f in subprocess.run(["git", "-C", root, "ls-files", "-z"],
         capture_output=True, check=True).stdout.split(b"\0") if f]
bad = [f for f in files if os.path.isfile(os.path.join(root, f))
       and b"\0" in open(os.path.join(root, f), "rb").read()]
if bad:
    sys.exit("files containing NUL, invisible to the secret scan: " + ", ".join(bad))
print(f"{len(files)} tracked files, none contain NUL")
PY
    sed 's/^/        /' "$TMP/nul.log"; ok "no tracked file is binary to the secret scan"
  else bad "$(cat "$TMP/nul.log")"; fi

  big=$(git -C "$ROOT" ls-files | while read -r f; do
          [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 1000000 ] && \
            echo "$f ($(stat -c%s "$f") bytes)"; done || true)
  if [ -z "$big" ]; then
    ok "no tracked file over 1 MB (largest: $(git -C "$ROOT" ls-files | while read -r f; do
          [ -f "$f" ] && printf '%s %s\n' "$(stat -c%s "$f")" "$f"; done | sort -rn | head -1))"
  else bad "large tracked files"; printf '%s\n' "$big" | head -3 | sed 's/^/        /'; fi

  if [ -z "$(git -C "$ROOT" ls-files | grep -E '\.(mp4|mkv|webm|mov|avi|gif)$' || true)" ]; then
    ok "no video binary is tracked (the sources are generated from code)"
  else bad "a video file is tracked"; fi
else
  bad "not a git repository"
fi

# The README is a claim like any other, and twelve verify scripts in this workspace pass
# without ever looking at theirs.
TESTS="$(cat "$TMP/testcount")"
if [ -f README.md ]; then
  miss=""
  grep -q '## Status' README.md || miss="$miss no-status-section"
  grep -q 'VERIFY OK' README.md || miss="$miss no-pasted-success-line"
  grep -qi 'TODO' README.md && miss="$miss unreplaced-TODO"
  grep -q "$TESTS unit tests passed" README.md || miss="$miss test-count-stale($TESTS)"
  if [ -n "$COUNT" ] && ! grep -q "$COUNT measurements" README.md; then
    miss="$miss measurement-count-stale($COUNT)"
  fi
  if [ -z "$miss" ]; then
    ok "README has a Status section whose numbers still match this run"
  else
    bad "README:$miss"
  fi
else
  bad "no README.md"
fi

echo
printf '%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ] || { echo "VERIFY FAILED"; exit 1; }
echo "VERIFY OK"
