/* Run docs/index.html in a real browser and check what it actually draws.
 *
 * Three claims can only be settled here.
 *
 * One: the page draws the same picture the fidelity numbers describe. The page has its own
 * decoder for the packed clip and its own compositor over the glyph raster, written in
 * JavaScript, and neither has anything to do with the Python that produced the numbers.
 * This pulls one frame off the canvas and compares it to `out/browser_expect.json`, which
 * Python composited. A page whose decoder dropped a delta or swapped a colour channel would
 * look plausible and be wrong, and no unit test would ever see it.
 *
 * Two: the script runs at all. An unbalanced parenthesis leaves a page that renders as
 * static HTML with every table intact and no animation, and the file still exists.
 *
 * Three: the numbers printed in the tables are the numbers in fidelity.json. They are
 * re-aggregated here, in JavaScript, from the raw results.
 *
 * Exits 2 when it cannot run, which is not the same as passing.
 */
import { readFileSync, existsSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const TITLE = 'Video, rendered to characters';
const require = createRequire(join(root, 'package.json'));
const tidy = (s) => String(s).split(homedir()).join('~');

const candidates = [
  process.env.PLAYWRIGHT_CORE,
  join(root, 'node_modules', 'playwright-core'),
  'playwright-core',
  // A sibling checkout, resolved relative to this repository rather than from an absolute
  // home path, so the check behaves the same wherever the repository is cloned.
  resolve(root, '..', 'a11y-sweep', 'node_modules', 'playwright-core'),
  resolve(root, '..', '..', 'a11y-sweep', 'node_modules', 'playwright-core'),
].filter(Boolean);

/* A candidate counts only if it LAUNCHES. A copy can import cleanly and still be unusable
 * because its pinned browser build is not in the local cache, and playwright-core is
 * CommonJS so its exports arrive either as named bindings or under `.default`. Accepting a
 * candidate on anything less than a running browser turns this check into a skip. */
let browser = null, resolvedFrom = null;
const attempts = [];
for (const c of candidates) {
  let mod;
  try {
    mod = await import(c.startsWith('/') ? `file://${require.resolve(c)}` : c);
  } catch { attempts.push(`${c}: not importable`); continue; }
  const chromium = mod.chromium ?? mod.default?.chromium ?? null;
  if (!chromium) { attempts.push(`${c}: no chromium export`); continue; }
  try { browser = await chromium.launch(); } catch (e) {
    attempts.push(`${c}: ${String(e).split('\n')[0].slice(0, 90)}`); continue;
  }
  resolvedFrom = c;
  break;
}
if (!browser) {
  console.error('no usable playwright-core with a matching browser build. Tried:');
  for (const a of attempts) console.error(`    ${tidy(a)}`);
  console.error('Install one with:');
  console.error('    npm install --no-save playwright-core && npx playwright install chromium');
  console.error('Without it, nothing checks that the page draws the frames it claims to.');
  process.exit(2);
}

const pagePath = join(root, 'docs', 'index.html');
const expectPath = join(root, 'out', 'browser_expect.json');
const reportPath = join(root, 'out', 'fidelity.json');
for (const p of [pagePath, expectPath, reportPath]) {
  if (!existsSync(p)) {
    console.error(`missing ${tidy(p)}. Run the study and scripts/build_docs.py first; `
      + 'this check has nothing to compare against.');
    await browser.close();
    process.exit(2);
  }
}
const html = readFileSync(pagePath, 'utf8');
const expect = JSON.parse(readFileSync(expectPath, 'utf8'));
const report = JSON.parse(readFileSync(reportPath, 'utf8'));

const server = createServer((q, r) => {
  r.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  r.end(html);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/`;
console.log(`  playwright-core from ${tidy(resolvedFrom)}`);

let pass = 0, fail = 0;
const ok = (m) => { console.log(`  ok    ${m}`); pass++; };
const bad = (m) => { console.log(`  FAIL  ${m}`); fail++; };

try {
  for (const width of [1200, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], requests = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('request', (r) => { if (r.url() !== url) requests.push(r.url()); });
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForSelector('body[data-page-ready="1"]', { timeout: 30000 })
      .catch(async () => {
        const err = await page.evaluate(() => document.body.dataset.pageError || null);
        throw new Error(err ? `the page script threw: ${err}` : 'the page never became ready');
      });

    // Identity first. The browser is shared between agents on this box, so a measurement
    // taken without checking which page it came from can silently describe another project.
    if ((await page.title()) === TITLE) {
      if (width === 1200) ok(`the served page is this project's ("${TITLE}")`);
    } else { bad(`wrong page at ${width}px: ${await page.title()}`); break; }

    if (!errors.length) { if (width === 1200) ok('no uncaught errors'); }
    else bad(`uncaught error: ${errors[0].slice(0, 160)}`);

    if (!requests.length) { if (width === 1200) ok('the page makes no external request'); }
    else bad(`unexpected request: ${requests[0]}`);

    const over = await page.evaluate(() => {
      const root = document.documentElement;
      const bad = [];
      for (const el of document.querySelectorAll('*')) {
        let p = el.parentElement, scrolls = false;
        while (p) {
          const ov = getComputedStyle(p).overflowX;
          if (ov === 'auto' || ov === 'scroll') { scrolls = true; break; }
          p = p.parentElement;
        }
        if (scrolls) continue;
        const r = el.getBoundingClientRect();
        if (r.right > root.clientWidth + 1 || r.left < -1) {
          bad.push(`${el.tagName.toLowerCase()}.${el.className || '-'} right=${
            Math.round(r.right)} vs ${root.clientWidth}`);
        }
      }
      return { bad: bad.slice(0, 3), scrollWidth: root.scrollWidth,
               clientWidth: root.clientWidth };
    });
    if (!over.bad.length && over.scrollWidth <= over.clientWidth + 1) {
      ok(`nothing overflows sideways at ${width}px`);
    } else {
      bad(`sideways overflow at ${width}px: ${over.bad.join('; ')} `
        + `(scrollWidth ${over.scrollWidth} vs ${over.clientWidth})`);
    }

    if (width !== 1200) { await page.close(); continue; }

    /* The canvas has to carry a real picture, and a different one per frame. */
    const meta = await page.evaluate(() => window.__asciivid.clips);
    if (meta.length >= 3 && meta.every((m) => m.frames > 1)) {
      ok(`${meta.length} clips decoded in the browser, ${meta[0].frames} frames each`);
    } else bad(`the page decoded ${JSON.stringify(meta)}`);

    const shot = async (clip, frame) => page.evaluate(([c, f]) => {
      window.__asciivid.frameAt(c, f);
      const cv = document.getElementById('screen');
      const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
      let bin = '';
      for (let i = 0; i < d.length; i += 4) {
        bin += String.fromCharCode(d[i], d[i + 1], d[i + 2]);
      }
      return { w: cv.width, h: cv.height, rgb: btoa(bin) };
    }, [clip, frame]);

    const a = await shot(0, expect.frame);
    const b = await shot(0, (expect.frame + 11) % meta[0].frames);
    if (a.rgb !== b.rgb) ok('consecutive frames differ, so the clip really animates');
    else bad('two different frames rendered identical pixels');

    if (a.w === expect.width && a.h === expect.height) {
      const got = Buffer.from(a.rgb, 'base64');
      const want = Buffer.from(expect.rgb, 'base64');
      let worst = 0, differing = 0;
      for (let i = 0; i < want.length; i++) {
        const d = Math.abs(got[i] - want[i]);
        if (d) { differing++; if (d > worst) worst = d; }
      }
      if (worst === 0) {
        ok(`canvas frame ${expect.frame} of ${expect.spec} is byte-identical to the `
          + `Python composite (${want.length} bytes)`);
      } else {
        bad(`the browser drew frame ${expect.frame} differently: ${differing} bytes differ, `
          + `worst by ${worst}`);
      }
      /* Negative control for the comparison itself: a different frame must NOT match. */
      const other = Buffer.from(b.rgb, 'base64');
      let same = true;
      for (let i = 0; i < want.length && same; i++) if (other[i] !== want[i]) same = false;
      if (!same) ok('the comparison rejects a frame that is not the one measured');
      else bad('a different frame compared equal, so the comparison proves nothing');
    } else {
      bad(`canvas is ${a.w}x${a.h}, the reference is ${expect.width}x${expect.height}`);
    }

    /* The tables must show the numbers in fidelity.json, re-aggregated here. */
    const shown = await page.evaluate(() => {
      const t = document.querySelectorAll('table')[0];
      const out = {};
      for (const tr of t.querySelectorAll('tbody tr')) {
        const td = [...tr.querySelectorAll('td')].map((x) => x.textContent.trim());
        out[td[0]] = Number(td[1]);
      }
      return out;
    });
    const byMode = {};
    for (const r of report.results) (byMode[r.mode] ||= []).push(r.ssim);
    let mismatched = [];
    for (const [mode, vals] of Object.entries(byMode)) {
      const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
      if (!(mode in shown)) { mismatched.push(`${mode} is missing from the page`); continue; }
      if (Math.abs(shown[mode] - mean) > 5e-5) {
        mismatched.push(`${mode}: page ${shown[mode]}, fidelity.json ${mean.toFixed(6)}`);
      }
    }
    if (!mismatched.length && Object.keys(shown).length === Object.keys(byMode).length) {
      ok(`all ${Object.keys(shown).length} colour-mode SSIM figures match a re-aggregation `
        + 'of fidelity.json');
    } else bad(`table disagrees with fidelity.json: ${mismatched.join('; ')}`);

    const imgs = await page.evaluate(() => [...document.querySelectorAll('img')].map(
      (i) => ({ ok: i.complete && i.naturalWidth > 0, alt: i.alt })));
    if (imgs.length >= 8 && imgs.every((i) => i.ok)) {
      ok(`${imgs.length} inline source and render stills all decoded`);
    } else bad(`a still failed to decode: ${JSON.stringify(imgs.filter((i) => !i.ok))}`);

    await page.close();
  }
} catch (e) {
  bad(`the check threw: ${String(e).split('\n')[0]}`);
} finally {
  await browser.close();
  server.close();
}

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
