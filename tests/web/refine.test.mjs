// Pariteit van docs/refine.js met alphamaker/refine.py, core.py, scipy en PIL.
// Fixtures: .venv/bin/python -I bench/make_web_fixtures.py   Draai: node --test tests/web/refine.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as R from '../../docs/refine.js';

const fx = JSON.parse(readFileSync(new URL('fixtures.json', import.meta.url), 'utf8'));
const TOL = 1e-4;
const bytes = s => new Uint8Array(Buffer.from(s, 'base64'));
const floats = s => new Float32Array(bytes(s).buffer);
const rgbaOf = rgb => {
  const n = rgb.length / 3, o = new Uint8ClampedArray(4 * n).fill(255);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) o[4 * i + c] = rgb[3 * i + c];
  return o;
};
function near(got, want, tol, what) {
  assert.equal(got.length, want.length, `${what}: lengte`);
  for (let i = 0; i < got.length; i++)
    if (!(Math.abs(got[i] - want[i]) <= tol)) assert.fail(`${what}: index ${i}: ${got[i]} i.p.v. ${want[i]}`);
}

// zelfde nep-modellen als FakeSeg/FakeMatte in make_web_fixtures.py
const fakeSeg = (S, amp, bias) => x => {
  const n = S * S, data = new Float32Array(n);
  for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) {
    const p = i * S + j;
    data[p] = amp * Math.sin(x[p] + 2 * x[n + p] + 3 * x[2 * n + p] + 0.3 * i + 0.1 * j) + bias;
  }
  return { data, dims: [1, 1, S, S] };
};
const fakeMatte = (x, hp, wp) => {
  const n = hp * wp, a = new Float32Array(n);
  for (let i = 0; i < hp; i++) for (let j = 0; j < wp; j++) {
    const p = i * wp + j;
    a[p] = 0.5 + 0.25 * Math.sin(x[p] + 2 * x[n + p] + 3 * x[2 * n + p] + 4 * x[3 * n + p] + 0.1 * i + 0.2 * j + 0.01 * hp + 0.03 * wp);
  }
  return a;
};
const specOf = c => ({ size: c.size, norm: { mean: c.mean, std: c.std }, post: c.post });

test('boxFilterReflect = scipy uniform_filter(mode=reflect), oneven en even k', () => {
  for (const c of fx.box) near(R.boxFilterReflect(floats(c.x), c.w, c.h, c.k), floats(c.out), TOL, `k=${c.k} ${c.w}x${c.h}`);
});

test('dilate/erode = scipy binary_dilation/binary_erosion (kruis, iterations=n)', () => {
  const { w, h } = fx.morph, m = bytes(fx.morph.mask);
  for (const c of fx.morph.cases) {
    assert.deepEqual(R.dilate(m, w, h, c.n), bytes(c.dil), `dilatie n=${c.n}`);
    assert.deepEqual(R.erode(m, w, h, c.n), bytes(c.ero), `erosie n=${c.n}`);
  }
});

test('trimap = refine.trimap (ook precies op de drempels)', () => {
  for (const t of fx.trimap) for (const c of t.cases)
    assert.deepEqual(R.trimap(floats(t.alpha), t.w, t.h, c.band), floats(c.tri), `${t.w}x${t.h} band ${c.band}`);
});

test('tileStarts/ramp = refine._starts/_ramp', () => {
  for (const c of fx.starts) assert.deepEqual(R.tileStarts(c.n, c.tile, c.overlap), c.out, `n=${c.n} tegel ${c.tile}`);
  for (const c of fx.ramp) near(R.ramp(c.n, c.overlap), c.out, 1e-12, `ramp ${c.n}`);
});

test('vitmatte: tegels samenvoegen = refine.vitmatte (nep-sessie)', async () => {
  const v = fx.vitmatte, rgba = rgbaOf(bytes(v.rgb)), tri = floats(v.tri);
  for (const c of v.cases) {
    let calls = 0;
    const got = await R.vitmatte(rgba, tri, v.w, v.h, (...a) => (calls++, fakeMatte(...a)), c.tile, c.overlap);
    near(got, floats(c.out), TOL, `tegel ${c.tile}`);
    if (c.tile === 24) assert.equal(calls, 6, 'tegels zonder onzekere pixels worden overgeslagen');
  }
});

test('foregroundBlurFusion = refine.foreground (blur fusion)', () => {
  for (const c of fx.foreground) {
    const n = c.w * c.h, got = new Float32Array(3 * n);
    R.foregroundBlurFusion(rgbaOf(bytes(c.rgb)), floats(c.alpha), c.w, c.h, (ch, F) => got.set(F, ch * n), c.ks);
    near(got, floats(c.out), TOL, `${c.w}x${c.h} k=${c.ks}`);
  }
});

test('resize = PIL Image.resize(BILINEAR), RGB en F', () => {
  for (const c of fx.resize) {
    const what = `${c.sw}x${c.sh} → ${c.dw}x${c.dh}`;
    near(R.resize(bytes(c.rgb), c.sw, c.sh, c.dw, c.dh, 3), bytes(c.rgb_out), 0, 'RGB ' + what);
    near(R.resize(floats(c.f), c.sw, c.sh, c.dw, c.dh), floats(c.f_out), TOL, 'F ' + what);
  }
});

test('segment = core.segment (nep-model: resize, normalisatie, nabewerking, terugschalen)', async () => {
  for (const c of fx.segment) {
    const got = await R.segment(rgbaOf(bytes(c.rgb)), c.w, c.h, specOf(c), fakeSeg(c.size, c.amp, c.bias));
    near(got, floats(c.out), TOL, `${c.model} (${c.post}) ${c.w}x${c.h}`);
  }
});

test('edgeBand = band uit core.cutout', () => {
  for (const c of fx.band) assert.equal(R.edgeBand(c.w, c.h), c.out, `${c.w}x${c.h}`);
});

test('cutout = core.cutout (nep-modellen), RGBA-bytes', async () => {
  for (const c of fx.cutout) {
    const got = await R.cutout(rgbaOf(bytes(c.rgb)), c.w, c.h, specOf(c), fakeSeg(c.size, c.amp, c.bias), c.refine ? fakeMatte : null);
    near(got, bytes(c.out), 1, `verfijning ${c.refine}`);
  }
});
