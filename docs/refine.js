// AlphaMaker in de browser: pure beeldfuncties zonder DOM. Port van alphamaker/refine.py plus de voor- en
// nabewerking uit core.segment/cutout; tests/web/refine.test.mjs vergelijkt alles met de Python-uitvoer.
export const TILE = 1024, OVERLAP = 128;
const f32 = Math.fround;
// numpy rekent hier in float32 en vergelijkt met float32(drempel): f32(0.05) > 0.05
const LO = f32(0.05), HI = f32(0.95), EPS = f32(1e-5);

/** Python's round(): halven naar het even getal (round(12.5) == 12). */
export function pyRound(v) {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 ? r - 1 : r;
}

// scipy.ndimage mode 'reflect' (spiegelen om de rand), ook als het venster langer is dan de lijn
const reflect = (j, n) => { const m = ((j % (2 * n)) + 2 * n) % (2 * n); return m < n ? m : 2 * n - 1 - m; };

/** scipy.ndimage.uniform_filter(src, size=k, mode='reflect') op een h×w-vlak. Venster per as:
 *  [i − k//2, i + k − k//2 − 1] (even k ligt een pixel naar links). Eerst langs de kolommen, met scipy's
 *  lopende som in float64 en float32 tussen de stappen. out mag niet src zijn. */
export function boxFilterReflect(src, w, h, k, out = new Float32Array(w * h)) {
  if (k <= 1) { out.set(src); return out; }
  const s1 = k >> 1, acc = new Float64Array(w);
  for (let t = 0; t < k; t++) { const r = reflect(t - s1, h) * w; for (let x = 0; x < w; x++) acc[x] += src[r + x]; }
  for (let x = 0; x < w; x++) out[x] = acc[x] /= k;
  for (let y = 1; y < h; y++) {
    const add = reflect(y - 1 + k - s1, h) * w, sub = reflect(y - 1 - s1, h) * w, o = y * w;
    for (let x = 0; x < w; x++) out[o + x] = acc[x] += (src[add + x] - src[sub + x]) / k;
  }
  const idx = Int32Array.from({ length: w + k - 1 }, (_, e) => reflect(e - s1, w)), line = new Float64Array(idx.length);
  for (let o = 0; o < w * h; o += w) {
    for (let e = 0; e < idx.length; e++) line[e] = out[o + idx[e]];
    let s = 0;
    for (let e = 0; e < k; e++) s += line[e];
    out[o] = s /= k;
    for (let x = 1; x < w; x++) out[o + x] = s += (line[x - 1 + k] - line[x - 1]) / k;
  }
  return out;
}

// Manhattan-afstand tot de dichtstbijzijnde zaadpixel; twee rasterpasses zijn exact voor deze metriek.
// Zaad = pixels met mask != inv. Bij erosie (inv) telt buiten het beeld als achtergrond, net als scipy.
function cityBlock(mask, w, h, inv) {
  const n = w * h, big = w + h + 2, edge = inv ? 0 : big, d = new Int32Array(n);
  for (let i = 0; i < n; i++) d[i] = !mask[i] === inv ? 0 : big;
  for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++)
    d[i] = Math.min(d[i], (y ? d[i - w] : edge) + 1, (x ? d[i - 1] : edge) + 1);
  for (let y = h - 1, i = n - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--, i--)
    d[i] = Math.min(d[i], (y < h - 1 ? d[i + w] : edge) + 1, (x < w - 1 ? d[i + 1] : edge) + 1);
  return d;
}

/** scipy binary_dilation(mask, iterations=n) met de standaard kruisstructuur = diamant met straal n. */
export function dilate(mask, w, h, n) {
  const d = cityBlock(mask, w, h, false), o = new Uint8Array(w * h);
  for (let i = 0; i < o.length; i++) o[i] = d[i] <= n;
  return o;
}

/** scipy binary_erosion(mask, iterations=n), border_value=0. */
export function erode(mask, w, h, n) {
  const d = cityBlock(mask, w, h, true), o = new Uint8Array(w * h);
  for (let i = 0; i < o.length; i++) o[i] = d[i] > n;
  return o;
}

/** refine.trimap: 1 = voorgrond, 0.5 = onzeker, 0 = achtergrond; band = halve breedte onzekere zone. */
export function trimap(alpha, w, h, band) {
  const n = w * h, hi = new Uint8Array(n), lo = new Uint8Array(n), t = new Float32Array(n);
  for (let i = 0; i < n; i++) { hi[i] = alpha[i] > HI; lo[i] = alpha[i] > LO; }
  const fg = erode(hi, w, h, band), near = dilate(lo, w, h, band);
  for (let i = 0; i < n; i++) t[i] = fg[i] ? 1 : near[i] ? 0.5 : 0;
  return t;
}

/** refine._starts: tegelbegin langs een as van n px. */
export function tileStarts(n, tile = TILE, overlap = OVERLAP) {
  if (n <= tile) return [0];
  const k = Math.ceil((n - overlap) / (tile - overlap));
  return Array.from({ length: k }, (_, i) => pyRound(i * (n - tile) / (k - 1)));
}

/** refine._ramp: lineaire 'feather'-weging over de overlap. */
export function ramp(n, overlap = OVERLAP) {
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) r[i] = Math.min(i + 1, n - i, overlap) / overlap;
  return r;
}

/** refine.vitmatte: ViTMatte alleen op tegels (max tile px) met onzekere pixels, op volle resolutie,
 *  opgevuld tot een veelvoud van 32. runTile(input, hp, wp, index, aantal) → alphas [hp·wp], met als input
 *  [4, hp, wp]: RGB (x − 0.5)/0.5 en de trimap als 4e kanaal. */
export async function vitmatte(rgba, tri, w, h, runTile, tile = TILE, overlap = OVERLAP) {
  const acc = new Float32Array(w * h), wsum = new Float32Array(w * h), jobs = [];
  const unknown = (y, x, th, tw) => {
    for (let r = y; r < y + th; r++) for (let c = x; c < x + tw; c++) if (tri[r * w + c] === 0.5) return true;
    return false;
  };
  for (const y of tileStarts(h, tile, overlap)) for (const x of tileStarts(w, tile, overlap)) {
    const th = Math.min(tile, h - y), tw = Math.min(tile, w - x);
    if (unknown(y, x, th, tw)) jobs.push([y, x, th, tw]);
  }
  for (const [j, [y, x, th, tw]] of jobs.entries()) {
    const hp = Math.ceil(th / 32) * 32, wp = Math.ceil(tw / 32) * 32, p = hp * wp, inp = new Float32Array(4 * p);
    for (let r = 0; r < th; r++) for (let c = 0; c < tw; c++) {
      const s = (y + r) * w + x + c, d = r * wp + c;
      for (let ch = 0; ch < 3; ch++) inp[ch * p + d] = (f32(rgba[4 * s + ch] / 255) - 0.5) / 0.5;
      inp[3 * p + d] = tri[s];
    }
    const a = await runTile(inp, hp, wp, j, jobs.length), rh = ramp(th, overlap), rw = ramp(tw, overlap);
    for (let r = 0; r < th; r++) for (let c = 0; c < tw; c++) {
      const f = f32(Math.min(rh[r], rw[c])), s = (y + r) * w + x + c;
      acc[s] += a[r * wp + c] * f;
      wsum[s] += f;
    }
  }
  for (let i = 0; i < acc.length; i++) acc[i] = tri[i] === 1 ? 1 : tri[i] === 0.5 && wsum[i] > 0 ? acc[i] / wsum[i] : 0;
  return acc;
}

/** refine.foreground: randkleurcorrectie 'blur fusion' (Forte & Pitié 2021) op de RGB van RGBA-bytes.
 *  De kanalen zijn onafhankelijk; om geheugen te sparen wordt elk kanaal afgemaakt en aan sink(c, F)
 *  gegeven (F = float32-vlak, wordt hergebruikt voor het volgende kanaal). */
export function foregroundBlurFusion(rgba, alpha, w, h, sink, ks = [90, 6]) {
  const n = w * h, F = new Float32Array(n), B = new Float32Array(n), tmp = new Float32Array(n);
  const bA = new Float32Array(n), b1 = new Float32Array(n);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) F[i] = B[i] = rgba[4 * i + c] / 255;
    for (const k of ks) {
      boxFilterReflect(alpha, w, h, k, bA);
      for (let i = 0; i < n; i++) tmp[i] = 1 - alpha[i];
      boxFilterReflect(tmp, w, h, k, b1);
      for (let i = 0; i < n; i++) tmp[i] = F[i] * alpha[i];
      boxFilterReflect(tmp, w, h, k, F);
      for (let i = 0; i < n; i++) tmp[i] = B[i] * (1 - alpha[i]);
      boxFilterReflect(tmp, w, h, k, B);
      for (let i = 0; i < n; i++) {
        const a = alpha[i], bf = f32(F[i] / f32(bA[i] + EPS)), bb = f32(B[i] / f32(b1[i] + EPS));
        F[i] = Math.min(1, Math.max(0, bf + a * (f32(rgba[4 * i + c] / 255) - a * bf - (1 - a) * bb)));
        B[i] = bb;
      }
    }
    sink(c, F);
  }
}

/** Nabewerking uit core.segment: 'sigmoid', 'minmax' of 'auto' (sigmoid alleen als waarden buiten [0, 1] vallen). */
export function postprocess(y, kind) {
  let lo = Infinity, hi = -Infinity;
  for (const v of y) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (kind === 'sigmoid' || (kind === 'auto' && (lo < 0 || hi > 1))) return y.map(v => 1 / (1 + Math.exp(-v)));
  if (kind === 'minmax') { const d = Math.max(f32(hi - lo), 1e-8); return y.map(v => (v - lo) / d); }
  return y;
}

// Gewichten van PIL's precompute_coeffs, bilineair: driehoek, bij verkleinen verbreed (antialias).
// 8-bit afbeeldingen rekenen in PIL met gehele gewichten van 22 bits.
function coeffs(inSize, outSize, u8) {
  const scale = inSize / outSize, fs = Math.max(1, scale), ss = 1 / fs, ks = Math.ceil(fs) * 2 + 1;
  const lo = new Int32Array(outSize), len = new Int32Array(outSize), k = new Float64Array(outSize * ks);
  for (let o = 0; o < outSize; o++) {
    const c = (o + 0.5) * scale, a = Math.max(0, Math.trunc(c - fs + 0.5)), n = Math.min(inSize, Math.trunc(c + fs + 0.5)) - a;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += k[o * ks + i] = Math.max(0, 1 - Math.abs((i + a - c + 0.5) * ss));
    for (let i = 0; i < n; i++) {
      const v = sum ? k[o * ks + i] / sum : k[o * ks + i];
      k[o * ks + i] = u8 ? Math.trunc(0.5 + v * 4194304) : v;
    }
    lo[o] = a; len[o] = n;
  }
  return { ks, lo, len, k };
}

/** PIL Image.resize((dw, dh), BILINEAR) op verweven data met ch kanalen: Float32Array zoals modus 'F',
 *  bytes zoals 'RGB' (vaste komma, afronding na elke pass). */
export function resize(src, sw, sh, dw, dh, ch = 1) {
  const u8 = !(src instanceof Float32Array), make = n => new src.constructor(n);
  let img = src, w = sw, h = sh;
  const horizontal = () => {
    if (w === dw) return;
    const { ks, lo, len, k } = coeffs(w, dw, u8), out = make(dw * h * ch);
    for (let y = 0; y < h; y++) for (let o = 0; o < dw; o++) for (let q = 0; q < ch; q++) {
      let s = u8 ? 2097152 : 0;
      for (let i = 0; i < len[o]; i++) s += img[(y * w + lo[o] + i) * ch + q] * k[o * ks + i];
      out[(y * dw + o) * ch + q] = u8 ? Math.min(255, s >> 22) : s;
    }
    img = out; w = dw;
  };
  const vertical = () => {
    if (h === dh) return;
    const { ks, lo, len, k } = coeffs(h, dh, u8), out = make(w * dh * ch);
    for (let o = 0; o < dh; o++) for (let x = 0; x < w; x++) for (let q = 0; q < ch; q++) {
      let s = u8 ? 2097152 : 0;
      for (let i = 0; i < len[o]; i++) s += img[((lo[o] + i) * w + x) * ch + q] * k[o * ks + i];
      out[(o * w + x) * ch + q] = u8 ? Math.min(255, s >> 22) : s;
    }
    img = out; h = dh;
  };
  // Pillow 13 (Resample.c) doet de verticale pass eerst als de hoogte veel sterker krimpt; bij 8 bit
  // geeft de volgorde een andere afronding
  if (sh - dh > 0 && sh - dh > (sw - dw) * 2) { vertical(); horizontal(); } else { horizontal(); vertical(); }
  return img === src ? src.slice() : img;
}

/** Normalisatie uit core.segment: bytes (ch kanalen per pixel) → float32 [3, h, w] van (v/255 − mean)/std. */
export function toNCHW(px, w, h, mean, std, ch = 4) {
  const n = w * h, out = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) out[c * n + i] = (f32(px[i * ch + c] / 255) - mean[c]) / std[c];
  return out;
}

/** core.segment: RGBA → masker 0..1 op w×h. run(x [1,3,size,size]) → {data, dims}. */
export async function segment(rgba, w, h, { size, norm, post }, run) {
  const x = toNCHW(resize(rgba, w, h, size, size, 4), size, size, norm.mean, norm.std);
  const { data, dims } = await run(x, size);
  const [oh, ow] = dims.slice(-2);
  const a = resize(postprocess(data.subarray(0, oh * ow), post), ow, oh, w, h);
  for (let i = 0; i < a.length; i++) a[i] = Math.min(1, Math.max(0, a[i]));
  return a;
}

/** Breedte onzekere zone uit core.cutout: ~1,25% van de lange zijde, minstens 4 px. */
export const edgeBand = (w, h) => Math.max(4, pyRound(Math.max(w, h) * 0.0125));

const toByte = v => f32(f32(v * 255) + 0.5) | 0;  // (rgba * 255 + 0.5).astype(uint8) in float32

/** core.cutout op RGBA-bytes, in place: segmenteren → (als runTile) ViTMatte → randkleuren.
 *  step(naam, tegel, aantal) meldt de voortgang: 'segment', 'refine' of 'colors'. */
export async function cutout(rgba, w, h, spec, runSeg, runTile, step = () => {}) {
  step('segment');
  let alpha = await segment(rgba, w, h, spec, runSeg);
  if (runTile) {
    const tri = trimap(alpha, w, h, edgeBand(w, h));
    alpha = null;  // het grove masker is nu niet meer nodig: geheugen vrij tijdens ViTMatte
    alpha = await vitmatte(rgba, tri, w, h, (inp, hp, wp, j, n) => { step('refine', j, n); return runTile(inp, hp, wp); });
  }
  step('colors');
  foregroundBlurFusion(rgba, alpha, w, h, (c, F) => { for (let i = 0; i < F.length; i++) rgba[4 * i + c] = toByte(F[i]); });
  for (let i = 0; i < alpha.length; i++) rgba[4 * i + 3] = toByte(alpha[i]);
  return rgba;
}
