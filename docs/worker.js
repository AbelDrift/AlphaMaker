// Rekenwerk buiten de UI-thread: onnxruntime-web (WebGPU of WASM), modeldownloads en de pijplijn uit refine.js.
import { MODELS, VITMATTE, MAX_PIXELS } from './models.js';
import { cutout } from './refine.js';

const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
const CACHE = 'alphamaker-modellen-v1';
const STEPS = { segment: 'Achtergrond herkennen…', refine: 'Randen verfijnen…', colors: 'Randkleuren corrigeren…' };
// zelfde geheugenspaarstand als core.session: geen arena, geen vooraf gepland blok, geen verpakte gewichten
const OPTIONS = { enableCpuMemArena: false, enableMemPattern: false, extra: { 'session.disable_prepacking': '1' } };
const post = (type, data = {}, transfer = []) => postMessage({ type, ...data }, transfer);
const progress = (text, frac = null) => post('progress', { text, frac });
const mb = bytes => Math.round(bytes / 2 ** 20);
const slots = { seg: {}, matte: {} };  // één sessie per soort; bij een wissel wordt de oude vrijgegeven
let ort, backend;

const ready = (async () => {
  ort = await import(ORT + 'ort.webgpu.bundle.min.mjs');
  ort.env.wasm.wasmPaths = ORT;
  const gpu = await navigator.gpu?.requestAdapter().catch(() => null);
  // de WebGPU-modellen zijn fp16: dat vraagt shader-f16 op de kaart en Float16Array in JS
  backend = gpu?.features.has('shader-f16') && typeof Float16Array === 'function' ? 'webgpu' : 'wasm';
  for (const key of await self.caches?.keys().catch(() => []) ?? [])
    if (key.startsWith('alphamaker-') && key !== CACHE) caches.delete(key).catch(() => {});  // oude modelversies opruimen
  post('ready', { backend, threads: self.crossOriginIsolated });
})();
ready.catch(e => post('error', { message: 'de rekenkern (onnxruntime-web) kon niet laden: ' + explain(e) }));

function explain(e) {
  const msg = String(e?.message ?? e);
  return e instanceof RangeError || /memory|alloc/i.test(msg)
    ? `te weinig geheugen (${msg}). Herlaad de pagina en probeer een kleinere foto of een lichter model.`
    : msg;
}

async function download({ url, bytes }, label) {
  const plausible = n => Math.abs(n - bytes) <= bytes / 4;  // vangt foutpagina's en afgebroken downloads
  const cache = await self.caches?.open(CACHE).catch(() => null);  // geen Cache API: elke keer downloaden
  const hit = await cache?.match(url);
  if (hit) {
    const b = new Uint8Array(await hit.arrayBuffer());
    if (plausible(b.length)) return b;
    await cache.delete(url);
  }
  let res;
  try {
    res = await fetch(url, { cache: 'no-store' });  // geen tweede kopie in de HTTP-cache
  } catch {
    throw new Error(`${label} downloaden mislukt: geen verbinding met Hugging Face.`);
  }
  if (!res.ok) throw new Error(`${label} downloaden mislukt (HTTP ${res.status}).`);
  const len = Number(res.headers.get('Content-Length')) || 0, total = len || bytes;
  if (len && !plausible(len)) throw new Error(`${label}: onverwachte bestandsgrootte (${mb(len)} MB in plaats van ${mb(bytes)} MB).`);
  let buf = new Uint8Array(total), got = 0, shown = -1;
  try {
    for (const reader = res.body.getReader(); ;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length > buf.length) {
        const b = new Uint8Array(Math.max(2 * buf.length, got + value.length));
        b.set(buf.subarray(0, got));
        buf = b;
      }
      buf.set(value, got);
      got += value.length;
      const pct = Math.min(100, Math.floor(100 * got / total));
      if (pct !== shown) progress(`${label} downloaden (eenmalig, ${mb(total)} MB)… ${shown = pct}%`, got / total);
    }
  } catch {
    throw new Error(`${label}: download onderbroken. Controleer je verbinding en probeer het opnieuw.`);
  }
  if (len ? got !== len : !plausible(got)) throw new Error(`${label}: download onvolledig (${mb(got)} MB). Probeer het opnieuw.`);
  buf = buf.subarray(0, got);
  try {
    await cache?.put(url, new Response(buf));
  } catch (e) {
    console.warn('AlphaMaker: model niet bewaard (opslag vol?), volgende keer opnieuw downloaden', e);
  }
  return buf;
}

async function session(slot, key, file, label) {
  if (slot.key === key) return slot.sess;
  await slot.sess?.release();
  slot.sess = slot.key = null;
  progress(`${label} laden…`);
  const bytes = await download(file, label);
  progress(`${label} starten…`);
  slot.sess = await ort.InferenceSession.create(bytes, { ...OPTIONS, executionProviders: [backend] });
  slot.key = key;
  return slot.sess;
}

// fp16-modellen: invoer als float16-bits (Uint16Array slikt elke ORT-versie), uitvoer terug naar float32
const tensor = (type, x, dims) => type === 'float16'
  ? new ort.Tensor('float16', new Uint16Array(new Float16Array(x).buffer), dims)
  : new ort.Tensor('float32', x, dims);
const float32 = t => t.type !== 'float16' ? t.data
  : Float32Array.from(t.data instanceof Uint16Array ? new Float16Array(t.data.buffer, t.data.byteOffset, t.data.length) : t.data);

async function infer(sess, feeds, name = sess.outputNames[0]) {
  const out = await sess.run(feeds);
  const r = { data: float32(out[name]), dims: out[name].dims };
  for (const t of [...Object.values(feeds), ...Object.values(out)]) t.dispose();
  return r;
}

async function run({ rgba: buffer, w, h, model, refine, matte }) {
  await ready;
  const t0 = performance.now(), rgba = new Uint8ClampedArray(buffer);
  if (!(w > 0 && h > 0 && w * h <= MAX_PIXELS && rgba.length === 4 * w * h)) throw new Error('ongeldige afbeelding.');
  const spec = MODELS[model], file = spec?.files[backend];
  if (!file) throw new Error('dit model werkt niet op dit apparaat.');
  const marks = [['laden', t0]];  // tijd per stap, voor de rooktest en om traagheid te vinden
  let modelMs = 0;
  const seg = await session(slots.seg, model, file, spec.label);
  const vm = refine ? await session(slots.matte, matte, VITMATTE[matte] ?? VITMATTE.q, 'Verfijningsmodel') : null;
  const meta = seg.inputMetadata?.[0], dim = meta?.shape?.[2];
  const size = Number.isInteger(dim) && dim > 0 ? dim : spec.size;  // vaste invoermaat van het model wint, zoals in core.segment
  await cutout(rgba, w, h, { ...spec, size },
    async x => { const t = performance.now(), r = await infer(seg, { [seg.inputNames[0]]: tensor(meta?.type, x, [1, 3, size, size]) }); modelMs += performance.now() - t; return r; },
    vm && ((x, hp, wp) => infer(vm, { pixel_values: new ort.Tensor('float32', x, [1, 4, hp, wp]) }, 'alphas').then(r => r.data)),
    (step, j, n) => { if (!j) marks.push([step, performance.now()]); progress(n ? `${STEPS[step]} tegel ${j + 1} van ${n}` : STEPS[step], n ? j / n : null); });
  const end = performance.now(), steps = { model: Math.round(modelMs) };  // model = alleen session.run, binnen 'segment'
  marks.forEach(([s, t], i) => { steps[s] = Math.round((marks[i + 1]?.[1] ?? end) - t); });
  post('done', { rgba: rgba.buffer, w, h, ms: end - t0, steps }, [rgba.buffer]);
}

// één foto tegelijk: een sessiewissel mag nooit een sessie vrijgeven die nog rekent
let queue = Promise.resolve();
self.onmessage = ({ data }) => { queue = queue.then(() => run(data)).catch(e => post('error', { message: explain(e) })); };
