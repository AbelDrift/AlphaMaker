// UI van de webversie: foto inlezen (max 12 MP), naar de worker sturen, resultaat tonen en downloaden.
import { MODELS, DEFAULT_MODEL, MAX_PIXELS } from './models.js';

const $ = id => document.getElementById(id);
const status = t => { $('status').textContent = t; };
const mb = bytes => Math.round(bytes / 2 ** 20);
const worker = new Worker(new URL('worker.js', import.meta.url), { type: 'module' });
let backend, cutout, busy, name = 'alphamaker', note = '';

function setBusy(b) {
  busy = b;
  for (const id of ['file', 'model', 'fijn']) $(id).disabled = b;
  $('groot').disabled = b || !$('fijn').checked;
}

function bar(frac) {  // undefined = verbergen, null = onbepaald
  const el = $('bar');
  el.hidden = frac === undefined;
  if (frac === null) el.removeAttribute('value');
  else if (frac !== undefined) el.value = frac;
}

function showLicense() {
  const spec = MODELS[$('model').value];
  $('license').textContent = 'Licentie: ' + spec.license;
  $('license').classList.toggle('warn', !!spec.nonCommercial);
}

function ready(m) {
  backend = m.backend;
  const sel = $('model');
  for (const [key, spec] of Object.entries(MODELS)) {
    const f = spec.files[backend];
    sel.add(Object.assign(new Option(`${spec.label} · ${f ? mb(f.bytes) + ' MB' : 'alleen met WebGPU'}`, key), { disabled: !f }));
  }
  const def = DEFAULT_MODEL[backend];
  sel.value = MODELS[def]?.files[backend] ? def : [...sel.options].find(o => !o.disabled).value;
  $('engine').textContent = backend === 'webgpu'
    ? 'Rekent op je grafische kaart (WebGPU).'
    : `Rekent op de processor (WebAssembly${m.threads ? '' : ', één kern'}); dat is trager. Je browser heeft geen WebGPU met fp16, daarom zijn de modellen "alleen met WebGPU" uitgeschakeld.`;
  $('fijn').checked = false;  // ViTMatte maakte de randen in de benchmark niet beter: experimenteel
  showLicense();
  setBusy(false);
  status('');
}

worker.onmessage = ({ data: m }) => {
  if (m.type === 'ready') return ready(m);
  if (m.type === 'progress') { status(m.text); return bar(m.frac); }
  setBusy(false);
  bar();
  if (m.type === 'error') return status('Mislukt: ' + m.message);
  cutout = new ImageData(new Uint8ClampedArray(m.rgba), m.w, m.h);
  box = subjectBox(cutout);
  render();
  $('result').hidden = false;
  $('download').disabled = false;
  status(`Klaar in ${(m.ms / 1000).toLocaleString('nl', { maximumFractionDigits: 1 })} s. ${note}`);
};
worker.onerror = e => { status('De rekenkern kon niet starten' + (e.message ? ': ' + e.message : '.')); bar(); };

async function handle(file) {
  if (!file?.type.startsWith('image/')) return status('Dat is geen afbeelding.');
  if (busy) return status(backend ? 'Even geduld, de vorige foto is nog bezig.' : 'Even geduld, de rekenkern start nog.');
  if (!backend) return status('De rekenkern is niet beschikbaar; herlaad de pagina.');
  setBusy(true);
  name = file.name.replace(/\.[^.]+$/, '') || 'alphamaker';
  if ($('orig').src) URL.revokeObjectURL($('orig').src);
  $('orig').src = URL.createObjectURL(file);
  $('download').disabled = true;
  try {
    status('Foto openen…');
    const bmp = await createImageBitmap(file);
    let w = bmp.width, h = bmp.height;
    note = '';
    if (w * h > MAX_PIXELS) {
      const s = Math.sqrt(MAX_PIXELS / (w * h)), w0 = w, h0 = h;
      w = Math.max(1, Math.floor(w * s));
      h = Math.max(1, Math.floor(h * s));
      note = `De foto is verkleind van ${w0}×${h0} naar ${w}×${h} pixels: in de browser kan maximaal ${MAX_PIXELS / 1e6} megapixel.`;
      status(note);
    }
    const ctx = Object.assign(document.createElement('canvas'), { width: w, height: h }).getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bmp, 0, 0, w, h);
    bmp.close();
    const { data } = ctx.getImageData(0, 0, w, h);
    ctx.canvas.width = ctx.canvas.height = 0;  // canvasgeheugen direct vrijgeven
    worker.postMessage({ rgba: data.buffer, w, h, model: $('model').value, refine: $('fijn').checked,
      matte: $('groot').checked ? 'fp32' : 'q' }, [data.buffer]);
  } catch (e) {
    setBusy(false);
    status('Kon de foto niet openen: ' + e.message);
  }
}

let bgImg = null, box = null, rect = null;  // achtergrondfoto, onderwerp-kader en getoonde uitsnede

function subjectBox({ width: w, height: h, data }) {  // kleinste rechthoek met alpha > 8, zoals core.finish
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (data[4 * (y * w + x) + 3] > 8) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

// afwerking zoals core.finish (bijsnijden + marge, achtergrondkleur/-foto, schaduw), alles lokaal op canvas
function render() {
  const src = Object.assign(document.createElement('canvas'), { width: cutout.width, height: cutout.height });
  src.getContext('2d').putImageData(cutout, 0, 0);
  let r = { x: 0, y: 0, w: cutout.width, h: cutout.height };
  if ($('crop').checked && box) {
    const m = Math.min(50, Math.max(0, Number($('margin').value) || 0)) / 100, mx = Math.round(box.w * m), my = Math.round(box.h * m);
    const x = Math.max(0, box.x - mx), y = Math.max(0, box.y - my);
    r = { x, y, w: Math.min(cutout.width, box.x + box.w + mx) - x, h: Math.min(cutout.height, box.y + box.h + my) - y };
  }
  const c = $('out'), ctx = c.getContext('2d');
  c.width = r.w; c.height = r.h;
  rect = r;
  const mode = $('bgmode').value;
  if (mode === 'color') { ctx.fillStyle = $('bgcolor').value; ctx.fillRect(0, 0, r.w, r.h); }
  if (mode === 'image' && bgImg) {  // 'cover': vullen en bijsnijden, zoals ImageOps.fit
    const s = Math.max(r.w / bgImg.width, r.h / bgImg.height), bw = bgImg.width * s, bh = bgImg.height * s;
    ctx.drawImage(bgImg, (r.w - bw) / 2, (r.h - bh) / 2, bw, bh);
  }
  if ($('shadow').checked) {  // zachte slagschaduw: ~1% van de lange zijde, 50% dekking
    const d = Math.max(2, Math.round(Math.max(r.w, r.h) * 0.01));
    Object.assign(ctx, { shadowColor: 'rgba(0,0,0,0.5)', shadowBlur: 2 * d, shadowOffsetX: d, shadowOffsetY: d });
  }
  ctx.drawImage(src, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
  src.width = src.height = 0;
}

setBusy(true);
status('Rekenkern laden…');
$('file').onchange = e => handle(e.target.files[0]);
$('model').onchange = showLicense;
$('fijn').onchange = () => { $('groot').disabled = !$('fijn').checked; };
$('bgmode').onchange = () => {
  $('bgcolor').hidden = $('bgmode').value !== 'color';
  $('bgimage').hidden = $('bgmode').value !== 'image';
  if (cutout) render();
};
$('bgcolor').oninput = () => cutout && render();
$('bgimage').onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  bgImg?.close();
  bgImg = await createImageBitmap(f).catch(() => null);
  if (!bgImg) status('Kon de achtergrondfoto niet openen.');
  if (cutout) render();
};
for (const id of ['crop', 'margin', 'shadow']) $(id).oninput = () => cutout && render();
// PNG (transparant), JPG (witte ondergrond) of alleen het masker (alfa als grijswaarden, zoals remove.bg channels=alpha)
$('download').onclick = () => {
  const fmt = $('format').value, c = $('out');
  let src = c, type = 'image/png', suffix = '-alpha.png';
  if (fmt !== 'png') {
    src = Object.assign(document.createElement('canvas'), { width: c.width, height: c.height });
    const x = src.getContext('2d');
    if (fmt === 'jpg') {
      x.fillStyle = '#ffffff'; x.fillRect(0, 0, c.width, c.height); x.drawImage(c, 0, 0);
      type = 'image/jpeg'; suffix = '-alpha.jpg';
    } else {
      const d = x.createImageData(rect.w, rect.h);
      for (let y = 0; y < rect.h; y++) for (let i = 0; i < rect.w; i++) {
        const a = cutout.data[4 * ((y + rect.y) * cutout.width + i + rect.x) + 3], j = 4 * (y * rect.w + i);
        d.data[j] = d.data[j + 1] = d.data[j + 2] = a; d.data[j + 3] = 255;
      }
      x.putImageData(d, 0, 0); suffix = '-masker.png';
    }
  }
  src.toBlob(b => {
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(b), download: name + suffix });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }, type, 0.92);
};
const drop = $('drop');
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); handle(e.dataTransfer.files[0]); };
addEventListener('paste', e => { const f = e.clipboardData.files[0]; if (f) handle(f); });
