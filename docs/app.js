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
  sel.value = MODELS[DEFAULT_MODEL].files[backend] ? DEFAULT_MODEL : [...sel.options].find(o => !o.disabled).value;
  $('engine').textContent = backend === 'webgpu'
    ? 'Rekent op je grafische kaart (WebGPU).'
    : `Rekent op de processor (WebAssembly${m.threads ? '' : ', één kern'}); dat is trager. Je browser heeft geen WebGPU met fp16, daarom zijn de modellen "alleen met WebGPU" uitgeschakeld.`;
  $('fijn').checked = backend === 'webgpu';  // ViTMatte op de processor duurt per tegel lang
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

function render() {
  const c = $('out'), ctx = c.getContext('2d');
  c.width = cutout.width; c.height = cutout.height;
  ctx.putImageData(cutout, 0, 0);
  if ($('bgmode').value === 'color') {
    ctx.globalCompositeOperation = 'destination-over';
    ctx.fillStyle = $('bgcolor').value;
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.globalCompositeOperation = 'source-over';
  }
}

setBusy(true);
status('Rekenkern laden…');
$('file').onchange = e => handle(e.target.files[0]);
$('model').onchange = showLicense;
$('fijn').onchange = () => { $('groot').disabled = !$('fijn').checked; };
$('bgmode').onchange = () => { $('bgcolor').hidden = $('bgmode').value !== 'color'; render(); };
$('bgcolor').oninput = render;
$('download').onclick = () => $('out').toBlob(b => {
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(b), download: name + '-alpha.png' });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
});
const drop = $('drop');
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); handle(e.dataTransfer.files[0]); };
addEventListener('paste', e => { const f = e.clipboardData.files[0]; if (f) handle(f); });
