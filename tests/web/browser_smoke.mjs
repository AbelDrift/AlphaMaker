// Rooktest van de webversie in een echte (headless) Firefox via WebDriver BiDi — geen extra pakketten.
// Gebruik: node tests/web/browser_smoke.mjs <url-van-pagina> <url-van-testfoto> [model]
// Verwacht: Firefox gestart met --remote-debugging-port $BIDI_PORT (standaard 9333).
const [url, photo, model = 'rmbg-1.4-licht'] = process.argv.slice(2);
const ws = new WebSocket(`ws://127.0.0.1:${process.env.BIDI_PORT || 9333}/session`);
let id = 0;
const pending = new Map(), logs = [];
ws.onmessage = ({ data }) => {
  const m = JSON.parse(data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  else if (m.method === 'log.entryAdded') {
    const f = m.params.stackTrace?.callFrames?.[0];
    logs.push(`${m.params.level}: ${m.params.text}` + (f ? ` (${f.url.split('/').pop()}:${f.lineNumber + 1})` : ''));
  }
};
const send = (method, params = {}) => new Promise((res, rej) => {
  const n = ++id;
  pending.set(n, m => m.type === 'error' ? rej(new Error(`${method}: ${m.error} ${m.message}`)) : res(m.result));
  ws.send(JSON.stringify({ id: n, method, params }));
});
const sleep = ms => new Promise(r => setTimeout(r, ms));
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

await send('session.new', { capabilities: {} });
const { contexts: [{ context }] } = await send('browsingContext.getTree');
await send('session.subscribe', { events: ['log.entryAdded'] });
const evaluate = async expression => (await send('script.evaluate',
  { expression, target: { context }, awaitPromise: true, resultOwnership: 'none' })).result?.value;
await send('browsingContext.navigate', { context, url, wait: 'complete' });

// de pagina herlaadt zichzelf eenmalig onder de service worker (coi.js): tijdens herladen faalt evaluate even
const status = () => evaluate(`document.getElementById('status').textContent`).catch(() => 'laden');
let t0 = Date.now(), s = '';
while (Date.now() - t0 < 60_000 && /laden/i.test(s = await status() ?? 'laden')) await sleep(1000);
await sleep(1500);
while (Date.now() - t0 < 60_000 && /laden/i.test(s = await status() ?? 'laden')) await sleep(1000);
console.log('rekenkern:', await evaluate(`document.getElementById('engine')?.textContent ?? ''`), '| geïsoleerd:', await evaluate('self.crossOriginIsolated').catch(() => '?'));

// lichtste model kiezen en de foto via het uploadveld aanbieden (zoals een gebruiker)
console.log('model:', await evaluate(`(() => { const s = document.getElementById('model'); s.value = ${JSON.stringify(model)}; s.dispatchEvent(new Event('change')); return s.value; })()`));
await evaluate(`(async () => {
  const b = await (await fetch(${JSON.stringify(photo)})).blob();
  const dt = new DataTransfer();  // twee foto's tegelijk: test ook de wachtrij (bulk)
  for (const n of ['test1.jpg', 'test2.jpg']) dt.items.add(new File([b], n, { type: 'image/jpeg' }));
  const inp = document.getElementById('file'); inp.files = dt.files; inp.dispatchEvent(new Event('change'));
  return 'aangeboden';
})()`);
t0 = Date.now(); let last = '';
while (Date.now() - t0 < 600_000) {
  s = await status() ?? '';
  if (s !== last && !/downloaden.*%/.test(s)) console.log('status:', s);
  last = s;
  if ((/^Klaar/.test(s) && !/wachtrij/.test(s)) || /^Mislukt|^Kon /.test(s)) break;
  await sleep(2000);
}
const stats = await evaluate(`(() => {
  const c = document.getElementById('out'); if (!c.width) return 'geen resultaat';
  const a = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let t = 0, h = 0; for (let i = 3; i < a.length; i += 4) { if (a[i] === 0) t++; else if (a[i] < 255) h++; }
  const n = a.length / 4;
  return c.width + 'x' + c.height + ', transparant ' + (100 * t / n).toFixed(1) + '%, halfdoorzichtig ' + (100 * h / n).toFixed(1) + '%';
})()`);
console.log('resultaat:', stats);
console.log('tijden per foto (ms):', await evaluate('JSON.stringify(self.__timings ?? [])'));
const listed = await evaluate(`document.querySelectorAll('#list button').length`);
console.log('in de lijst:', listed);
// afwerking zoals remove.bg: bijsnijden (5% marge) + schaduw + witte achtergrond → kleiner en volledig dekkend
const finish = await evaluate(`(() => {
  const set = (id, prop, v, ev) => { const e = document.getElementById(id); e[prop] = v; e.dispatchEvent(new Event(ev)); };
  set('crop', 'checked', true, 'input'); set('shadow', 'checked', true, 'input'); set('bgmode', 'value', 'color', 'change');
  const c = document.getElementById('out'), a = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let t = 0; for (let i = 3; i < a.length; i += 4) if (a[i] < 255) t++;
  return c.width + 'x' + c.height + ', niet-dekkend ' + t;
})()`);
console.log('afwerking (bijsnijden+schaduw+kleur):', finish);
const fmt = await evaluate(`(() => { const f = document.getElementById('format'); return [...f.options].map(o => o.value).join(',') + ' / knop: ' + document.getElementById('download').textContent; })()`);
console.log('formaten:', fmt);
const brush = await evaluate(`(async () => {
  const $ = id => document.getElementById(id), c = $('out');
  for (const [id, prop, v, ev] of [['crop', 'checked', false, 'input'], ['shadow', 'checked', false, 'input'], ['bgmode', 'value', 'none', 'change'], ['brush', 'value', 'erase', 'change']]) { $(id)[prop] = v; $(id).dispatchEvent(new Event(ev)); }
  await new Promise(r => requestAnimationFrame(r));
  const b = c.getBoundingClientRect(), x = b.left + b.width / 2, y = b.top + b.height / 2;
  const alpha = () => c.getContext('2d').getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data[3];
  const fire = t => c.dispatchEvent(new PointerEvent(t, { clientX: x, clientY: y, pointerId: 1, bubbles: true }));
  const before = alpha();
  fire('pointerdown'); fire('pointerup'); await new Promise(r => setTimeout(r, 300));
  const erased = alpha();
  $('brush').value = 'restore'; $('brush').dispatchEvent(new Event('change'));
  fire('pointerdown'); fire('pointerup'); await new Promise(r => setTimeout(r, 1000));
  return before + '>' + erased + '>' + alpha();
})()`);
console.log('penseel (alpha midden: voor>gum>herstel):', brush);
const finishOk = /niet-dekkend 0$/.test(finish ?? '') && finish.split(',')[0] !== stats.split(',')[0];
console.log('console:', logs.length ? '\n  ' + logs.join('\n  ') : 'geen meldingen');
await send('session.end').catch(() => {});
ws.close();
process.exit(/^Klaar/.test(s) && /transparant [1-9]/.test(stats) && finishOk && listed === 2 && /^\d+>0>255$/.test(brush ?? '') ? 0 : 1);
