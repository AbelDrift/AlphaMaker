// Service worker die cross-origin-isolatie aanzet op hosts zonder eigen headers (GitHub/Codeberg Pages):
// dan mag WebAssembly meerdere processorkernen gebruiken (vooral sneller in browsers zonder WebGPU, zoals Firefox).
// Alle externe bronnen (jsDelivr, Hugging Face) worden met CORS geladen, dus 'require-corp' blokkeert niets.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const r = e.request;
  if (r.cache === 'only-if-cached' && r.mode !== 'same-origin') return;
  e.respondWith(fetch(r).then(res => {
    if (res.status === 0) return res;  // ondoorzichtig antwoord: ongewijzigd laten
    const h = new Headers(res.headers);
    h.set('Cross-Origin-Embedder-Policy', 'require-corp');
    h.set('Cross-Origin-Opener-Policy', 'same-origin');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});
