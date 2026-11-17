"""Gebruik:
  python -m alphamaker foto.jpg [meer.jpg ...] [-o map] [--model NAAM] [--fijne-randen]
      [--bijsnijden [--marge 0.05]] [--alleen-masker] [--schaduw] [--achtergrond '#fff' | --achtergrondfoto pad]
      [--formaat png|jpg|webp]
  python -m alphamaker serve [--port 8092] [--host 127.0.0.1]"""
import argparse, io, json, sys
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from PIL import Image
from . import core

MAX_UPLOAD = 40 * 1024 * 1024
PAGE = (Path(__file__).parent / 'static/index.html').read_bytes()


class Handler(BaseHTTPRequestHandler):
    model = core.DEFAULT_MODEL

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'): return self._send(200, PAGE, 'text/html; charset=utf-8')
        if self.path == '/api/models':
            return self._send(200, json.dumps({'models': core.SEGMENTERS, 'default': self.model}).encode(), 'application/json')
        self._send(404, b'niet gevonden', 'text/plain')

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != '/api/cutout': return self._send(404, b'niet gevonden', 'text/plain')
        q = parse_qs(url.query)
        model = q.get('model', [self.model])[0]
        if model not in core.SEGMENTERS: return self._send(400, b'onbekend model', 'text/plain')
        n = int(self.headers.get('Content-Length') or 0)
        if not 0 < n <= MAX_UPLOAD: return self._send(413, b'bestand te groot (max 40 MB)', 'text/plain')
        try:
            img = Image.open(io.BytesIO(self.rfile.read(n)))
            img.load()
        except Exception:
            return self._send(400, b'geen geldige afbeelding', 'text/plain')
        buf = io.BytesIO()
        try:
            core.cutout(img, model, refine_edges='fijn' in q).save(buf, 'PNG')
        except ValueError as e:
            return self._send(413, str(e).encode(), 'text/plain; charset=utf-8')
        self._send(200, buf.getvalue(), 'image/png')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='alphamaker')
    ap.add_argument('files', nargs='+', help="afbeeldingen, of 'serve' voor de webpagina")
    ap.add_argument('-o', '--out', type=Path, help='uitvoermap (standaard naast het origineel)')
    ap.add_argument('--model', choices=core.SEGMENTERS, help=f'standaard: {core.DEFAULT_MODEL}')
    ap.add_argument('--type', choices=['auto', 'persoon', 'object'], default='auto',
                    help='zoals remove.bg type: persoon → birefnet-general (beste voor haar, traag), '
                         'object → rmbg-1.4 (beste en snelste voor objecten, niet-commercieel)')
    ap.add_argument('--fijne-randen', action='store_true', help='experimenteel: ViTMatte-randverfijning (was in de benchmark niet beter)')
    ap.add_argument('--bijsnijden', action='store_true', help='bijsnijden tot het onderwerp (zoals remove.bg crop)')
    ap.add_argument('--marge', type=float, default=0.0, help='marge bij bijsnijden, als fractie van het onderwerp (0.05 = 5%%)')
    ap.add_argument('--alleen-masker', action='store_true', help='alleen het alfamasker (grijswaarden) opslaan')
    ap.add_argument('--schaduw', action='store_true', help='zachte slagschaduw toevoegen')
    ap.add_argument('--achtergrond', help="achtergrondkleur, bv. '#ffffff' of 'white'")
    ap.add_argument('--achtergrondfoto', type=Path, help='foto als nieuwe achtergrond')
    ap.add_argument('--formaat', choices=['png', 'jpg', 'webp'], default='png', help='uitvoerformaat (jpg zonder achtergrond krijgt wit)')
    ap.add_argument('--port', type=int, default=8092)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args(argv)
    by_type = {'persoon': 'birefnet-general' if core.pt.available() else None, 'object': 'rmbg-1.4'}
    a.model = a.model or by_type.get(a.type) or core.DEFAULT_MODEL
    if a.files == ['serve']:
        Handler.model = a.model
        print(f'AlphaMaker op http://{a.host}:{a.port}  (Ctrl+C stopt)', flush=True)
        # één verzoek tegelijk: houdt het geheugengebruik voorspelbaar
        HTTPServer((a.host, a.port), Handler).serve_forever()
    bg = a.achtergrond or ('white' if a.formaat == 'jpg' and not a.achtergrondfoto and not a.alleen_masker else None)
    bg_image = Image.open(a.achtergrondfoto) if a.achtergrondfoto else None
    files = list(map(Path, a.files))
    for n, f in enumerate(files, 1):
        dst = (a.out or f.parent) / f'{f.stem}-alpha.{a.formaat}'
        dst.parent.mkdir(parents=True, exist_ok=True)
        cut = core.cutout(Image.open(f), a.model, refine_edges=a.fijne_randen)
        core.finish(cut, a.bijsnijden, a.marge, a.alleen_masker, bg, bg_image, a.schaduw).save(dst, quality=95)
        print(f'[{n}/{len(files)}] {dst}' if len(files) > 1 else dst, flush=True)


if __name__ == '__main__':
    sys.exit(main())
