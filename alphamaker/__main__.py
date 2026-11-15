"""Gebruik:
  python -m alphamaker foto.jpg [meer.jpg ...] [-o map] [--model NAAM] [--snel]
  python -m alphamaker serve [--port 8092] [--host 127.0.0.1]"""
import argparse, io, sys
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
        self._send(404, b'niet gevonden', 'text/plain')

    def do_POST(self):
        if self.path.split('?')[0] != '/api/cutout': return self._send(404, b'niet gevonden', 'text/plain')
        n = int(self.headers.get('Content-Length') or 0)
        if not 0 < n <= MAX_UPLOAD: return self._send(413, b'bestand te groot (max 40 MB)', 'text/plain')
        try:
            img = Image.open(io.BytesIO(self.rfile.read(n)))
            img.load()
        except Exception:
            return self._send(400, b'geen geldige afbeelding', 'text/plain')
        fast = 'snel=1' in self.path
        buf = io.BytesIO()
        try:
            core.cutout(img, self.model, refine_edges=not fast).save(buf, 'PNG')
        except ValueError as e:
            return self._send(413, str(e).encode(), 'text/plain; charset=utf-8')
        self._send(200, buf.getvalue(), 'image/png')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='alphamaker')
    ap.add_argument('files', nargs='+', help="afbeeldingen, of 'serve' voor de webpagina")
    ap.add_argument('-o', '--out', type=Path, help='uitvoermap (standaard naast het origineel)')
    ap.add_argument('--model', default=core.DEFAULT_MODEL, choices=[m for m in core.MODELS if m != 'vitmatte'])
    ap.add_argument('--snel', action='store_true', help='zonder randverfijning (sneller, minder fijn haar)')
    ap.add_argument('--port', type=int, default=8092)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args(argv)
    if a.files == ['serve']:
        Handler.model = a.model
        print(f'AlphaMaker op http://{a.host}:{a.port}  (Ctrl+C stopt)', flush=True)
        # één verzoek tegelijk: houdt het geheugengebruik voorspelbaar
        HTTPServer((a.host, a.port), Handler).serve_forever()
    for f in map(Path, a.files):
        dst = (a.out or f.parent) / (f.stem + '-alpha.png')
        dst.parent.mkdir(parents=True, exist_ok=True)
        core.cutout(Image.open(f), a.model, refine_edges=not a.snel).save(dst)
        print(dst, flush=True)


if __name__ == '__main__':
    sys.exit(main())
