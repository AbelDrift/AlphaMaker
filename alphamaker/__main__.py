"""Gebruik:
  python -m alphamaker foto.jpg [meer.jpg ...] [-o map] [--model NAAM] [--fijne-randen]
      [--bijsnijden [--marge 0.05]] [--alleen-masker] [--schaduw] [--achtergrond '#fff' | --achtergrondfoto pad]
      [--formaat png|jpg|webp]
  python -m alphamaker serve [--port 8092] [--host 127.0.0.1]"""
import argparse, io, json, sys
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from PIL import Image, ImageOps
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
            return self._send(200, json.dumps({'models': core.SEGMENTERS, 'default': self.model, 'noncommercial': sorted(core.NONCOMMERCIAL)}).encode(), 'application/json')
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
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(self.rfile.read(n))))
            img.load()
        except Exception:
            return self._send(400, b'geen geldige afbeelding', 'text/plain')
        buf = io.BytesIO()
        try:
            core.cutout(img, model, refine_edges='fijn' in q).save(buf, 'PNG')
        except ValueError as e:
            return self._send(413, str(e).encode(), 'text/plain; charset=utf-8')
        self._send(200, buf.getvalue(), 'image/png')


def load(f, max_px=None):
    """Foto openen, rechtop gezet volgens EXIF (telefoons); optioneel verkleinen tot max_px pixels."""
    img = ImageOps.exif_transpose(Image.open(f))
    if max_px and img.width * img.height > max_px:
        s = (max_px / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(prog='alphamaker')
    ap.add_argument('files', nargs='+', help="afbeeldingen, of 'serve' voor de webpagina")
    ap.add_argument('-o', '--out', type=Path, help='uitvoermap (standaard naast het origineel)')
    ap.add_argument('--model', choices=core.SEGMENTERS,
                    help=f'één model voor alle foto\'s (standaard: {core.DEFAULT_MODEL}, MIT). NIET voor commercieel gebruik: '
                         + ', '.join(sorted(core.NONCOMMERCIAL)))
    ap.add_argument('--type', choices=['auto', 'persoon', 'object'],
                    help='zoals remove.bg type: persoon → birefnet-general (MIT; beste voor haar, traag), '
                         'object → inspyrenet (MIT, de standaard), auto → per foto kiezen met de router van withoutBG: '
                         'haar → persoon, anders object (350 MB extra download; de router valt onder Meta\'s DINOv3-licentie, '
                         'met gebruiksbeperkingen zoals geen militair gebruik)')
    ap.add_argument('--voorkeur', choices=['snel', 'precies'],
                    help='snel: foto\'s boven 3 MP eerst verkleinen (zoals 50%% bij een telefoonfoto; zelfde model); '
                         'precies: birefnet-general (MIT; beste op haar en drukke foto\'s, traag), op volle resolutie')
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
    by_type = {'persoon': 'birefnet-general' if core.pt.available() else None, 'object': 'inspyrenet'}
    if a.voorkeur == 'precies' and not a.type: a.model = a.model or by_type['persoon']  # zonder torch: standaard
    def warn(models):  # niet-commerciële modellen altijd uitdrukkelijk benoemen
        for m in sorted(set(models) & core.NONCOMMERCIAL.keys()):
            print(f'Let op: {m} mag niet commercieel gebruikt worden ({core.NONCOMMERCIAL[m]}), '
                  'dus niet voor productfoto\'s voor verkoop.', file=sys.stderr)
    routed = a.type == 'auto' and not a.model and core.pt.available()  # zonder BiRefNet verandert de router niets
    if not (a.model or core.pt.available()) and (a.type in ('auto', 'persoon') or a.voorkeur == 'precies'):
        print("Let op: BiRefNet heeft PyTorch nodig (pip install -e '.[beste]'); nu wordt overal inspyrenet gebruikt.", file=sys.stderr)
    a.model = a.model or by_type.get(a.type) or core.DEFAULT_MODEL
    if a.files == ['serve']:
        Handler.model = a.model; warn([a.model])
        print(f'AlphaMaker op http://{a.host}:{a.port}  (Ctrl+C stopt)', flush=True)
        # één verzoek tegelijk: houdt het geheugengebruik voorspelbaar
        HTTPServer((a.host, a.port), Handler).serve_forever()
    bg = a.achtergrond or ('white' if a.formaat == 'jpg' and not a.achtergrondfoto and not a.alleen_masker else None)
    bg_image = load(a.achtergrondfoto) if a.achtergrondfoto else None
    files = list(map(Path, a.files))
    kinds = {}
    if routed:  # eerst alles routeren, dan de router vrijgeven (geheugen)
        kinds = {f: core.kind(load(f)) for f in files}
        core._sessions.pop('wbg-router', None)
    warn({by_type.get(kinds.get(f)) or a.model for f in files})
    if routed: print('Let op: de router van withoutBG valt onder Meta\'s DINOv3-licentie, met gebruiksbeperkingen '
                     '(zoals geen militair gebruik).', file=sys.stderr)
    for n, f in enumerate(files, 1):
        dst = (a.out or f.parent) / f'{f.stem}-alpha.{a.formaat}'
        dst.parent.mkdir(parents=True, exist_ok=True)
        model = by_type.get(kinds.get(f)) or a.model
        cut = core.cutout(load(f, 3_000_000 if a.voorkeur == 'snel' else None), model, refine_edges=a.fijne_randen)
        core.finish(cut, a.bijsnijden, a.marge, a.alleen_masker, bg, bg_image, a.schaduw).save(dst, quality=95)
        info = f'  ({kinds[f]} → {model})' if f in kinds else ''
        print(f'[{n}/{len(files)}] {dst}{info}' if len(files) > 1 else f'{dst}{info}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
