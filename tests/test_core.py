"""Snelle tests zonder modellen: python tests/test_core.py (of pytest)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from alphamaker import core


def cut():  # rood vierkant (40x40 op 20,10) op transparant 100x80
    c = Image.new('RGBA', (100, 80), (0, 0, 0, 0)); c.paste((255, 0, 0, 255), (20, 10, 60, 50)); return c


def test_crop_and_margin():
    assert core.finish(cut(), crop=True).size == (40, 40)
    assert core.finish(cut(), crop=True, margin=0.1).size == (48, 48)


def test_alpha_only():
    m = core.finish(cut(), alpha_only=True)
    assert m.mode == 'L' and m.getpixel((30, 30)) == 255 and m.getpixel((0, 0)) == 0


def test_backgrounds_and_shadow():
    w = core.finish(cut(), bg='white')
    assert w.mode == 'RGB' and w.getpixel((0, 0)) == (255, 255, 255) and w.getpixel((30, 30)) == (255, 0, 0)
    s = core.finish(cut(), shadow=True)
    assert s.getpixel((30, 30))[:3] == (255, 0, 0) and s.getpixel((61, 51))[3] > 0
    b = core.finish(cut(), bg_image=Image.new('RGB', (10, 10), (0, 0, 255)))
    assert b.size == (100, 80) and b.getpixel((0, 0)) == (0, 0, 255)


def test_cli_type_kiest_model():
    import alphamaker.__main__ as cli
    seen = {}
    core_cutout, core.cutout = core.cutout, lambda img, model, **kw: (seen.setdefault('model', model), cut())[1]
    try:
        out = Path(__file__).resolve().parent / '_tmp.png'
        Image.new('RGB', (8, 8)).save(out)
        persoon = 'birefnet-general' if core.pt.available() else core.DEFAULT_MODEL
        core_kind, core.kind = core.kind, lambda img: 'persoon'  # nep-router: geen download in de test
        for t, expected in (('persoon', persoon), ('object', 'inspyrenet'), ('auto', persoon), (None, core.DEFAULT_MODEL),
                            ('--voorkeur=snel', core.DEFAULT_MODEL), ('--voorkeur=precies', persoon)):
            seen.clear()
            cli.main([str(out), '-o', str(out.parent)] + ([t] if t and t.startswith('--') else ['--type', t] if t else []))
            assert seen['model'] == expected, (t, seen)
        core.kind = lambda img: 'object'
        for t, expected in ((['--type', 'auto'], 'inspyrenet'), (['--voorkeur=precies'], persoon),
                            (['--voorkeur=precies', '--type', 'object'], 'inspyrenet'), (['--model', 'rmbg-1.4'], 'rmbg-1.4')):
            seen.clear()
            cli.main([str(out), '-o', str(out.parent)] + t)
            assert seen['model'] == expected, (t, seen)
        core.kind = core_kind
    finally:
        core.cutout = core_cutout
        for f in out.parent.glob('_tmp*.png'): f.unlink()


def test_load_exif_en_verkleinen():
    """Telefoonfoto met EXIF-draaiing komt rechtop binnen; --voorkeur snel verkleint tot ~3 MP."""
    import alphamaker.__main__ as cli
    f = Path(__file__).resolve().parent / '_tmp_exif.jpg'
    exif = Image.Exif(); exif[0x0112] = 6  # 90 graden gedraaid opgeslagen
    Image.new('RGB', (4000, 3000)).save(f, exif=exif)
    try:
        assert cli.load(f).size == (3000, 4000)
        w, h = cli.load(f, 3_000_000).size
        assert (w, h) == (1500, 2000), (w, h)
    finally:
        f.unlink()


def test_serve_api():
    """Server: modellenlijst, geldig model → PNG, onbekend model → 400, geen foto → 400 (nep-cutout, geen model)."""
    import io, json, threading, urllib.error, urllib.request
    from http.server import HTTPServer
    import alphamaker.__main__ as cli
    srv = HTTPServer(('127.0.0.1', 0), cli.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'
    core_cutout, core.cutout = core.cutout, lambda img, model, **kw: cut()
    try:
        info = json.load(urllib.request.urlopen(base + '/api/models'))
        assert info['default'] in info['models']
        buf = io.BytesIO(); Image.new('RGB', (8, 8)).save(buf, 'PNG')
        post = lambda q, data: urllib.request.urlopen(urllib.request.Request(base + '/api/cutout' + q, data=data, method='POST'))
        r = post('?model=' + info['models'][0], buf.getvalue())
        assert r.headers['Content-Type'] == 'image/png' and Image.open(io.BytesIO(r.read())).mode == 'RGBA'
        for q, data in (('?model=bestaat-niet', buf.getvalue()), ('', b'geen foto')):
            try: post(q, data); raise AssertionError('verwacht 400')
            except urllib.error.HTTPError as e: assert e.code == 400, e.code
    finally:
        core.cutout = core_cutout
        srv.shutdown()


def test_geen_achtergrond():  # close-up: object loopt het beeld uit
    import numpy as np
    a = np.zeros((100, 100), np.float32)
    a[15:85, 15:85] = 1                       # gewone productfoto: object met ruimte eromheen (fg 0,49)
    assert not core.geen_achtergrond(a)
    assert core.geen_achtergrond(np.ones_like(a))  # alles voorgrond
    b = np.zeros_like(a); b[:, :70] = 1       # loopt links, boven en onder het beeld uit (fg 0,7)
    assert core.geen_achtergrond(b)
    c = np.zeros_like(a); c[:, :20] = 1       # smal object langs de rand: te weinig voorgrond (fg 0,2)
    assert not core.geen_achtergrond(c)


def test_agree():  # keuzeregel objecten-precies: IoU ≥ 0,75 = eens
    import numpy as np
    a = np.zeros((10, 10), np.float32); a[2:8, 2:8] = 1
    b = a.copy(); b[2:8, 7] = 0  # IoU 30/36 ≈ 0,83
    assert core.agree(a, a) and core.agree(a, b)
    assert not core.agree(a, a * (np.arange(10) < 5)) and not core.agree(a, np.zeros_like(a))  # IoU 0,5 en 0



def test_objecten_precies():  # welk masker komt terug, met nepmodellen (geen downloads)
    import numpy as np
    seg, dm = core.segment, core.DEFAULT_MODEL
    a = np.zeros((10, 10), np.float32); a[2:8, 2:8] = 1
    fake = {'rmbg-1.4': a, 'isnet-general': a.copy(), 'inspyrenet': np.full_like(a, 0.5), 'birefnet-general': np.full_like(a, 0.4)}
    core.segment = lambda img, name: seg(img, name) if name == 'objecten-precies' else fake[name]
    try:
        assert core.segment(None, 'objecten-precies') is fake['rmbg-1.4']  # eens → rmbg-1.4
        fake['isnet-general'] = np.zeros_like(a)  # oneens → standaardmodel
        for core.DEFAULT_MODEL in ('inspyrenet', 'birefnet-general'):
            assert core.segment(None, 'objecten-precies') is fake[core.DEFAULT_MODEL], core.DEFAULT_MODEL
        core.DEFAULT_MODEL = 'isnet-general'  # terugval is altijd het standaardmodel, wat dat ook is
        assert core.segment(None, 'objecten-precies') is fake['isnet-general']
    finally:
        core.segment, core.DEFAULT_MODEL = seg, dm


def test_niet_commercieel_gemeld():  # eis: RMBG-1.4 e.d. alleen expliciet en altijd met melding
    import io, contextlib, alphamaker.__main__ as cli
    core_cutout, core.cutout = core.cutout, lambda img, model, **kw: cut()
    out = Path(__file__).resolve().parent / '_tmp.png'
    Image.new('RGB', (8, 8)).save(out)
    try:
        for args, nc in ((['--model', 'rmbg-1.4'], True), (['--model', 'objecten-precies'], True), ([], False),
                         (['--type', 'object'], False), (['--voorkeur', 'snel'], False)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err): cli.main([str(out), '-o', str(out.parent)] + args)
            assert ('niet commercieel' in err.getvalue()) == nc, (args, err.getvalue())
        assert core.DEFAULT_MODEL not in core.NONCOMMERCIAL
    finally:
        core.cutout = core_cutout
        for f in out.parent.glob('_tmp*.png'): f.unlink()


def test_auto_zonder_pytorch():  # zonder BiRefNet geen router (die zou niets veranderen), wel een melding
    import io, contextlib, alphamaker.__main__ as cli
    seen = {}
    def no_router(img): raise AssertionError('router aangeroepen')
    saved = core.cutout, core.kind, core.pt.available
    core.cutout = lambda img, model, **kw: (seen.setdefault('model', model), cut())[1]
    core.kind, core.pt.available = no_router, lambda: False
    out = Path(__file__).resolve().parent / '_tmp.png'
    Image.new('RGB', (8, 8)).save(out)
    try:
        for args in (['--type', 'auto'], ['--voorkeur', 'precies'], ['--type', 'persoon']):
            seen.clear(); err = io.StringIO()
            with contextlib.redirect_stderr(err): cli.main([str(out), '-o', str(out.parent)] + args)
            assert seen['model'] == core.DEFAULT_MODEL and 'PyTorch' in err.getvalue(), (args, seen, err.getvalue())
    finally:
        core.cutout, core.kind, core.pt.available = saved
        for f in out.parent.glob('_tmp*.png'): f.unlink()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'): fn(); print('ok', name)
