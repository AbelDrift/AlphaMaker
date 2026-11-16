"""Pariteitsfixtures voor de webversie (docs/refine.js): kleine willekeurige invoer (vast zaad, max 64x64)
met de uitvoer van alphamaker.refine en alphamaker.core (nep-sessies, geen modellen), scipy.ndimage en PIL.
Schrijft tests/web/fixtures.json. Draai: .venv/bin/python -I bench/make_web_fixtures.py"""
import base64, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, uniform_filter
from alphamaker import core, refine

rng = np.random.default_rng(20261116)
f32 = lambda a: base64.b64encode(np.ascontiguousarray(a, '<f4').tobytes()).decode()
u8 = lambda a: base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()
img = lambda h, w: rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def smooth_alpha(h, w, sigma=3):
    """Vloeiend masker met voorgrond, achtergrond en overgangen."""
    return ((gaussian_filter(rng.random((h, w)), sigma) - 0.5) * 40 + 0.5).clip(0, 1).astype(np.float32)


class FakeSeg:
    """Nep-segmentatiemodel, zelfde formule als in refine.test.mjs: A·sin(x0 + 2x1 + 3x2 + 0.3i + 0.1j) + B."""
    def __init__(self, size, amp, bias): self.size, self.amp, self.bias = size, amp, bias
    def get_inputs(self): return [type('Input', (), {'name': 'input', 'shape': [1, 3, self.size, self.size]})]
    def run(self, _, feeds):
        x = next(iter(feeds.values()))[0].astype(np.float64)
        i, j = np.mgrid[:self.size, :self.size]
        y = self.amp * np.sin(x[0] + 2 * x[1] + 3 * x[2] + 0.3 * i + 0.1 * j) + self.bias
        return [y[None, None].astype(np.float32)]


class FakeMatte:
    """Nep-ViTMatte: 0.5 + 0.25·sin(c0 + 2c1 + 3c2 + 4c3 + 0.1i + 0.2j + 0.01hp + 0.03wp)."""
    def run(self, _, feeds):
        x = feeds['pixel_values'][0].astype(np.float64)
        hp, wp = x.shape[1:]
        i, j = np.mgrid[:hp, :wp]
        a = 0.5 + 0.25 * np.sin(x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3] + 0.1 * i + 0.2 * j + 0.01 * hp + 0.03 * wp)
        return [a[None, None].astype(np.float32)]


def fake_sessions(seg):
    core.session = lambda name, *_: FakeMatte() if name.startswith('vitmatte') else seg


fx = {}

fx['box'] = []
for h, w in ((23, 37), (4, 3)):  # 4x3 met k=90: venster veel langer dan de lijn
    x = rng.random((h, w), dtype=np.float32)
    for k in (1, 2, 3, 6, 7, 90):
        fx['box'].append(dict(w=w, h=h, k=k, x=f32(x), out=f32(uniform_filter(x, size=k, mode='reflect'))))

m = gaussian_filter(rng.random((31, 40)), 1.5) > 0.5
m[0, :6] = m[:5, -1] = True  # pixels tegen de rand: scipy telt buiten het beeld als False
fx['morph'] = dict(w=40, h=31, mask=u8(m), cases=[
    dict(n=n, dil=u8(binary_dilation(m, iterations=n)), ero=u8(binary_erosion(m, iterations=n))) for n in (1, 2, 5)])

a = smooth_alpha(48, 40)
edge = np.zeros((9, 9), np.float32)
edge[4, 4] = np.float32(0.05)  # precies op de drempel
edge[1, 1] = np.float32(0.95)
fx['trimap'] = [dict(w=w, h=h, alpha=f32(al), cases=[dict(band=b, tri=f32(refine.trimap(al, b))) for b in (1, 3, 4)])
                for al in (a, edge) for h, w in [al.shape]]

fx['starts'] = [dict(n=n, tile=1024, overlap=128, out=refine._starts(n)) for n in (1, 500, 1024, 1025, 1500, 2001, 2047, 4000, 12345)]
fx['ramp'] = [dict(n=n, overlap=128, out=refine._ramp(n).tolist()) for n in (1, 7, 300)]

# ViTMatte: een schijf linksboven, zodat een deel van de tegels geen onzekere pixels heeft
h, w = 50, 41
yy, xx = np.mgrid[:h, :w]
tri = refine.trimap(np.clip(8 - np.hypot(yy - 10, xx - 10), 0, 1).astype(np.float32), 3)
rgb = img(h, w)
vm = []
for tile, ov in ((24, 8), (1024, 128)):
    refine.TILE, refine.OVERLAP = tile, ov
    vm.append(dict(tile=tile, overlap=ov, out=f32(refine.vitmatte(FakeMatte(), rgb.astype(np.float32) / 255, tri))))
    if tile == 24:
        fx['starts'] += [dict(n=n, tile=24, overlap=8, out=refine._starts(n)) for n in (5, 24, 25, 41, 50, 64)]
        fx['ramp'] += [dict(n=n, overlap=8, out=refine._ramp(n).tolist()) for n in (1, 5, 24)]
refine.TILE, refine.OVERLAP = 1024, 128
fx['vitmatte'] = dict(w=w, h=h, rgb=u8(rgb), tri=f32(tri), cases=vm)

fx['foreground'] = []
for (h, w), r in (((40, 33), (90, 6)), ((17, 9), (7, 4))):
    rgb, al = img(h, w), smooth_alpha(h, w, 2)
    F = refine.foreground(rgb.astype(np.float32) / 255, al, r)
    fx['foreground'].append(dict(w=w, h=h, ks=list(r), rgb=u8(rgb), alpha=f32(al), out=f32(F.transpose(2, 0, 1))))

fx['resize'] = []
sizes = [((37, 29), (16, 16)), ((11, 9), (16, 16)), ((37, 29), (50, 10)), ((37, 29), (37, 10))]
sizes += [(tuple(rng.integers(3, 40, 2)), tuple(rng.integers(3, 64, 2))) for _ in range(30)]  # beide pass-volgordes
for (sw, sh), (dw, dh) in [((int(a), int(b)), (int(c), int(d))) for (a, b), (c, d) in sizes]:
    rgb, y = img(sh, sw), rng.random((sh, sw), dtype=np.float32)
    fx['resize'].append(dict(sw=sw, sh=sh, dw=dw, dh=dh, rgb=u8(rgb), f=f32(y),
                             rgb_out=u8(np.asarray(Image.fromarray(rgb).resize((dw, dh), Image.BILINEAR))),
                             f_out=f32(np.asarray(Image.fromarray(y).resize((dw, dh), Image.BILINEAR)))))

# core.segment zelf, met een nep-sessie van invoermaat 16: zowel verkleinen als vergroten.
# BiRefNet/RMBG-2.0 lopen in Python via PyTorch; de webversie gebruikt hun ONNX-export met deze
# voorbewerking, dus toetsen we die met test-configuraties in ONNX-vorm (nooit een echt model).
core.MODELS['test-imnet-sigmoid'] = (None, 1024, core.IMNET, 'sigmoid', None)
core.MODELS['test-imnet-auto'] = (None, 1024, core.IMNET, 'auto', None)
fx['segment'] = []
for name, (w, h), amp, bias in (('rmbg-1.4', (37, 29), 3, 0), ('test-imnet-sigmoid', (11, 9), 6, 0),
                                ('test-imnet-auto', (37, 29), 6, 0), ('test-imnet-auto', (23, 31), 0.4, 0.5)):
    rgb = img(h, w)
    fake_sessions(FakeSeg(16, amp, bias))
    _, _, (mean, std), post, _ = core.MODELS[name]
    fx['segment'].append(dict(model=name, w=w, h=h, size=16, mean=list(mean), std=list(std), post=post, amp=amp, bias=bias,
                              rgb=u8(rgb), out=f32(core.segment(Image.fromarray(rgb), name))))

fx['cutout'] = []
for refine_edges in (False, True):
    w, h = 64, 56
    # gladde foto: het nepmasker volgt dan vooral de positie, met echte voor-/achtergrondzones
    rgb = (gaussian_filter(rng.random((h, w, 3)), (4, 4, 0)) * 255).astype(np.uint8)
    fake_sessions(FakeSeg(16, 6, 0))
    _, _, (mean, std), post, _ = core.MODELS['test-imnet-auto']
    out = np.asarray(core.cutout(Image.fromarray(rgb), 'test-imnet-auto', refine_edges=refine_edges))
    fx['cutout'].append(dict(refine=refine_edges, w=w, h=h, size=16, mean=list(mean), std=list(std), post=post,
                             amp=6, bias=0, rgb=u8(rgb), out=u8(out)))

# band uit core.cutout (staat daar inline): Python's round() rondt halven naar even af
fx['band'] = [dict(w=w, h=h, out=max(4, round(max(w, h) * 0.0125)))
              for w, h in ((45, 38), (1000, 800), (1040, 600), (1080, 300), (3000, 4000), (1400, 1400), (1800, 20), (200, 300))]

dst = ROOT / 'tests/web/fixtures.json'
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(fx))
print(dst, f'{dst.stat().st_size // 1024} KB')
