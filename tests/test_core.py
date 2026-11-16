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


def test_cli_type_persoon_kiest_haarmodel():
    import alphamaker.__main__ as cli
    seen = {}
    core_cutout, core.cutout = core.cutout, lambda img, model, **kw: (seen.setdefault('model', model), cut())[1]
    try:
        out = Path(__file__).resolve().parent / '_tmp.png'
        Image.new('RGB', (8, 8)).save(out)
        cli.main([str(out), '--type', 'persoon', '-o', str(out.parent)])
        expected = 'birefnet-general' if core.pt.available() else core.DEFAULT_MODEL
        assert seen['model'] == expected, seen
    finally:
        core.cutout = core_cutout
        for f in out.parent.glob('_tmp*.png'): f.unlink()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'): fn(); print('ok', name)
