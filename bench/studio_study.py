"""Nagebootste studiofoto's: object (met het echte masker) op een egale, licht verlopende achtergrond met ruis.
Vraag: is U²-Net (Apache-2.0, snel) daar ongeveer zo goed als IS-Net? InSPyReNet (medium) ter vergelijking.
Gebruik: python bench/studio_study.py   (schrijft niets weg; meet met dezelfde maat als bench/bench.py)"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from alphamaker import core

def metrics(pred, gt):  # = bench/bench.py metrics(): rand-MAE en IoU
    hard = gt > 0.5
    band = binary_dilation(hard, iterations=8) & binary_dilation(~hard, iterations=8)
    edge = ((gt > 0.02) & (gt < 0.98)) | band
    p = pred > 0.5
    return float(np.abs(pred - gt)[edge].mean()) if edge.any() else 0.0, float((p & hard).sum() / max((p | hard).sum(), 1))

rng = np.random.default_rng(0)
def studio(img, gt, kleur):  # egale achtergrond: verloop van 6% van boven naar onder, ruis sigma 2
    h, w = gt.shape
    bg = np.array(kleur, np.float32)[None, None] * np.linspace(1.0, 0.94, h, dtype=np.float32)[:, None, None]
    bg = np.clip(bg + rng.normal(0, 2, (h, w, 1)).astype(np.float32), 0, 255)
    fg = np.asarray(img, np.float32)
    return Image.fromarray((fg * gt[..., None] + bg * (1 - gt[..., None])).astype(np.uint8))

MODELLEN = sys.argv[1:] or ['u2net', 'isnet-general', 'inspyrenet']
GROEPEN = {'alledaags (S3OD)': [p for s in ('test_syn', 'test_syn2') for p in sorted((ROOT / 'data' / s).glob('*.jpg'))],
           'DIS-objecten': [p for s in ('test_clean', 'test_clean2') for p in sorted((ROOT / 'data' / s).glob('dis_*.jpg'))]}
for achtergrond, kleur in (('wit', (245, 245, 245)), ('grijs', (150, 150, 150))):
    for groep, paden in GROEPEN.items():
        if achtergrond == 'grijs' and groep == 'DIS-objecten': continue  # grijs alleen voor alledaagse objecten
        uit = {m: [] for m in MODELLEN}
        for p in paden:
            img = Image.open(p).convert('RGB')
            gt = np.asarray(Image.open(p.with_name(p.stem + '_gt.png')).convert('L'), np.float32) / 255
            comp = studio(img, gt, kleur)
            for m in MODELLEN:
                if m == 'inspyrenet' and groep == 'DIS-objecten': continue  # besmet op DIS, en traag
                t = time.time(); a = core.segment(comp, m); dt = time.time() - t
                uit[m].append((*metrics(a, gt), dt))
        cel = ' | '.join(f"{m} {np.mean([r[0] for r in v]):.3f}/{sum(r[1] < .8 for r in v):2d} ({np.mean([r[2] for r in v]):.1f} s)"
                         for m, v in uit.items() if v)
        print(f"{achtergrond:5} {groep:17} n={len(paden):3d} | {cel}", flush=True)
