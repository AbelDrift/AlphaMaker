"""Offline toets van een keuzeregel voor objecten, zonder modellen te draaien:
rmbg-1.4, tenzij rmbg-1.4 en isnet-general het oneens zijn (IoU van de maskers < t) -> inspyrenet.
Gebruikt de bewaarde maskers in preds/ en de scores per foto in results/.
DIS-foto's: inspyrenet (Plus Ultra) is daarop getraind, dus ook 'schoon' = scores van inspyrenet-dis5k."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
res = lambda s, m: {r['img']: r for r in map(json.loads, open(ROOT / f'results/{s}/{m}.jsonl'))}
mask = lambda s, m, n: np.asarray(Image.open(ROOT / f'preds/{s}/{m}/{n}.png').convert('L')) > 127

rows = []
for s in ('test_clean', 'test_clean2', 'test_syn'):
    R = {m: res(s, m) for m in ('rmbg-1.4', 'isnet-general', 'inspyrenet')}
    R['schoon'] = R['inspyrenet'] if s == 'test_syn' else {n: (r if n.startswith('dis') else R['inspyrenet'][n]) for n, r in res(s, 'inspyrenet-dis5k').items()}  # alleen DIS is besmet
    for n in R['rmbg-1.4']:
        a, b = mask(s, 'rmbg-1.4', n), mask(s, 'isnet-general', n)
        assert a.shape == b.shape, (s, n, a.shape, b.shape)
        rows.append(dict(set='syn' if s == 'test_syn' else 'dis' if n.startswith('dis') else 'ppm', img=n,
                         iou=(a & b).sum() / max((a | b).sum(), 1), ratio=a.sum() / max(b.sum(), 1),
                         **{m: R[m][n] for m in R}))

def stats(rs, pick):
    e = [r[pick(r)]['edge_mae'] for r in rs]
    return f"{np.mean(e):.4f}/{sum(r[pick(r)]['iou'] < 0.8 for r in rs):2d}"

sets = {k: [r for r in rows if r['set'] == k] for k in ('dis', 'ppm', 'syn')}
print({k: len(v) for k, v in sets.items()}, '  cel = rand-MAE/missers(IoU<0,8)')
print(f"{'regel':34} {'DIS (besmet)':>13} {'DIS (schoon)':>13} {'portret':>13} {'syn':>13}  naar 2e model dis/ppm/syn")
def line(name, pick):
    clean = lambda r: 'schoon' if pick(r) == 'inspyrenet' else pick(r)
    n = [sum(pick(r) != 'rmbg-1.4' for r in sets[k]) for k in ('dis', 'ppm', 'syn')]
    print(f"{name:34} {stats(sets['dis'], pick):>13} {stats(sets['dis'], clean):>13} {stats(sets['ppm'], pick):>13} {stats(sets['syn'], pick):>13}  {n[0]:3d}/{n[1]:3d}/{n[2]:3d}")
for m in ('rmbg-1.4', 'isnet-general', 'inspyrenet'):
    line(f'altijd {m}', lambda r, m=m: m)
for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
    line(f'IoU(rmbg,isnet) < {t} -> insp.', lambda r, t=t: 'inspyrenet' if r['iou'] < t else 'rmbg-1.4')
for t in (0.6, 0.7, 0.8):
    line(f'IoU < {t} -> isnet (zonder torch)', lambda r, t=t: 'isnet-general' if r['iou'] < t else 'rmbg-1.4')
line('orakel (beste van rmbg/insp.)', lambda r: min(('rmbg-1.4', 'inspyrenet'), key=lambda m: r[m]['edge_mae']))
