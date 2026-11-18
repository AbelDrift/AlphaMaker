"""Commercieel toegestane keuzeregel (voor SellYourShit): isnet-general (Apache-2.0), tenzij isnet en u2net
(Apache-2.0) het oneens zijn -> inspyrenet (MIT). Naast de losse modellen en de niet-commerciele rmbg-regel.
DIS: inspyrenet is daarop getraind; kolom 'dis schoon' telt omgeleide DIS-foto's met inspyrenet-dis5k."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
res = lambda s, m: {r['img']: r for r in map(json.loads, open(ROOT / f'results/{s}/{m}.jsonl'))}
fg = lambda s, m, n: np.asarray(Image.open(ROOT / f'preds/{s}/{m}/{n}.png').convert('L')) > 127
iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)

rows = []
for s in ('test_clean', 'test_clean2', 'test_syn', 'test_syn2'):
    R = {m: res(s, m) for m in ('rmbg-1.4', 'isnet-general', 'inspyrenet', 'u2net')}
    clean = res(s, 'inspyrenet-dis5k') if not s.startswith('test_syn') else R['inspyrenet']
    for n in R['isnet-general']:
        i, u, r = fg(s, 'isnet-general', n), fg(s, 'u2net', n), fg(s, 'rmbg-1.4', n)
        rows.append(dict(set=s[5:] if s.startswith('test_syn') else 'dis' if n.startswith('dis') else 'ppm',
                         iu=iou(i, u), ri=iou(r, i), schoon=clean[n] if n.startswith('dis') else R['inspyrenet'][n],
                         **{m: R[m][n] for m in R}))

G = ('dis', 'ppm', 'syn', 'syn2')
sets = {g: [r for r in rows if r['set'] == g] for g in G}
def stats(rs, pick):
    return f"{np.mean([r[pick(r)]['edge_mae'] for r in rs]):.4f}/{sum(r[pick(r)]['iou'] < 0.8 for r in rs):2d}"
def line(name, pick, base):
    clean = lambda r: 'schoon' if pick(r) == 'inspyrenet' else pick(r)
    n = '/'.join(f"{sum(pick(r) != base for r in sets[g]):2d}" for g in G)
    print(f"{name:32}", f"{stats(sets['dis'], clean):>12}", *(f"{stats(sets[g], pick):>12}" for g in G[1:]), ' omgeleid', n)
print({g: len(v) for g, v in sets.items()}, ' cel = rand-MAE/missers; dis met schone inspyrenet-scores')
print(f"{'regel':32}", *(f'{g:>12}' for g in G))
for m in ('u2net', 'isnet-general', 'inspyrenet', 'rmbg-1.4'):
    line(f'altijd {m}', lambda r, m=m: m, m)
for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
    line(f'IoU(isnet,u2net) < {t} -> insp.', lambda r, t=t: 'inspyrenet' if r['iu'] < t else 'isnet-general', 'isnet-general')
line('NC: IoU(rmbg,isnet) < 0.75', lambda r: 'inspyrenet' if r['ri'] < 0.75 else 'rmbg-1.4', 'rmbg-1.4')
