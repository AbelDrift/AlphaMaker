"""Kan rmbg-1.4 zelf (zonder isnet) zien dat een foto druk is? Signalen uit zijn eigen masker als keuzeregel
'signaal > q -> inspyrenet', naast de isnet-regel. Groepen zoals bench/route_study.py; syn2 = controle."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
res = lambda s, m: {r['img']: r for r in map(json.loads, open(ROOT / f'results/{s}/{m}.jsonl'))}
alpha = lambda s, m, n: np.asarray(Image.open(ROOT / f'preds/{s}/{m}/{n}.png').convert('L'), np.float32) / 255

rows = []
for s in ('test_clean', 'test_clean2', 'test_syn', 'test_syn2'):
    R = {m: res(s, m) for m in ('rmbg-1.4', 'isnet-general', 'inspyrenet')}
    for n in R['rmbg-1.4']:
        a, b = alpha(s, 'rmbg-1.4', n), alpha(s, 'isnet-general', n)
        fg = a > 0.5
        lab, k = ndimage.label(fg)
        sizes = np.bincount(lab.ravel())[1:] if k else np.array([0])
        edge = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
        rows.append(dict(set=s[5:] if s.startswith('test_syn') else 'dis' if n.startswith('dis') else 'ppm',
                         iou=((fg & (b > .5)).sum() / max((fg | (b > .5)).sum(), 1)),
                         soft=((a > .05) & (a < .95)).sum() / max(fg.sum(), 1),   # onzekere rand t.o.v. onderwerp
                         area=fg.mean(),                                           # deel van het beeld voorgrond
                         comps=int((sizes > 0.01 * max(fg.sum(), 1)).sum()),       # stukken van >1% van de voorgrond
                         border=edge.mean(),                                       # deel van de beeldrand voorgrond
                         **{m: R[m][n] for m in R}))

G = ('dis', 'ppm', 'syn', 'syn2')
sets = {g: [r for r in rows if r['set'] == g] for g in G}
def stats(rs, pick):
    return f"{np.mean([r[pick(r)]['edge_mae'] for r in rs]):.4f}/{sum(r[pick(r)]['iou'] < 0.8 for r in rs):2d}"
def line(name, pick):
    n = '/'.join(f"{sum(pick(r) != 'rmbg-1.4' for r in sets[g]):2d}" for g in G)
    print(f"{name:30}", *(f"{stats(sets[g], pick):>12}" for g in G), ' omgeleid', n)
print({g: len(v) for g, v in sets.items()}, ' cel = rand-MAE/missers; DIS hier met besmette inspyrenet-scores')
print(f"{'regel':30}", *(f'{g:>12}' for g in G))
line('altijd rmbg-1.4', lambda r: 'rmbg-1.4')
line('IoU(rmbg,isnet) < 0.75', lambda r: 'inspyrenet' if r['iou'] < 0.75 else 'rmbg-1.4')
for key, qs in (('soft', (0.15, 0.25, 0.35, 0.5)), ('area', (0.3, 0.4, 0.5, 0.6)),
                ('comps', (1, 2, 3, 5)), ('border', (0.1, 0.2, 0.3, 0.4))):
    for q in qs:
        line(f'{key} > {q}', lambda r, key=key, q=q: 'inspyrenet' if r[key] > q else 'rmbg-1.4')
