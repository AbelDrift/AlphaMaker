"""Standaardmodus per foto (voorstel) tegen de huidige standaard (altijd inspyrenet):
router (core.kind, results/<set>/router.json) zegt haar -> inspyrenet; anders objecten-precies
(rmbg-1.4, bij IoU(rmbg,isnet) < 0,75 inspyrenet). Tijd = som van de gemeten s/beeld per stap.
DIS: inspyrenet is daarop getraind; 'schoon' rekent omgeleide DIS-foto's met inspyrenet-dis5k."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
res = lambda s, m: {r['img']: r for r in map(json.loads, open(ROOT / f'results/{s}/{m}.jsonl'))}
fg = lambda s, m, n: np.asarray(Image.open(ROOT / f'preds/{s}/{m}/{n}.png').convert('L')) > 127
rows = []
for s in ('test_clean', 'test_clean2', 'test_syn', 'test_syn2'):
    R = {m: res(s, m) for m in ('rmbg-1.4', 'isnet-general', 'inspyrenet')}
    clean = res(s, 'inspyrenet-dis5k') if not s.startswith('test_syn') else R['inspyrenet']
    route = json.loads((ROOT / f'results/{s}/router.json').read_text())
    for n in R['rmbg-1.4']:
        a, b = fg(s, 'rmbg-1.4', n), fg(s, 'isnet-general', n)
        rows.append(dict(g=s[5:] if s.startswith('test_syn') else 'dis' if n.startswith('dis') else 'ppm',
                         kind=route['kinds'][n], router=route['sec_per_image'], iou=(a & b).sum() / max((a | b).sum(), 1),
                         schoon=clean[n] if n.startswith('dis') else R['inspyrenet'][n], **{m: R[m][n] for m in R}))

def nieuw(r):  # welk model levert het masker, en welke stappen kosten tijd
    if r['kind'] == 'persoon': return 'inspyrenet', r['router'] + r['inspyrenet']['sec']
    t = r['router'] + r['rmbg-1.4']['sec'] + r['isnet-general']['sec']
    return ('inspyrenet', t + r['inspyrenet']['sec']) if r['iou'] < 0.75 else ('rmbg-1.4', t)
nu = lambda r: ('inspyrenet', r['inspyrenet']['sec'])

G = ('dis', 'ppm', 'syn', 'syn2')
print({g: sum(r['g'] == g for r in rows) for g in G}, ' cel = rand-MAE/missers (s/foto)')
for name, pick in (('nu: altijd inspyrenet', nu), ('voorstel: router + objecten-precies', nieuw)):
    cells = []
    for g in G:
        rs = [r for r in rows if r['g'] == g]
        m = [pick(r)[0] for r in rs]
        e = [(r['schoon'] if (g == 'dis' and k == 'inspyrenet') else r[k]) for r, k in zip(rs, m)]
        cells.append(f"{np.mean([x['edge_mae'] for x in e]):.4f}/{sum(x['iou'] < 0.8 for x in e):2d} ({np.mean([pick(r)[1] for r in rs]):4.1f} s)")
    print(f'{name:36}', *cells)
print('voorstel naar inspyrenet per groep:', {g: sum(nieuw(r)[0] == 'inspyrenet' for r in rows if r['g'] == g) for g in G})
