"""Vat results/*.jsonl samen tot results/SUMMARY.md (gesorteerd op rand-MAE, lager = beter)."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / 'results'
rows = []
for f in sorted(R.glob('*.jsonl')):
    d = [json.loads(l) for l in f.read_text().splitlines()]
    if not d: continue
    m = lambda k, pre='': np.mean([x[k] for x in d if x['img'].startswith(pre)] or [np.nan])
    st = (R / (f.stem + '.status')).read_text().strip() if (R / (f.stem + '.status')).exists() else 'bezig'
    rows.append((f.stem, len(d), m('mae'), m('edge_mae'), m('edge_mae', 'dis_'), m('edge_mae', 'ppm_'),
                 m('iou'), m('sec'), max(x['peak_mb'] for x in d), st))
rows.sort(key=lambda r: r[3])
out = ['| model | n | MAE | rand-MAE | rand DIS | rand PPM (haar) | IoU | s/beeld | piek MB | status |',
       '|---|---|---|---|---|---|---|---|---|---|']
out += [f'| {r[0]} | {r[1]} | {r[2]:.4f} | **{r[3]:.4f}** | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.3f} | {r[7]:.1f} | {r[8]} | {r[9]} |' for r in rows]
(R / 'SUMMARY.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
