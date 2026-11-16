"""Meet het gemiddelde van opgeslagen maskers van meerdere modellen (geen modelberekening nodig).
Gebruik: ensemble.py model1 model2 [...]  → results/ens(model1,model2).jsonl"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from bench import metrics, DATA, RES, PRED  # bench/bench.py (volgt TESTSET)

names = sys.argv[1:]
assert len(names) >= 2, 'geef minstens twee modellen'
tag = 'ens(' + ','.join(names) + ')'
out = RES / (tag + '.jsonl')
pd = PRED / tag
pd.mkdir(parents=True, exist_ok=True)
rows = []
# tijd = som van de losse modellen, geheugen = grootste piek (ze draaien na elkaar)
comp = {n: {json.loads(l)['img']: json.loads(l) for l in (RES / (n + '.jsonl')).read_text().splitlines()} for n in names}
for p in sorted(DATA.glob('*.jpg')):
    gt = np.asarray(Image.open(p.with_name(p.stem + '_gt.png')).convert('L'), np.float32) / 255
    a = np.mean([np.asarray(Image.open(PRED / n / (p.stem + '.png')), np.float32) / 255 for n in names], 0)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(pd / (p.stem + '.png'))
    mae, emae, iou = metrics(a, gt)
    rows.append(dict(img=p.stem, mae=mae, edge_mae=emae, iou=iou,
                     sec=sum(comp[n][p.stem]['sec'] for n in names), peak_mb=max(comp[n][p.stem]['peak_mb'] for n in names)))
out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
(RES / (tag + '.status')).write_text('ok (ensemble)\n')
print(tag, 'MAE', np.mean([r['mae'] for r in rows]).round(4), 'rand-MAE', np.mean([r['edge_mae'] for r in rows]).round(4))
