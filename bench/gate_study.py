"""Kan het snelle model zelf zien wanneer het twijfelt? Per foto: IoU (tegen leermeester/GT) en twee twijfelsignalen.
Gebruik: python bench/gate_study.py LICHT.onnx [VALMAP]  → bench/gate_study.json + tabel: hoeveel missers vang je bij x% terugval."""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'train'))
import kit
import onnxruntime as ort

so = ort.SessionOptions(); so.intra_op_num_threads = 4
s = ort.InferenceSession(sys.argv[1], so, providers=['CPUExecutionProvider'])
sets = []
if len(sys.argv) > 2:
    d = Path(sys.argv[2])
    sets.append(('eigen', [(d / 'images' / n, d / 'masks' / (n[:-4] + '.png')) for n in (d / 'val.txt').read_text().split()]))
for t in ('test_syn2', 'test_clean2'):
    sets.append((t, [(p, p.with_name(p.stem + '_gt.png')) for p in sorted((kit.ROOT / 'data' / t).glob('*.jpg'))]))


def iou(a, b):
    a, b = a > .5, b > .5
    return float((a & b).sum() / max((a | b).sum(), 1))


rows = []
for naam, paren in sets:
    for img, gtp in paren:
        im = Image.open(img).convert('RGB')
        x = ((np.asarray(im.resize((1024, 1024), Image.BILINEAR), np.float32) / 255 - kit.MEAN) / kit.STD).transpose(2, 0, 1)[None]
        t0 = time.time()
        y = s.run(None, {'input': x})[0][0, 0]
        yf = s.run(None, {'input': x[..., ::-1].copy()})[0][0, 0, :, ::-1]  # gespiegeld
        dt = time.time() - t0
        gt = np.asarray(Image.open(gtp).convert('L').resize((1024, 1024), Image.BILINEAR), np.float32) / 255
        fg = max((y > .5).sum(), 1)
        rows.append({'set': naam, 'img': img.name, 'iou': iou(y, gt), 'soft': float(((y > .1) & (y < .9)).sum() / fg),
                     'flip': 1 - iou(y, yf), 's': dt})
    print(naam, 'klaar', flush=True)
Path(__file__).with_name('gate_study.json').write_text(json.dumps(rows))

for sig in ('soft', 'flip'):
    for naam in dict.fromkeys(r['set'] for r in rows):
        r = [x for x in rows if x['set'] == naam]
        miss = sum(x['iou'] < .8 for x in r)
        order = sorted(r, key=lambda x: -x[sig])
        out = []
        for frac in (.1, .2, .3, .5):
            k = round(frac * len(r))
            out.append(f'{frac:.0%} terug: {sum(x["iou"] < .8 for x in order[:k])}/{miss}')
        print(f'{sig:5} {naam:12} n={len(r):3d}  ' + '  '.join(out))
