"""Meet verfijning bovenop opgeslagen maskers. Gebruik: refine_bench.py <basismodel> <variant>
variant: vm<band> (ViTMatte fp32, band px), vq<band> (ViTMatte gekwantiseerd), gf<r> (guided filter), of combinatie 'vm16+gf4'."""
import json, re, sys, time, resource
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault('ALPHAMAKER_MODELS', str(ROOT / 'models'))
from alphamaker import core
from alphamaker.refine import trimap, vitmatte, guided_filter
sys.path.insert(0, str(ROOT / "bench"))
from bench import metrics  # bench/bench.py

base, variant = sys.argv[1], sys.argv[2]
name = f'{base}+{variant}'
out = ROOT / 'results' / (name + '.jsonl')
done = {json.loads(l)['img'] for l in out.read_text().splitlines()} if out.exists() else set()
for p in sorted((ROOT / 'data/test').glob('*.jpg')):
    if p.stem in done: continue
    rgb = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255
    gt = np.asarray(Image.open(p.with_name(p.stem + '_gt.png')).convert('L'), np.float32) / 255
    a = np.asarray(Image.open(ROOT / 'preds' / base / (p.stem + '.png')), np.float32) / 255
    t = time.time()
    for step in variant.split('+'):
        kind, k = re.fullmatch(r'(vm|vq|gf)(\d+)', step).groups()
        if kind in ('vm', 'vq'):
            a = vitmatte(core.session('vitmatte' if kind == 'vm' else 'vitmatte-q'), rgb, trimap(a, int(k)))
        else:
            a = guided_filter(rgb, a, int(k))
    dt = time.time() - t
    mae, emae, iou = metrics(a, gt)
    row = dict(img=p.stem, mae=mae, edge_mae=emae, iou=iou, sec=dt,
               peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    with out.open('a') as fh: fh.write(json.dumps(row) + '\n')
    print(row, flush=True)
(ROOT / 'results' / (name + '.status')).write_text('ok\n')
