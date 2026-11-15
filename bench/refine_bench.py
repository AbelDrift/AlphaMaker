"""Meet verfijning bovenop opgeslagen maskers. Gebruik: refine_bench.py <basismodel> <variant>
variant: vm<band> (ViTMatte, band px), gf<r> (guided filter), of combinatie 'vm16+gf4'."""
import json, re, sys, time, resource
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from alphamaker.refine import trimap, vitmatte, guided_filter
from bench.bench import metrics

base, variant = sys.argv[1], sys.argv[2]
name = f'{base}+{variant}'
out = ROOT / 'results' / (name + '.jsonl')
done = {json.loads(l)['img'] for l in out.read_text().splitlines()} if out.exists() else set()
sess = None
for p in sorted((ROOT / 'data/test').glob('*.jpg')):
    if p.stem in done: continue
    rgb = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255
    gt = np.asarray(Image.open(p.with_name(p.stem + '_gt.png')).convert('L'), np.float32) / 255
    a = np.asarray(Image.open(ROOT / 'preds' / base / (p.stem + '.png')), np.float32) / 255
    t = time.time()
    for step in variant.split('+'):
        kind, k = re.fullmatch(r'(vm|gf)(\d+)', step).groups()
        if kind == 'vm':
            if sess is None:
                import onnxruntime as ort
                so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.enable_cpu_mem_arena = False
                sess = ort.InferenceSession(str(ROOT / 'models/vitmatte-small-c1k.onnx'), so, providers=['CPUExecutionProvider'])
            a = vitmatte(sess, rgb, trimap(a, int(k)))
        else:
            a = guided_filter(rgb, a, int(k))
    dt = time.time() - t
    mae, emae, iou = metrics(a, gt)
    row = dict(img=p.stem, mae=mae, edge_mae=emae, iou=iou, sec=dt,
               peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    with out.open('a') as fh: fh.write(json.dumps(row) + '\n')
    print(row, flush=True)
(ROOT / 'results' / (name + '.status')).write_text('ok\n')
