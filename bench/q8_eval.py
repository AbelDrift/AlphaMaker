"""int8-InSPyReNet (dynamisch gekwantiseerd uit de MIT-ONNX) tegen de gemeten fp32/torch-uitvoer.
Gebruik: python bench/q8_eval.py models/inspyrenet-plus-ultra-q8.onnx test_syn2 test_clean2"""
import json, sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
def metrics(pred, gt):  # = bench/bench.py metrics()
    hard = gt > 0.5
    band = binary_dilation(hard, iterations=8) & binary_dilation(~hard, iterations=8)
    edge = ((gt > 0.02) & (gt < 0.98)) | band
    p = pred > 0.5
    return float(np.abs(pred - gt)[edge].mean()) if edge.any() else 0.0, float((p & hard).sum() / max((p | hard).sum(), 1))

so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
s = ort.InferenceSession(sys.argv[1], so, providers=['CPUExecutionProvider'])
mean, std = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)
for st in sys.argv[2:]:
    ref = {r['img']: r for r in map(json.loads, open(ROOT / f'results/{st}/inspyrenet.jsonl'))}
    rows = []
    for f in sorted((ROOT / 'data' / st).glob('*.jpg')):
        img = Image.open(f).convert('RGB')
        gt = np.asarray(Image.open(f.with_name(f.stem + '_gt.png')).convert('L'), np.float32) / 255
        x = ((np.asarray(img.resize((1024, 1024), Image.BILINEAR), np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
        t = time.time(); y = s.run(None, {'input': x})[0][0, 0]; dt = time.time() - t
        a = np.asarray(Image.fromarray(y.astype(np.float32)).resize(img.size, Image.BILINEAR)).clip(0, 1)
        e, iou = metrics(a, gt)
        fp = np.asarray(Image.open(ROOT / f'preds/{st}/inspyrenet/{f.stem}.png').convert('L'), np.float32) / 255
        same = float(((a > .5) & (fp > .5)).sum() / max(((a > .5) | (fp > .5)).sum(), 1))
        rows.append((f.stem, e, iou, dt, same, ref[f.stem]['edge_mae'], ref[f.stem]['iou']))
    for grp in ('dis', 'ppm', 'syn'):
        g = [r for r in rows if r[0].startswith(grp)]
        if g:
            print(f'{st} {grp} n={len(g)}: q8 rand {np.mean([r[1] for r in g]):.4f} missers {sum(r[2] < .8 for r in g)} | '
                  f'fp32 rand {np.mean([r[5] for r in g]):.4f} missers {sum(r[6] < .8 for r in g)} | '
                  f'IoU q8~fp32 gem {np.mean([r[4] for r in g]):.4f} min {min(r[4] for r in g):.3f} | {np.mean([r[3] for r in g]):.1f} s', flush=True)
