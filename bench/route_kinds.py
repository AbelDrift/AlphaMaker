"""Router van withoutBG (core.kind) per testbeeld: results/<set>/router.json = {beeld: 'persoon'|'object', ...} + tijd."""
import json, sys, time
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from alphamaker import core

for s in sys.argv[1:]:
    out, kinds, t = ROOT / f'results/{s}/router.json', {}, time.time()
    for p in sorted((ROOT / 'data' / s).glob('*.jpg')):
        kinds[p.stem] = core.kind(Image.open(p).convert('RGB'))
    sec = (time.time() - t) / max(len(kinds), 1)
    out.write_text(json.dumps({'sec_per_image': round(sec, 2), 'kinds': kinds}, indent=0))
    print(s, len(kinds), 'beelden,', sum(v == 'persoon' for v in kinds.values()), 'persoon,', f'{sec:.2f} s/beeld', flush=True)
