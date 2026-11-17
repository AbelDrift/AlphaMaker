"""Controleset data/test_syn2: 60 S3OD-beelden (okupyn/s3od_dataset, MIT, FLUX 2025) die niet in test_syn zitten.
Voor een eerlijke toets van keuzeregels die op test_syn zijn afgesteld. 10 plekken × 6 via datasets-server."""
import io, json, re, urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data/test_syn2'
API = 'https://datasets-server.huggingface.co/rows?dataset=okupyn/s3od_dataset&config=default&split=train'
ASSETS = 'https://datasets-server.huggingface.co/cached-assets/okupyn/s3od_dataset/'
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'bench-client'}), timeout=60).read()

OUT.mkdir(parents=True, exist_ok=True)
old = {p.stem[4:] for p in (ROOT / 'data/test_syn').glob('syn_*.jpg')}
have = {p.stem[4:] for p in OUT.glob('syn_*.jpg')}  # hervatbaar: wat er al staat telt mee
for off in range(7000, 140000, 14000):  # andere plekken dan test_syn; dubbele beelden vallen hoe dan ook af
    n = 0
    for r in json.loads(get(f'{API}&offset={off}&length=12'))['rows']:
        if n >= 6: break
        r = r['row']; i = r['image_id']
        if not re.fullmatch(r'[0-9a-f]{32}', i) or i in old: continue  # id komt in een bestandsnaam: alleen hex
        if i in have: n += 1; continue
        src, msk = r['image']['src'], r['mask']['src']
        assert src.startswith(ASSETS) and msk.startswith(ASSETS), (src, msk)
        img = Image.open(io.BytesIO(get(src))).convert('RGB')
        gt = Image.open(io.BytesIO(get(msk))).convert('L')
        if gt.size != img.size: continue
        img.save(OUT / f'syn_{i}.jpg', quality=95)
        gt.save(OUT / f'syn_{i}_gt.png')
        have.add(i); n += 1
    print(off, n, flush=True)
print(len(list(OUT.glob('*.jpg'))), 'beelden in', OUT)
