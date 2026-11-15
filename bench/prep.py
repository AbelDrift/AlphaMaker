"""Maakt data/test/: 10 beelden uit DIS5K-TE1 + 8 uit PPM-100, max 1280 px, met _gt.png."""
import io, zipfile
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data/test'; OUT.mkdir(parents=True, exist_ok=True)
MAX = 1280


def save(name, img, gt):
    img, gt = img.convert('RGB'), gt.convert('L')
    assert img.size == gt.size, (name, img.size, gt.size)
    s = MAX / max(img.size)
    if s < 1:
        sz = (round(img.width * s), round(img.height * s))
        img, gt = img.resize(sz, Image.LANCZOS), gt.resize(sz, Image.LANCZOS)
    img.save(OUT / f'{name}.jpg', quality=95)
    gt.save(OUT / f'{name}_gt.png')


f = pq.ParquetFile(ROOT / 'data/raw/dis_te1.parquet')
for g in range(f.metadata.num_row_groups):  # één rijgroep (~100 rijen) tegelijk in geheugen
    t = f.read_row_group(g, columns=['image', 'label', 'image_name']).to_pylist()
    for i in (0, 50):
        r = t[i]
        save('dis_' + Path(r['image_name']).stem[:40],
             Image.open(io.BytesIO(r['image']['bytes'])), Image.open(io.BytesIO(r['label']['bytes'])))

z = zipfile.ZipFile(ROOT / 'data/raw/ppm100.zip')
names = sorted(n for n in z.namelist() if n.startswith('PPM-100/val/fg/') and n.endswith('.jpg'))
for n in names[:8]:
    m = n.replace('/fg/', '/alpha/')
    m = next(x for x in z.namelist() if x.rsplit('.', 1)[0] == m.rsplit('.', 1)[0])
    save('ppm_' + Path(n).stem, Image.open(z.open(n)), Image.open(z.open(m)))

print(len(list(OUT.glob('*.jpg'))), 'testbeelden in', OUT)
