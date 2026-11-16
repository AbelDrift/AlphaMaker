"""Maakt data/test_hr/: dezelfde portretten (PPM-100 val) + 4 DIS-objecten op hoge resolutie (max 3000 px)."""
import io, sys, zipfile
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'bench'))
import prep  # noqa: E402  (alleen save(); met andere map en maximale maat)
prep.OUT = ROOT / 'data/test_hr'; prep.OUT.mkdir(parents=True, exist_ok=True)
prep.MAX = 3000
f = pq.ParquetFile(ROOT / 'data/raw/dis_te1.parquet')
for g in range(2):
    t = f.read_row_group(g, columns=['image', 'label', 'image_name']).to_pylist()
    for i in (0, 50):
        r = t[i]
        prep.save('dis_' + Path(r['image_name']).stem[:40],
                  Image.open(io.BytesIO(r['image']['bytes'])), Image.open(io.BytesIO(r['label']['bytes'])))
z = zipfile.ZipFile(ROOT / 'data/raw/ppm100.zip')
for n in sorted(n for n in z.namelist() if n.startswith('PPM-100/val/fg/') and n.endswith('.jpg'))[:8]:
    m = n.replace('/fg/', '/alpha/')
    m = next(x for x in z.namelist() if x.rsplit('.', 1)[0] == m.rsplit('.', 1)[0])
    prep.save('ppm_' + Path(n).stem, Image.open(z.open(n)), Image.open(z.open(m)))
for p in sorted(prep.OUT.glob('*.jpg')):
    print(p.name[:50], Image.open(p).size)
