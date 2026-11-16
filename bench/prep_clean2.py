"""Maakt data/test_clean2/: 60 andere DIS-VD-beelden (dan test_clean) + 40 PPM-100-train-portretten."""
import io, sys, zipfile
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'bench'))
import prep  # noqa: E402  (alleen save(); andere map)
prep.OUT = ROOT / 'data/test_clean2'; prep.OUT.mkdir(parents=True, exist_ok=True)
f = pq.ParquetFile(ROOT / 'data/raw/dis_vd.parquet')
step, k = f.metadata.num_rows // 40, 0  # zelfde stap als test_clean, maar andere rijen
for g in range(f.metadata.num_row_groups):
    for r in f.read_row_group(g, columns=['image', 'label', 'image_name']).to_pylist():
        if k % step in (step // 3, 2 * step // 3) and len(list(prep.OUT.glob('dis_*.jpg'))) < 60:
            prep.save('dis_' + Path(r['image_name']).stem[:40],
                      Image.open(io.BytesIO(r['image']['bytes'])), Image.open(io.BytesIO(r['label']['bytes'])))
        k += 1
z = zipfile.ZipFile(ROOT / 'data/raw/ppm100.zip')
for nm in sorted(x for x in z.namelist() if x.startswith('PPM-100/train/fg/') and x.endswith('.jpg'))[:40]:
    m = nm.replace('/fg/', '/alpha/')
    m = next((x for x in z.namelist() if x.rsplit('.', 1)[0] == m.rsplit('.', 1)[0]), None)
    if m: prep.save('ppm_' + Path(nm).stem, Image.open(z.open(nm)), Image.open(z.open(m)))
print(len(list(prep.OUT.glob('dis_*.jpg'))), 'DIS-VD +', len(list(prep.OUT.glob('ppm_*.jpg'))), 'PPM')
