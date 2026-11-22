"""Snel, commercieel bruikbaar achtergrondmodel: een klein MobileNetV4-U-Net leert de maskers van InSPyReNet (MIT)
na op je EIGEN productfoto's (distillatie). Geen voorgetrainde gewichten, geen S3OD (FLUX-licentie). Zie train/README.md.

  python train/kit.py prep   --src FOTOMAP --out WERKMAP        # lokaal: rechtzetten, verkleinen, EXIF weg, splitsen
  python train/kit.py label  --data WERKMAP                     # GPU: maskers van InSPyReNet (leermeester)
  python train/kit.py train  --data WERKMAP [--epochs 60]       # GPU: leerling trainen (hervatbaar)
  python train/kit.py export --data WERKMAP --out licht.onnx    # naar ONNX (invoer 1024, uitvoer 0..1)
  python train/kit.py eval   --onnx licht.onnx [--data WERKMAP] # lokaal: tegen InSPyReNet en de testsets
"""
import argparse, hashlib, io, json, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MEAN, STD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)
TEACHER = ('https://huggingface.co/OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX/resolve/'
           'fe628d2d1465f775c707368a80248a2d5b5ad71f/onnx/model.onnx',
           'c108dd92b3ddfe3a2d9e9ac2b74730cc5acbb2ddc7ea863330c43f56ae832aa3')


def prep(a):
    """Foto's rechtop (EXIF), lange zijde <= --max, zonder metadata (GPS!) als JPEG; dubbele eruit; train/val."""
    src, out = Path(a.src), Path(a.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    seen, names = set(), []
    paden = sorted(p for p in src.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    if a.sample: paden = sorted(random.Random(1).sample(paden, min(a.sample, len(paden))))  # steekproef over alles
    paden = paden[:a.limit] if a.limit else paden
    for i, p in enumerate(paden):
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
        except Exception as e:  # kapot of geen afbeelding: overslaan
            print('overgeslagen:', p.name, e); continue
        im.thumbnail((a.max, a.max), Image.LANCZOS)
        h = hashlib.md5(np.asarray(im.resize((32, 32))).tobytes()).hexdigest()  # zelfde foto (ook als kopie)
        if h in seen: continue
        seen.add(h); naam = f'{len(names):05d}.jpg'
        im.save(out / 'images' / naam, 'JPEG', quality=92)  # zonder exif= : alle metadata weg
        names.append(naam)
        if i % 200 == 0: print(f'{i}/{len(paden)}', flush=True)
    random.Random(0).shuffle(names)
    nval = max(1, len(names) // 10)
    (out / 'val.txt').write_text('\n'.join(sorted(names[:nval])))
    (out / 'train.txt').write_text('\n'.join(sorted(names[nval:])))
    print(f'{len(names)} foto\'s ({len(paden) - len(names)} dubbel/kapot), val {nval}')


def teacher_session(cache):
    import onnxruntime as ort, urllib.request
    f = Path(cache) / 'inspyrenet.onnx'
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEACHER[0], f.with_suffix('.part'))
        if hashlib.sha256(f.with_suffix('.part').read_bytes()).hexdigest() != TEACHER[1]:
            raise SystemExit('controlesom leermeester klopt niet')
        f.with_suffix('.part').rename(f)
    so = ort.SessionOptions(); so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False  # cpu: 2,2 i.p.v. >4,5 GB
    return ort.InferenceSession(str(f), so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def label(a):
    """Masker van de leermeester per foto (PNG, zelfde maat) + statistiek: voorgrond, onzeker, randen bezet."""
    d = Path(a.data); (d / 'masks').mkdir(exist_ok=True)
    s = teacher_session(a.cache); stats = {}
    rem = None
    try:  # vindt onnxruntime-gpu CUDA niet: zelfde model (transparent-background, MIT) via torch op de GPU
        import torch
        if 'CUDAExecutionProvider' not in s.get_providers() and torch.cuda.is_available():
            from transparent_background import Remover
            rem = Remover(mode='base', device='cuda'); print('leermeester via transparent-background (cuda)', flush=True)
    except ImportError:
        pass
    old = d / 'labels.jsonl'
    if old.exists(): stats = {r['img']: r for r in map(json.loads, old.read_text().splitlines())}
    names = (d / 'train.txt').read_text().split() + (d / 'val.txt').read_text().split()
    with old.open('a') as log:
        for i, n in enumerate(names):
            if n in stats: continue
            im = Image.open(d / 'images' / n).convert('RGB')
            if rem is not None:
                m = np.asarray(rem.process(im, type='map').convert('L'), np.float32) / 255
            else:
                x = ((np.asarray(im.resize((1024, 1024), Image.BILINEAR), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
                y = s.run(None, {'input': x})[0][0, 0].astype(np.float32)
                m = np.asarray(Image.fromarray(y).resize(im.size, Image.BILINEAR)).clip(0, 1)
            Image.fromarray((m * 255 + .5).astype(np.uint8)).save(d / 'masks' / (n[:-4] + '.png'))
            rand = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]) > .5
            r = {'img': n, 'fg': round(float((m > .5).mean()), 4), 'unsure': round(float(((m > .1) & (m < .9)).mean()), 4),
                 'border': round(float(rand.mean()), 4)}
            log.write(json.dumps(r) + '\n'); log.flush()
            if i % 100 == 0: print(f'{i}/{len(names)}', r, flush=True)


# ---------------- leerling ----------------
def build_student():
    import timm, torch, torch.nn as nn, torch.nn.functional as F

    def block(i, o):  # dieptescheidbaar, licht voor de cpu
        return nn.Sequential(nn.Conv2d(i, i, 3, padding=1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                             nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = timm.create_model('mobilenetv4_conv_medium', pretrained=False, features_only=True)  # geen ImageNet
            ch = self.enc.feature_info.channels()                     # [32, 48, 80, 160, 960] op 1/2 .. 1/32
            w = [64, 64, 96, 128]
            self.top = nn.Sequential(nn.Conv2d(ch[4], 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
            self.ctx = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch[4], 128, 1), nn.ReLU(inplace=True))
            self.lat = nn.ModuleList(nn.Conv2d(c, k, 1) for c, k in zip(ch[:4], w))
            self.dec = nn.ModuleList([block(128 + 128, 128), block(128 + 96, 96), block(96 + 64, 64), block(64 + 64, 64)])
            self.head = nn.Sequential(nn.Conv2d(64 + 3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 3, padding=1))

        def forward(self, x):
            f = self.enc(x)
            y = self.top(f[4]) + self.ctx(f[4])
            for k, lvl in enumerate((3, 2, 1, 0)):  # van grof naar fijn
                y = F.interpolate(y, size=f[lvl].shape[2:], mode='bilinear', align_corners=False)
                y = self.dec[k](torch.cat([y, self.lat[lvl](f[lvl])], 1))
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
            return self.head(torch.cat([y, x], 1))  # volle resolutie: scherpe randen met de beeldinformatie
    return Student()


def loss_fn(logit, t):
    import torch, torch.nn.functional as F
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, t)
    inter, union = (p * t).sum((1, 2, 3)), (p + t - p * t).sum((1, 2, 3))
    iou = (1 - (inter + 1) / (union + 1)).mean()
    rand = (F.max_pool2d(t, 7, 1, 3) + F.max_pool2d(-t, 7, 1, 3)) > .05  # band rond de rand van de leermeester
    edge = ((p - t).abs() * rand).sum() / rand.sum().clamp(min=1)
    gx = lambda z: z[..., :, 1:] - z[..., :, :-1]; gy = lambda z: z[..., 1:, :] - z[..., :-1, :]
    grad = (gx(p) - gx(t)).abs().mean() + (gy(p) - gy(t)).abs().mean()
    return bce + iou + edge + grad


def achtergrond(size, rng, fotos):
    """Willekeurige nieuwe achtergrond: effen, verloop, ruis of (een stuk van) een andere eigen foto."""
    k = rng.random()
    if k < .3: return np.full((size, size, 3), rng.integers(0, 256, 3), np.float32)
    if k < .55:
        a, b = rng.integers(0, 256, 3).astype(np.float32), rng.integers(0, 256, 3).astype(np.float32)
        t = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
        return (a * (1 - t) + b * t) * np.ones((1, size, 1), np.float32)
    if k < .65: return rng.normal(128, 40, (size, size, 3)).clip(0, 255).astype(np.float32)
    im = Image.open(rng.choice(fotos)).convert('RGB')
    return np.asarray(ImageOps.fit(im, (size, size), Image.BILINEAR), np.float32)


class Data:
    def __init__(self, d, split, size, train):
        self.d, self.size, self.train = Path(d), size, train
        lab = {r['img']: r for r in map(json.loads, (self.d / 'labels.jsonl').read_text().splitlines())}
        names = (self.d / f'{split}.txt').read_text().split()
        # leermeester twijfelt te veel (veel halfdoorzichtig): niet om van te leren
        self.names = [n for n in names if n in lab and lab[n]['unsure'] < .25]
        self.fotos = [self.d / 'images' / n for n in self.names]

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        import torch
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        n = self.names[i]
        im = Image.open(self.d / 'images' / n).convert('RGB'); m = Image.open(self.d / 'masks' / (n[:-4] + '.png'))
        if self.train:  # willekeurige uitsnede (60-100%), spiegelen, licht/kleur, achtergrond wisselen, jpeg
            s = rng.uniform(.6, 1); W, H = im.size; cw, chh = int(W * s), int(H * s)
            x0, y0 = rng.integers(0, W - cw + 1), rng.integers(0, H - chh + 1)
            im, m = im.crop((x0, y0, x0 + cw, y0 + chh)), m.crop((x0, y0, x0 + cw, y0 + chh))
            if rng.random() < .5: im, m = ImageOps.mirror(im), ImageOps.mirror(m)
        im = np.asarray(im.resize((self.size, self.size), Image.BILINEAR), np.float32)
        m = np.asarray(m.resize((self.size, self.size), Image.BILINEAR), np.float32) / 255
        if self.train:
            if rng.random() < .5:
                im = im * m[..., None] + achtergrond(self.size, rng, self.fotos) * (1 - m[..., None])
            im = im * rng.uniform(.6, 1.4) + rng.uniform(-30, 30, 3)  # belichting en kleurzweem, zoals amateurfoto's
            if rng.random() < .3:
                buf = io.BytesIO(); Image.fromarray(im.clip(0, 255).astype(np.uint8)).save(buf, 'JPEG', quality=int(rng.integers(40, 90)))
                im = np.asarray(Image.open(buf), np.float32)
        x = ((im.clip(0, 255) / 255 - MEAN) / STD).astype(np.float32).transpose(2, 0, 1)  # augmentatie maakt float64
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(m[None].copy())


def train(a):
    import torch
    from torch.utils.data import DataLoader
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = Path(a.data); ck = d / 'student.pt'
    tr, va = Data(d, 'train', a.size, True), Data(d, 'val', a.size, False)
    dl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch, persistent_workers=a.workers > 0)
    net = build_student().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=.05)
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
    start = 0
    if ck.exists():  # hervatten na onderbreking (gehuurde GPU)
        s = torch.load(ck, map_location=dev, weights_only=True)
        net.load_state_dict(s['net']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); start = s['epoch'] + 1
    for ep in range(start, a.epochs):
        net.train(); t0, tot = time.time(), 0.
        for k, (x, m) in enumerate(dl):
            x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16 if dev == 'cuda' else torch.bfloat16, enabled=dev == 'cuda'):
                out = net(x)
            loss = loss_fn(out.float(), m)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
            if a.max_steps and k + 1 >= a.max_steps: break
        net.eval(); ious = []
        with torch.no_grad():
            for x, m in DataLoader(va, 8):
                p = torch.sigmoid(net(x.to(dev))) > .5; t = m.to(dev) > .5
                ious += ((p & t).sum((1, 2, 3)) / (p | t).sum((1, 2, 3)).clamp(min=1)).tolist()
                if a.max_steps: break
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep}, ck)
        print(f'epoch {ep + 1}/{a.epochs} verlies {tot / max(1, k + 1):.4f} val-IoU {np.mean(ious):.4f} ({time.time() - t0:.0f} s)', flush=True)


def export(a):
    import torch
    net = build_student(); net.load_state_dict(torch.load(Path(a.data) / 'student.pt', map_location='cpu', weights_only=True)['net'])
    net.eval()
    wrap = torch.nn.Sequential(net, torch.nn.Sigmoid())  # uitvoer 0..1, net als de andere AlphaMaker-modellen
    with torch.no_grad():  # vrije maat; klein voorbeeld houdt het exporteren zuinig (AlphaMaker rekent op 1024)
        torch.onnx.export(wrap, torch.zeros(1, 3, 512, 512), a.out, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {2: 'h', 3: 'w'}, 'output': {2: 'h', 3: 'w'}}, opset_version=17, dynamo=False)
    print(a.out, f'{Path(a.out).stat().st_size / 2**20:.0f} MB, sha256', hashlib.sha256(Path(a.out).read_bytes()).hexdigest())


def evaluate(a):
    """Rand-MAE/missers (zoals bench/bench.py) en tijd: leerling tegen leermeester op val, en tegen GT op testsets."""
    import onnxruntime as ort
    from scipy.ndimage import binary_dilation

    def metrics(pred, gt):
        hard = gt > .5
        edge = ((gt > .02) & (gt < .98)) | (binary_dilation(hard, iterations=8) & binary_dilation(~hard, iterations=8))
        p = pred > .5
        return float(np.abs(pred - gt)[edge].mean()) if edge.any() else 0., float((p & hard).sum() / max((p | hard).sum(), 1))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(a.onnx, so, providers=['CPUExecutionProvider'])
    sets = []
    if a.data:
        d = Path(a.data)
        sets.append(('eigen val (tegen leermeester)', [(d / 'images' / n, d / 'masks' / (n[:-4] + '.png')) for n in (d / 'val.txt').read_text().split()]))
    for t in ('test_syn2', 'test_clean2'):
        g = sorted((ROOT / 'data' / t).glob('*.jpg'))
        if g: sets.append((t, [(p, p.with_name(p.stem + '_gt.png')) for p in g[:a.limit or None]]))
    for naam, paren in sets:
        r = []
        for img, gtp in paren:
            im = Image.open(img).convert('RGB')
            x = ((np.asarray(im.resize((1024, 1024), Image.BILINEAR), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
            t0 = time.time(); y = s.run(None, {'input': x})[0][0, 0]; dt = time.time() - t0
            pred = np.asarray(Image.fromarray(y.astype(np.float32)).resize(im.size, Image.BILINEAR)).clip(0, 1)
            gt = np.asarray(Image.open(gtp).convert('L'), np.float32) / 255
            r.append((*metrics(pred, gt), dt))
        print(f'{naam:30} n={len(r):4d} rand-MAE {np.mean([x[0] for x in r]):.4f} missers {sum(x[1] < .8 for x in r)} '
              f'{np.mean([x[2] for x in r]):.2f} s/foto', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep'); p.add_argument('--src', required=True); p.add_argument('--out', required=True)
    p.add_argument('--max', type=int, default=1280); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--sample', type=int, default=0, help='willekeurige steekproef van zoveel foto\'s')
    p = sub.add_parser('label'); p.add_argument('--data', required=True); p.add_argument('--cache', default=str(Path.home() / '.cache/alphamaker'))
    p = sub.add_parser('train'); p.add_argument('--data', required=True); p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=16); p.add_argument('--size', type=int, default=768); p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--max-steps', type=int, default=0, help='alleen voor een proefrun')
    p = sub.add_parser('export'); p.add_argument('--data', required=True); p.add_argument('--out', default='licht.onnx')
    p = sub.add_parser('eval'); p.add_argument('--onnx', required=True); p.add_argument('--data'); p.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    {'prep': prep, 'label': label, 'train': train, 'export': export, 'eval': evaluate}[a.cmd](a)
