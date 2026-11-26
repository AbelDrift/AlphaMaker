"""AlphaMaker-kern: segmentatie (ONNX) → verfijning op volle resolutie → randkleurcorrectie."""
import hashlib, os, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from . import pt, refine

REMBG = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/'
HF = 'https://huggingface.co/'
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
NONE = ((0, 0, 0), (1, 1, 1))
# ONNX-modellen; naam: (url, invoergrootte, normalisatie, nabewerking, sha256 of None).
# De BiRefNet-familie en RMBG-2.0 draaien via PyTorch (pt.py): ONNX Runtime heeft daarvoor >4,5 GB nodig.
MODELS = {
    # InSPyReNet Plus Ultra (MIT) als ONNX: zelfde maskers als transparent-background, sneller en zuiniger, zonder torch
    'inspyrenet':       (HF + 'OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX/resolve/fe628d2d1465f775c707368a80248a2d5b5ad71f/onnx/model.onnx',
                         1024, IMNET, 'auto', 'c108dd92b3ddfe3a2d9e9ac2b74730cc5acbb2ddc7ea863330c43f56ae832aa3'),
    # eigen snel model (MIT; train/kit.py, eigen foto's met InSPyReNet-maskers, ±0,8 s). Alleen via 'licht' hieronder.
    'licht-net':        ('https://github.com/AbelDrift/AlphaMaker/releases/download/licht-v1/licht-net.onnx',
                         1024, IMNET, None, '2b55ffe1e63824347ebc18f3c929793be23219da21a47a7baab1705d0ae3e598'),
    'u2net':          (REMBG + 'u2net.onnx', 320, IMNET, 'minmax', None),
    'isnet-general':    (REMBG + 'isnet-general-use.onnx', 1024, HALF, 'minmax', None),
    'rmbg-1.4':         (HF + 'briaai/RMBG-1.4/resolve/main/onnx/model.onnx', 1024, HALF, 'minmax', None),
    'ben2':             (HF + 'PramaLLC/BEN2/resolve/main/BEN2_Base.onnx', 1024, NONE, 'minmax', None),
    # ViTMatte (verfijning): dynamische invoermaat, ook vanuit de browser te laden
    'vitmatte':         (HF + 'Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model.onnx', None, None, None, None),
    'vitmatte-q':       (HF + 'Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model_quantized.onnx', None, None, None, None),
    # router van withoutBG v4 (Apache-2.0 + DINOv3-voorwaarden), alleen voor --type auto: kiest per foto haar- of objectmodel
    'wbg-router':       (HF + 'withoutbg/withoutbg-openweights-onnx/resolve/232aa520ccb6f94316cd638a7d916832479ec9a8/withoutbg-open-weights-backbone.onnx',
                         448, None, None, 'a64ba43cc25557f568ad6f35ed1139049dd43f5568385fa4f9848e0b4c7123ff'),
}
# niet voor commercieel gebruik (dus niet voor productfoto's voor verkoop): alleen als uitdrukkelijke keuze, met melding
NONCOMMERCIAL = {'rmbg-1.4': 'licentie bria-rmbg-1.4', 'isnet-general': 'gewichten zonder licentie, getraind op DIS5K',
                 'objecten-precies': 'gebruikt RMBG-1.4 en IS-Net'}
CACHE = Path(os.environ.get('ALPHAMAKER_MODELS', Path.home() / '.cache/alphamaker'))
_sessions = {}
_pt = {}


def model_path(name):
    url, *_, sha = MODELS[name]
    f = CACHE / (name + '.onnx')
    if not f.exists():
        if not url: raise FileNotFoundError(f'{f} ontbreekt: dit model wordt niet gedownload')
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.part')
        print(f'model {name} downloaden…', flush=True)
        urllib.request.urlretrieve(url, tmp)
        if sha:
            h = hashlib.sha256()
            with tmp.open('rb') as fh:
                for blk in iter(lambda: fh.read(1 << 20), b''): h.update(blk)
            if h.hexdigest() != sha:
                tmp.unlink(); raise RuntimeError(f'controlesom klopt niet voor {name}')
        tmp.rename(f)
    return f


def session(name, threads=4):
    if name not in _sessions:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.enable_cpu_mem_arena = False  # lagere geheugenpiek
        so.enable_mem_pattern = False       # geen vooraf gepland groot blok
        so.add_session_config_entry('session.disable_prepacking', '1')  # geen verpakte kopie van gewichten
        _sessions[name] = ort.InferenceSession(str(model_path(name)), so, providers=['CPUExecutionProvider'])
    return _sessions[name]


def agree(a, b, t=0.75):
    """Zijn twee maskers het eens (IoU ≥ t bij alpha > 0,5)? Drempel: bench/route_study.py."""
    m, n = a > 0.5, b > 0.5
    return (m & n).sum() >= t * max((m | n).sum(), 1)


def geen_achtergrond(alpha, fg=0.45, rand=0.15):
    """Close-up zonder achtergrond: het object loopt het beeld uit (gemiddeld >= 15% van de beeldrand bezet) en vult
    >= 45% van het beeld. Gekalibreerd op eigen productfoto's: close-ups rand 0,23-0,67, gewone foto's <= 0,07."""
    m = np.asarray(alpha) > 0.5
    return bool(m.mean() >= fg and (m[0].mean() + m[-1].mean() + m[:, 0].mean() + m[:, -1].mean()) / 4 >= rand)


def twijfel(alpha, t=0.12):
    """Twijfelt het snelle model? Halfdoorzichtige pixels t.o.v. de voorgrond > t. Eigen val (bench/gate_study.py):
    15% terugval, missers 27 → 6 van 448."""
    a = np.asarray(alpha)
    return bool(((a > .1) & (a < .9)).sum() > t * max((a > .5).sum(), 1))


def segment(img, name):
    """img: PIL RGB → alpha float32 0..1 op volle resolutie (bilineair opgeschaald)."""
    if name == 'objecten-precies':  # rmbg-1.4; oneens met isnet-general (drukke foto) → standaardmodel
        a, b = segment(img, 'rmbg-1.4'), segment(img, 'isnet-general')
        if agree(a, b): return a
        return segment(img, DEFAULT_MODEL)
    if name == 'licht':  # snel model; twijfelt het, dan het standaardmodel
        a = segment(img, 'licht-net')
        return segment(img, DEFAULT_MODEL) if twijfel(a) else a
    if name in ('inspyrenet', *pt.PT) and name not in (_pt if name in pt.PT else _sessions):
        _pt.clear(); _sessions.pop('inspyrenet', None); __import__('gc').collect()  # één zwaar model tegelijk (2,2–4,3 GB)
    if name in pt.PT:
        if name not in _pt: _pt[name] = pt.load(name, CACHE)
        return _pt[name](img)
    _, size, (mean, std), post, _ = MODELS[name]
    sess = session(name)
    shp = sess.get_inputs()[0].shape
    if isinstance(shp[2], int): size = shp[2]
    x = np.asarray(img.resize((size, size), Image.BILINEAR), np.float32) / 255
    x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0].astype(np.float32)
    if post == 'sigmoid' or (post == 'auto' and (y.min() < 0 or y.max() > 1)):
        y = 1 / (1 + np.exp(-y))
    elif post == 'minmax':
        y = (y - y.min()) / max(y.max() - y.min(), 1e-8)
    return np.asarray(Image.fromarray(y).resize(img.size, Image.BILINEAR), np.float32).clip(0, 1)


def _bilinear(x, height, width):
    """(C, H, W) → (C, height, width) zoals torch bilinear (align_corners=False, geen antialias); uit de withoutBG-README."""
    def axis(src, dst):
        pos = np.maximum((np.arange(dst) + 0.5) * (src / dst) - 0.5, 0.0)
        i0 = pos.astype(np.int64)
        return i0, np.minimum(i0 + 1, src - 1), (pos - i0).astype(np.float32)
    r0, r1, rl = axis(x.shape[1], height)
    c0, c1, cl = axis(x.shape[2], width)
    rows = x[:, r0] * (1 - rl)[:, None] + x[:, r1] * rl[:, None]
    return rows[:, :, c0] * (1 - cl) + rows[:, :, c1] * cl


def kind(img):
    """'persoon' als de withoutBG-router haar ziet (klasse fine_strand), anders 'object'.
    Schone test: 37 van 50 portretten fine_strand, 0 van 100 objecten."""
    x = _bilinear(np.asarray(img.convert('RGB'), np.float32).transpose(2, 0, 1) / 255, 448, 448)
    logits = session('wbg-router').run(['route_logits'], {'rgb_matting': x[None]})[0][0]
    return 'persoon' if int(np.argmax(logits)) == 0 else 'object'  # klassen: fine_strand, soft_detail, transparency, ...


DEFAULT_MODEL = 'inspyrenet'  # MIT, ook commercieel; 0 missers op portretten, beste vrije model op drukke foto's na BiRefNet
SEGMENTERS = [m for m in MODELS if not m.startswith(('vitmatte', 'wbg', 'licht'))] + list(pt.PT) + ['objecten-precies', 'licht']
MAX_PIXELS = 50_000_000              # zelfde plafond als remove.bg
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def cutout(img, model=DEFAULT_MODEL, refine_edges=False, fix_colors=True):
    # refine_edges (ViTMatte) maakte de randen in de benchmark slechter: experimentele optie
    """PIL-afbeelding → RGBA met transparante achtergrond."""
    if img.width * img.height > MAX_PIXELS:
        raise ValueError(f'afbeelding te groot (max {MAX_PIXELS // 1_000_000} MP)')
    img = img.convert('RGB')
    rgb = np.asarray(img, np.float32) / 255
    alpha = segment(img, model)
    if refine_edges:
        band = max(4, round(max(img.size) * 0.0125))  # onzekere zone ~1,25% van lange zijde
        alpha = refine.vitmatte(session('vitmatte'), rgb, refine.trimap(alpha, band))
    out = refine.foreground(rgb, alpha) if fix_colors else rgb
    rgba = np.dstack([out, alpha])
    return Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), 'RGBA')


def finish(cut, crop=False, margin=0.0, alpha_only=False, bg=None, bg_image=None, shadow=False):
    """Afwerking zoals de opties van remove.bg: bijsnijden tot het onderwerp (+marge, fractie van de
    onderwerpgrootte), alleen het masker, schaduw, achtergrondkleur of -foto. cut: RGBA (uit cutout)."""
    from PIL import ImageColor, ImageFilter, ImageOps
    if crop:
        box = cut.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()
        if box:
            mx, my = round((box[2] - box[0]) * margin), round((box[3] - box[1]) * margin)
            box = (max(0, box[0] - mx), max(0, box[1] - my), min(cut.width, box[2] + mx), min(cut.height, box[3] + my))
            cut = cut.crop(box)
    if alpha_only:
        return cut.getchannel('A')
    if shadow:  # zachte slagschaduw: vervaagd masker, iets verschoven, 50% dekking
        a = cut.getchannel('A')
        r = max(2, round(max(cut.size) * 0.01))
        sh = Image.new('RGBA', cut.size, (0, 0, 0, 0))
        sh.putalpha(a.filter(ImageFilter.GaussianBlur(r)).point(lambda v: v // 2))
        base = Image.new('RGBA', cut.size, (0, 0, 0, 0))
        base.alpha_composite(sh, (r, r))
        base.alpha_composite(cut)
        cut = base
    if bg_image is not None:
        back = ImageOps.fit(bg_image.convert('RGBA'), cut.size, Image.LANCZOS)
        back.alpha_composite(cut)
        return back.convert('RGB')
    if bg is not None:
        back = Image.new('RGBA', cut.size, ImageColor.getrgb(bg) + (255,))
        back.alpha_composite(cut)
        return back.convert('RGB')
    return cut
