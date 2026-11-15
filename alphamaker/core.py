"""AlphaMaker-kern: segmentatie (ONNX) → verfijning op volle resolutie → randkleurcorrectie."""
import hashlib, os, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from . import refine

REMBG = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/'
HF = 'https://huggingface.co/'
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
NONE = ((0, 0, 0), (1, 1, 1))
# naam: (url, invoergrootte, normalisatie, nabewerking, sha256 of None)
MODELS = {
    'u2net':            (REMBG + 'u2net.onnx', 320, IMNET, 'minmax', None),
    'isnet-general':    (REMBG + 'isnet-general-use.onnx', 1024, HALF, 'minmax', None),
    'rmbg-1.4':         (HF + 'briaai/RMBG-1.4/resolve/main/onnx/model.onnx', 1024, HALF, 'minmax', None),
    'birefnet-lite':    (REMBG + 'BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx', 1024, IMNET, 'sigmoid', None),
    'birefnet-general': (REMBG + 'BiRefNet-general-epoch_244.onnx', 1024, IMNET, 'sigmoid', None),
    'birefnet-portrait':(REMBG + 'BiRefNet-portrait-epoch_150.onnx', 1024, IMNET, 'sigmoid', None),
    'birefnet-hr-matting': (REMBG + 'BiRefNet_HR-matting-epoch_135.onnx', 2048, IMNET, 'sigmoid', None),
    'rmbg-2.0':         (HF + 'mohantesting/remove_background/resolve/main/onnx/model.onnx', 1024, IMNET, 'auto', None),
    'ben2':             (HF + 'PramaLLC/BEN2/resolve/main/BEN2_Base.onnx', 1024, NONE, 'minmax', None),
    'vitmatte':         (REMBG + 'vitmatte-small-composition-1k.onnx', 1024, None, None, None),
}
CACHE = Path(os.environ.get('ALPHAMAKER_MODELS', Path.home() / '.cache/alphamaker'))
_sessions = {}


def model_path(name):
    url, *_, sha = MODELS[name]
    f = CACHE / (name + '.onnx')
    if not f.exists():
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


def segment(img, name):
    """img: PIL RGB → alpha float32 0..1 op volle resolutie (bilineair opgeschaald)."""
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


DEFAULT_MODEL = 'birefnet-general'  # wordt bijgesteld na de benchmark (zie NOTES.md)
MAX_PIXELS = 50_000_000              # zelfde plafond als remove.bg
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def cutout(img, model=DEFAULT_MODEL, refine_edges=True, fix_colors=True):
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
