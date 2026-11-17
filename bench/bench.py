"""Benchmark één achtergrond-model op data/test. Gebruik: bench.py <model>
Hervatbaar: elke afbeelding wordt direct naar results/<model>.jsonl geschreven en overgeslagen bij herstart."""
import json, os, sys, time, resource, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
SET = os.environ.get('TESTSET', 'test')  # 'test' (≤1280 px) of 'test_hr' (≤3000 px)
DATA = ROOT / 'data' / SET
RES = ROOT / 'results' if SET == 'test' else ROOT / 'results' / SET
PRED = ROOT / 'preds' if SET == 'test' else ROOT / 'preds' / SET
REMBG = 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/'
HF = 'https://huggingface.co/'
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
MODNET = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
NONE = ((0, 0, 0), (1, 1, 1))
# naam: (url, invoergrootte, normalisatie, nabewerking)
MODELS = {
    'u2net':            (REMBG + 'u2net.onnx', 320, IMNET, 'minmax'),
    'isnet-general':    (REMBG + 'isnet-general-use.onnx', 1024, HALF, 'minmax'),
    'rmbg-1.4':         (HF + 'briaai/RMBG-1.4/resolve/main/onnx/model.onnx', 1024, HALF, 'minmax'),
    # nieuwe kandidaten (testplan-2026-11.json), vaste revisies
    'mvanet':           (HF + 'onnx-community/MVANet-ONNX/resolve/43ec3427514b8d9164eea02df93ca2f1b036bb7b/onnx/model.onnx', 1024, IMNET, 'auto'),
    'mvanet-int8':      (HF + 'onnx-community/MVANet-ONNX/resolve/43ec3427514b8d9164eea02df93ca2f1b036bb7b/onnx/model_int8.onnx', 1024, IMNET, 'auto'),
    'finegrain-box':    (HF + 'MarcinEU/finegrain-box-segmenter-ONNX/resolve/544d5cc37109d3e8d318e9368a2467fcd99ac8c5/onnx/model.onnx', 1024, IMNET, 'sigmoid'),
    'modnet':        (HF + 'Xenova/modnet/resolve/fa2fa546052fba4c08921230a26cc69a333fca12/onnx/model.onnx', -512, MODNET, 'auto'),  # portretten, Apache-2.0
    'rmbg-1.4-q':       (HF + 'briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx', 1024, HALF, 'minmax'),  # webversie 'licht'
    'birefnet-lite':    (REMBG + 'BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx', 1024, IMNET, 'sigmoid'),
    'birefnet-general': (REMBG + 'BiRefNet-general-epoch_244.onnx', 1024, IMNET, 'sigmoid'),
    'birefnet-portrait':(REMBG + 'BiRefNet-portrait-epoch_150.onnx', 1024, IMNET, 'sigmoid'),
    'birefnet-hr-matting': (REMBG + 'BiRefNet_HR-matting-epoch_135.onnx', 2048, IMNET, 'sigmoid'),
    'rmbg-2.0':         (HF + 'mohantesting/remove_background/resolve/main/onnx/model.onnx', 1024, IMNET, 'auto'),
    'ben2':             (HF + 'PramaLLC/BEN2/resolve/main/BEN2_Base.onnx', 1024, NONE, 'minmax'),
}


def metrics(pred, gt):
    """pred, gt: float32 0..1, zelfde vorm. MAE overal, MAE in randzone, IoU@0.5."""
    mae = float(np.abs(pred - gt).mean())
    hard = gt > 0.5
    # randzone = halfdoorzichtige pixels + band van 8 px rond de harde contour
    band = binary_dilation(hard, iterations=8) & binary_dilation(~hard, iterations=8)
    edge = ((gt > 0.02) & (gt < 0.98)) | band
    emae = float(np.abs(pred - gt)[edge].mean()) if edge.any() else 0.0
    p, g = pred > 0.5, hard
    iou = float((p & g).sum() / max((p | g).sum(), 1))
    return mae, emae, iou


def predict_onnx(sess, cfg, img):
    _, size, (mean, std), post = cfg
    shp = sess.get_inputs()[0].shape
    if isinstance(shp[2], int): size = shp[2]
    # negatieve size: kortste zijde = -size, zijden veelvoud van 32, verhouding behouden (zoals MODNet)
    wh = tuple(max(32, round(d * -size / min(img.size) / 32) * 32) for d in img.size) if size < 0 else (size, size)
    x = np.asarray(img.resize(wh, Image.BILINEAR), np.float32) / 255
    x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0].astype(np.float32)
    if post == 'sigmoid' or (post == 'auto' and (y.min() < 0 or y.max() > 1)):
        y = 1 / (1 + np.exp(-y))
    elif post == 'minmax':
        y = (y - y.min()) / max(y.max() - y.min(), 1e-8)
    return np.asarray(Image.fromarray(y).resize(img.size, Image.BILINEAR), np.float32).clip(0, 1)


# withoutBG open weights v4: hulpfuncties overgenomen uit het standalone-script in hun HF-README
# (withoutbg/withoutbg-openweights-onnx @232aa520, Apache-2.0)
def wbg_bilinear(x, height, width):
    """(C, H, W) -> (C, height, width); torch bilinear, align_corners=False, no antialias."""
    def axis(src, dst):
        pos = np.maximum((np.arange(dst) + 0.5) * (src / dst) - 0.5, 0.0)
        i0 = pos.astype(np.int64)
        return i0, np.minimum(i0 + 1, src - 1), (pos - i0).astype(np.float32)
    _, h, w = x.shape
    r0, r1, rl = axis(h, height)
    c0, c1, cl = axis(w, width)
    rows = x[:, r0] * (1 - rl)[:, None] + x[:, r1] * rl[:, None]
    return rows[:, :, c0] * (1 - cl) + rows[:, :, c1] * cl


def wbg_feed(rgb, spec):
    """Square-stretch native (H, W, 3) float RGB as each graph input's spec says."""
    feeds = {}
    for name, s in spec["inputs"].items():
        if s["interpolation"] == "bicubic":  # antialiased; PIL "F" mode matches torch
            x = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]))
                                     .resize((s["size"],) * 2, Image.Resampling.BICUBIC))
                          for c in range(3)])
        else:
            x = wbg_bilinear(np.ascontiguousarray(rgb.transpose(2, 0, 1)), s["size"], s["size"])
        feeds[name] = np.ascontiguousarray(x[None], dtype=np.float32)
    return feeds


def load(name):
    if name.startswith('pt-'):  # BiRefNet-familie via PyTorch (zie pt_models.py)
        from pt_models import load_pt
        return load_pt(name)
    if name in ('inspyrenet-dis5k', 'inspyrenet-nightly'):  # schone InSPyReNet-varianten (zie testplan-2026-11.json)
        from transparent_background import Remover
        r = Remover(mode='base', device='cpu', ckpt=str(ROOT / 'models' / (name + '.pth')))
        return lambda img: np.asarray(r.process(img, type='map').convert('L'), np.float32) / 255
    if name in ('s3od-dis', 's3od-synth'):  # S3OD (ICLR 2026; MIT + DINOv3-licentie), alleen in ~/venvs/am-kand (PY=...)
        import torch
        from s3od import BackgroundRemoval
        torch.set_num_threads(4)
        tl = torch.load  # hun predictor.py laadt met weights_only=False (volledige pickle): forceer True
        torch.load = lambda f, *a, **k: tl(f, map_location='cpu', weights_only=True)
        try:
            m = BackgroundRemoval(model_id=str(ROOT / 'models' / (name + '.pt')), device='cpu')
        finally:
            torch.load = tl
        return lambda img: m.remove_background(img).predicted_mask.astype(np.float32).clip(0, 1)
    if name == 'pdfnet':  # PDFNet_Best (CVPR 2026, MIT) + Depth Anything V2 Small (Apache-2.0); code in ~/am-kand/PDFNet
        import argparse, cv2, torch
        sys.path.insert(0, str(Path.home() / 'am-kand/PDFNet'))
        from args import get_args_parser
        from models.PDFNet import PDFNet_process, PDF_decoder, PDF_depth_decoder
        from models.swin_transformer import SwinB
        from DAM_V2.depth_anything_v2.dpt import DepthAnythingV2
        torch.set_num_threads(4)
        a = argparse.ArgumentParser(parents=[get_args_parser()]).parse_args(['--device', 'cpu'])
        net = PDFNet_process(encoder=SwinB(args=a, in_chans=3, pretrained=False), decoder=PDF_decoder(args=a),
                             depth_decoder=PDF_depth_decoder(args=a), device='cpu', args=a)
        miss, unexp = net.load_state_dict(torch.load(ROOT / 'models/pdfnet-best.pth', map_location='cpu', weights_only=True), strict=False)
        # hun demo laadt met strict=False (verbergt fouten); alleen depth_decoder.* mag over zijn: die zit alleen in de
        # training (forward), in PDFNet_process.__init__ uitgecommentarieerd en niet in inference()
        assert not miss and all(k.startswith('depth_decoder.') for k in unexp), (miss[:5], [k for k in unexp if not k.startswith('depth_decoder.')][:5])
        net.eval()
        dav2 = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        dav2.load_state_dict(torch.load(ROOT / 'models/dav2-vits.pth', map_location='cpu', weights_only=True)); dav2.eval()
        mean, std = np.array(IMNET[0], np.float32), np.array(IMNET[1], np.float32)
        def predict(img):
            rgb = np.asarray(img); H, W = rgb.shape[:2]
            with torch.inference_mode():
                depth = dav2.infer_image(np.ascontiguousarray(rgb[:, :, ::-1]))  # infer_image verwacht BGR
                x = (cv2.resize(rgb, (1024, 1024)).astype(np.float32) / 255 - mean) / std
                x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
                d = torch.from_numpy(cv2.resize(depth, (1024, 1024)))[None, None].float()
                y = net.inference(x, d)[0][0, 0].numpy()
            return cv2.resize(y, (W, H)).clip(0, 1)
        return predict
    if name == 'inspyrenet':  # transparent-background
        from transparent_background import Remover
        r = Remover(mode='base', device='cpu')
        return lambda img: np.asarray(r.process(img, type='map').convert('L'), np.float32) / 255
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4          # laat 4 threads over voor de desktop
    so.enable_cpu_mem_arena = False      # lagere geheugenpiek
    so.enable_mem_pattern = False       # geen vooraf gepland groot blok
    so.add_session_config_entry('session.disable_prepacking', '1')  # geen verpakte kopie van gewichten
    if name == 'wbg-matting':  # withoutBG v4, altijd de matting-tak (testplan #3); bestanden in models/withoutbg
        d = ROOT / 'models' / 'withoutbg'
        sc = json.loads((d / 'withoutbg-open-weights.onnx.json').read_text())
        S = {k: ort.InferenceSession(str(d / sc[k]['file']), so, providers=['CPUExecutionProvider']) for k in ('router', 'coarse')}
        r, c = sc['router'], sc['coarse']
        def predict(img):
            rgb = np.asarray(img, np.float32) / 255
            rf = wbg_feed(rgb, r)
            logits, *feats = S['router'].run([r['logits_name'], *r['feature_names']], rf)
            print('route', r['categories'][int(np.argmax(logits[0]))], flush=True)
            f = {**rf, **wbg_feed(rgb, {'inputs': {'rgb_depth': c['inputs']['rgb_depth']}}), **dict(zip(c['feature_names'], feats))}
            a = S['coarse'].run([c['output_name']], f)[0][0]
            return np.clip(wbg_bilinear(a, *rgb.shape[:2]), 0, 1)[0]
        return predict
    cfg = MODELS[name]
    f = ROOT / 'models' / (name + '.onnx')
    if not f.exists():
        f.parent.mkdir(exist_ok=True)
        subprocess.run(['curl', '-sfL', '-o', str(f) + '.part', cfg[0]], check=True)
        Path(str(f) + '.part').rename(f)
    sess = ort.InferenceSession(str(f), so, providers=['CPUExecutionProvider'])
    return lambda img: predict_onnx(sess, cfg, img)


def main(name):
    out = RES / (name + '.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['img'] for l in out.read_text().splitlines()} if out.exists() else set()
    imgs = sorted(DATA.glob('*.jpg'))
    todo = [p for p in imgs if p.stem not in done]
    if not todo: return print(name, 'al klaar')
    t0 = time.time(); run = load(name); print(name, f'geladen in {time.time()-t0:.1f}s', flush=True)
    pd = PRED / name; pd.mkdir(parents=True, exist_ok=True)
    for p in todo:
        img = Image.open(p).convert('RGB')
        gt = np.asarray(Image.open(p.with_name(p.stem + '_gt.png')).convert('L'), np.float32) / 255
        t = time.time(); a = run(img); dt = time.time() - t
        Image.fromarray((a * 255).astype(np.uint8)).save(pd / (p.stem + '.png'))
        mae, emae, iou = metrics(a, gt)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        row = dict(img=p.stem, mae=mae, edge_mae=emae, iou=iou, sec=dt, peak_mb=rss)
        with out.open('a') as fh: fh.write(json.dumps(row) + '\n')
        print(row, flush=True)


if __name__ == '__main__':
    # zelftest metriek: perfecte voorspelling = 0 fout, IoU 1
    g = np.zeros((40, 40), np.float32); g[10:30, 10:30] = 1
    assert metrics(g, g) == (0.0, 0.0, 1.0)
    main(sys.argv[1])
