"""BiRefNet-familie via PyTorch (ONNX Runtime op CPU heeft >4,5 GB nodig; PyTorch rekent zuiniger).
Code + gewichten worden per model op een vaste revisie gedownload naar models/hf/<naam>/ en lokaal
geladen (geen trust_remote_code van internet). De code is vooraf nagelezen (zie NOTES.md)."""
import importlib.util, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMNET = (np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32))
# naam: (HF-repo, vaste revisie, invoergrootte)
PT = {
    'pt-birefnet-general':    ('ZhengPeng7/BiRefNet', 'e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4', 1024),
    'pt-birefnet-lite':       ('ZhengPeng7/BiRefNet_lite', 'aa62cd87eafb9cc43056d08ef3615a14628b831d', 1024),
    'pt-birefnet-portrait':   ('ZhengPeng7/BiRefNet-portrait', 'b6561965a70070d9143fd9e558f6ca3c481510db', 1024),
    'pt-birefnet-hr-matting': ('ZhengPeng7/BiRefNet_HR-matting', '5d6b6f8adcb5b417c871b1d84ceaae9871355b7f', 2048),
    'pt-rmbg-2.0':            ('mohantesting/remove_background', 'ddd449c55570b4effc5bce03a603d085299eec29', 1024),
}


def _fetch(repo, rev, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in ('birefnet.py', 'BiRefNet_config.py', 'model.safetensors'):
        if not (dst / f).exists():
            subprocess.run(['curl', '-sfL', '-o', str(dst / f) + '.part',
                            f'https://huggingface.co/{repo}/resolve/{rev}/{f}'], check=True)
            Path(str(dst / f) + '.part').rename(dst / f)
    (dst / '__init__.py').touch()


def load_pt(name, threads=4):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(threads)
    repo, rev, size = PT[name]
    d = ROOT / 'models/hf' / name
    _fetch(repo, rev, d)
    pkg = 'hf_' + name.replace('-', '_').replace('.', '_')
    spec = importlib.util.spec_from_file_location(pkg, d / '__init__.py', submodule_search_locations=[str(d)])
    sys.modules[pkg] = importlib.util.module_from_spec(spec); spec.loader.exec_module(sys.modules[pkg])
    mod = importlib.import_module(pkg + '.birefnet')
    cfg = importlib.import_module(pkg + '.BiRefNet_config').BiRefNetConfig(bb_pretrained=False)
    model = mod.BiRefNet(config=cfg)  # bb_pretrained=False: geen losse backbone-gewichten laden
    missing, unexpected = model.load_state_dict(load_file(d / 'model.safetensors'), strict=False, assign=True)
    assert not missing and not unexpected, (name, missing[:5], unexpected[:5])
    model = model.float().eval()  # sommige repos slaan fp16 op; CPU rekent in fp32

    def predict(img):
        x = np.asarray(img.resize((size, size), Image.BILINEAR), np.float32) / 255
        x = torch.from_numpy(((x - IMNET[0]) / IMNET[1]).transpose(2, 0, 1)[None].copy())
        with torch.inference_mode():
            y = model(x)[-1].sigmoid()[0, 0].numpy()
        return np.asarray(Image.fromarray(y).resize(img.size, Image.BILINEAR), np.float32).clip(0, 1)
    return predict
