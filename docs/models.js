// Modellen voor de browser. Alleen Hugging Face: daar werkt CORS (GitHub-releases niet).
// bytes = omvang op Hugging Face (nov 2026), gebruikt voor de plausibiliteitscontrole van de download.
const HF = 'https://huggingface.co/';
const file = (repo, name, bytes, rev = 'main') => ({ url: `${HF}${repo}/resolve/${rev}/onnx/${name}`, bytes });
const IMAGENET = { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
const HALF = { mean: [0.5, 0.5, 0.5], std: [1, 1, 1] };
const INSPY = 'OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX', INSPY_REV = 'fe628d2d1465f775c707368a80248a2d5b5ad71f';

// Standaard InSPyReNet Plus Ultra (MIT, ook commercieel; zelfde maskers als de CLI). RMBG-1.4 is NIET voor
// commercieel gebruik en blijft alleen als uitdrukkelijke keuze. BiRefNet (beter voor haar) past niet in WASM.
export const DEFAULT_MODEL = { webgpu: 'inspyrenet', wasm: 'inspyrenet' };
export const MAX_PIXELS = 12_000_000;  // grotere foto's eerst verkleinen: geheugen van browser en telefoon

// files per rekenkern: 'wasm' (processor) en 'webgpu' (fp16 op de grafische kaart). Geen bestand = kan daar niet.
// post: nabewerking van de modeluitvoer (zie refine.postprocess)
export const MODELS = {
  'inspyrenet': {  // vastgezette revisie; fp32 voor beide kernen (fp16 is in de browser niet aangetoond)
    label: 'InSPyReNet (MIT, ook commercieel)', license: 'MIT: vrij te gebruiken, ook commercieel',
    size: 1024, norm: IMAGENET, post: 'auto',
    files: {
      wasm: file(INSPY, 'model.onnx', 395316574, INSPY_REV),
      webgpu: file(INSPY, 'model.onnx', 395316574, INSPY_REV),
    },
  },
  'birefnet-lite': {
    label: 'BiRefNet lite', license: 'MIT', size: 1024, norm: IMAGENET, post: 'auto',
    files: {
      webgpu: file('onnx-community/BiRefNet_lite-ONNX', 'model_fp16.onnx', 114538221),
    },
  },
  'rmbg-1.4': {
    label: 'RMBG-1.4 (NIET voor commercieel gebruik)',
    license: 'bria-rmbg-1.4: NIET voor commercieel gebruik, dus niet voor productfoto\'s voor verkoop',
    nonCommercial: true, size: 1024, norm: HALF, post: 'minmax',
    files: {
      wasm: file('briaai/RMBG-1.4', 'model.onnx', 176153355),
      webgpu: file('briaai/RMBG-1.4', 'model_fp16.onnx', 88217533),
    },
  },
  'rmbg-1.4-licht': {
    label: 'RMBG-1.4 licht, kleine download (NIET voor commercieel gebruik)',
    license: 'bria-rmbg-1.4: NIET voor commercieel gebruik, dus niet voor productfoto\'s voor verkoop',
    nonCommercial: true, size: 1024, norm: HALF, post: 'minmax',
    files: {
      wasm: file('briaai/RMBG-1.4', 'model_quantized.onnx', 44403226),
      webgpu: file('briaai/RMBG-1.4', 'model_quantized.onnx', 44403226),
    },
  },
  'ben2': {
    label: 'BEN2', license: 'MIT', size: 1024, norm: IMAGENET, post: 'auto',
    files: { webgpu: file('onnx-community/BEN2-ONNX', 'model_fp16.onnx', 219121675) },
  },
  'birefnet': {
    label: 'BiRefNet (beste voor haar, zwaar)', license: 'MIT', size: 1024, norm: IMAGENET, post: 'auto',
    files: { webgpu: file('onnx-community/BiRefNet-ONNX', 'model_fp16.onnx', 489666272) },
  },
};

// ViTMatte voor 'Fijne randen': invoer pixel_values [1,4,H,W] (RGB + trimap), uitvoer alphas
export const VITMATTE = {
  q: file('Xenova/vitmatte-small-composition-1k', 'model_quantized.onnx', 27499369),
  fp32: file('Xenova/vitmatte-small-composition-1k', 'model.onnx', 103885865),
};
