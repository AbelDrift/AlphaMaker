"""Zet InSPyReNet (transparent-background 'base', MIT) om naar ONNX: models/inspyrenet-base.onnx.
Invoer 'input' [1,3,1024,1024] (ImageNet-genormaliseerd), uitvoer 'output' [1,1,1024,1024] in 0..1."""
import sys
from pathlib import Path
import torch
from transparent_background import Remover

dst = Path(__file__).resolve().parent.parent / 'models/inspyrenet-base.onnx'
model = Remover(mode='base', device='cpu').model.eval()
x = torch.rand(1, 3, 1024, 1024)
with torch.no_grad():
    try:
        torch.onnx.export(model, x, str(dst), input_names=['input'], output_names=['output'], opset_version=17, dynamo=False)
    except Exception as e:  # nieuwe exporter als de oude ontbreekt
        print('oude exporter faalde:', str(e)[:200], file=sys.stderr)
        torch.onnx.export(model, (x,), str(dst), input_names=['input'], output_names=['output'], opset_version=18, dynamo=True)
print(dst, dst.stat().st_size // 2**20, 'MB')
