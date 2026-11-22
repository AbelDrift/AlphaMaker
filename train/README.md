# Een snel, commercieel bruikbaar achtergrondmodel trainen

**Doel:** een "licht" model voor AlphaMaker en SellYourShit dat in ongeveer 1 s per foto op de processor draait
(InSPyReNet, de huidige standaard, kost ±10 s), met randen die zo dicht mogelijk bij InSPyReNet komen, en dat
**commercieel gebruikt mag worden**.

**Hoe:** een klein netwerk (MobileNetV4-U-Net, 7 miljoen parameters) leert de maskers van InSPyReNet na
(distillatie), op je eigen productfoto's. Alles staat in `train/kit.py`.

## Waarom zo (licenties)

- **Alleen eigen foto's.** S3OD (140.000 synthetische beelden, "MIT") valt af: de beelden zijn met FLUX gemaakt,
  en de FLUX-licentie verbiedt om die uitvoer te gebruiken voor het trainen van commerciële modellen.
- **Vanaf nul, zonder voorgetrainde gewichten.** Gewichten die op ImageNet zijn voorgetraind, hebben geen
  aantoonbaar commerciële status. De code (timm, PyTorch) is Apache-2.0/BSD.
- **De leermeester is InSPyReNet (MIT, code én gewichten).** Ook InSPyReNet is deels getraind op DIS5K, waarvan de
  voorwaarden niet-commercieel zijn; zijn gewichten hebben wel een uitdrukkelijke MIT-licentie. Of zulke
  datasetvoorwaarden doorwerken op een model, is juridisch niet uitgemaakt; dat geldt voor elk open model.
- **Privacy:** `prep` haalt alle metadata weg (ook GPS-locatie) voordat de foto's de deur uit gaan.

## Kosten (prijzen 22 november 2026, excl. btw)

Met ±4.500 eigen foto's is het werk klein:

| Stap | RTX 4090 | Kosten bij RunPod Community (€0,31/u) | bij Vast.ai EU (€0,38–0,46/u) |
|---|---|---|---|
| Labelen (InSPyReNet op alle foto's) | ±10–15 min | €0,10 | €0,10 |
| Trainen (150 epochs, 768 px) | ±2–4 u | €0,60–1,25 | €0,75–1,85 |
| Opzetten, uploaden, exporteren | ±0,5–1 u | €0,15–0,30 | €0,20–0,45 |
| **Eén run** | **±3–5 u** | **±€1–2** | **±€1,50–2,50** |

Reken op 3 tot 5 rondes om de instellingen goed te krijgen: **samen ±€5–10**. Een RTX 5090 (€0,62/u) is ongeveer
1,4× zo snel en kost dus ongeveer hetzelfde. Grotere kaarten (A100, H100) zijn hier niet nodig.

**Zelf een GPU kopen?** Voor dit model is 16 GB videogeheugen genoeg; 24 GB of meer geeft ruimte om later ook grote
modellen bij te trainen. Je huidige laptop kan geen GPU krijgen, dus het wordt een nieuwe pc. Door het
geheugentekort van 2026 zijn kaarten nu duur; voor deze klus is huren veel goedkoper.

## Stappen

**1. Lokaal: foto's klaarzetten** (rechtop volgens EXIF, lange zijde ≤ 1280 px, zonder metadata, dubbele eruit,
10% apart als controleset):

```bash
python train/kit.py prep --src /pad/naar/ProductFoto/Database --out /pad/naar/ProductFoto/alphamaker-train
```

**2. Een GPU huren.** Bijvoorbeeld RunPod (Community Cloud, RTX 4090, sjabloon "PyTorch") of Vast.ai (EU, RTX 4090,
"verified"). Kies een machine met minstens 8 processorkernen: de augmentatie draait op de processor.

**3. De werkmap uploaden** (±1–2 GB), bijvoorbeeld met `scp -r alphamaker-train root@SERVER:/workspace/` of de
uploadtool van de aanbieder.

**4. Op de GPU:**

```bash
git clone https://github.com/AbelDrift/AlphaMaker && cd AlphaMaker
pip install timm onnx onnxruntime-gpu pillow scipy
python train/kit.py label --data /workspace/alphamaker-train        # maskers van de leermeester
python train/kit.py train --data /workspace/alphamaker-train --epochs 150 --batch 16 --size 768
python train/kit.py export --data /workspace/alphamaker-train --out /workspace/licht.onnx
```

`train` bewaart na elke epoch een checkpoint (`student.pt`) en gaat bij een nieuwe start verder waar hij was. Valt de
server weg, start dan hetzelfde commando opnieuw. Download aan het eind `licht.onnx` (±30 MB) en, als je wilt,
`student.pt` en `labels.jsonl`. **Stop daarna de server**, anders loopt de rekening door.

**5. Lokaal beoordelen:**

```bash
python train/kit.py eval --onnx licht.onnx --data /pad/naar/ProductFoto/alphamaker-train
```

Dit geeft per set de randfout en missers (dezelfde maat als de benchmark) en de tijd per foto: op je eigen
controlefoto's tegen InSPyReNet, en op de testsets van AlphaMaker tegen de echte maskers. Pas als het goed genoeg is,
komt het model als "licht" in AlphaMaker en SellYourShit.

## Proefrun

Op 22 november liep alles één keer op de processor met 8 foto's: `prep`, `label` (echte leermeester, 2,2 GB),
`train` (ook hervatten), `export` en `eval`. Het netwerk kost ±0,9 s per foto bij 1024 px op deze laptop
(i7-4700MQ), tegen ±10 s voor InSPyReNet.

## Close-ups zonder achtergrond

`label` schrijft per foto in `labels.jsonl` hoeveel van het beeld voorgrond is (`fg`) en hoeveel van de beeldrand
door het object wordt geraakt (`border`). Daarmee is de regel gekalibreerd die close-ups zonder achtergrond herkent;
SellYourShit slaat het weghalen van de achtergrond voor zulke foto's over.
