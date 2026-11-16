# AlphaMaker

Achtergrond verwijderen met de beste open modellen, **op je eigen apparaat**: als Python-programma (commando of lokale webpagina) of direct in je browser. Je foto gaat nergens heen.

**Webversie:** <https://abeldrift.github.io/AlphaMaker/> (rekent in je browser; WebGPU aanbevolen)

*English summary below.*

## Wat het doet

1. **Segmentatie** met het best gemeten open model (standaard BiRefNet, MIT-licentie) op 1024×1024. Het masker wordt daarna teruggeschaald naar de originele maat.
2. **Randkleurcorrectie** ("blur fusion", Forte & Pitié 2021): de kleur van de oude achtergrond wordt uit haar en randen gehaald, zodat er geen gekleurde halo overblijft op een nieuwe achtergrond. remove.bg doet iets vergelijkbaars ("edge color corrections").
3. Optioneel, experimenteel: **fijne randen** met ViTMatte, in tegels op volle resolutie. In onze benchmark werd dat niet beter (zie hieronder), daarom staat het standaard uit.

## Benchmark

18 testbeelden met echte alpha-maskers: 10 lastige objecten uit DIS5K-TE1 en 8 portretten met haar uit PPM-100, verkleind tot hooguit 1280 px. Alles draaide op een oude CPU (Intel 4e generatie, geen GPU), elk model apart en begrensd op 4,5 GB geheugen.

| model | rand-MAE ↓ | MAE ↓ | IoU ↑ | s/beeld (CPU) | piekgeheugen | licentie |
|---|---|---|---|---|---|---|
| **BiRefNet general** (PyTorch) | **0,0629** | **0,0033** | **0,980** | 38 | 4,0 GB | MIT |
| RMBG-2.0 (PyTorch, alleen benchmark) | 0,0710 | 0,0040 | 0,976 | 41 | 4,2 GB | CC BY-NC 4.0 |
| BiRefNet lite (PyTorch) | 0,0769 | 0,0047 | 0,971 | 28 | 2,6 GB | MIT |
| InSPyReNet (transparent-background) | 0,0792 | 0,0053 | 0,976 | 15 | 4,2 GB | MIT |
| RMBG-1.4 (ONNX) | 0,0867 | 0,0252 | 0,934 | 2,3 | 1,1 GB | niet-commercieel |
| BEN2 (ONNX) | 0,1086 | 0,0084 | 0,955 | 19 | 3,9 GB | MIT |
| IS-Net general (ONNX) | 0,1137 | 0,0084 | 0,962 | 1,7 | 1,1 GB | Apache-2.0 |
| U²-Net (ONNX) | 0,2762 | 0,1058 | 0,684 | 0,6 | 0,8 GB | Apache-2.0 |

- **rand-MAE** is de gemiddelde fout in de randzone (halfdoorzichtige pixels plus een band van 8 px rond de contour). Daar zie je verschil in haar en fijne details. **MAE** is de fout over het hele beeld, **IoU** de overlap van het masker.
- **Randverfijning hielp niet.** BiRefNet lite + ViTMatte gaf een rand-MAE van 0,084–0,090 (afhankelijk van de bandbreedte) tegenover 0,077 zonder. Een guided filter gaf 0,094. Bij foto's van veel meer dan 1280 px kan verfijning op volle resolutie wel helpen, maar dat is niet gemeten.
- **Hoge resolutie** (12 foto's van ~3000 px, `data/test_hr`): BiRefNet general bleef het beste, met een rand-MAE van 0,122 in 37 s. Het model nogmaals draaien op randtegels van 1024 px op volle resolutie hielp BiRefNet lite iets bij haar (0,132 → 0,119), maar maakte objecten slechter en BiRefNet general in totaal slechter (0,137), terwijl het 10× zo lang duurde. ViTMatte (0,140) en een guided filter (0,150) hielpen ook hier niet.
- **Combineren hielp niet:** het gemiddelde van InSPyReNet met een zwakker model was niet beter dan InSPyReNet alleen.
- **BiRefNet-portrait** is sterk op haar (0,075), maar zwak op objecten. **BiRefNet HR-matting** (2048 px) paste niet in 4,5 GB.
- **Beperkingen:** 18 beelden is een kleine set en elke variant is één keer gemeten. Het verschil tussen de top-3 is klein; met meer beelden kan de volgorde veranderen.
- De `mohantesting/remove_background`-repo op Hugging Face is een ongegatede kopie van RMBG-2.0: zelfde bestanden en groottes, zelfde niet-commerciële licentie.

## Hoe remove.bg werkt (voor zover openbaar)

remove.bg is gebouwd door Kaleido (Wenen) en sinds 2021 van Canva. Het model zelf is gesloten en draait op hun servers. De website en app sturen alleen je foto op, dus het model valt van buitenaf niet te achterhalen. Hun publieke API laat wel zien wat hun pijplijn doet:

- een **soort voorgrond** kiezen (persoon, product, auto, dier, grafisch): aparte afstemming per categorie;
- **halfdoorzichtigheid** (glas, haar) als aparte optie;
- **randkleurcorrectie**: hun voorgrondkleur wijkt af van de invoer;
- uitvoer **tot 25–50 megapixel**: het masker wordt dus op hoge resolutie gemaakt, niet alleen opgeschaald;
- bijsnijden, regio, schaal, positie, schaduw en achtergrondkleur of -foto.

Een onafhankelijke-achtige test (Runflow, feb 2026, 5000 productfoto's) gaf remove.bg 95 punten en RMBG-2.0 93. In onze test scoort BiRefNet general beter dan RMBG-2.0.

AlphaMaker is niet met remove.bg vergeleken, en is ook niet op hun dienst gebaseerd. Hun voorwaarden (art. 5.2) verbieden reverse engineering en het gebruik van de dienst voor benchmarks of voor het bouwen van een concurrerend product. Daarom zijn alleen openbare bronnen gebruikt: documentatie, aankondigingen en voorwaarden.

## Installeren en gebruiken (Python)

```bash
git clone https://github.com/AbelDrift/AlphaMaker && cd AlphaMaker
python3 -m venv .venv && . .venv/bin/activate
# PyTorch voor de CPU (veel kleiner dan de standaardversie met CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm kornia einops "transformers<5" safetensors
pip install -e .

alphamaker foto.jpg                          # → foto-alpha.png
alphamaker *.jpg -o uitvoer/ --model birefnet-lite
alphamaker serve                             # webpagina op http://127.0.0.1:8092

# afwerken zoals bij remove.bg (alles lokaal):
alphamaker foto.jpg --bijsnijden --marge 0.05        # bijsnijden tot het onderwerp, 5% marge
alphamaker foto.jpg --achtergrond '#ffffff' --formaat jpg
alphamaker foto.jpg --achtergrondfoto strand.jpg --schaduw
alphamaker foto.jpg --alleen-masker                  # alleen het alfamasker (grijswaarden)
```

Zonder PyTorch werkt AlphaMaker ook, maar dan met het lichtere IS-Net (ONNX). Modellen worden bij eerste gebruik gedownload naar `~/.cache/alphamaker` (code en gewichten van vaste revisies). Reken voor BiRefNet general op zo'n 4 GB vrij geheugen; `--model birefnet-lite` heeft genoeg aan 2,6 GB.

## Webversie

Staat in [`docs/`](docs/) en heeft geen bouwstap: onnxruntime-web 1.31.1 met WebGPU (grafische kaart) of WebAssembly (processor). Modellen komen van Hugging Face en worden in de browser bewaard.

- Met WebGPU is BiRefNet de standaard (466 MB, eenmalig downloaden).
- Zonder WebGPU wordt het RMBG-1.4 (niet-commercieel). De BiRefNet-modellen passen in ONNX niet binnen de 4 GB die WebAssembly maximaal mag gebruiken.
- Foto's groter dan 12 megapixel worden eerst verkleind.
- Afwerken zoals in de Python-app: achtergrondkleur of -foto, bijsnijden tot het onderwerp (met marge) en schaduw.

De rekenstappen zijn getest tegen de Python-versie (`node --test tests/web/`). Een rooktest in headless Firefox (`tests/web/run_browser_smoke.sh`) verwerkt een echte foto: zonder WebGPU klaar in 27 s, zonder consolefouten.

## Benchmark zelf draaien

```bash
python bench/prep.py                 # testset uit DIS5K-TE1 + PPM-100 (eerst downloaden naar data/raw)
bench/run_all.sh                     # elk model apart, met geheugengrens
python bench/summarize.py            # → results/SUMMARY.md
```

Zie [`NOTES.md`](NOTES.md) voor het volledige logboek: onderzoek, keuzes, mislukte pogingen.

## Licenties

De code van AlphaMaker valt onder MIT. De modellen hebben hun eigen licenties, zie de tabel. RMBG-1.4 en RMBG-2.0 zijn alleen voor niet-commercieel gebruik. RMBG-2.0 is alleen in de benchmark gebruikt, niet in de app. RMBG-1.4 is in de webversie de standaard zonder WebGPU; voor commercieel gebruik kies je daar een ander model of een licentie van Bria.

---

## English summary

AlphaMaker removes image backgrounds locally, as a Python CLI/local web page or fully in-browser (GitHub Pages, WebGPU/WASM). On an 18-image benchmark with real alpha mattes (DIS5K-TE1 + PPM-100), **BiRefNet general (MIT)** had the lowest boundary error (0.063), ahead of RMBG-2.0 (0.071), BiRefNet lite (0.077) and InSPyReNet (0.079). It is the default.

Two things did not help on this set: ViTMatte edge refinement and guided filtering made results worse, and averaging models did not improve on the best single model. Foreground colour decontamination ("blur fusion") is applied to avoid halos. The code is MIT; model licenses vary (RMBG = non-commercial).
