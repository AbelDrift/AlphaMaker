# AlphaMaker

Achtergrond verwijderen met de beste open modellen, **op je eigen apparaat**: als Python-programma (commando of lokale webpagina) of direct in je browser. Je foto gaat nergens heen.

**Webversie:** <https://abeldrift.github.io/AlphaMaker/> (rekent in je browser; WebGPU aanbevolen)

*English summary below.*

## Wat het doet

1. **Segmentatie** met het best gemeten open model: standaard InSPyReNet (`transparent-background`, MIT), voor portretten met veel haar BiRefNet general. Het masker wordt op de originele maat opgeleverd.
2. **Randkleurcorrectie** ("blur fusion", Forte & Pitié 2021): de kleur van de oude achtergrond wordt uit haar en randen gehaald, zodat er geen gekleurde halo overblijft op een nieuwe achtergrond. remove.bg doet iets vergelijkbaars ("edge color corrections").
3. Optioneel, experimenteel: **fijne randen** met ViTMatte, in tegels op volle resolutie. In onze benchmark werd dat niet beter (zie hieronder), daarom staat het standaard uit.

## Benchmark

**Schone testset** (`data/test_clean`): 40 objecten uit DIS5K-VD en 10 portretten met haar uit PPM-100. Voor zover bekend zit geen van deze beelden in de training van de geteste modellen. Alles draaide op een oude CPU (Intel 4e generatie, geen GPU), elk model apart en begrensd op 4,5 GB.

| model | rand-MAE ↓ | objecten | portretten (haar) | MAE ↓ | IoU ↑ | s/beeld | piek | licentie |
|---|---|---|---|---|---|---|---|---|
| **InSPyReNet** (transparent-background) | 0,079 | 0,079 | 0,081 | **0,010** | **0,947** | 15 | 3,8 GB | MIT |
| RMBG-1.4 | **0,075** | **0,067** | 0,110 | 0,016 | 0,923 | 1,7 | 1,1 GB | niet-commercieel |
| RMBG-2.0 (alleen benchmark) | 0,089 | 0,092 | 0,076 | 0,013 | 0,914 | 42 | 4,3 GB | CC BY-NC 4.0 |
| BiRefNet general | 0,096 | 0,103 | **0,069** | 0,020 | 0,888 | 41 | 4,3 GB | MIT |
| IS-Net general | 0,103 | 0,094 | 0,137 | 0,013 | 0,932 | 1,6 | 1,0 GB | Apache-2.0 |
| BEN2 | 0,123 | 0,125 | 0,116 | 0,025 | 0,856 | 16 | 3,9 GB | MIT |
| BiRefNet lite | 0,125 | 0,137 | 0,077 | 0,034 | 0,825 | 28 | 2,5 GB | MIT |

**Bevestigd op een tweede schone set** (60 andere DIS-VD-beelden en 40 PPM-train-portretten, samen 150 beelden). De rand-MAE over alle 150 beelden: InSPyReNet 0,078 (objecten 0,079 / haar 0,075, MAE 0,009, IoU 0,954), RMBG-1.4 0,079 (0,066 / 0,105), BiRefNet general 0,090 (0,102 / **0,067**), IS-Net 0,108.

- **InSPyReNet is de beste allrounder.** Het heeft de laagste totale fout en de beste overlap, is even goed op objecten als op haar, en is 3× sneller dan BiRefNet. Daarom is het de standaard in de app.
- **RMBG-1.4** maakt de mooiste randen op objecten, maar mist vaker een heel stuk (lagere IoU) en mag niet commercieel gebruikt worden. Het is klein en snel, en daarom de standaard in de webversie, met de licentie erbij vermeld.
- **BiRefNet general** is het beste voor haar (0,069). Voor portretten: `--model birefnet-general`, vergelijkbaar met de keuze `type=person` bij remove.bg.
- **Combineren** (maskers middelen) leverde niets op. InSPyReNet + RMBG-1.4 gaf iets betere randen (0,075) maar een hogere totale fout; met BiRefNet erbij werd het slechter.
- **rand-MAE** is de gemiddelde fout in de randzone (halfdoorzichtige pixels plus een band van 8 px rond de contour); daar zit het verschil in haar en fijne details. **MAE** is de fout over het hele beeld, **IoU** de overlap van het masker.
- **Beperkingen:** 50 beelden, één meting per model. Het verschil tussen de top-2 is klein.

**Les: eerst controleren waarop een model getraind is.** De eerste meting gebruikte DIS5K-TE1. Daar won BiRefNet general ruim (rand-MAE 0,063 tegen 0,079 voor InSPyReNet). Volgens de model zoo van BiRefNet zijn de general-gewichten echter getraind op onder meer "DIS-TEs", dus op die testbeelden zelf. Op de schone set zakt BiRefNet op objecten van 0,054 naar 0,103.

**Andere experimenten** (eerste testset en hoge resolutie):
- **Randverfijning hielp niet.** ViTMatte op volle resolutie en een guided filter maakten de randen steeds slechter (BiRefNet lite: 0,077 → 0,084–0,094).
- **Hoge resolutie** (12 foto's van ~3000 px): het model opnieuw draaien op randtegels van 1024 px hielp BiRefNet lite iets bij haar, maar maakte objecten slechter en duurde 10× zo lang. Het blijft daarom alleen als onderzoeksfunctie (`refine.tile_refine`).
- **BiRefNet HR-matting** (2048 px) past niet in 4,5 GB. De ONNX-versies van BiRefNet hebben op de CPU meer dan 4,5 GB nodig; via PyTorch is het 2,5–4,3 GB.
- De `mohantesting/remove_background`-repo op Hugging Face is een ongegatede kopie van RMBG-2.0: dezelfde bestanden, dezelfde niet-commerciële licentie.

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
pip install transparent-background timm kornia einops "transformers<5" safetensors
pip install -e .

alphamaker foto.jpg                          # → foto-alpha.png
alphamaker *.jpg -o uitvoer/
alphamaker portret.jpg --type persoon       # beste model voor haar (BiRefNet general)
alphamaker serve                             # webpagina op http://127.0.0.1:8092

# afwerken zoals bij remove.bg (alles lokaal):
alphamaker foto.jpg --bijsnijden --marge 0.05        # bijsnijden tot het onderwerp, 5% marge
alphamaker foto.jpg --achtergrond '#ffffff' --formaat jpg
alphamaker foto.jpg --achtergrondfoto strand.jpg --schaduw
alphamaker foto.jpg --alleen-masker                  # alleen het alfamasker (grijswaarden)
```

Zonder PyTorch werkt AlphaMaker ook, maar dan met het lichtere IS-Net (ONNX). Modellen worden bij eerste gebruik gedownload: InSPyReNet via `transparent-background`, BiRefNet naar `~/.cache/alphamaker` (code en gewichten van vaste revisies). Reken op zo'n 4 GB vrij geheugen; `--model isnet-general` heeft genoeg aan 1 GB.

## Webversie

Staat in [`docs/`](docs/) en heeft geen bouwstap: onnxruntime-web 1.31.1 met WebGPU (grafische kaart) of WebAssembly (processor). Modellen komen van Hugging Face en worden in de browser bewaard.

- InSPyReNet bestaat niet als browsermodel. De standaard is daarom RMBG-1.4: beste randen op de schone testset, klein (84–167 MB) en snel, maar niet-commercieel (staat erbij).
- Met WebGPU kun je ook BiRefNet kiezen, het beste voor haar (466 MB). Zonder WebGPU past BiRefNet niet in de 4 GB die WebAssembly maximaal mag gebruiken.
- Foto's groter dan 12 megapixel worden eerst verkleind.
- Afwerken zoals in de Python-app: achtergrondkleur of -foto, bijsnijden tot het onderwerp (met marge) en schaduw; downloaden als PNG, JPG of alleen het masker.
- Meerdere foto's tegelijk: ze worden na elkaar verwerkt en zijn samen te downloaden.
- Met de hand bijwerken: een gum en een herstelpenseel (originele pixels terug), zoals in de editor van remove.bg.

De rekenstappen zijn getest tegen de Python-versie (`node --test tests/web/refine.test.mjs`). Een rooktest in headless Firefox (`tests/web/run_browser_smoke.sh`) verwerkt een echte foto: zonder WebGPU klaar in 27 s, zonder consolefouten.

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

AlphaMaker removes image backgrounds locally, as a Python CLI/local web page or fully in-browser (GitHub/Codeberg Pages, WebGPU/WASM).

On a clean 50-image benchmark (DIS5K-VD + PPM-100, not in the models' training data), **InSPyReNet (transparent-background, MIT)** was the best all-rounder: lowest MAE (0.010), best IoU (0.947) and a boundary error of 0.079. It is the default. RMBG-1.4 had the best object edges (0.075, non-commercial) and is the browser default. BiRefNet general was best on hair (0.069).

An earlier test on DIS5K-TE1 favoured BiRefNet, but its weights were trained on those test images. Edge refinement (ViTMatte, guided filter, high-res tile re-segmentation) and model averaging did not help. Foreground colour decontamination ("blur fusion") is applied to avoid halos.

The code is MIT; model licenses vary.
