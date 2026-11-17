# AlphaMaker

Achtergrond verwijderen met de beste open modellen, **op je eigen apparaat**: als Python-programma (commando of lokale webpagina) of direct in je browser. Je foto gaat nergens heen.

**Webversie:** <https://abeldrift.github.io/AlphaMaker/> (rekent in je browser; WebGPU aanbevolen)

*English summary below.*

## Snel aan de slag (zonder installeren)

1. Open <https://abeldrift.github.io/AlphaMaker/>. Chrome of Edge is het snelst (WebGPU); Firefox werkt ook, maar trager.
2. Sleep een of meer foto's in het vak, of plak er een met Ctrl+V. De eerste keer wordt een model gedownload (± 85–170 MB), daarna gaat het sneller.
3. Kies eventueel een achtergrond, bijsnijden of schaduw, werk randen bij met de gum of het herstelpenseel, en klik op **Download**.

Je foto's verlaten je apparaat niet: alles wordt in je eigen browser berekend.

## Wat het doet

1. **Segmentatie** met het best gemeten open model: standaard InSPyReNet (`transparent-background`, MIT) als veilig compromis; voor portretten met veel haar BiRefNet general (`--type persoon`), voor objecten en producten RMBG-1.4 (`--type object`: beter én sneller, wel niet-commercieel). Het masker wordt op de originele maat opgeleverd.
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

> ⚠️ **Correctie (17 nov 2026): de objectscore van InSPyReNet is besmet.** Het standaard-checkpoint van transparent-background ("Plus Ultra") is volgens de model zoo van InSPyReNet getraind op onder meer DIS-VD, dus op onze objecttestbeelden. De schone InSPyReNet-versie (alleen DIS-TR) haalt op objecten 0,115 en op haar 0,108. Eerlijk gemeten is **RMBG-1.4 het beste op objecten (0,066)** en **BiRefNet general het beste op haar (0,067)**. De haarscore van InSPyReNet (0,075) is wel schoon. Zie de tabel hieronder voor de schone scores per soort foto.

**Schoon gemeten per soort foto** (150 beelden over twee sets: 100 objecten uit DIS-VD en 50 portretten uit PPM-100). De objectscores van InSPyReNet staan tussen haakjes, omdat ze besmet en dus te gunstig zijn.

| model | objecten: rand | MAE | IoU | missers | haar: rand | MAE | IoU | missers | s/beeld (CPU) |
|---|---|---|---|---|---|---|---|---|---|
| **RMBG-1.4** (NC) | **0,066** | **0,008** | **0,939** | 2 | 0,105 | 0,024 | 0,932 | 4 | **1,7** |
| RMBG-1.4 licht (NC) | 0,067 | 0,009 | 0,937 | 2 | 0,109 | 0,024 | 0,931 | 4 | 4,1 |
| **InSPyReNet** (MIT) | (0,079) | (0,011) | (0,936) | (3) | 0,075 | 0,005 | 0,989 | 0 | 14 |
| **BiRefNet general** (MIT) | 0,102 | 0,023 | 0,864 | 15 | **0,067** | **0,003** | **0,990** | 0 | 41 |
| MVANet (MIT) | 0,112 | 0,028 | 0,833 | 23 | 0,115 | 0,019 | 0,960 | 1 | 12 |
| Finegrain box (MIT) | 0,149 | 0,056 | 0,739 | 49 | 0,149 | 0,051 | 0,885 | 10 | 13 |

Een misser is een beeld met IoU < 0,8: er mist een flink stuk van het onderwerp, of er staat een stuk achtergrond bij.

- **Objecten en producten: RMBG-1.4.** Het scoort het best op alle maten en is ook het snelst, maar het is niet-commercieel. Gebruik `--type object`. Het is ook de standaard in de webversie.
- **Portretten en haar: BiRefNet general** is het nauwkeurigst, maar traag. Gebruik `--type persoon`, vergelijkbaar met `type=person` bij remove.bg. **InSPyReNet** is bijna even goed en 3× sneller.
- **Standaard in de app: InSPyReNet.** Als je niet weet wat er op de foto staat, is dit het veiligste compromis: geen enkele misser op portretten, redelijk op objecten, en een MIT-licentie. De echte objectscore is onbekend, maar zelfs met het voordeel van de besmetting verliest het daar van RMBG-1.4. RMBG-1.4 is 8× sneller, maar mist op portretten vaker een stuk (MAE 5× hoger).
- **Automatisch kiezen: `--type auto`.** De router van withoutBG (open weights v4) kijkt per foto of er haar op staat (ongeveer 1 s). Bij haar gaat de foto naar BiRefNet general, anders naar RMBG-1.4. Op de schone test levert dat objecten 0,066 en portretten 0,075 op (2 missers), dus op beide soorten het beste van twee werelden. De router herkende 37 van de 50 portretten als haar en geen enkel object. Het kost 350 MB extra download, en objecten lopen via RMBG-1.4 (niet-commercieel). Daarom is het een optie en niet de standaard.
- **Nieuwe kandidaten (nov 2026)** uit `bench/testplan-2026-11.json`: MVANet (licht en zwaar) en Finegrain scoren op beide soorten foto's slechter. MODNet (portretten) heeft slechtere haarranden dan RMBG-1.4 (0,127 tegen 0,105). De matting-tak van withoutBG is op portretten beter dan RMBG-1.4 (0,096, 1 misser), maar blijft achter bij InSPyReNet en BiRefNet. BiRefNet_dynamic past niet in 3,9 GB en wacht op meer vrij geheugen. S3OD en PDFNet worden nu getest.
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
alphamaker product.jpg --type object       # beste en snelste voor objecten (RMBG-1.4, niet-commercieel)
alphamaker *.jpg --type auto               # per foto kiezen: haar → BiRefNet general, anders RMBG-1.4
alphamaker *.jpg --voorkeur snel           # snelst: RMBG-1.4, foto's boven 3 MP eerst verkleinen
alphamaker *.jpg --voorkeur precies        # nauwkeurigst: per foto het beste model, volle resolutie
alphamaker serve                             # webpagina op http://127.0.0.1:8092 (model kiezen, afwerken)

# afwerken zoals bij remove.bg (alles lokaal):
alphamaker foto.jpg --bijsnijden --marge 0.05        # bijsnijden tot het onderwerp, 5% marge
alphamaker foto.jpg --achtergrond '#ffffff' --formaat jpg
alphamaker foto.jpg --achtergrondfoto strand.jpg --schaduw
alphamaker foto.jpg --alleen-masker                  # alleen het alfamasker (grijswaarden)
```

**Snelheid of precisie kies je zelf.** Een telefoonfoto van 12 MP kost op een gewone pc met RMBG-1.4 op volle maat 12,5 s, en met `--voorkeur snel` 4,8 s (uitvoer 2000×1500, ruim genoeg voor een advertentie). Het model zelf kost maar ~2 s; de rest is nabewerking op volle resolutie. Foto's worden rechtop gezet volgens de EXIF-gegevens van de telefoon.

Zonder PyTorch werkt AlphaMaker ook, maar dan met het lichtere IS-Net (ONNX). Modellen worden bij eerste gebruik gedownload: InSPyReNet via `transparent-background`, BiRefNet naar `~/.cache/alphamaker` (code en gewichten van vaste revisies). Reken op zo'n 4 GB vrij geheugen; `--model isnet-general` heeft genoeg aan 1 GB.

## Webversie

Staat in [`docs/`](docs/) en heeft geen bouwstap: onnxruntime-web 1.31.1 met WebGPU (grafische kaart) of WebAssembly (processor). Modellen komen van Hugging Face en worden in de browser bewaard.

- InSPyReNet bestaat niet als browsermodel. De standaard is daarom RMBG-1.4: beste randen op de schone testset, klein (84–167 MB) en snel, maar niet-commercieel (staat erbij).
- RMBG-1.4 is er ook in een lichte versie (42 MB). Die is 2,8% minder nauwkeurig (rand-MAE 0,081 tegen 0,079 op 150 beelden) en op oudere processors niet sneller (Firefox: 8,2 s tegen 7,0 s). Handig bij een trage verbinding.
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

De router voor `--type auto` komt uit withoutBG open weights v4 (`withoutbg/withoutbg-openweights-onnx`, revisie 232aa520): Apache-2.0 plus de voorwaarden van DINOv3. Built with DINOv3. Hun modelpagina noemt daarnaast een aparte commerciële licentie; check die zelf bij commercieel gebruik.

---

## English summary

AlphaMaker removes image backgrounds locally, as a Python CLI/local web page or fully in-browser (GitHub/Codeberg Pages, WebGPU/WASM).

On clean benchmarks (150 images: DIS5K-VD objects and PPM-100 portraits, kept out of the models' training data), **RMBG-1.4** (non-commercial) is the best and fastest on objects (boundary error 0.066, 1.7 s per image) and is the browser default. **BiRefNet general** (MIT) is the best on hair (0.067), but slow. **InSPyReNet** (transparent-background, MIT) comes close on hair (0.075) at 3× the speed of BiRefNet, and is the CLI default as the safe compromise. Use `--type object` or `--type persoon` to pick the specialist. InSPyReNet's default checkpoint was trained on DIS-VD, so its object score is not reliable.

An earlier test on DIS5K-TE1 favoured BiRefNet, but its weights were trained on those test images. Edge refinement (ViTMatte, guided filter, high-res tile re-segmentation) and model averaging did not help. Foreground colour decontamination ("blur fusion") is applied to avoid halos.

The code is MIT; model licenses vary.
