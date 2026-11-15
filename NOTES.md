# AlphaMaker – logboek (lees dit eerst na een crash)

Doel: achtergrondverwijderaar bouwen op basis van het best presterende open model.
Stappen: (1) onderzoek remove.bg, (2) kandidaat-modellen verzamelen, (3) elk model
testen op een kleine testset met echte alpha-maskers, (4) evalueren, (5) bouwen.

## Crash-veiligheid (regels)
- Pc: i-4e gen, 8 threads, geen GPU, 11 GB RAM, swap 5 GB (stond al ~vol).
- Elk model draait **apart, één tegelijk**, via
  `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 ...`
  → bij te veel geheugen wordt alleen dat proces gekild, niet de pc.
- Resultaten per model worden direct naar `results/<model>.json` geschreven;
  een model met bestaand resultaat wordt overgeslagen (hervatten = script opnieuw draaien).
- Geen browser-tests met zware modellen.

## Voortgang
- [x] 2026-11-15 project aangemaakt

## Onderzoek remove.bg (2026-11-15)
- Gebouwd door Kaleido AI (Wenen), sinds 2021 van Canva. Model draait op hun servers
  (geen open gewichten); website/app = upload-client. Niet te reverse-engineeren tot model.
- Publieke API laat zien wat hun pijplijn doet: `type` (person/product/car/animal/graphic)
  → aparte modellen/routing per soort voorgrond; `semitransparency` (glas, haar);
  **"edge color corrections"**: color.jpg wijkt af van input = kleurdecontaminatie
  van randen (voorgrondkleur schatten); `crop`/`roi`/`scale`/`position`; schaduw;
  output tot 25–50 MP → masker wordt op hoge resolutie verfijnd, niet alleen 1024² opgeschaald.
- Onafh.-ish benchmark (Runflow, feb 2026, 5000 productfoto's): remove.bg 95, RMBG-2.0 93.

## Kandidaten
| model | licentie | opmerking |
|---|---|---|
| BiRefNet (general, lite, HR-matting) | MIT | basis van RMBG-2.0 |
| RMBG-2.0 (briaai) | CC BY-NC | gated |
| mohantesting/remove_background | CC BY-NC | = kopie RMBG-2.0 (zelfde bestanden/groottes), ongegated |
| RMBG-1.4 | niet-commercieel | IS-Net basis |
| BEN2 (PramaLLC) | MIT | ONNX beschikbaar |
| transparent-background (InSPyReNet) | MIT | pip, torch |
| rembg: u2net, isnet-general-use | MIT/Apache | klassiekers, licht |
Licentie: voor privégebruik zijn alle prima; voor verkoop alleen MIT/Apache.

## Benchmark-opzet
- `bench/bench.py <model>`: één model, schrijft per afbeelding naar `results/<model>.jsonl`
  (hervat automatisch), maskers naar `preds/<model>/`. Modellen in `models/` (gecachet).
- `bench/run_all.sh [modellen]`: alle modellen na elkaar, elk in cgroup met
  MemoryMax=4500M, swap 0, nice 10, 4 threads. Status in `results/<m>.status`.
  **Herstel na crash: `bench/run_all.sh` opnieuw starten** (verwijder `results/<m>.status`
  om een model opnieuw te doen).
- Testset `data/test/`: 10× DIS5K-TE1 + 8× PPM-100 (portret/haar), max 1280 px.
  Metrieken: MAE, rand-MAE (halfdoorzichtig + 8px-band rond contour), IoU@0.5, sec, piek-MB.
- Bij pauze door tokens/limiet: niets nodig, alles staat op schijf; volgende sessie: lees dit bestand.

## Einddoel publicatie
- Abel wil AlphaMaker als nieuwe repo op **Codeberg én GitHub** (vragen: openbaar/privé, vóór pushen).
