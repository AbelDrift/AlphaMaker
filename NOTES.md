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
- Benchmark draait als systemd-taak (overleeft crash van Claude/terminal):
  `systemctl --user status alphamaker-bench`, log: `results/run.log`.
  Opnieuw starten (na reboot): `systemd-run --user --unit=alphamaker-bench --working-directory=$HOME/claude/AlphaMaker -p OOMPolicy=continue bash -c 'bench/run_all.sh >> results/run.log 2>&1'`
- Samenvatting: `.venv/bin/python bench/summarize.py` → `results/SUMMARY.md`.
- 2026-11-16 00:07: birefnet-lite gekild op 4,5 GB (rc 137) — limiet werkte, pc heel.
  Daarna in bench.py + core.py: `enable_mem_pattern=False` en `session.disable_prepacking=1`
  (ORT plant anders alle tussenresultaten vooraf + dupliceert gewichten). Mislukte modellen
  opnieuw: `rm results/<m>.status` en run_all.sh opnieuw. Desktop gebruikt zelf ~6 GB.
- App-skelet (nog niet getest): `alphamaker/core.py` (modellen, segment, cutout),
  `alphamaker/refine.py` (trimap, ViTMatte in tegels op volle res, guided filter,
  randkleurcorrectie blur-fusion), `alphamaker/__main__.py` (CLI + `serve` op 127.0.0.1:8092),
  `alphamaker/static/index.html` (UI uit Knipsel). DEFAULT_MODEL voorlopig birefnet-general.
- Volgende stap na benchmark: `bench/refine_bench.py <beste-model> vm8|vm16|gf4` → beste
  verfijning kiezen, DEFAULT_MODEL + band in core.py zetten, end-to-end test, README, repo's.
- claude-eyes: bekeken, niet geïnstalleerd (is voor video; foto's kan Claude zelf lezen).

## Publicatie-afspraken (2026-11-16)
- Repo's **openbaar**. Abel maakt zelf lege repo's AbelDrift/AlphaMaker op GitHub + Codeberg;
  ik push via SSH (key werkt voor beide, user AbelDrift). Geen push vóór code klaar is.
- Webversie in `docs/` (GitHub Pages: main /docs) + `pages`-branch voor Codeberg Pages.
  Browser kan alleen modellen van Hugging Face laden (CORS), niet van GitHub-releases.
- 2026-11-16 00:25: ook rmbg-2.0 en birefnet-general (ONNX) gekild op 4,5 GB → BiRefNet-familie
  via **PyTorch** (`bench/pt_models.py`, namen `pt-*`). Code van ZhengPeng7 (officieel) en
  mohantesting nagelezen: alleen versie-/opmerkingverschillen, geen netwerk/systeemaanroepen;
  vaste revisies; gewichten als safetensors. Tweede taak `alphamaker-bench2` start vanzelf
  na `alphamaker-bench` (status: `systemctl --user status alphamaker-bench2`).

## Webversie (GitHub/Codeberg Pages) — 2026-11-16 00:30
- Gebouwd door workflow `alphamaker-webversie` (bouwer → 4 reviewers → verificatie → fixer),
  run-id `wf_6241ea65-43a`. Hervatten na crash: Workflow met
  scriptPath `~/.claude/projects/-home-abel-claude-algemeen/63eeee7a-6517-415a-ba3c-8a163a78d5ff/workflows/scripts/alphamaker-webversie-wf_6241ea65-43a.js`
  en resumeFromRunId `wf_6241ea65-43a` (klaar-agents komen uit cache).
- Ontwerp: `docs/` (index.html, app.js, worker.js, refine.js, models.js), onnxruntime-web 1.31.1,
  WebGPU of WASM, modellen alleen van Hugging Face (Cache API), max 12 MP,
  pariteitstests `tests/web/` tegen `alphamaker/refine.py` via `bench/make_web_fixtures.py`.
- Browsermodellen: birefnet-lite (MIT), rmbg-1.4 (NC), ben2 + birefnet (alleen WebGPU),
  ViTMatte Xenova (q 26 MB / fp32 99 MB). onnx-community/BEN2 gebruikt ImageNet-normalisatie
  (anders dan PramaLLC-onnx in de Python-bench, die zonder normalisatie werkt).

## Stand 2026-11-16 04:35 (na pauze door gebruikslimiet)
- Benchmark (rand-MAE ↓): pt-rmbg-2.0 0.0710 (41 s, 4,2 GB, NC) · pt-birefnet-lite 0.0769
  (28 s, 2,6 GB, MIT) · inspyrenet 0.0792 (15 s, 4,2 GB, MIT) · rmbg-1.4 0.0867 · ben2 0.1086 ·
  isnet 0.1137 · portrait goed op haar maar slecht op objecten. Ensembles met zwakkere modellen: niet beter.
- pt-birefnet-general/hr faalden op fp16-gewichten → `model.float()`; herkansing in `alphamaker-bench3`
  (`bench/queue2.sh`: general, dan verfijning vq16/vm16/vm8/gf4 op lite + vm16 op rmbg-2.0, dan HR).
  `alphamaker-bench4`: ONNX birefnet-lite opnieuw (meet of hij in browser-WASM (4 GB) past).
- Webversie-workflow gestopt na de bouwer (reviewers liepen vast op limiet); zelf nagekeken:
  tests 10/10, ORT-bestanden bestaan, geen innerHTML. Open: standaardmodel per rekenkern.
- App: BiRefNet-familie + RMBG-2.0 nu via PyTorch (`alphamaker/pt.py`), standaard birefnet-lite
  (terugval isnet-general zonder torch). ONNX-varianten daarvan uit core verwijderd (te veel geheugen).

## Besluiten 2026-11-16 05:00 (na volledige benchmark)
- pt-birefnet-general wint (rand-MAE 0.0629, MIT, 38 s, 4,0 GB) → standaard in Python-app.
- ViTMatte/guided filter maken randen slechter (lite 0.077 → 0.084–0.094) → verfijning standaard
  UIT, optie `--fijne-randen` / vinkje "experimenteel". Kanttekening: testbeelden ≤1280 px.
- ONNX birefnet-lite ook met zuinige ORT-opties >4,5 GB → past niet in browser-WASM (4 GB):
  web-standaard WebGPU = birefnet (fp16), WASM = rmbg-1.4 (NC, duidelijk vermeld).
- HR-matting (2048) past niet in 4,5 GB. Codeberg Pages vraagt nu een git-pages-webhook (Abel).
- End-to-end getest: CLI met birefnet-general op portret (51 s, 4,3 GB, visueel schoon haar);
  `serve` met isnet: GET 200, POST foto → RGBA-PNG, ongeldige invoer → 400.
- Open: webversie nog niet in echte browser getest; remove.bg-vergelijking vraagt API-sleutel.

## Onderzoek remove.bg, deel 2 (2026-11-16 05:15)
- **Voorwaarden remove.bg (remove.bg/tos, art. 5.2):** verboden: (ii) reverse-engineeren/decompileren,
  (iii) gebruik voor prestatiebenchmarks, (iv) gebruik om een concurrerend product te bouwen,
  (vi) data mining/scraping. → GEEN vergelijking via hun API (eerder voorstel ingetrokken), geen
  analyse van hun client/dienst. Alleen openbare bronnen. Art. 6.9: remove.bg krijgt een eeuwigdurende
  licentie op uploads (privacy-argument voor lokaal werken).
- Openbaar gevonden: Kaleido bouwde voort op bestaande ontwerpen (trendingtopics); Scale-case study
  (via zoekfragment): randgevallen aangepakt met ~500 echte + ~15.000 synthetische beelden;
  50 MP-aankondiging: originele resolutie tot 50 MP (ZIP/JPG), PNG max 10 MP. Geen octrooien of papers
  van Kaleido gevonden (USPTO-zoekresultaten waren van andere bedrijven).
- Volgende experiment: hoge resolutie (testset `data/test_hr`, ≤3000 px): basis vs. BiRefNet op
  randtegels op volle resolutie ('st') vs. ViTMatte vs. guided filter.
- HR-experiment draait: unit `alphamaker-hr` (`bench/queue_hr.sh`, TESTSET=test_hr), resultaten in
  `results/test_hr/`, samenvatting: `TESTSET=test_hr .venv/bin/python bench/summarize.py`.
  Nieuwe functie `refine.tile_refine` (basismodel opnieuw op randtegels op volle resolutie).
- 2026-11-16 ~07:00: webversie LIVE op https://abeldrift.github.io/AlphaMaker/ (Pages: main /docs) en
  https://abeldrift.codeberg.page/AlphaMaker/ (git-pages-webhook; deploy = nieuwe push naar `pages`:
  `git subtree split --prefix docs -b pages && git push -f codeberg pages`). Abel besloot: repo's blijven openbaar.

## Hoge resolutie + browsertest (2026-11-16 07:45)
- test_hr (12×~3000 px), rand-MAE: general 0.1224 (37 s) · lite+st24 0.1346 (285 s; haar 0.119 maar
  objecten 0.165) · general+st24 0.1374 (392 s) · lite 0.1391 · lite+vm24 0.1404 · lite+gf8 0.1496.
  → Standaard (general, geen verfijning) blijft het beste, ook op hoge resolutie. tile_refine blijft
  alleen als onderzoeksfunctie (niet in de app-opties).
- Webversie heeft nu ook afwerkopties (achtergrondfoto, bijsnijden+marge, schaduw); rooktest controleert
  bijsnijden+schaduw+kleur (761×853, volledig dekkend).
- Browser-rooktest (headless Firefox 153 ESR, WebDriver BiDi, WASM 1 kern, rmbg-1.4-licht):
  klaar in 27 s, 1280×853, 63% transparant, 4,7% halfdoorzichtig, 0 consolemeldingen. Script:
  `tests/web/run_browser_smoke.sh [foto] [model]` (max 3 GB). WebGPU-pad niet getest (geen GPU hier).

## Besmetting testset ontdekt (2026-11-16 08:00)
- BiRefNet-README (model zoo): general-gewichten getraind op DIS5K-TR + **DIS-TEs** + DUTS/HRSOD/UHRSD/HRS10K
  (TR+TE) + P3M-10k + TE-P3M-500 + TR-humans. → onze dis_-beelden (DIS-TE1) zitten in de training van
  BiRefNet general/lite: DIS-scores van BiRefNet zijn te rooskleurig. PPM-100 staat er niet in (schoon).
  Makers testen zelf op DIS-VD.
- Nieuwe schone set `data/test_clean`: 40× DIS-VD + 11× PPM-100-val (≤1280 px), opnieuw meten.
- Schone benchmark draait: unit `alphamaker-clean` (TESTSET=test_clean; 40 DIS-VD + 10 PPM), resultaten
  `results/test_clean/`, samenvatting `TESTSET=test_clean .venv/bin/python bench/summarize.py`.

## Schone benchmark — eindstand (2026-11-16 10:00)
- test_clean (40 DIS-VD + 10 PPM), rand-MAE: rmbg-1.4 0.0751 · **inspyrenet 0.0793 (MAE 0.0096, IoU 0.947,
  15 s, MIT)** · rmbg-2.0 0.0892 · birefnet-general 0.0962 (haar 0.0694 = beste) · isnet 0.1026 ·
  ben2 0.1233 · birefnet-lite 0.1253. Ensembles: insp+rmbg1.4 0.0748 (MAE slechter), insp+birefnet 0.0860.
- BESLUIT: app-standaard **inspyrenet** (transparent-background); birefnet-general aanbevolen voor portretten.
  Web-standaard **rmbg-1.4** (beide backends; NC vermeld), BiRefNet optioneel ("beste voor haar").
- E2E: CLI met inspyrenet + --bijsnijden op DIS-fietsframe: 25 s, 3,7 GB, RGBA bijgesneden. README herschreven.
