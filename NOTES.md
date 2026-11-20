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
- 2026-11-16 ~10:30: CLI `--type persoon` (→ birefnet-general); `tests/test_core.py` (afwerking + CLI, zonder
  modellen). Webversie: downloaden als PNG/JPG/alleen masker. Lokale `serve`-pagina heeft nu dezelfde
  afwerking (code overgenomen uit docs/app.js — houd gelijk); getest in headless Firefox tegen serve (isnet):
  2,4 s, afwerking ok, 0 consolemeldingen. Rooktest werkt ook tegen serve (foto als data-URL ≤128 KB).
- 2026-11-16 ~11:00: webversie bulk (wachtrij, lijst, 'Alles downloaden', max 20 in geheugen); rooktest biedt 2 foto's aan (2 in lijst ok). Live GitHub+Codeberg doorstaan de rooktest (26 s, 0 fouten).

## Stand en volgende stappen (2026-11-16 11:00)
- KLAAR: onderzoek (openbaar; remove.bg-ToS verbiedt RE/benchmark), schone benchmark (inspyrenet standaard,
  birefnet-general voor haar, web rmbg-1.4), app (CLI/serve) + webversie met afwerking/formaten/bulk,
  alles getest (CLI, serve, lokale en live webversie in headless Firefox), README eerlijk.
- KLAAR 11:15: gum/herstelpenseel in webversie (rooktest: alpha 15>0>255).
- IDEEËN: WebGPU-pad testen op een pc met GPU; InSPyReNet naar ONNX voor de webversie (MIT-alternatief
  voor rmbg-1.4); grotere schone testset (meer DIS-VD/P3M); automatische persoonsdetectie voor --type.
- 2026-11-16 11:20: grotere schone set test_clean2 (60 DIS-VD andere rijen + 40 PPM-train), unit alphamaker-clean2 (isnet, rmbg-1.4, inspyrenet, birefnet-general).
- 2026-11-16 ~13:00: test_clean2 klaar. Gepoold 150 beelden rand-MAE: inspyrenet 0.0776 (MAE 0.0090, IoU 0.954) · rmbg-1.4 0.0789 (obj 0.0660) · birefnet-general 0.0903 (haar 0.0670) · isnet 0.1084 → keuzes bevestigd.
- 2026-11-16 ~13:30: GitHub Actions CI groen (syntax, pariteit web↔Python, Python-tests; Node 22 vraagt expliciet testbestand). Tegoed ~96%: werk staat volledig gepusht; volgende sessie: zie 'Stand en volgende stappen'.
- 2026-11-16 ~13:45: InSPyReNet→ONNX-export (`bench/export_inspyrenet.py`, invoer 1024², ImageNet-norm., uitvoer 0..1)
  gekild op 4,5 GB na 22 s. Niet hoger gezet: swap 100% vol, 5,3 GB vrij → crashrisico. Conclusie: ook in de
  browser waarschijnlijk alleen via WebGPU haalbaar (WASM max 4 GB). Export op pc met ≥16 GB RAM; daarna
  ORT-geheugen meten, fp16 maken, op HF zetten (Abels account) en in docs/models.js opnemen.
- 2026-11-16 ~14:00: serve-pagina met modelkeuze (/api/models, ?model= gevalideerd → 400 bij onbekend); core geeft vorig zwaar model vrij bij wissel. README snelstart; CLI-voortgang [n/N].
- 2026-11-16 ~14:15: zelfreview app.js: 2 fouten hersteld (Alles downloaden zette originele foto niet om → herstelpenseel gebruikte verkeerde foto; kader niet herberekend na penseel). Rooktest groen.
- 2026-11-16 ~14:30: serve-pagina heeft nu ook gum/herstelpenseel (zelfde code als docs/app.js); getest in Firefox: alpha 145>0>255, 0 consolemeldingen. CI groen (e03b924).

## Nieuwe modellen zoeken (2026-11-16, na reset weektegoed)
- Workflow `alphamaker-nieuwe-modellen` (run `wf_ceeab066-8ef`): 4 zoekers (HF, papers, GitHub/ComfyUI,
  bedrijven) → adversariële verificatie per kandidaat → testplan. Hervatten: Workflow met scriptPath
  `~/.claude/projects/-home-abel-claude-algemeen/63eeee7a-6517-415a-ba3c-8a163a78d5ff/workflows/scripts/alphamaker-nieuwe-modellen-wf_ceeab066-8ef.js`
  + resumeFromRunId. Daarna: kandidaten één voor één benchmarken op test_clean + test_clean2 (150 beelden).
- 2026-11-16 ~19:45: webversie: (1) volle resolutie tot 50 MP (rekenen op 12 MP, alfa/randkleur opschalen, dekkende pixels uit origineel; test 17 MP → 5100×3400), (2) service worker coi.js → crossOriginIsolated → WASM multithread: 27,1 → 11,8 s in Firefox, (3) unieke namen in lijst, race bij opschalen voorkomen (busy tot klaar).
- 2026-11-16 ~20:30: lichte RMBG-1.4 op 150 beelden: rand 0.0811 vs zwaar 0.0789 (−2,8%), Python 4,1 s vs 1,7 s; Firefox licht 8,2 s. Zware versie faalde in browser: download via service worker afgebroken → coi.js laat externe bronnen nu met rust (test loopt). Workflow nieuwe modellen GEPAUZEERD (5-uurslimiet 97%): hervatten met resumeFromRunId wf_ceeab066-8ef.
- Zware RMBG-1.4 werkt weer in browser (7,0 s; licht 8,2 s) → standaard blijft zwaar; label licht aangepast.
- 2026-11-16 20:18: PAUZE op verzoek van Abel. Workflow nieuwe modellen gestopt; hervatten: Workflow scriptPath ~/.claude/projects/-home-abel-claude-algemeen/63eeee7a-6517-415a-ba3c-8a163a78d5ff/workflows/scripts/alphamaker-nieuwe-modellen-wf_ceeab066-8ef.js + resumeFromRunId wf_ceeab066-8ef. Daarna kandidaten benchmarken (licht én nauwkeurig: Abel wil beide opties).
- 2026-11-17 00:47: geplande hervatting; workflow nieuwe modellen hervat (task wtnrf4ntc). Swap 4,8/5,1 GB vol, 4,2 GB vrij → nieuwe benchmarks met lagere grens (≤3,5 GB) of eerst geheugen checken.

## Testplan nieuwe modellen (2026-11-17 01:20) — `bench/testplan-2026-11.json`
- 34 agents, 29 kandidaten, 20 bruikbaar. BELANGRIJK: transparent-background ckpt_base.pth (sha256 0a6fe2a7…)
  = Plus_Ultra, getraind op o.a. **DIS-VD + DIS-TE** → InSPyReNet-DIS-score (0.0787) BESMET; alleen PPM 0.0754 schoon.
  Rapporteer DIS en PPM altijd apart. Doelen: objecten < 0.0660 (rmbg-1.4), haar < 0.0670 (birefnet-general).
- Volgorde: 1 inspyrenet-dis5k (schone nulmeting, DIS-TR only; sha256 ac057287… ok) · 2 S3OD dis/synth (ICLR 2026,
  eigen venv, git-commit 11385423, torch.load weights_only=True forceren) · 3 withoutBG v4 matting · 4 PDFNet+DAv2-S ·
  5 MVANet-ONNX · 6 FeyNobg · 7 BiRefNet_dynamic · 8 Finegrain · 9 MGD-SAM2-HRSOD · 10 SAM2Matting (NC) · 11 ZIM (NC).
- run_all.sh heeft nu CAP (standaard 4500M). Unit `alphamaker-kand1`: inspyrenet-dis5k, CAP=3900M (4,2 GB vrij, swap vol).
- 01:40: S3OD-installatie (pip git+https van KupynOrest/s3od) GEWEIGERD door auto-mode-beveiliging → wacht op Abel
  (zelfde geldt waarschijnlijk voor withoutBG #3 en PDFNet #4: externe code). Zonder nieuwe code verder: unit
  `alphamaker-kand2` (na kand1): finegrain-box, mvanet-int8, mvanet (ONNX), pt-birefnet-dynamic (code = nagelezen general).
- 02:25: nulmeting inspyrenet-dis5k (schoon): DIS 0.1152, PPM 0.1078, IoU 0.874 → bevestigt besmetting standaard-InSPyReNet op DIS. Eerlijk: objecten rmbg-1.4 0.0660 (NC), haar birefnet-general 0.0670 (MIT). README gecorrigeerd; standaardkeuze herzien na kand2.
- 2026-11-17 05:02: kand2 klaar (150 schone beelden). Per soort, rand/MAE/IoU/missers(IoU<0,8)/s:
  objecten: rmbg-1.4 0.0660/0.0083/0.939/2/1.7 (beste + snelste); inspyrenet (besmet) 0.0787; birefnet-general 0.1019 (15 missers).
  haar: birefnet-general 0.0670/0.0034/0.990/0/40.5; inspyrenet 0.0754/0.0048/0.989/0/14.1; rmbg-1.4 0.1049/0.0237/0.932/4.
  Afgevallen: mvanet 0.1123/0.1148, mvanet-int8 0.1127/0.1132 (42 s), finegrain-box 0.1493/0.1492.
  pt-birefnet-dynamic: OOM bij 3900M → unit alphamaker-dyn wacht op MemAvailable ≥ 6000 MB, draait dan met CAP 4500M (OOMScoreAdjust=800).
  Besluit: app-standaard blijft InSPyReNet (compromis: 0 missers op haar, MIT); nieuw --type object → rmbg-1.4; --type persoon → birefnet-general.
  Rest (S3OD, withoutBG, PDFNet, FeyNobg, ZIM, SAM2Matting, MGD-SAM2) vraagt externe code → wacht op Abel.
- 2026-11-17 09:53: Firefox (WASM) heeft geen goed haarmodel (birefnet-lite ONNX > 4,5 GB, zie 00:25). Kandidaat MODNet (Xenova/modnet @fa2fa546, Apache-2.0, ~25 MB, kortste zijde 512 /32) toegevoegd aan bench.py; draait op test_clean + test_clean2 (CAP 1500M). Goed op haar → als portretoptie in de webversie.
- 2026-11-17 09:59: MODNet afgevallen (portret rand 0.1274 vs rmbg-1.4 0.1049; wel minder missers 1 vs 4; objecten onbruikbaar). Gemiddelde modnet+rmbg-1.4 op portret: rand 0.1038, 0 missers → kleine winst, niet ingebouwd. withoutBG v4 (wbg-matting: router + matting-tak, ONNX @232aa520, sha256 ok, eigen code naar hun README, geen pip) draait op beide sets (CAP 3000M, log results/<set>/wbg-matting.log).
- 2026-11-17 10:04: withoutBG wbg-matting: objecten 0.1427 (32 missers, hoort naar hun BiRefNet-tak), portret rand 0.0960 / MAE 0.0159 / 1 misser / 1,5 s / 831 MB → beter dan rmbg-1.4 op portret, slechter dan inspyrenet. Router als type-detectie (fine_strand → haarmodel, rest → rmbg-1.4), berekend uit bestaande jsonl's: met birefnet-general objecten 0.0660 / portret 0.0747 (2 missers, 30 s); met inspyrenet portret 0.0817. Router: portretten 37/50 fine_strand, objecten 0/100 fine_strand. Vraag aan Abel: auto-keuze als standaard (objecten dan NC) + toestemming S3OD/PDFNet.
- 2026-11-17 10:51: Abel: auto-keuze alleen als extra optie (standaard blijft InSPyReNet); S3OD + PDFNet installeren mag. S3OD @11385423 nagelezen (665 regels, lokaal, geen netwerk/shell; pickle → weights_only=True geforceerd), venv ~/venvs/am-kand (.pth naar .venv, s3od --no-deps), gewichten sha256 ok. Unit alphamaker-kand3: s3od-dis + s3od-synth op beide sets (CAP 3800M). alphamaker-dyn gestopt (komt na PDFNet).
- 2026-11-17 10:55: --type auto ingebouwd (core.kind: withoutBG-router, fine_strand → persoon-model, anders rmbg-1.4; CLI routeert eerst alles en geeft de router vrij; opgegeven --model gaat voor). Getest: 6/6 gelijk aan bench-routes, ~1,2 s/foto; tests groen. README bijgewerkt.
- 2026-11-17 11:01: PDFNet @f4e218b + DAv2-code uit Space @b1a1fe6b nagelezen (geen netwerk/shell; torch.load alleen in pretrained=True-takken, niet gebruikt), matplotlib (+numpy/pillow gepind) in am-kand, gewichten sha256 ok. Laadtest: alleen depth_decoder.* over (alleen training) → assert daarop beperkt. Unit alphamaker-kand4: wacht op kand3 → pdfnet beide sets (CAP 3900M) → wacht MemAvailable ≥ 6000 → pt-birefnet-dynamic (4500M).
- 2026-11-17 11:37: Abel vraagt: snelste optie? lichte variant veranderd? nog NC? → lichte web-variant ongewijzigd (RMBG-1.4 int8, NC). Workflow alphamaker-commercieel-licht (run wf_71a4b396-896) zoekt commercieel bruikbare browsermodellen (WASM ≤ 4 GB): 4 zoekers → verificatie per kandidaat → testplan. Hervatten: Workflow scriptPath ~/.claude/projects/-home-abel-claude-algemeen/63eeee7a-6517-415a-ba3c-8a163a78d5ff/workflows/scripts/alphamaker-commercieel-licht-wf_71a4b396-896.js + resumeFromRunId wf_71a4b396-896. Benchmark loopt door (kand3 S3OD, daarna kand4).
- 2026-11-17 11:38: tussenstand s3od-dis (150): objecten 0.1036 / MAE 0.0232 / 23 missers, portret 0.1277 → haalt rmbg-1.4 niet. OPEN VRAAG: rmbg-1.4 (trainingsdata niet openbaar) scoort op DIS-VD 3× beter in MAE dan BiRefNet/S3OD/MVANet (0.0083 vs ~0.023); ook isnet-general-use opvallend goed (0.0133) → mogelijke besmetting niet uit te sluiten; toetsen op een objectset van na feb 2024.
- 2026-11-17 11:50: RICHTING van Abel: volledig focussen op niet-commerciële prestaties; snelheid en nauwkeurigheid voor OBJECTEN leidend. Commercieel spoor eruit tenzij het iets toevoegt; --type auto (router + BiRefNet, MIT voor haar) blijft als optie voor portretten/haar. Prestaties van RMBG-1.4 'nabouwen via reverse engineering' voor een commerciële versie: niet doen (Bria-licentie verbiedt commercieel gebruik van model en afgeleiden). Workflow wf_71a4b396-896 (commercieel licht) GESTOPT; alleen hervatten als dat spoor weer nodig is.
  PAUZE tot het tegoed reset (81% om 11:50, reset ~1 uur). Benchmark-units lopen door zonder tokens: kand3 (S3OD) → kand4 (PDFNet → wacht ≥6 GB → BiRefNet_dynamic). Uitslag: per soort uit results/test_clean*/<model>.jsonl (zie eerdere scripts).
  Daarna (objecten, NC toegestaan): (1) uitslag S3OD-synth/PDFNet; (2) RMBG-1.4 sneller in Firefox: tijden per stap meten (voorbewerking, model, opschalen, randkleur) in headless Firefox; (3) nauwkeuriger op objecten: test-time augmentation (spiegelen, gemiddelde) en multischaal voor RMBG-1.4 in bench; (4) open vraag besmetting RMBG-1.4 toetsen op objectset van na feb 2024.
- 2026-11-17 13:01: Abel: in SellYourShit (Python-app, levert AlphaMaker-kern mee, ~/Downloads/sellyourshit-2.9.tar.gz) is 'licht' 40 s vs 'zwaar' ~2 min; daar is licht = isnet-general, zwaar = inspyrenet; SYS verkleint standaard naar 50%. Advies: licht → rmbg-1.4 (objecten 0.066 vs 0.094, even snel). Gemeten 12 MP: model ~1,9 s, randkleur 6–9 s, png 3 s. Randkleur alleen rond onderwerp: zelfde uitkomst maar niet sneller → teruggedraaid. NIEUW: --voorkeur snel (rmbg-1.4 + verkleinen tot 3 MP) / precies (--type auto); EXIF-draaiing (load() in CLI, uploads, achtergrondfoto); import zonder torch (find_spec): import 5,0 → 0,5 s. CLI 12 MP jpg: rmbg-1.4 volle maat 16,6 → 12,5 s, --voorkeur snel 4,8 s. Tests 6/6 ok.
  Benchmark: s3od-synth test_clean2 rc 137 bij beeld 71 (3800M), pdfnet rc 137 op beide sets (3900M) → niet verder gejaagd (objecten tellen; s3od-dis haalde rmbg-1.4 niet). kand4 wacht nog op ≥6 GB voor BiRefNet_dynamic.
- 2026-11-17 13:02: stap 1 klaar: s3od-synth objecten 0.1329 (29 missers, 100 beelden; portret n=21 0.1120), s3od-dis 0.1036, pdfnet OOM → geen enkele haalt rmbg-1.4 (0.0660) op objecten. Tegoed om 12:51 op 99% (reset ~13:37) → hervatting opnieuw ingepland 13:47; dan stap 2 (Firefox per stap meten) en 3 (TTA/multischaal rmbg-1.4).
- 2026-11-17 13:16: stap 2 (Firefox per stap, 12 MP, warm): openen 0,1 s · model (session.run) 5,1 s zwaar / 5,4 s licht · voor/na masker 0,9 s · randkleur 4,8 s · eerste keer laden 34 s zwaar / 11 s licht. Licht is in Firefox per foto niet sneller, alleen kleinere download. numThreads=8 gaf niets (4 echte kernen) → teruggedraaid. Randkleur: vervaagde alpha één keer per straal en box(1−a) = 1−box(a) (JS + Python, pariteit 10/10 ok): 4,8 → 3,7 s. Worker/app meten nu tijden per stap (self.__timings, rooktest toont ze). Rest alleen via lager rekenen (voorkeur snel) of WebGPU.
- 2026-11-17 13:20: web: optie 'Snel' (rekenen op 3 MP, terugschalen naar volle maat). 12 MP in Firefox: randkleur 3,7 → 0,9 s, terugschalen 0,3 s, model ~5–6 s blijft; resultaat vrijwel gelijk (61,2% vs 61,1% transparant). RMBG-1.4 heeft vaste invoer 1024; models/rmbg-1.4-dyn.onnx = zelfde gewichten met vrije maat (1024 identiek, max verschil 0): CPU 512 0,54 s · 768 1,0 s · 1024 1,7 s · 1280 2,7 s. Stap 3 gestart: unit alphamaker-kand5 (rmbg-1.4@768, @1280, @flip, @ms op beide sets, CAP 2500M). kand4 (BiRefNet_dynamic, haar) gestopt: past niet bij focus objecten, één job tegelijk.
- 2026-11-17 13:26: stap 4: modelkaart briaai/RMBG-1.4 (README r.82–88, r.128): 'trained exclusively on a professional-grade dataset', '12,000 ... fully licensed images' (stock, e-commerce, gaming, advertising), 'developed on IS-Net ... proprietary dataset'. DIS5K = Flickr met onderzoekslicentie, past daar niet in; IS-Net zelf is op DIS-TR getraind (niet VD). → besmetting op DIS-VD onwaarschijnlijk, niet te bewijzen zonder nieuwe objectset (later eventueel: set van na 2024, bv. synthetische S3OD-beelden). Opvallend: isnet-general-use ook sterk op VD (0.0133) — die 'general use'-training is onbekend.
- 2026-11-17 13:28: stap 3 tussenstand (40 objecten): @768 0.1244 (MAE 0.052), @1280 0.0847, @flip 0.0658 (vs 1024 0.0665, 2× tijd) → RMBG-1.4 werkt alleen goed op 1024. kand5 gestopt; kand6: @ms afmaken op test_clean, @flip op test_clean2. Daarna verslag .txt voor SellYourShit (Abel vraagt; lokaal houden, niet in openbare repo).
- 2026-11-17 13:39: stap 3 klaar (150 beelden): @flip objecten 0.0653 vs 0.0660 (−1%), portret MAE slechter, 2× tijd → niet zinvol; @ms (40 obj., ontdubbeld) 0.0887 vs 0.0665 → slechter; @768/@1280 slechter. Python 12 MP: model 1,8 s + randkleur 6,5–9,7 s + wit/jpg 0,1 s; 3 MP: 1,6 + 1,2 + 0,0 = 2,9 s. Verslag voor SellYourShit: ~/claude/AlphaMaker/verslag-sellyourshit.txt (lokaal, in .git/info/exclude). Let op: systemctl stop van een unit stopt de scope (systemd-run --scope) van de lopende bench niet → die liep door; dubbele regels in @ms.jsonl ontdubbeld.
- 2026-11-17 13:52: run_all.sh geeft TERM/INT door aan de bench in zijn scope (getest: unit stoppen stopt nu ook bench.py; eerder liep die door). README webversie: 'Snel' en 12→50 MP-gedrag.
- 2026-11-17 13:54: stap 4 (besmetting): schone objectset data/test_syn = 60 beelden uit okupyn/s3od_dataset (MIT, FLUX-gegenereerd 2025, dus na RMBG-1.4; maskers van hun eigen teacher-model, niet van RMBG/BiRefNet/InSPyReNet; 0,7% halfdoorzichtig), 10 plekken × 6 via datasets-server. Unit alphamaker-kand7: rmbg-1.4, isnet-general, inspyrenet, pt-birefnet-general (CAP 4500M). S3OD zelf niet (getraind op deze set).
- 2026-11-17 14:11: review-workflow alphamaker-review-vandaag (run wf_fabff6ca-3df) over e9fc582..HEAD: 5 invalshoeken (cli-core, web, numeriek, bench-shell, beveiliging/docs) → adversariële verificatie per bevinding. Hervatten: Workflow scriptPath ~/.claude/projects/-home-abel-claude-algemeen/63eeee7a-6517-415a-ba3c-8a163a78d5ff/workflows/scripts/alphamaker-review-vandaag-wf_fabff6ca-3df.js + resumeFromRunId wf_fabff6ca-3df. Tegelijk loopt kand7 (besmettingstoets).
- 2026-11-17 17:51: BESMETTINGSTOETS (test_syn, 60 S3OD-synthbeelden 2025): rmbg-1.4 rand 0.1915 / MAE 0.1068 / IoU 0.746 / 23 missers (SLECHTST) · isnet 0.1752 / 0.0443 / 14 · inspyrenet 0.1345 / 0.0186 / 5 · birefnet-general (n=22, daarna OOM 4500M) 0.1126 / 0.0266 / 1. Op DIS-VD was rmbg-1.4 juist best (0.0660) → DIS-score rmbg-1.4 (en isnet) verdacht. Twee verklaringen: (a) besmetting/DIS-domein in training, (b) maskers van S3OD-teacher definiëren het onderwerp anders. VOLGENDE: slechtste rmbg-beelden in test_syn bekijken (masker vs GT), dan README + verslag SellYourShit (advies 'licht → rmbg-1.4' staat op losse schroeven). README kreeg waarschuwing.
  Review-workflow wf_fabff6ca-3df GESTOPT (tegoed 100%, reset ~21:00); hervatten: Workflow scriptPath ~/.claude/projects/-home-abel-claude-algemeen/63eeee7a-6517-415a-ba3c-8a163a78d5ff/workflows/scripts/alphamaker-review-vandaag-wf_fabff6ca-3df.js + resumeFromRunId wf_fabff6ca-3df. Hervatting ingepland 21:07.
- 2026-11-17 18:00: verslag SellYourShit gecorrigeerd: advies 'licht → rmbg-1.4' INGETROKKEN (NC + op test_syn niet beter dan IS-Net); IS-Net (Apache-2.0) en InSPyReNet (MIT) mogen commercieel; zwaar = beste kwaliteit. Open: --voorkeur snel gebruikt nog rmbg-1.4 → na de avondcheck eventueel isnet-general (Apache, even snel).
- 2026-11-17 18:04: test_syn bekeken (montage 4 slechtste): rmbg-1.4 neemt alles op de voorgrond mee (zonnebloem i.p.v. lieveheersbeestjes, hele aanrecht i.p.v. pan, rots bij stoepa, bladeren bij slang). >20% te veel: rmbg 18/60, inspyrenet 2, isnet 4; >20% te weinig: rmbg 7, inspyrenet 3, isnet 10. → geen bewijs voor besmetting; wel: bij rommelige foto's is InSPyReNet veel betrouwbaarder. README-waarschuwing aangepast; verslag SYS aangevuld.
- 2026-11-17 19:45: review-workflow wf_fabff6ca-3df staat op 'killed' (5 agents, ~0,8M tokens). Hervatten met cache kan alleen in de sessie die hem startte. Wekker op 21:07: daarna de review opnieuw draaien (zuiniger: Sonnet, of de 5 invalshoeken zelf nalopen).
- 2026-11-17 23:10: review klaar (hervat als wf_83dc4234-93c, pauze 19:50–22:50 door tegoed; 19 agents, 2,5M tokens): 9 van 15 bevindingen bevestigd, alle hersteld. Web: Snel boven 50 MP gaf 3 MP i.p.v. 12 (Snel doet daar nu niets); penseel tijdens opschalen tekende in het tussenresultaat (resultaat lokaal tot klaar) + Herstellen-race (guard in paint); lijst kiezen tijdens bulk hernoemde de rekenende foto (naam pas bij 'done'). Core: standaard inspyrenet alleen als ook torch er is. run_all.sh: Ctrl-C stopte een aanroepende lus niet (INT opnieuw opwerpen; getest: INT 130, TERM 143). Hulptekst/README: RMBG-1.4 'goed bij rustige productfoto's' i.p.v. 'beste'/'nauwkeurigst'; S3OD-uitslag (obj 0,104/0,133, haar 0,128 tegen rmbg 0,066/0,105); PDFNet en BiRefNet_dynamic niet getest; BiRefNet op test_syn n=22. 6 afgewezen (o.a. serve negeert --type auto/--voorkeur; rmbg-1.4 van unpinned main, niet nieuw). Tests: refine 10/10, test_core ok. Open: --type object/--voorkeur snel/precies blijven op rmbg-1.4 (op drukke foto's slechter dan de standaard) — keuze Abel.
- 2026-11-17 23:25: browser-rooktest na de reviewfixes groen in Firefox: gewoon (portret, 2× in lijst, penseel 15>0>255) en met Snel op 6,75 MP (berekend op 3 MP, uitvoer 3000×2250, opschalen 185 ms, penseel 255>0>255), 0 consolemeldingen. results/*/SUMMARY.md opnieuw gemaakt (S3OD, PDFNet, test_syn). Lokale branch pages = split van docs (450cae6); codeberg pages stond op df002f9 = split van 63c374c. Push (main naar github+codeberg, pages naar codeberg) geweigerd door auto-mode → wacht op Abel.
- 2026-11-17 23:40: keuzeregel objecten getoetst op bewaarde maskers (bench/route_study.py, geen modellen nodig): rmbg-1.4, tenzij IoU(rmbg-1.4, isnet-general) < 0,75 → inspyrenet. DIS 0,066/2 (gelijk, 0 van 100 omgeleid → besmetting telt niet), portret 0,096/0 (rmbg 0,105/4), syn 0,137/7 (rmbg 0,191/23, altijd inspyrenet 0,135/5); zonder torch → isnet: syn 0,159/12. Plateau 0,6–0,8. Echte proef: core.segment = bench-preds (IoU ≥0,993), keuze gelijk aan offline; CLI --model objecten-precies op drukste syn-foto = inspyrenet-masker, 27 s, piek 4,2 GB. Gebouwd op branch objecten-precies (--voorkeur precies → objecten-precies; tests groen), NIET op main: wacht op Abel.
- 2026-11-18 00:05: CONTROLESET test_syn2 (60 nieuwe S3OD-beelden, bench/prep_syn2.py, 0 overlap met test_syn; unit alphamaker-kand8): rmbg-1.4 0,1656/24 missers · isnet 0,1756/18 · inspyrenet 0,1316/5. Regel objecten-precies met VASTE drempel 0,75: 0,1344/6 (24 van 60 omgeleid); zonder torch (→ isnet) 0,1608/17. → afstelling op test_syn houdt stand. Tegoed 5-uur 99% → gepauzeerd. NA RESET: op branch objecten-precies README-alinea + route_study.py aanvullen met test_syn2, dan Abel laten kiezen (merge).
- 2026-11-18 03:54: na tegoedreset hervat: branch objecten-precies op main gezet (rebase, nooit gepusht) en aangevuld: bench/route_study.py met kolom test_syn2 (controle) en terugval op 0,75 zoals de code; README-alinea met de controlecijfers. Tests groen. Wacht op Abel: (1) push main (+ pages), (2) branch overnemen met git merge objecten-precies.
- 2026-11-18 03:56: kan rmbg-1.4 zelf drukke foto's herkennen (zonder isnet)? bench/signal_study.py, signalen uit zijn eigen masker (onzekere rand, voorgrondoppervlak, aantal stukken, beeldrand): NEE. Beste (oppervlak > 0,3 / rand > 0,1) syn 0,150–0,155 en syn2 0,145–0,146 met 12–18 missers, en ze sturen 17–18 DIS-foto's om; isnet-regel syn 0,137/7, syn2 0,134/6, 0 DIS. → de 1,6 s voor isnet blijft nodig. Branch objecten-precies: test voor het teruggegeven masker erbij (4bc8858).
- 2026-11-18 04:02: (1) rmbg-1.4 en isnet tegelijk (2 threads) i.p.v. na elkaar: 3,25 → 3,04 s/foto (−6%) → niet ingebouwd. (2) u2net (Apache-2.0) op alle 4 sets (unit alphamaker-kand9): DIS 0,373/91 missers, portret 0,175/2, syn 0,240/17, syn2 0,248/23. Commerciële keuzeregel 'isnet, tenzij IoU(isnet,u2net) < t → inspyrenet' (bench/commercial_study.py) WERKT NIET: u2net is het op DIS bijna altijd oneens (61–97 van 100 omgeleid, schone inspyrenet daar 0,115/28); pas bij t=0,9 winst op drukke foto's, dan = altijd inspyrenet. Advies SellYourShit blijft: medium = inspyrenet bij drukke foto's (syn2 0,132/5 tegen isnet 0,176/18); verslag lokaal aangevuld. SUMMARY.md's met u2net bijgewerkt, test_syn2 erbij.
- 2026-11-18 04:05: op dezelfde 22 syn-beelden: pt-birefnet-general 0,1126/1 (44,8 s) tegen inspyrenet 0,1480/1 (18,9 s) → misschien betere terugval voor objecten-precies (als pt.available()). Unit alphamaker-kand10: pt-birefnet-general op test_syn (rest) en test_syn2, CAP 4500M, herstart tot 6× na OOM (bench is hervatbaar). Daarna: regel met birefnet-terugval toetsen (syn + syn2). Web: birefnet(-lite) alleen via WebGPU, hier niet te toetsen.
- 2026-11-18 05:06: pt-birefnet-general nu op alle 60+60 syn-beelden (kand10 ok): altijd birefnet syn 0,1121/5, syn2 0,1145/7 (beste randen op drukke foto's), maar DIS 0,1019/15 en 40 s/foto. Als terugval in objecten-precies: syn 0,1276/7, syn2 0,1289/8 tegen inspyrenet-terugval 0,1372/7 en 0,1344/6; gemiddeld 17,8 tegen 8,7 s/foto (syn2), vraagt pt + liep eerder op 4,5 GB vast → NIET overstappen, terugval blijft inspyrenet. SUMMARY.md test_syn/test_syn2 bijgewerkt.
- 2026-11-18 05:08: eigen controle van de branch-diff: terugval bij onenigheid is nu het standaardmodel (inspyrenet, anders birefnet-general: syn 0,128, anders isnet: 0,159) i.p.v. altijd isnet zonder transparent-background. Branch opnieuw op main gezet (7924315, 4 commits), route_study met birefnet-regel, 8 tests groen. Open kandidaten FeyNobg/ZIM/SAM2Matting/MGD-SAM2: allemaal externe code (FeyNobg bovendien getraind op S3OD-synth → besmet voor test_syn/test_syn2) → wacht op Abel.
- 2026-11-18 19:00: Abel koos: (1) pushen, (2) branch objecten-precies OVERGENOMEN (fast-forward, 4 commits; --voorkeur precies → objecten-precies), (3) laatste kandidaten FeyNobg/ZIM/SAM2Matting/MGD-SAM2 OVERGESLAGEN → testplan 2026-11 afgesloten. CI-stappen op main groen (syntax, pariteit 10/10, test_core 8/8, refine-zelftest).
- 2026-11-18 19:08: STANDAARDMODUS PER FOTO getoetst (Abel: ja, onderzoeken). Router (bench/route_kinds.py, 0,7 s/beeld; persoon: clean 5/50, clean2 32/100, syn 10/60, syn2 4/60) → haar: inspyrenet, anders objecten-precies. bench/default_study.py tegen nu (altijd inspyrenet): DIS 0,066/2 in 3,9 s (nu 0,079 besmet – 0,115/28 schoon, 14,3 s); portret 0,0785/0 12,5 s (nu 0,0754/0 14,1 s); syn 0,138/7 12,7 s (nu 0,1345/5 20,1 s); syn2 0,134/6 9,6 s (nu 0,132/5 13,8 s). Nadelen: objecten via RMBG-1.4 (niet-commercieel; standaard was MIT), ~700 MB extra download (router+rmbg+isnet). → branch standaard-per-foto, keuze Abel.
- 2026-11-18 19:10: branch standaard-per-foto gebouwd: zonder opties → router; haar → DEFAULT_MODEL, anders objecten-precies. Tests 8/8 (nieuw: zonder opties + object → objecten-precies). Echte proef CLI zonder opties: DIS-object → objecten-precies, portret → inspyrenet, 31,6 s voor 2 foto's incl. laden, piek 4,3 GB. Wacht op Abel (licentie: objecten dan via NC RMBG-1.4).
- 2026-11-19 10:14: Abel over branch standaard-per-foto: 'men moet alphamaker licentievrij kunnen gebruiken dus niet gebonden zijn aan beperkingen voor het doel van gebruik' → NIET overgenomen (branch blijft staan). Gevolg: geen niet-commercieel model als standaard. Let op: de webversie heeft RMBG-1.4 (NC) als standaard, en zonder WebGPU is RMBG-1.4 daar het ENIGE model (ben2/birefnet-lite/birefnet alleen webgpu) → vraag aan Abel of opt-in NC-opties mogen blijven en hoe de web-standaard vrij wordt.
- 2026-11-19 10:20: Abel: AlphaMaker is voor productfoto's van spullen die verkocht worden → commercieel bruikbaar. 'Standaarden vrij, opties mogen'; 'rembg' (= RMBG-1.4) mag optie blijven, nooit standaard, en moet expliciet als niet-commercieel benoemd worden. Plan: (1) voorinstellingen (--type object/auto, --voorkeur snel/precies) naar vrije modellen; RMBG-1.4/objecten-precies alleen via --model, met melding; (2) web-standaard vrij op wasm én webgpu (kandidaat isnet-general Apache-2.0, kopie op HF nodig; sha256 lokaal 60920e99…); (3) README: productadvies met vrije modellen voorop. Onderzoek-workflow wf_b2ab98d2-bd9; unit alphamaker-kand12 meet pt-birefnet-lite op clean2/syn/syn2 (ONNX-variant OOM, test_clean al: DIS 0,137, 28 s/foto).
- 2026-11-19 10:50: ONDERZOEK wf_b2ab98d2-bd9 (7 agents; volledige uitkomst: bench/onderzoek-licenties-2026-11-19.json). (1) IS-NET NIET AANTOONBAAR COMMERCIEEL: DIS-README: Apache-2.0 dekt alleen 'code and evaluation metric'; isnet-general-use-gewichten zonder licentie (Google Drive/Baidu); DIS5K-voorwaarden niet-commercieel 'ook na bewerking'; vraag DIS#150 onbeantwoord. Alle 3 HF-kopieën (BritishWerewolf/IS-Net @9783722, skillsafe-ai, jellybox; sha256 = rembg) daarom AFGEWEZEN. (2) InSPyReNet bevestigd: OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX @fe628d2d14, MIT (repo + upstream code én gewichten); fp32 onnx/model.onnx 395 MB (draait volgens ZeroBG op webgpu én wasm), fp16 199 MB (niet aangetoond in browser). (3) BiRefNet_lite-ONNX: fp32 224 MB te zwaar voor wasm, fp16 faalt volgens meldingen op webgpu → web-item birefnet-lite waarschijnlijk stuk. (4) --type auto-router (withoutBG) heeft DINOv3-backbone onder Meta's DINOv3-licentie (geen MIT/Apache) → nakijken. (5) Webcode: file() pint 'main' (revisie-parameter nodig); worker infer() haalt alle uitvoer op (sess.run(feeds, [naam])). GEVOLGEN: isnet-general niet als vrije terugval/standaard/snel; advies SellYourShit ('IS-Net Apache, commercieel') moet gecorrigeerd. NA RESET: plan herzien met InSPyReNet (MIT) als vrije basis, ook web (ONNX 395 MB), 'snel' via transparent-background mode=fast nagaan.
- 2026-11-19 13:00: (a) InSPyReNet-ONNX (OS-Software @fe628d2d, onnx/model.onnx 395 MB, sha256 c108dd92…, MIT) = zelfde Plus Ultra als transparent-background: IoU 0,998–0,9999, gem. verschil <0,001 (4 syn2-beelden); ORT 10 s/beeld (torch 14–20 s), piek 2,2 GB (torch 3,9 GB), geen torch nodig → wordt 'inspyrenet' in core en web-standaard. (b) withoutBG-router: Apache-2.0 + Meta DINOv3-licentie met gebruiksbeperkingen (o.a. militair/wapens) en verplichte vermelding 'Built with DINOv3' → --type auto alleen als expliciete keuze met melding; --voorkeur precies → birefnet-general (MIT) zonder router. (c) pt-birefnet-lite klaar op alle sets (kand12).
- 2026-11-19 18:06: OMBOUW COMMERCIEEL BRUIKBAAR klaar: 'inspyrenet' = ONNX (OS-Software @fe628d2d, sha256 c108dd92…, MIT) in core en web, standaard overal (CLI, serve, web wasm+webgpu); transparent-background niet meer nodig (pyproject 'beste' alleen voor BiRefNet). Voorinstellingen vrij: --type object → inspyrenet, --voorkeur precies → birefnet-general (zonder PyTorch: inspyrenet + melding), --voorkeur snel = alleen verkleinen. core.NONCOMMERCIAL (rmbg-1.4, isnet-general, objecten-precies) alleen via --model, met melding in CLI, serve-pagina en web ('NIET voor commercieel gebruik'); router (--type auto) alleen met PyTorch en met DINOv3-melding. Review wf_f55a99bd-db0 (32 agents): 14 bevestigd (4 uniek: router zonder PyTorch, één-zwaar-model-regel, README-getallen RMBG/MB, 'vaste revisies'), alle hersteld. Tests 10/10 + pariteit 10/10; Firefox-WASM-rooktest groen (28 s/foto); CLI standaard 14 s/2,4 GB, objecten-precies piek 2,5 GB. SellYourShit-verslag lokaal gecorrigeerd (IS-Net niet commercieel).
- 2026-11-20 11:56: WEB SNELLER? int8 van InSPyReNet-ONNX (onnxruntime.quantization.quantize_dynamic, QUInt8): volledig 125 MB, alleen MatMul/Gemm 137 MB (fp32 395 MB). Kwaliteit test_syn2: q8 0,1335/5 missers tegen fp32 0,1316/5 (IoU q8~fp32 gem 0,992). Snelheid (ORT CPU, 4 threads, 3 beelden): fp32 10,3 s, q8-matmul 11,5 s, q8 16,2 s → int8 is NIET sneller, alleen kleinere download. Kleinere invoer (bv. 768) vraagt een nieuwe export uit PyTorch (liep hier op 4,5 GB vast). → web blijft fp32; sneller kan alleen via WebGPU. bench/q8_eval.py toegevoegd. SELLYOURSHIT 3.5 gebouwd in ~/Marktplaats/sellyourshit-3.5 (3.4 onaangeroerd): AlphaMaker a94f279 meegeleverd, stand 'licht' (IS-Net) weg → medium (InSPyReNet ONNX) standaard, oude keuze 'licht' → medium (migratie getest: 7 gevallen), installers/Windows/README/BRON.txt bijgewerkt; helper-proef met app-venv: 2 foto's 23,8 s, piek 2,4 GB. Review wf_16fa34bc-056 loopt.
- 2026-11-20 16:21: SELLYOURSHIT 3.5 review wf_16fa34bc-056 (23 agents): 14 bevestigd = 5 uniek, alle hersteld: (1) geheugenmelding medium vooraf zonder zinloze 50%-tip (tip alleen bij vol tijdens het werk); (2) Windows levert ckpt_base.pth (370 MB, door niets meer gebruikt) niet meer mee: build.ps1, start_sellyourshit.pyw, test_windows.py, iss ruimt het op, CI-cache inspyrenet.onnx i.p.v. isnet; (3) README (licht weg, geheugen medium 2,8/zwaar 6 GB, installer ~450 MB); (4) teksten foto-editor; (5) reden IS-Net = gewichten zonder licentie (InSPyReNet/BiRefNet: uitdrukkelijk MIT). Opgeleverd in ~/Marktplaats: sellyourshit-3.5.tar.gz en sellyourshit-3.4-naar-3.5.patch (14 bestanden). Niet geïnstalleerd (wacht op Abel); Windows-build hier niet te testen.
- 2026-11-20 18:00: LICHTE STAND VOOR SELLYOURSHIT? U²-Net (Apache-2.0) tegen IS-Net op egale achtergronden. (a) egaalste echte achtergronden: 10% (22) u2net 0,313/15 missers, isnet 0,104/3, inspyrenet 0,083/2. (b) nagebootste studio (bench/studio_study.py: object met echt masker op egaal verloop + ruis): wit 120 alledaags u2net 0,143/10 (1,0 s), isnet 0,057/10 (1,8 s), inspyrenet 0,046/3 (11,7 s); wit 100 DIS u2net 0,269/58, isnet 0,052/4; grijs 120 alledaags u2net 0,152/12, isnet 0,064/11, inspyrenet 0,049/2. → U²-Net mist even vaak op egale achtergrond maar randen 2,5× slechter (rekent op 320 px), faalt bij fijne details → NIET ongeveer zo goed als IS-Net → volgens Abel: voorlopig geen lichte stand (3.5 blijft medium + zwaar). Abel draait install.sh zelf (vraagt sudo).
