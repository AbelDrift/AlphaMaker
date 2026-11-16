#!/bin/bash
# Draait docs/ in headless Firefox (eigen tijdelijk profiel) met een echte foto; alles samen max 3 GB.
# Gebruik: tests/web/run_browser_smoke.sh [foto] [model]   (standaard: eerste portret uit data/test, rmbg-1.4-licht)
cd "$(dirname "$0")/../.."
FOTO=${1:-$(ls data/test/ppm_*.jpg | head -1)}; MODEL=${2:-rmbg-1.4-licht}
T=$(mktemp -d); mkdir -p "$T/site" "$T/profiel"; cp -r docs/. "$T/site/"; cp "$FOTO" "$T/site/test.jpg"
cleanup() { kill $SRV $FF 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
python3 -m http.server --bind 127.0.0.1 8093 -d "$T/site" >/dev/null 2>&1 & SRV=$!
firefox-esr --headless --no-remote -profile "$T/profiel" --remote-debugging-port 9333 about:blank >/dev/null 2>&1 & FF=$!
for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/9333) 2>/dev/null && break; sleep 1; done
node tests/web/browser_smoke.mjs http://127.0.0.1:8093/ http://127.0.0.1:8093/test.jpg "$MODEL"
