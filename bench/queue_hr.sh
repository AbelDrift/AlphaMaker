#!/bin/bash
# Hoge-resolutie-experiment (data/test_hr, ≤3000 px). Hervatbaar: opnieuw starten slaat klaar werk over.
cd "$(dirname "$0")/.."
export TESTSET=test_hr
R=results/test_hr; mkdir -p $R
cap() { systemd-run --user --scope -q -p MemoryMax=4500M -p MemorySwapMax=0 nice -n 10 "$@"; }
bench/run_all.sh pt-birefnet-lite pt-birefnet-general
for v in "pt-birefnet-lite st24" "pt-birefnet-lite vm24" "pt-birefnet-lite gf8" "pt-birefnet-general st24"; do
  set -- $v; [ -f "$R/$1+$2.status" ] && continue
  echo "== hr verfijning $1 + $2  $(date +%T)"
  cap .venv/bin/python bench/refine_bench.py "$1" "$2" || echo "fout rc=$? (137=geheugenlimiet)" > "$R/$1+$2.status"
done
