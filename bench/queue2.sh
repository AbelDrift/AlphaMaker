#!/bin/bash
# Tweede reeks: herkansing BiRefNet-general, verfijningsvarianten, als laatste HR-matting.
cd "$(dirname "$0")/.."
cap() { systemd-run --user --scope -q -p MemoryMax=4500M -p MemorySwapMax=0 nice -n 10 "$@"; }
bench/run_all.sh pt-birefnet-general
for v in "pt-birefnet-lite vq16" "pt-birefnet-lite vm16" "pt-rmbg-2.0 vm16" "pt-birefnet-lite vm8" "pt-birefnet-lite gf4"; do
  set -- $v; [ -f "results/$1+$2.status" ] && continue
  echo "== verfijning $1 + $2  $(date +%T)"
  cap .venv/bin/python bench/refine_bench.py "$1" "$2" || echo "fout rc=$? (137=geheugenlimiet)" > "results/$1+$2.status"
done
bench/run_all.sh pt-birefnet-hr-matting
