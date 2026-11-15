#!/bin/bash
# Draait elk model apart met harde geheugenlimiet. Hervatbaar: gewoon opnieuw starten.
cd "$(dirname "$0")/.."
MODELS="${*:-u2net isnet-general rmbg-1.4 birefnet-lite ben2 rmbg-2.0 birefnet-general birefnet-portrait inspyrenet birefnet-hr-matting}"
for m in $MODELS; do
  [ -f "results/$m.status" ] && { echo "$m: al gedaan ($(cat results/$m.status))"; continue; }
  echo "== $m  $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=4500M -p MemorySwapMax=0 \
    nice -n 10 .venv/bin/python bench/bench.py "$m"
  rc=$?
  [ $rc -eq 0 ] && echo ok > "results/$m.status" || echo "fout rc=$rc (137=geheugenlimiet)" > "results/$m.status"
  echo "$m: $(cat results/$m.status)"
done
