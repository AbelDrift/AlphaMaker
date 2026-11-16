#!/bin/bash
# Draait elk model apart met harde geheugenlimiet. Hervatbaar: gewoon opnieuw starten.
cd "$(dirname "$0")/.."
R=results; [ -n "$TESTSET" ] && [ "$TESTSET" != test ] && R=results/$TESTSET; mkdir -p $R
MODELS="${*:-u2net isnet-general rmbg-1.4 birefnet-lite ben2 rmbg-2.0 birefnet-general birefnet-portrait inspyrenet birefnet-hr-matting}"
for m in $MODELS; do
  [ -f "$R/$m.status" ] && { echo "$m: al gedaan ($(cat $R/$m.status))"; continue; }
  echo "== $m  $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=4500M -p MemorySwapMax=0 \
    nice -n 10 .venv/bin/python bench/bench.py "$m"
  rc=$?
  [ $rc -eq 0 ] && echo ok > "$R/$m.status" || echo "fout rc=$rc (137=geheugenlimiet)" > "$R/$m.status"
  echo "$m: $(cat $R/$m.status)"
done
