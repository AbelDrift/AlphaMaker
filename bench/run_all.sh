#!/bin/bash
# Draait elk model apart met harde geheugenlimiet. Hervatbaar: gewoon opnieuw starten.
cd "$(dirname "$0")/.."
R=results; [ -n "$TESTSET" ] && [ "$TESTSET" != test ] && R=results/$TESTSET; mkdir -p $R
MODELS="${*:-u2net isnet-general rmbg-1.4 birefnet-lite ben2 rmbg-2.0 birefnet-general birefnet-portrait inspyrenet birefnet-hr-matting}"
for m in $MODELS; do
  [ -f "$R/$m.status" ] && { echo "$m: al gedaan ($(cat $R/$m.status))"; continue; }
  echo "== $m  $(date +%T)"
  # de scope is een eigen cgroup: bij het stoppen van de unit krijgt hij geen signaal → zelf doorgeven
  systemd-run --user --scope -q -p MemoryMax=${CAP:-4500M} -p MemorySwapMax=0 \
    nice -n 10 ${PY:-.venv/bin/python} bench/bench.py "$m" &
  # INT opnieuw opwerpen i.p.v. exit: anders ziet een aanroepende lus geen Ctrl-C en start die het volgende model
  job=$!; trap 'kill $job 2>/dev/null; exit 143' TERM; trap 'kill $job 2>/dev/null; trap - INT; kill -INT $$' INT
  wait $job; rc=$?
  [ $rc -eq 0 ] && echo ok > "$R/$m.status" || echo "fout rc=$rc (137=geheugenlimiet)" > "$R/$m.status"
  echo "$m: $(cat $R/$m.status)"
done
