#!/bin/bash
# Lokaal: upload, start label+train+export op een RunPod-pod, wacht, download. Gebruik: train/pod.sh HOST POORT DATA UIT
# Veiligheid: de pod stopt zichzelf 45 min na KLAAR-ALLES of een crash (Traceback), niet bij andere meldingen.
set -u
H=$1; PORT=$2; P=$(dirname "$3"); NAME=$(basename "$3"); R=$4; mkdir -p "$R"
S="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p $PORT root@$H"
tar -C "$P" -cf - "$NAME" | $S "mkdir -p /workspace && tar -C /workspace -xf - && rm -f /workspace/$NAME/student.pt" || { echo UPLOAD-MISLUKT; exit 1; }
printf '%s\n' 'set -e' 'cd /workspace && git clone -q https://github.com/AbelDrift/AlphaMaker && cd AlphaMaker' \
  'pip install -q --break-system-packages timm onnx onnxruntime-gpu scipy transparent-background' "D=/workspace/$NAME" \
  'python train/kit.py label --data $D' \
  'python train/kit.py train --data $D --epochs 100 --batch 16 --size 768 --workers 10 --pretrained' \
  'python train/kit.py export --data $D --out /workspace/licht.onnx' 'echo KLAAR-ALLES' \
  | $S 'cat > /workspace/run.sh; nohup bash /workspace/run.sh > /workspace/run.log 2>&1 &'
printf '%s\n' 'export $(tr "\0" "\n" < /proc/1/environ | grep -E "^RUNPOD_(API_KEY|POD_ID)=")' \
  'while ! grep -q -e "^KLAAR-ALLES" -e "^Traceback" /workspace/run.log; do sleep 60; done' 'sleep 2700' 'runpodctl stop pod "$RUNPOD_POD_ID"' \
  | $S 'cat > /workspace/autostop.sh; (nohup bash /workspace/autostop.sh > /workspace/autostop.log 2>&1 &)'
for i in $(seq 1 600); do $S 'grep -q -e "^KLAAR-ALLES" -e "^Traceback" /workspace/run.log' && break; sleep 60; done
$S 'grep -E "^epoch|^Traceback|KLAAR" /workspace/run.log | tail -3'
scp -q -P "$PORT" root@$H:/workspace/licht.onnx root@$H:/workspace/$NAME/student.pt root@$H:/workspace/run.log "$R"/; ls "$R"
