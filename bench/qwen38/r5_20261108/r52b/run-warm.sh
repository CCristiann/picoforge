#!/bin/zsh
# R5.2b: steady-state sync cost (warm pool, 0 misses after warmup): resident vs wait vs event.
set -u
cd /Users/cristiancirje/Desktop/Dev/picoforge-perf-decode
PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
O=build/r5-checks/r52b-warm-${TAG}-$(git rev-parse --short HEAD)-dirty; mkdir -p $O
P='Scrivi cinque consigli per verificare un backup.'
for mode in ${=MODES}; do
  pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' && { echo "FOREIGN PROCESS: stop"; exit 9; }
  pmset -g batt | sed -n 2p
  if [ $mode = res ]; then ENVS=(Q38_X=0); elif [ $mode = notify ]; then ENVS=(PICOFORGE_STREAM_EXPERTS=24576 PICOFORGE_STREAM_NOTIFY=1); else ENVS=(PICOFORGE_STREAM_EXPERTS=24576 PICOFORGE_STREAM_SYNC=$mode); fi
  $PY -B tools/qwen38_gpu_lock.py --timeout 900 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py \
    $O/$mode ${ENVS[@]} -- --qwen38 $M --chat "$P" --max-new 64 --context 2048 --chunk 128 --warmup 2 --repeat 5 \
    --runs-jsonl '{d}/runs.jsonl' || echo "RUN $mode FAILED"
done
for mode in ${=MODES}; do printf "%-6s " $mode; jq -s -r 'map(select(.warmup==false)) | map(1000*.decode_s/.decode_tokens) | sort | "decode ms/token min \(.[0]*10|round/10) median \(.[2]*10|round/10) max \(.[4]*10|round/10)"' $O/$mode/runs.jsonl; grep -o "[0-9]* misses.*" $O/$mode/process/stderr.log | tail -1; done
