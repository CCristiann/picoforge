#!/bin/zsh
# R5.1c: real Q2, resident vs streamed pool at 10/50/100% of the routed experts.
set -u
cd /Users/cristiancirje/Desktop/Dev/picoforge-perf-decode
PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
O=build/r5-checks/r51c-$(git rev-parse --short HEAD); mkdir -p $O
P='Scrivi cinque consigli per verificare un backup.'
for name in res s2458 s12288 s24576; do
  pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' && { echo "FOREIGN PROCESS: stop"; exit 9; }
  pmset -g batt | sed -n 2p; df -h /System/Volumes/Data | tail -1
  ENVS=(Q38_REF_KERNELS=gemm)
  [ $name != res ] && ENVS+=(PICOFORGE_STREAM_EXPERTS=${name#s})
  $PY -B tools/qwen38_gpu_lock.py --timeout 900 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py \
    $O/$name ${ENVS[@]} -- --qwen38 $M --chat "$P" --max-new 16 --context 2048 --chunk 128 \
    --logits '{d}/logits.f32' --token-ids '{d}/ids.json' --runs-jsonl '{d}/runs.jsonl' || echo "RUN $name FAILED"
done
for n in s2458 s12288 s24576; do
  [ -f $O/$n/logits.f32 ] || { echo "$n: missing"; continue; }
  if cmp -s $O/res/logits.f32 $O/$n/logits.f32 && cmp -s <(jq -c .generated_ids $O/res/ids.json) <(jq -c .generated_ids $O/$n/ids.json); then echo "$n: logits+ids BIT-IDENTICAL"; else echo "$n: DIFFER"; fi
done
