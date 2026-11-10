#!/bin/zsh
# R5.2b: per-layer sync: wait vs event gates, real Q2, 50% pool, alternating pairs.
set -u
cd /Users/cristiancirje/Desktop/Dev/picoforge-perf-decode
PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
O=build/r5-checks/r52b-$(git rev-parse --short HEAD)-dirty; mkdir -p $O
P='Scrivi cinque consigli per verificare un backup.'
for rep in 1 2; do for rd in wait event; do
  pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' && { echo "FOREIGN PROCESS: stop"; exit 9; }
  pmset -g batt | sed -n 2p
  $PY -B tools/qwen38_gpu_lock.py --timeout 900 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py \
    $O/r$rd-$rep Q38_REF_KERNELS=gemm PICOFORGE_STREAM_EXPERTS=12288 PICOFORGE_STREAM_SYNC=$rd -- --qwen38 $M --chat "$P" --max-new 64 --context 2048 --chunk 128 \
    --logits '{d}/logits.f32' --token-ids '{d}/ids.json' --runs-jsonl '{d}/runs.jsonl' || echo "RUN r$rd-$rep FAILED"
done; done
for d in $O/r*; do printf "%s\t" ${d:t}; grep -o "[0-9.]* misses.*" $d/process/stderr.log | tail -1 | tr '\n' '\t'; jq -r '[.prefill_s,.decode_s,.decode_tokens]|@tsv' $d/runs.jsonl | tr '\n' '\t'; jq -s -r 'map(.disk_read_bytes)|max/1073741824*100|round/100' $d/process/memory.jsonl; shasum -a 256 $d/logits.f32 | cut -c1-12; done
