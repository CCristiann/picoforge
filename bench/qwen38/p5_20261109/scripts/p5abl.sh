#!/bin/zsh
# P5.0 ablation: 8K half profile with a metallib. usage: p5abl.sh OUTDIR METALLIB
cd /Users/cristiancirje/Desktop/Dev/picoforge
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 900 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $1 . -- \
  build/p5/picoforge --qwen38 $M --tokens "$(cat build/p4/tok8k.txt)" --teacher-forced 16 --context 8192 --chunk 1024 \
  --metallib $2 --prefill-precision half --runs-jsonl {d}/runs.jsonl --profile {d}/profile.csv
