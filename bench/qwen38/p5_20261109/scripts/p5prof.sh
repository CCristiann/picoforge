#!/bin/zsh
# P5.0: 8K profile (half and fp32) on main 78ea2a8. usage: p5prof.sh OUTROOT
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=$1; mkdir -p $OUT
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
for prec in half fp32; do
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 900 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/profile8k-$prec . -- \
    build/p5/picoforge --qwen38 $M --tokens "$(cat build/p4/tok8k.txt)" --teacher-forced 16 --context 8192 --chunk 1024 \
    --metallib build/p5/picoforge.metallib --prefill-precision $prec --runs-jsonl {d}/runs.jsonl --profile {d}/profile.csv || exit 1
done
