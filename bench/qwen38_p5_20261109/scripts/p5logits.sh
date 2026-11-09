#!/bin/zsh
# P5.1: 8K + 16 forced logits, FP32 and half, MMA split (default) vs R2.1 split (Q38_QSA_SPLIT=1).
# usage: p5logits.sh OUTROOT
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=$1; mkdir -p $OUT
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
run() { name=$1; prec=$2; shift 2
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 900 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/$name . "$@" -- \
    build/p5/picoforge --qwen38 $M --tokens "$(cat build/p4/tok8k.txt)" --teacher-forced 16 --context 8192 --chunk 1024 \
    --metallib build/p5/picoforge.metallib --prefill-precision $prec --runs-jsonl {d}/runs.jsonl --logits $OUT/$name.f32 || exit 1
}
for prec in fp32 half; do
  run $prec-mma $prec
  run $prec-split $prec Q38_QSA_SPLIT=1
done
