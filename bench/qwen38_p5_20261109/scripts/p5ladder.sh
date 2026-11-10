#!/bin/zsh
# P5.1: FP32 logits at prompt lengths N (+16 forced), MMA split vs R2.1 split. usage: p5ladder.sh OUTROOT N...
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=$1; shift; mkdir -p $OUT
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
for n in "$@"; do
  cut -d, -f1-$((n+16)) build/p4/tok8k.txt | tr -d "\n" > $OUT/tok$n.txt
  for mode in mma split; do
    extra=(); [[ $mode == split ]] && extra=(Q38_QSA_SPLIT=1)
    ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 900 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/n$n-$mode . $extra -- \
      build/p5/picoforge --qwen38 $M --tokens "$(cat $OUT/tok$n.txt)" --teacher-forced 16 --context 8192 --chunk 1024 \
      --metallib build/p5/picoforge.metallib --prefill-precision fp32 --runs-jsonl {d}/runs.jsonl --logits $OUT/n$n-$mode.f32 || exit 1
  done
done
