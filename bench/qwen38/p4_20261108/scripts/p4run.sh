#!/bin/zsh
# P4 GDN scan: before (metallib at 90a3393) vs after, same picoforge binary (host unchanged).
# usage: p4run.sh OUTROOT LOGITDIR
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=$1; LG=$2; mkdir -p $OUT
M=models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
run() { name=$1; lib=$2; prec=$3; toks=$4; shift 4
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 600 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/$name . -- \
    ./picoforge --qwen38 $M --tokens "$(cat build/p4/$toks)" --teacher-forced 16 --context 8192 --chunk 1024 \
    --metallib build/p4/$lib.metallib --prefill-precision $prec --runs-jsonl {d}/runs.jsonl --token-ids {d}/ids.json \
    --logits $LG/$name.f32 "$@" || exit 1
}
# profile at N 1000 (+16 forced)
for prec in fp32 half; do for lib in before after; do
  run profile-$prec-$lib $lib $prec tok1000.txt --profile {d}/profile.csv
done; done
# 8K prefill, alternating pairs
for i in 1 2; do for prec in fp32 half; do for lib in before after; do
  run 8k-$i-$prec-$lib $lib $prec tok8k.txt
done; done; done
