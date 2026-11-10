#!/bin/zsh
cd /Users/cristiancirje/Desktop/Dev/picoforge
S=/private/tmp/claude-501/-Users-cristiancirje-Desktop-Dev-picoforge/060dd1c8-1a33-428e-b40f-fe333108e36c/scratchpad
OUT=bench/qwen38/p3_20261108/cross; mkdir -p $OUT
run() { name=$1; f=$2; shift 2
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 600 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/$name . PICOFORGE_GEMM_MIN_TILES=80 "$@" -- \
    ./picoforge -m q2 --chat-file $f -n 2 --temp 0 --no-mtp --profile {d}/profile.csv --logits {d}/logits.f32
}
for p in p300 p700 p2000; do
  run $p-group $S/$p.txt
  run $p-nogroup $S/$p.txt PICOFORGE_GROUPED_MIN_PAIRS=100000000
done
