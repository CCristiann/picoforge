#!/bin/zsh
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=bench/qwen38/p3_20261108/shape
run() { name=$1; shift
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 600 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/$name . "$@" -- \
    ./picoforge -m q2 -p "list all the activation functions in machine learning" --thinking -n 4 --temp 0 --no-mtp --profile {d}/profile.csv --logits {d}/logits.f32
}
mkdir -p $OUT
run base
run nogroup PICOFORGE_GROUPED_MIN_PAIRS=100000000
run nogroup-t80 PICOFORGE_GROUPED_MIN_PAIRS=100000000 PICOFORGE_GEMM_MIN_TILES=80
run nogroup-t160 PICOFORGE_GROUPED_MIN_PAIRS=100000000 PICOFORGE_GEMM_MIN_TILES=160
