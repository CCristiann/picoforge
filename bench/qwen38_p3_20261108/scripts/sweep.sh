#!/bin/zsh
# Short chat prompt prefill vs PICOFORGE_GEMM_MIN_ROWS, real Q2, guarded.
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=bench/qwen38_p3_20261108/gemm-threshold; mkdir -p $OUT
for rows in 32 16 8; do
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 600 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/rows$rows . PICOFORGE_GEMM_MIN_ROWS=$rows -- \
    ./picoforge bench -m q2 -p "list all the activation functions in machine learning" --thinking -n 16 --warmup 1 --repeat 5 --bench-csv {d}/bench.csv
done
