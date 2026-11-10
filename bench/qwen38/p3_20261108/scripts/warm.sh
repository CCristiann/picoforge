#!/bin/zsh
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=bench/qwen38/p3_20261108/warmup; mkdir -p $OUT
run() { name=$1; shift
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 900 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/$name . "$@" -- \
    ./picoforge -m q2 -p "list all the activation functions in machine learning" --thinking -n 8 --temp 0 --mtp --repeat 3 --runs-jsonl {d}/runs.jsonl
}
run warm
run nowarm PICOFORGE_NO_WARMUP=1
