#!/bin/zsh
cd /Users/cristiancirje/Desktop/Dev/picoforge
OUT=bench/qwen38/p3_20261108/overhead; mkdir -p $OUT
run() { name=$1; shift
  ./tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 600 -- ./tools/venv/bin/python -B build/r5-checks/launch-any.py $OUT/$name . -- \
    ./picoforge -m q2 -p "list all the activation functions in machine learning" --thinking -n 8 --temp 0 --repeat 4 --warmup 1 --runs-jsonl {d}/runs.jsonl "$@"
}
run plain --no-mtp
run mtp --mtp
