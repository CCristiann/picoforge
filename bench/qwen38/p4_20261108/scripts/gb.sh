#!/bin/zsh
# usage: gb.sh METALLIB args... : one isolated-kernel timing under the GPU lock
lib=$1; shift
cd /Users/cristiancirje/Desktop/Dev/picoforge
exec tools/venv/bin/python -B tools/qwen38_gpu_lock.py --timeout 120 -- env Q38_PACKED_LIBRARY=$lib build/qwen38-r1/test_fast_kernels "$@"
