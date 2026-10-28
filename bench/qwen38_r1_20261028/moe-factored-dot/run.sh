#!/bin/sh
# R1.9c same-session A/B: before = 1f995d6, after = d62f163 (harness + metallib built from
# each tree into $1/before and $1/after). Run under the GPU lock from the worktree root:
#   python -B tools/qwen38_gpu_lock.py --timeout 60 -- sh <this> <dir> > runs.jsonl
# Each spec is 3 warm-up + 20 measured command buffers (20 encodes each) by the harness;
# 3 rounds, order alternating (before, after / after, before / before, after).
D=$1
specs="dbench:fused:10:2560:768:640:512:1:10:20 dbench:fused:10:2560:768:640:512:4:10:20
dbench:fused:39:2560:640:640:512:1:10:20 gbench:fused:16:640:2560:2560:512:10:10:20
gbench:fused:16:640:2560:2560:512:40:10:20 gbench:fused:12:640:2560:2560:512:10:10:20"
for round in 1 2 3; do
  order="before after"; [ $round = 2 ] && order="after before"
  for spec in $specs; do for v in $order; do
    out=$(Q38_PACKED_LIBRARY="$D/$v/picoforge.metallib" "$D/$v/test_fast_kernels" $(echo $spec | tr : " ")) || exit 1
    echo "{\"round\": $round, \"tree\": \"$v\", ${out#\{}"
  done; done
done
