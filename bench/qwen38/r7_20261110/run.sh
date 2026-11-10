#!/bin/zsh
# R7: one guarded teacher-forced run of Q4 and of Q2 per corpus window, reduced to
# per-position metrics (tools/eval/qwen38_r7_quality.py), raw logits deleted after.
# usage: run.sh "prose:0 prose:1 code:0 ..."     (from the main checkout, on AC)
set -u
cd /Users/cristiancirje/Desktop/Dev/picoforge
Q4ENV=${Q4ENV:-PICOFORGE_STREAM_EXPERTS=10000}   # Q4 pool: logits are bit-identical at any size (R5.3a); 10000 slots leave room for the desktop
PY=tools/venv/bin/python; C=build/r7/corpus.json; O=build/r7/runs; B=bench/qwen38/r7_20261110
mkdir -p $O $B/windows
guard() { pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' -e 'qwen38_r7_' && { echo "FOREIGN PROCESS: stop"; exit 9; }; pmset -g batt | sed -n 2p; }
for spec in ${=1}; do
  s=${spec%%:*}; w=${spec#*:}; n=$s-$w
  [ -f $B/windows/$n.npz ] && { echo "$n: done"; continue; }
  TOK=$($PY -c "import json;print(','.join(map(str,json.load(open('$C'))['streams']['$s']['windows'][$w]['ids'])))")
  N=$(( $(echo $TOK | tr ',' '\n' | wc -l) - 1 ))
  for m in q4 q2; do
    envs=(Q38_X=0); [ $m = q4 ] && envs=(${=Q4ENV})
    rm -rf "build/r7/runs/${n:?}-${m:?}"; sleep 15; guard
    $PY -B tools/qwen38_gpu_lock.py --timeout 3600 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py $O/$n-$m ${envs[@]} \
      -- -m $m --tokens $TOK --teacher-forced $N --context 2048 \
      --logits '{d}/logits.f32' --runs-jsonl '{d}/runs.jsonl' || { echo "RUN $n-$m FAILED"; exit 1; }
  done
  $PY tools/eval/qwen38_r7_quality.py reduce $C $s $w $O/$n-q4/logits.f32 $O/$n-q2/logits.f32 $B/windows/$n.npz || exit 1
  for m in q4 q2; do
    shasum -a 256 $O/$n-$m/logits.f32 | sed "s|$O/||" >> $B/logits-sha256.txt
    cp $O/$n-$m/conditions.json $B/windows/$n-$m-conditions.json; cp $O/$n-$m/runs.jsonl $B/windows/$n-$m-runs.jsonl
    rm $O/$n-$m/logits.f32
  done
done
