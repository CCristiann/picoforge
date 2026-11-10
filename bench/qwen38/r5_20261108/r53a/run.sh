#!/bin/zsh
# R5.3: FP32 CPU reference (DwarfStar fc80bd6 --first-token-test) against picoforge teacher-forced logits.
# usage: r53-oracle.sh TAG MODEL "RUNS"   RUNS: name:ENV=VAL,ENV=VAL ... (res = no env)
set -u
cd /Users/cristiancirje/Desktop/Dev/picoforge-perf-decode
PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
DS=/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline
TAG=$1; M=$(cd ${2:h} && pwd)/${2:t}; RUNS=$3
TOK=248045,846,198,7734,12654,220,18,1970,25471,10562,56761,2515
O=build/r5-checks/r53-$TAG-$(git rev-parse --short HEAD); mkdir -p $O
guard() { pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' -e r53-oracle && { echo "FOREIGN PROCESS: stop"; exit 9; }; pmset -g batt | sed -n 2p; }
if [ ! -f $O/ref/fp32-logits.f32 ]; then
  guard
  $PY -B tools/qwen38_gpu_lock.py --timeout 3600 -- caffeinate -i -s $PY -B build/r5-checks/launch-any.py $O/ref $DS \
    DS4_QWEN4_FT_TOKENS=$TOK DS4_QWEN4_FT_OUT='{d}/fp32-logits.f32' -- $DS/ds4 --cpu --first-token-test -m $M -p x || { echo "REFERENCE FAILED"; exit 1; }
fi
for spec in ${=RUNS}; do
  name=${spec%%:*}; envs=(Q38_X=0); [ "$spec" != "$name" ] && envs=(${(s:,:)${spec#*:}})
  guard
  $PY -B tools/qwen38_gpu_lock.py --timeout 3600 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py \
    $O/$name ${envs[@]} -- --qwen38 $M --tokens $TOK --teacher-forced 11 --context 64 --chunk 16 \
    --logits '{d}/logits.f32' --token-ids '{d}/ids.json' --runs-jsonl '{d}/runs.jsonl' || echo "RUN $name FAILED"
done
V=$(( $(stat -f %z $O/ref/fp32-logits.f32) / 48 ))
for spec in ${=RUNS}; do name=${spec%%:*}; printf "%-10s " $name; grep -h "expert streaming: [0-9]* hits" $O/$name/process/stderr.log | tr '\n' ' '; echo; $PY -B build/r5-checks/compare-logits.py $O/$name/logits.f32 $O/ref/fp32-logits.f32 $V $O/$name/report.json; done
runs=(${=RUNS}); first=${runs[1]%%:*}
for spec in ${=RUNS}; do name=${spec%%:*}; cmp -s $O/$first/logits.f32 $O/$name/logits.f32 && echo "$name == $first: BIT-IDENTICAL" || echo "$name vs $first: differ"; done
