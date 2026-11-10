#!/bin/zsh
# A/B for the slow 10000-slot pool: A=14421 slots (set skipped), B=10000 (set attached), C=10000 without the set.
# 512 teacher-forced tokens of code-1, order A B C C B A, 15 s apart.
cd /Users/cristiancirje/Desktop/Dev/picoforge; PY=tools/venv/bin/python; O=build/r7/ab
mkdir -p $O
TOK=$($PY -c "import json;print(','.join(map(str,json.load(open('build/r7/corpus.json'))['streams']['code']['windows'][1]['ids'][:513])))")
i=0
for c in A B C C B A; do
  i=$((i+1))
  case $c in A) envs=(PICOFORGE_STREAM_EXPERTS=14421);; B) envs=(PICOFORGE_STREAM_EXPERTS=10000);; C) envs=(PICOFORGE_STREAM_EXPERTS=10000 PICOFORGE_NO_RESIDENCY=1);; esac
  sleep 15
  pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' -e 'ab.sh' && { echo FOREIGN; exit 9; }
  $PY -B tools/qwen38_gpu_lock.py --timeout 3600 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py $O/$i-$c ${envs[@]} -- \
    -m q4 --tokens $TOK --teacher-forced 512 --context 1024 --logits '{d}/logits.f32' --runs-jsonl '{d}/runs.jsonl'
  d=$O/$i-$c
  echo "$i $c decode_s=$($PY -c "import json;print(round(json.loads(open('$d/runs.jsonl').readline())['decode_s'],1))" 2>/dev/null) $(grep -ho 'residency set [a-z]*' $d/process/stderr.log) | $(grep -ho '[0-9]* misses[^,]*, [0-9]* evictions, [0-9.]* GiB read, [0-9.]* s waiting' $d/process/stderr.log) | sha $(shasum -a 256 $d/logits.f32 | cut -c1-12)"
  rm -f $d/logits.f32
done
