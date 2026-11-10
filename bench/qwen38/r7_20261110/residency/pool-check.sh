#!/bin/zsh
# Q4 logits of code-0 with a 10000-slot pool, against the 14421-slot run's sha256
cd /Users/cristiancirje/Desktop/Dev/picoforge; PY=tools/venv/bin/python; O=build/r7/pool-check
TOK=$($PY -c "import json;print(','.join(map(str,json.load(open('build/r7/corpus.json'))['streams']['code']['windows'][0]['ids'])))")
pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' -e pool-check && exit 9
$PY -B tools/qwen38_gpu_lock.py --timeout 3600 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py $O PICOFORGE_STREAM_EXPERTS=10000 -- -m q4 --tokens $TOK --teacher-forced 2047 --context 2048 --logits '{d}/logits.f32' || exit 1
shasum -a 256 $O/logits.f32; grep code-0-q4/ bench/qwen38/r7_20261110/logits-sha256.txt; rg -o "expert streaming: [0-9]+ slots[^,]*|expert streaming: [0-9]+ hits.*" $O/process/stderr.log
