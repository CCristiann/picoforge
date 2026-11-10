#!/bin/zsh
# R5.3: Q4 decode speed with streamed experts. Per prompt, one 128-token and one 512-token run (cold pool);
# (decode512 - decode128) / 384 is the marginal cost of a token once the pool is warm.
set -u
cd /Users/cristiancirje/Desktop/Dev/picoforge-perf-decode
PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
M=${MODEL:-models/Qwen3.8-Flash-Next-Q4/Qwen3.8-Flash-Next-Q4.gguf}; SLOTS=${SLOTS:-14244}
O=build/r5-checks/r53-speed-${TAG:-q4}-$(git rev-parse --short HEAD); mkdir -p $O
typeset -A P
P[code]='Scrivi una funzione Python che legge un file CSV e restituisce la media di ogni colonna numerica. Rispondi solo con il codice.'
P[chat]='Scrivi cinque consigli per verificare un backup.'
P[story]='Racconta una breve storia su un guardiano del faro che trova una lettera in una bottiglia.'
for k in code chat story; do for n in 128 512; do
  pgrep -fl 'ds4|picoforge --|picoforge -m' | grep -v -e t3-claude -e 'kill -TERM 0' && { echo "FOREIGN PROCESS: stop"; exit 9; }
  pmset -g batt | sed -n 2p
  $PY -B tools/qwen38_gpu_lock.py --timeout 3600 -- caffeinate -i -s $PY -B build/r4-checks/launch-cmd.py \
    $O/$k-$n PICOFORGE_STREAM_EXPERTS=$SLOTS -- --qwen38 $M --chat "${P[$k]}" --max-new $n --context 4096 --chunk 128 \
    --token-ids '{d}/ids.json' --runs-jsonl '{d}/runs.jsonl' || echo "RUN $k-$n FAILED"
done; done
$PY - $O <<'PY'
import json, re, sys, glob, os
O = sys.argv[1]; rows = {}
for d in sorted(glob.glob(O + '/*-*')):
    r = json.loads(open(d + '/runs.jsonl').readline()); s = open(d + '/process/stderr.log').read()
    m = re.search(r'(\d+) hits, (\d+) misses .* ([\d.]+) s waiting', s)
    mem = [json.loads(l) for l in open(d + '/process/memory.jsonl')]
    rows[os.path.basename(d)] = r
    print(f"{os.path.basename(d):10s} prefill {r['prefill_s']:.2f}s decode {r['decode_tokens']} tok {r['decode_s']:.2f}s = {1000*r['decode_s']/r['decode_tokens']:.1f} ms/tok;"
          f" misses {m.group(2)}/{int(m.group(1))+int(m.group(2))}, acquire {m.group(3)} s; swap {max(x['system_swap_used_bytes'] for x in mem)}, pressure {max(x['system_pressure_level'] for x in mem)},"
          f" wired {max(x['system_wired_pages'] for x in mem)*16384/2**30:.1f} GiB, stop {r['stop_reason']}")
for k in ('code', 'chat', 'story'):
    a, b = rows.get(k + '-128'), rows.get(k + '-512')
    if a and b and b['decode_tokens'] > a['decode_tokens']:
        dt = b['decode_tokens'] - a['decode_tokens']
        print(f"{k}: marginal {1000*(b['decode_s']-a['decode_s'])/dt:.1f} ms/token over tokens {a['decode_tokens']+1}-{b['decode_tokens']}")
PY
