"""M3 summary: per prompt and mode, tok/s, identity with plain, depth use, acceptance by depth, costs."""
import json, sys
from pathlib import Path
O = Path(sys.argv[1])
def run(d): return json.loads(Path(str(d) + '.jsonl').read_text().splitlines()[-1])
print(f"{'prompt':6} {'mode':5} {'tok/s':>6} {'x plain':>7} same  depth[k=0..3]      acc_at/prop_at          draft_ms/call verify_ms keep_ms")
for k in ('code', 'chat', 'story', 'count'):
    if not Path(str(O/f'{k}-plain') + '.jsonl').exists(): continue
    base = run(O/f'{k}-plain'); bt = base['decode_tokens']/base['decode_s']
    for mode in ('plain', '0', '1', '2', '3', 'auto'):
        d = O/f'{k}-{mode}'
        if not Path(str(d) + '.jsonl').exists(): continue
        r = run(d); m = r['mtp']; ts = r['decode_tokens']/r['decode_s']
        same = r['generated_ids'] == base['generated_ids']
        calls = max(1, sum(m.get('depth', [0])) if mode != 'plain' else 1)
        cyc = max(1, sum(m.get('depth', [0])[1:]))
        acc = ' '.join(f"{a}/{p}" for a, p in zip(m.get('accepted_at', []), m.get('proposed_at', [])))
        print(f"{k:6} {mode:5} {ts:6.2f} {ts/bt:7.3f} {str(same):5} {str(m.get('depth')):18} {acc:23} "
              f"{1e3*m['draft_s']/calls:8.2f} {1e3*m['verify_s']/cyc:9.2f} {1e3*m['replay_s']/cyc:7.3f}")
