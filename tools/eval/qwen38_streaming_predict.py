"""Predicted decode tok/s of Qwen3.8 Q4K with streamed experts (R5.0), from the cache simulator.

    qwen38_streaming_predict.py cache_sim.csv

A model, not a measurement: token time = compute + I/O, with I/O = misses x expert size / SSD
bandwidth. `serial` adds them (no overlap, the pessimistic bound), `overlap` takes the larger
(perfect hiding, the optimistic bound). Compute is the resident decode time: 38.5 ms is the Q2
graph after Decode II; the Q4K graph reads ~7.7 GB per token instead of ~6.1 GB, so 43 ms (the extra
0.65 GB at ~150 GB/s) is the central guess and 49 ms (7.7 GB at the 158 GB/s the Q2 graph reached
before Decode II) the conservative one. The SSD figure is the measured ~13 GB/s for 2.6 MB random
reads with 2+ threads (bench/qwen38_r5_20261105/ssd_bench.txt).
"""
import csv
import sys
from collections import defaultdict

EXPERT_MB, PICKS, SSD_GBPS = 2.58, 48 * 10, 13.0
rows = list(csv.DictReader(open(sys.argv[1])))
miss = defaultdict(list)
for r in rows:
    if r['policy'] == 'lru_global':
        miss[int(r['cache_gib'])].append(float(r['miss_rate']))
print(f'{"cache GiB":>9} {"miss":>6} {"IO ms":>6} | ' + ' | '.join(f'Tc {tc} ms: serial / overlap' for tc in (38.5, 43.0, 49.0)))
for gib in sorted(miss):
    m = sum(miss[gib]) / len(miss[gib])
    io = m * PICKS * EXPERT_MB / 1024 / SSD_GBPS * 1000
    cells = [f'{1000 / (tc + io):5.1f} / {1000 / max(tc, io):5.1f}' for tc in (38.5, 43.0, 49.0)]
    print(f'{gib:9d} {m:6.3f} {io:6.1f} | ' + '   |   '.join(f'{c:>20}' for c in cells))
