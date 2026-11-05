"""Expert-cache simulator for Qwen3.8 routed-expert streaming (R5.0).

Replays the routing traces written by `Q38_ROUTE_LOG` (decode appends only) against cache
policies and reports the miss rate per cache size. A miss is one routed expert (layer,
expert) that is not in the cache when the router picks it; it costs one read of that expert.

    qwen38_expert_cache_sim.py OUT.csv TRACE_DIR [TRACE_DIR ...]

A trace dir holds route.bin and ids.json (prompt_ids). route.bin is a sequence of records
`uint32 pos, count, layers, top_k` then `uint32 [layer][row][top_k]`. Decode appends are the
records with count == 1 and pos >= len(prompt_ids); the first half of each trace warms the
cache, the second half is measured (steady state), the way a long chat session would see it.

Policies: lru_global (one LRU over every layer), lru_layer (capacity split evenly per layer),
static_other (the C most frequent experts of the OTHER traces, never replaced: what a profile
built beforehand could pin), belady (offline optimum over all layers: a lower bound on any
policy, not an implementation).
"""
import collections
import csv
import heapq
import json
import struct
import sys
from pathlib import Path

EXPERT_MB_Q4K = 2.58          # 62 GiB / (48 layers x 512 experts), from the roadmap; Q2 is 1.42
LAYERS_TOTAL = 48 * 512


def load(trace):
    prompt = len(json.loads((trace / 'ids.json').read_text())['prompt_ids'])
    data = (trace / 'route.bin').read_bytes()
    pos, accesses = 0, []                       # one entry per decode token: list of (layer, expert)
    while pos < len(data):
        start, count, layers, k = struct.unpack_from('<4I', data, pos)
        pos += 16
        n = layers * count * k
        ids = struct.unpack_from(f'<{n}I', data, pos)
        pos += 4 * n
        if count != 1 or start < prompt:
            continue
        accesses.append([(l, ids[l * k + j]) for l in range(layers) for j in range(k)])
    return accesses


def lru(stream, capacity, split, layers):
    """stream: flat list of (layer, expert). Returns hit flags."""
    per = max(1, capacity // layers) if split else capacity
    caches = [collections.OrderedDict() for _ in range(layers if split else 1)]
    hits = []
    for key in stream:
        c = caches[key[0] if split else 0]
        if key in c:
            c.move_to_end(key); hits.append(True)
        else:
            hits.append(False)
            c[key] = None
            if len(c) > per:
                c.popitem(last=False)
    return hits


def belady(stream, capacity):
    nxt, last = [0] * len(stream), {}
    for i in range(len(stream) - 1, -1, -1):
        nxt[i] = last.get(stream[i], float('inf')); last[stream[i]] = i
    cache, heap, hits = {}, [], []
    for i, key in enumerate(stream):
        if key in cache:
            hits.append(True)
        else:
            hits.append(False)
            while len(cache) >= capacity:
                n, k = heapq.heappop(heap)
                if cache.get(k) == -n:
                    del cache[k]
            cache[key] = nxt[i]
            heapq.heappush(heap, (-nxt[i], key)); continue
        cache[key] = nxt[i]; heapq.heappush(heap, (-nxt[i], key))
    return hits


def static_hits(stream, pinned):
    return [k in pinned for k in stream]


def main():
    out, traces = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    data = {t.name: load(t) for t in traces}
    sizes_gib = [4, 8, 16, 24, 32, 36, 40, 48, 56, 62]
    rows = []
    for name, tokens in data.items():
        stream = [a for tok in tokens for a in tok]
        half = len(stream) // 2
        others = collections.Counter(a for n2, t2 in data.items() if n2 != name for tok in t2 for a in tok)
        order = [k for k, _ in others.most_common()]
        for gib in sizes_gib:
            cap = min(LAYERS_TOTAL, int(gib * 1024 / EXPERT_MB_Q4K))
            for policy, hits in (('lru_global', lru(stream, cap, False, 48)), ('lru_layer', lru(stream, cap, True, 48)),
                                 ('static_other', static_hits(stream, set(order[:cap]))), ('belady', belady(stream, cap))):
                steady = hits[half:]
                rows.append(dict(trace=name, policy=policy, cache_gib=gib, slots=cap, tokens=len(tokens),
                                 accesses=len(steady), miss_rate=round(1 - sum(steady) / len(steady), 5)))
    with out.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'{"cache GiB":>9} ' + ' '.join(f'{p:>13}' for p in ('lru_global', 'lru_layer', 'static_other', 'belady')) + '   (mean miss rate over traces, steady state)')
    for gib in sizes_gib:
        cells = []
        for p in ('lru_global', 'lru_layer', 'static_other', 'belady'):
            v = [r['miss_rate'] for r in rows if r['policy'] == p and r['cache_gib'] == gib]
            cells.append(f'{sum(v) / len(v):13.3f}')
        print(f'{gib:9d} ' + ' '.join(cells))


if __name__ == '__main__':
    main()
