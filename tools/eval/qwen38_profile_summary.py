"""R0.3 summary of `picoforge --qwen38 --profile` CSVs: what one decode token costs, per op.

Usage: qwen38_profile_summary.py CSV... --out DIR [--runs-jsonl FILE]

Every append starts with its `embed` row, the only reliable separator: (phase,
token) repeats with --mtp and --repeat. Only decode appends are summarized and
"per token" means per decode append (with --mtp a verify append counts once).
weight_bytes are the weight tensors a group's kernels read: linears their whole
matrix, embed its rows, gdn its linears, ple key/value plus the staged n-gram
rows, moe router + shared + top_k experts. qsa.core reads the KV/index caches,
not weights, so it reports 0 bytes. Writes summary.csv, totals.json and
summary.md into a new DIR. --runs-jsonl takes the runs file of a NORMAL-mode
(unprofiled) run on the same prompt, to compare the sum of groups with gpu_s.
Python standard library only.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

HEADER = ['phase', 'token', 'layer', 'op', 'format', 'weight_bytes', 'dispatches', 'barriers', 'gpu_us', 'gbps']
COLUMNS = ['op', 'format', 'calls_per_token', 'us_per_token', 'share', 'weight_bytes_per_token', 'gbps']
FLOOR_US = 1.4        # measured cost of one tiny dispatch inside a command buffer (devlog, Step 4.1)
CEILING_GBPS = 307.0  # M5 Pro memory bandwidth


def appends(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        if next(reader, None) != HEADER:
            raise SystemExit(f'{path}: not a qwen38 profile CSV')
        group = None
        for row in reader:
            r = dict(zip(HEADER, row, strict=True))
            if r['op'] == 'embed':
                if group:
                    yield group
                group = []
            elif group is None or (r['phase'], r['token']) != (group[0]['phase'], group[0]['token']):
                raise SystemExit(f'{path}: row outside its append: {row}')
            if int(r['dispatches']) < 0 or int(r['barriers']) < 0:
                raise SystemExit(f'{path}: no dispatch counts (a pre-R0.3 CSV)')
            group.append(r)
        if group:
            yield group


def quantile(xs, q):
    xs = sorted(xs); i = (len(xs) - 1) * q; lo = int(i); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summarize(paths, runs):
    decode = [a for p in paths for a in appends(p) if a[0]['phase'] == 'decode']
    if not decode:
        raise SystemExit('no decode appends')
    n, ops = len(decode), {}
    for r in (r for a in decode for r in a):
        o = ops.setdefault(r['op'], {'formats': set(), 'calls': 0, 'us': 0.0, 'bytes': 0})
        o['formats'].add(r['format']); o['calls'] += 1; o['us'] += float(r['gpu_us']); o['bytes'] += int(r['weight_bytes'])
    total_us = sum(o['us'] for o in ops.values()) / n
    rows = []
    for op, o in sorted(ops.items(), key=lambda kv: -kv[1]['us']):
        us, nbytes = o['us'] / n, o['bytes'] / n
        rows.append({'op': op, 'format': '+'.join(sorted(o['formats'])), 'calls_per_token': o['calls'] / n,
                     'us_per_token': us, 'share': us / total_us, 'weight_bytes_per_token': nbytes,
                     'gbps': nbytes / (us * 1e3) if us else 0.0})
    per_append = [sum(float(r['gpu_us']) for r in a) for a in decode]
    def per_token(key):
        return sum(int(r[key]) for a in decode for r in a) / n
    nbytes, dispatches = per_token('weight_bytes'), per_token('dispatches')
    totals = {'decode_appends': n, 'groups_per_token': sum(map(len, decode)) / n, 'us_per_token': total_us,
              'us_per_token_median': statistics.median(per_append), 'us_per_token_p10': quantile(per_append, 0.1),
              'us_per_token_p90': quantile(per_append, 0.9), 'weight_bytes_per_token': nbytes,
              'effective_gbps': nbytes / (total_us * 1e3), 'bandwidth_floor_us_per_token': nbytes / (CEILING_GBPS * 1e3),
              'dispatches_per_token': dispatches, 'barriers_per_token': per_token('barriers'),
              'dispatch_floor_us_per_token': dispatches * FLOOR_US}
    if runs:
        done = [json.loads(line) for line in Path(runs).read_text().splitlines() if line.strip()]
        done = [d for d in done if d.get('event') == 'done' and not d.get('warmup')]
        tokens = sum(d['mtp']['target_decode_appends'] for d in done)
        if not tokens:
            raise SystemExit(f'{runs}: no measured decode appends')
        normal = sum(d['decode_profile']['gpu_s'] for d in done) / tokens * 1e6
        totals.update(normal_decode_gpu_us_per_token=normal, groups_over_normal_gpu=total_us / normal)
    return rows, totals


def markdown(rows, totals, paths):
    t = totals
    lines = ['# Qwen3.8 decode profile, per decode token', '',
             f"Sources: {', '.join(Path(p).name for p in paths)}; {t['decode_appends']} decode appends.", '',
             f"- GPU time (sum of groups): {t['us_per_token']:.1f} µs (median {t['us_per_token_median']:.1f}, "
             f"p10 {t['us_per_token_p10']:.1f}, p90 {t['us_per_token_p90']:.1f})",
             f"- Weight bytes: {t['weight_bytes_per_token'] / 1e6:.2f} MB, {t['effective_gbps']:.1f} GB/s; "
             f"floor at {CEILING_GBPS:.0f} GB/s: {t['bandwidth_floor_us_per_token']:.1f} µs",
             f"- Dispatches: {t['dispatches_per_token']:.0f} (floor x {FLOOR_US} µs = {t['dispatch_floor_us_per_token']:.1f} µs); "
             f"barriers: {t['barriers_per_token']:.0f}"]
    if 'groups_over_normal_gpu' in t:
        lines.append(f"- Normal-mode decode gpu_s: {t['normal_decode_gpu_us_per_token']:.1f} µs; "
                     f"sum of groups / normal: {t['groups_over_normal_gpu']:.4f}")
    lines += ['', '| op | format | calls/token | µs/token | share | weight MB/token | GB/s |', '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['op']} | {r['format']} | {r['calls_per_token']:g} | {r['us_per_token']:.1f} | {100 * r['share']:.1f}% "
              f"| {r['weight_bytes_per_token'] / 1e6:.3f} | {r['gbps']:.2f} |" for r in rows]
    lines += ['', 'qsa.core reads the KV and index caches, not weights, so its weight bytes are 0. '
              'moe bytes are exact for one token (top_k experts each); for multi-token appends they are an upper bound.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csv', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--runs-jsonl')
    args = ap.parse_args()
    rows, totals = summarize(args.csv, args.runs_jsonl)
    out = Path(args.out)
    try:
        out.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f'{out} exists; every summary goes to a new directory')
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator='\n'); w.writeheader(); w.writerows(rows)
    (out / 'totals.json').write_text(json.dumps(totals, indent=2) + '\n')
    (out / 'summary.md').write_text(markdown(rows, totals, args.csv))


if __name__ == '__main__':
    main()
