# R1.5d parallel router top-k: protocol bench (2026-10-23)

Command: `caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_20261023/moe-route`
at ee67690 (manifest.txt: commit, `git status --porcelain`, harness and metallib SHA256; the
only entry is this untracked folder plus the uncommitted devlog draft, not part of the
measured code). Synthetic logits (the rbench generator), 512 experts, top 10, normalized,
never the checkpoint. 3 warm-ups + 20 measured command buffers per spec, GPU time per
dispatch (`GPUEndTime - GPUStartTime` of the buffer / 20), 20 dispatches (each followed by a
barrier) per buffer as in the decode graph. oracle = `q38f_route` (one thread per token,
64-thread groups); fast = `q38_moe_route_fast` (one 256-thread threadgroup per token). Spec
order alternates between token counts (oracle, fast / fast, oracle / oracle, fast / fast,
oracle). The error word is checked to be 0 after every measured buffer.

Conditions (before.txt / after.txt, 12:27:43Z to 12:27:46Z): AC power (`pmset -g batt`:
'AC Power', 90%, not charging), `pmset -g therm` no thermal, performance or CPU power
warning, NSProcessInfo.thermalState nominal, swap 0 before and after. `powermetrics` needs
sudo and was not used.

| tokens | q38f_route µs median (p10/p90) | q38_moe_route_fast µs median (p10/p90) | oracle / fast |
|---|---|---|---|
| 1 | 1227.3 (1224.7/1229.6) | 23.9 (23.8/24.7) | 51.4x |
| 4 | 1230.6 (1228.5/1232.3) | 24.0 (23.9/24.0) | 51.3x |
| 32 | 1294.8 (1294.0/1295.4) | 24.6 (24.6/25.0) | 52.6x |
| 128 | 1310.6 (1310.5/1311.1) | 38.7 (38.7/38.7) | 33.9x |

Per decoded token (48 MoE layers, t = 1; an estimate from the per-dispatch time, not an
end-to-end measurement): 48 x 1227.3 = 58.91 ms with the oracle, 48 x 23.9 = 1.15 ms with the
fast kernel (48 x 23.8 = 1.14 ms by the p10, 48 x 24.7 = 1.19 ms by the p90).

Prefill: no slowdown of the router dispatch (synthetic logits, not a prefill run). The fast kernel is faster at every measured token count, including
32 and 128 (24.6 and 38.7 µs against 1294.8 and 1310.6 µs), so it is the default with no
threshold; the oracle stays selectable with `Q38_REF_KERNELS=moe` (or `1`).

## runs.jsonl

One JSON line per spec, in execution order. `mode`: `oracle` or `fast`. `n`: top-k slots
(10). `k`: experts (512). `m`: tokens. `type`: always 0 (the route mode does not set it).
`reps`: dispatches per command buffer (20). `bytes`: the harness's nominal traffic,
4 x tokens x (2 x experts + slots) (4136 at 1 token): logits and probabilities counted once
each, plus ids and weights; not a measured byte count. `median_us`, `p10_us`, `p90_us`:
per-dispatch GPU time over the 20 measured buffers. `gbps`: `bytes` / median; at these sizes
the kernel is not bandwidth-bound, so it is not a bandwidth result.

## Reproduce

On AC power, with the Python venv of the main checkout:
`caffeinate -i make bench-qwen38-matvec BENCH_OUT=bench/qwen38_r1_<AAAAMMGG>/moe-route PYTHON=<venv>/bin/python`
(a new folder every time; an existing one fails). `make -n` with the same arguments lists
the 8 `rtbench` specs (checked).
The bit-identity of the fast kernel with `q38f_route` is checked by
`make test-qwen38-fast-kernels`, not by this bench.
