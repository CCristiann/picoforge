# R5.3b — Q4 decode speed with streamed experts, 2026-11-08

`run.sh`: three chat prompts (code, chat, story), each run once with 128 and once with 512 new tokens, cold
pool, greedy, context 4096, guarded launcher on AC (90%, no thermal warning). Q4 with a 36 GiB pool
(`PICOFORGE_STREAM_EXPERTS=14244`, 58% of the trunk experts, event gates, 8 readers). Same script on the Q2
with a 50% pool for comparison. The marginal cost (decode512 - decode128) / extra tokens removes the
cold-start misses of the first 128 tokens; a run that stopped early covers fewer tokens.

| | Q4, 36 GiB pool | Q2, 50% pool | Q2 resident (R5.2b) |
|---|---|---|---|
| marginal ms/token (code / chat / story) | 56.1 / 52.9 / 53.9 | 47.4 / 47.9 / 48.1 | 38.7 |
| ≈ tok/s once warm | 18-19 | ~21 | ~25.8 |
| whole 128-token run, ms/token | 64.3-68.5 | 52.4-56.6 | |

Q4 misses after the first 128 tokens: ~2.0% of lookups (story: 3708 of 184320), ~10 experts (27 MB) per
token; the time in acquire grows ~4.6 ms/token. The rest of the gap to the Q2 is the gates (~5.7 ms/token,
R5.2b) and the heavier Q4_K/MXFP4 expert kernels. R5.0 predicted 19-24 tok/s from a model; measured is
just under that range. Peak wired 47.4-48.3 GiB, pressure 1, swap unchanged (11.4 MiB, from before).
One run per cell: NOT the 3+20 protocol, and speed was not the target of this step.
