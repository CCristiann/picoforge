# R1 — chiusura (ruling R43)

M5 Pro 64 GiB, 28 ottobre 2026, alimentazione da rete, swap 0, nessun avviso termico.
Codice: `56760ab` (stesso codice motore di `f7bbeed`; dopo ci sono solo evidenze e documenti). Suite
`bench/qwen38_r0_20261017/suite`, harness `tools/eval/qwen38_compare.py` (alternato, `--block 1`).

**Esito in una riga:** R1 porta il decode da 1,06 a 20,7 tok/s (19,5×), ma si ferma a
**0,62×** DwarfStar sul prompt corto e a **0,37×** a 8K. La soglia d'uscita 0,95× non è
raggiunta; R1 si chiude qui per scelta (R43). gdn.hc_pre, gdn.scan e le matvec HC F16
passano al backlog R2.

| Frontiera | Motore | Warmup+misure | tok/s mediana | p10 | p90 | vs DwarfStar |
|---|---|---|---:|---:|---:|---:|
| short | picoforge | 3+20 | 20,716 | 20,692 | 20,741 | 0,623× |
| short | DwarfStar | 3+20 | 33,230 | 33,030 | 33,300 | — |
| 8K (8063 + 128) | picoforge | 1+5 | 11,715 | 11,133 | 11,745 | 0,372× |
| 8K (8063 + 128) | DwarfStar | 3+20 | 31,475 | 30,650 | 32,140 | — |
| 8K, R0 | picoforge | profilo R0, non appaiato | ≈ 1,03 | — | — | — |

- `paired-short-t17/` e `paired-8k-t17/`: campioni grezzi (`samples.csv`), manifest e processi.
- `profile-t18/`: profilo per gruppo (44,99 ms/token di GPU, 135,7 GB/s); vedi la ladder in
  `../qwen38_r1_20261019/ladder.csv`.
- `moe-down/`, `moe-gate-up/`, `moe-factored-dot/`: bench isolati di Task 18 (devlog R1.9b/c).
- 8K-R0 non rieseguito (R43): il valore R0 a 8K viene da `../qwen38_r0_20261017/profile-8k`
  e non è appaiato. Il picoforge 8K usa 1+5 ripetizioni (R22: ogni run a 8K è lungo).
- A 8K il divario cresce. Non c'è un profilo R1 a 8K, quindi la causa non è misurata. Il
  candidato più probabile è `qsa.core`, l'attenzione piena su 12 layer, che scala con il
  contesto: in R0 era 13 volte più lento a 8K (R0 README). Un profilo a 8K è la prima voce del
  backlog R2.
