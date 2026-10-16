# G2 — D QSA sul device

2026-10-13, Apple M5 Pro. Ripresa del Passo 3 nel worktree
`picoforge-g1-cpu`, ramo `coord/qwen38-g1-cpu`, base `38e3785`.
Evidenza di correttezza, non benchmark o generazione GPU completa.

`q38d_layer_encode` replica il percorso CPU sugli ingressi già proiettati:
norma/gate → RoPE → cache → pool dei blocchi completati → norma/RoPE del
pool → score → selezione sul device → attention gated. Nessun readback degli
ID fra gli stadi, allocazione per append, cast half o modifica delle formule.
Il caller conserva submission e pubblicazione dello stato attraverso G0.

## Risultati

- Build/link riusciti, warning come errori nei harness, ASan/UBSan,
  Metal API validation e lock GPU comune.
- **52 controlli G2**: 12 MoE preesistenti e 40 QSA. Prefissi da 3, 17, 40
  e 2053 token, chunk 1/13/1023, budget 8/2048 e score in pareggio.
  Scarto scalato massimo output QSA CPU/Metal **1,192e-7** (gate 3e-5);
  ogni intermedio ≤1,777e-7. ID esatti; output GPU bit-identici tra
  one-shot, chunk/decode e reset/replay.
- Preflight malformati e T=0 non scrivono i buffer. NaN su V dopo mutazione
  invalida la sessione G0 senza pubblicare posizione; reset/replay recupera
  la stessa uscita. Input, cache e scratch sono dichiarati al ledger G0.
- **13 test D Metal** contro CPU e oracle indipendente: 39 casi di selezione
  random/tie/near-tie con score FP32 identici e margini registrati; NaN/Inf;
  confine 2051; selezione → attention nello stesso comando a N=32768,
  con K/V non selezionate avvelenate. La slice Q2 esistente attraversa gli
  stadi e la selezione GPU; le sue proiezioni rimangono quelle della fixture.
- Regressioni: **44 test D CPU/oracle**, **5 G1**, **4 contratti** e
  **6 gruppi ledger + 5 sessione G0**, senza skip.

`parts.json` contiene i confronti G2; `attention.json` le misure per stadio,
la slice e i margini. `manifest.json` identifica base, macchina, test,
metallib, fixture e SHA-256. I log conservano i comandi completi.
`regression.log` include una precedente esecuzione G2, prima del riordino
del calcolo dei flag bitwise nel report; `gpu.log` è la verifica finale.
Il warning PyTorch in G1 riguarda una view NumPy in sola lettura già
presente nell'oracle; nessun test fallisce.

## Riproduzione

Dalla radice del worktree, con asset locali già preparati:

```sh
Q38_PY=/path/to/picoforge/tools/venv/bin/python
make -j4 picoforge
make test-qwen38-g2-parts test-qwen38-d-gpu PYTHON="$Q38_PY"
make test-qwen38-d-cpu test-qwen38-g0 test-qwen38-g1 test-qwen38-contracts PYTHON="$Q38_PY"
```

I target GPU acquisiscono il lock; non aggiungere un secondo wrapper.
Output atteso: righe `pass:true`, 13/44/5/4 test `OK` e gruppi G0 passati.
`make test-all` non è stato eseguito in questo incremento; le regressioni
elencate sono quelle pertinenti a D e alla composizione G0/G1/G2.

## Limiti

Ranking/compattazione conservativi O((N/ratio)²) per query e scratch
O(N/ratio + budget): nessuna matrice T×T, ma nessuna promessa di velocità.
32K è una prova bounded di selezione/attention su forme ridotte, non un
prefill del checkpoint intero. Non dimostra un budget RAM del modello.
Tie e near-tie sono certificati sul contratto FP32 a score identici;
non implicano equivalenza con un top-k HF instabile sui pareggi.
Restano proiezioni/output B intorno a D, collegamento del grafo GPU completo
e confronto dei logits reali. Vedi la consegna in `docs/specs/qwen38/D.md`.
