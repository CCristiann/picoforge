# Qwen3.8 G2/G3 — esecuzione ordinaria GPU

2026-10-13, M5 Pro 64 GiB, branch `coord/qwen38-g1-cpu`, base `458f8e5`.
Il commit contenente questo report identifica la consegna; gli hash dei sorgenti
e degli eseguibili verificati sono in [manifest.json](manifest.json).

Implementati PLE Metal, lookup embedding e composizione B/C/D/E/F sopra G0,
con selezione QSA e routing MoE sul device. La CLI `picoforge --qwen38` esegue
chat/completamento greedy sul GGUF reale. Contratto e uso in
[G2.md](../../docs/specs/qwen38/G2.md). MTP escluso dal grafo ordinario.

## Gate numerici

Errore scalato: `max(abs(actual-reference)) / max(1, max(abs(reference)))`.
Le metriche sull'intero tensore e per posizione usano denominatori diversi.
La soglia end-to-end resta **1e-3**, senza rilassamenti dopo le misure.

| Verifica | Risultato |
|---|---|
| Componenti GPU | 75 controlli numerici PASS, più assert su lifecycle/preflight/errori |
| PLE Delta vs CPU | massimo scalato 1,737e-7; gate 3e-5 |
| QSA | ID esatti, anche oltre p=2051; chunk/reset bit-identici |
| Grafo GPU vs oracle upstream | 5 test PASS; slice 0–4, tutti i confini, decode/chunk/replay |
| Stato e fault GPU | 72 token, microbatch MoE 32, chunk 17: output e history bit-identici; errore tardivo non pubblica logits/hash/posizione |
| CPU picoforge, checkpoint reale | 388 confini; massimo scalato 4,847e-6; logits 1,879e-6; argmax 12/12 |
| DwarfStar CPU FP32, checkpoint reale | tutti i 48 residui: massimo 2,560e-6; logits 2,241e-6 sull'intero tensore, 2,404e-6 massimo per token |
| DwarfStar CPU FP32, scelte token | argmax e top-5 ordinato 12/12; margine argmax > 2×errore in 12/12 |
| CLI | 3 test PASS: greedy vs CPU a ogni step, ultimo logit, chat/completamento e rifiuti input |
| Regressioni | B GPU 25 test, G0 11 gruppi, G1 5 test, E CPU 17 test, contratti 4 test, observer 6 test, `make test-all` PASS |

Log finali: [validation.log](validation.log), [cli.log](cli.log),
[e-cpu.log](e-cpu.log), [legacy-test.log](legacy-test.log).
[initial-validation.log](initial-validation.log) conserva i 75 controlli
componenti e il primo errore CLI: la fixture numerica era priva di tokenizer.
È stata aggiunta una fixture separata con vocabolario byte e template fissato;
i controlli di ammissione del runtime sono invariati. Il successivo test CLI
passa. [parts.json](parts.json) estrae soltanto i risultati dei componenti.

I piccoli harness C/Metal usano ASan/UBSan fatali, warning come errori e Metal
API validation. CLI e prova full-checkpoint sono build normali, senza sanitizer;
la CLI supera anche il controllo sintattico con `-Werror`. Gli operatori FP32
mantengono `-ffp-contract=off -fno-fast-math`.

## Checkpoint e riferimento indipendente

GGUF esistente di 147207127040 byte, hash già verificato nelle fasi precedenti
`b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
Nessun nuovo checksum integrale, download o conversione. N-gram BF16 da SSD.
DwarfStar è il checkout **non modificato**
`fc80bd695da76ee14bcb86820bf900a5cdfda806`.

[g3.json](g3.json) contiene i 12 token fissati, argv/env, gate, 48 errori hidden,
errori e margini per posizione, top-5 e misure di memoria.
[g3-artifacts.json](g3-artifacts.json) identifica le tracce binarie conservate
localmente in `build/qwen38-g3-final/`; non sono incluse nel repository.
Le classifiche top-5 sono state aggiunte con `logits_report` sugli stessi file
di logits conservati, senza rieseguire il modello.

Il gate G3 mantiene il lock del worktree picoforge anche nei figli che cambiano
cwd al checkout DwarfStar. Il riferimento primario è il percorso CPU FP32
`--cpu --first-token-test`, con dump hidden/logits abilitati dalle opzioni
diagnostiche stock. Il piccolo harness Metal osservazionale è linkato solo
agli oggetti DwarfStar, mai al motore picoforge.

## Controllo negativo stock Metal

Il backend DwarfStar Metal usa cache K/V FP16; picoforge mantiene FP32.
Il confronto stock **FAIL** è conservato: errore logits scalato 0,0324888
sull'intero tensore, massimo per token 0,0413082 alla posizione 11.
Argmax e top-5 coincidono comunque in 12/12 posizioni.

La divergenza cresce al MoE del layer 7, posizione 11. Al confine top-10,
picoforge seleziona 107, DwarfStar 227. Gap 10/11: 0,000214100 e 0,000111580,
entrambi sotto 2×errore massimo degli score (0,000737190).
[router-near-tie.json](router-near-tie.json) registra i ranking e gli hash del
router/header letto in modo bounded. [near-tie-scores.f32](near-tie-scores.f32)
contiene due righe di 512 FP32 little-endian, native poi DwarfStar.

Replay CPU e kernel GPU di F sugli **stessi** score: ID bit-identici, errore
pesi 1,49e-8 entro 2e-6 ([log](near-tie-replay.log)). Nessun arrotondamento
FP16 inserito nel motore e nessuna tolleranza aumentata per far passare lo
stock Metal. Il FAIL resta distinto dal PASS a pari precisione CPU FP32.

## Generazione e memoria

Prompt: «Rispondi in italiano: quanto fa 2 + 2?».
Output: `2 + 2 fa **4**.`; 25 token di prompt, 9 generati incluso stop.
[ID](chat-ids.json), [stdout](chat/stdout.log), [stderr](chat/stderr.log).

Contesto allocato 8192, chunk massimo 1024, una riga logits. Misure in
[memory.json](memory.json), dati grezzi in [chat/memory.jsonl](chat/memory.jsonl).

| Contatore | Misura |
|---|---:|
| Metal `currentAllocatedSize` a fine generazione | 43,116 GiB |
| Metal working set raccomandato | 51,840 GiB |
| Buffer Metal propri del grafo | 1,395 GiB |
| Picco lifetime footprint del processo | 1,577 GiB |
| Picco RSS campionato | 0,626 GiB |
| Picco wired dell'intero sistema | 47,164 GiB |
| Pressione nei 41 campioni (intervallo 0,25 s) | normale |
| Crescita swap e contatori swap-in/out | 0 |

Footprint, RSS, allocazioni Metal e wired di sistema contano insiemi diversi:
non vanno sommati né scambiati. Il footprint di processo non rappresenta
l'intero working set GPU. Il wired comprende anche sistema e altre app.
Questa prova usa un prompt breve con capacità 8K allocata: non certifica
un'intera generazione di 8192 token, qualità generale o vantaggi di velocità.
I tempi nei log servono alla riproducibilità, non sono benchmark prestazionali.

## Riproduzione

Dalla root del worktree, con gli asset già preparati e il venv esistente:

```sh
make test-qwen38-g2-parts test-qwen38-g2 test-qwen38-g2-state test-qwen38-cli \
  PYTHON=/percorso/venv/bin/python
make test-qwen38-e-cpu test-qwen38-b-gpu test-qwen38-g0 test-qwen38-g1 test-qwen38-contracts \
  PYTHON=/percorso/venv/bin/python
make qwen38-g3 G3_OUTPUT=build/qwen38-g3-nuovo \
  PYTHON=/percorso/venv/bin/python
```

I target GPU acquisiscono il lock comune: non annidare wrapper. `G3_OUTPUT`
deve essere nuovo; eventuali errori restano nei log. I percorsi predefiniti
si possono sostituire con `Q38_REFERENCE_ROOT`, `Q38_CHECKPOINT`, `Q38_DWARFSTAR`.
Il target G3 riporta separatamente `pass_fp32` e `stock_metal_pass`.

Per ripetere solo il near-tie salvato:

```sh
/percorso/venv/bin/python tools/qwen38_gpu_lock.py -- env MTL_DEBUG_LAYER=1 \
  ./build/qwen38-g2/test_parts build/qwen38-g1/graph.gguf "$PWD/picoforge.metallib" \
  bench/qwen38_g2_g3_20261013/near-tie-scores.f32
```

La suite legacy è stata eseguita sui modelli Qwen3 già presenti, con `make -o`
per i tre target quantizzati condivisi, evitando di rigenerarli. Nessuna nuova
dipendenza, modifica al wired limit, merge o push su main.
