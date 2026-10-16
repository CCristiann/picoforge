# Ripartenza C/D/E/F dalla BASE_INTEGRATA

Sostituire `<SHA_BASE_INTEGRATA>` con lo SHA completo restituito dal
coordinatore. Tutte le consegne precedenti A–F sono già nella base: creare
nuovi branch/worktree, senza cherry-pick duplicati né modifiche ai vecchi.
Il nome BASE_CONCORDATA resta lo storico contratto `c674b23...`; il punto
di partenza operativo per questi incrementi è BASE_INTEGRATA.

## Prefisso comune da inviare con ciascun incarico

```text
Lavora in /path/to/picoforge.
MODALITA=ESEGUI, BASE_INTEGRATA=<SHA_BASE_INTEGRATA>.
Leggi le istruzioni applicabili, docs/qwen38-q2-session-prompt.md,
docs/specs/qwen38/integration.md (incluso §11), la tua spec e la consegna.
Crea un nuovo branch/worktree dalla BASE_INTEGRATA indicata, preservando tutti
quelli esistenti. Non reimportare oracle o commit delle consegne già integrate.
Mantieni Qwen3.8-Flash-Next Q2, C11/Metal, M5 Pro 64 GiB.
Implementa gli incrementi assegnati sotto, con commit piccoli e verificati;
non fermarti alla pianificazione. Usa le dipendenze A/B/C presenti nella base.
Rispetta l'ownership §7; non modificare header pubblici, Makefile, lock,
grafo o file altrui. Per una modifica condivisa consegna una proposta concreta
separata, continuando gli incrementi indipendenti già autorizzati.
Riutilizza venv e checkpoint esistenti in sola lettura, senza nuove dipendenze,
nuovo checksum completo, dequantizzazione o mapping dell'intero modello.
Prepara fixture/slice bounded solo nel build del nuovo worktree, con manifest
di provenienza e hash; conserva cap e tolleranze della spec e di integration.md.
Nessuno skip vale come PASS e nessuna build vale come parità numerica.
GPU soltanto col lock tools/qwen38_gpu_lock.py del git-common-dir; non annidare
il wrapper attorno ai target GPU integrati, che lo acquisiscono già.
Consegna SHA completi, file, comandi ripetibili, conteggi/metriche, budget,
limiti e dipendenze residue. Nessun merge/push su main.
```

## C — HC/GDN composto

```text
SPEC_ID=C, nuovo branch impl/qwen38-c-v2 e worktree ../picoforge-c-v2.
Parti da C4: comporre HC e un layer GDN CPU usando A per config/ruoli e B
per proiezioni F16/F32/Q8_0. Riusa le primitive C integrate, senza un nuovo
matmul/decoder. Prima fixture piccole non degeneri, poi slice bounded:
controlli conv/gain/stato, un mixer HC e un layer GDN (C §slice/budget).
Mantieni header privato qwen38_c_internal.h e file C già assegnati.
Controlla ogni proiezione contro FP64 sul medesimo input, ogni stadio contro
upstream indipendente e l'uscita/residuo/stato del layer completo ai gate §6.
Verifica token/chunk e reset/epoch/errori, senza usare il residuo per nascondere
errori del contributo. Head tiled h%Hk, stato [Hv,Dv,Dk], A già trasformata,
normalizzazione Q/K e gate sigmoid restano invariati.
Poi prova la composizione Metal privata usando B encode e i kernel C comuni,
senza commit/wait per operatore e senza inventare un ABI pubblico di stato.
Conserva il test alias no-copy corretto in integrazione. Se il lifecycle GPU
pubblico richiede il coordinatore, consegna prima C4 CPU completo e verificato.
```

## D — prep pubblica e QSA completo su input proiettati

```text
SPEC_ID=D, nuovo branch impl/qwen38-d-v2 e worktree ../picoforge-d-v2.
Implementa qwen38_attention_cpu con RMS e attivazioni reali di C, riusando
il core e il lifecycle D. Input q/gate [T,H,2,D], gain diretto, RoPE parziale
half-split. Aggrega IKraw prima della norma/RoPE alla posizione iniziale
del blocco; completa i blocchi attraverso append non allineati.
Aggiungi fixture con gain non unitario, gate non nullo, causalità/tail,
soglia 2051, tie esatti e near-tie con margine registrato. Confronta tutti
gli intermedi e prefill/decode/chunk/reset contro l'oracle indipendente.
Successivamente usa A+B per slice/proiezioni bounded di un layer QSA;
nessun RMS/sigmoid/packed decoder duplicato. Mantieni cache/scratch FP32,
budget esplicito e pubblicazione della lunghezza soltanto a finish.
Dopo i gate CPU avanza con Metal conservativo nei file assegnati, harness
privato e lock comune. Consegna i comandi di link aggiuntivi al coordinatore;
non toccare Makefile o header pubblici per sbloccare il solo incremento GPU.
```

## E — store/job e PLE completo

```text
SPEC_ID=E, nuovo branch impl/qwen38-e-v2 e worktree ../picoforge-e-v2.
Implementa le API pubbliche store/state/job del contratto sopra il reader
bounded esistente: fd duplicato, config copiata, lease/lifetime/BUSY,
prepare/read/rows/commit/abort/release e T=0 senza I/O/avanzamento.
Usa C conv_state_copy per history candidata: niente memcpy di handle opachi.
Poi qwen38_ple_cpu e workspace: B Q8_0 per key/value, RMS C e gate E,
conv dilatata C con storia raw oldest-first (9 token), output Delta.
Il grafo aggiunge R+Delta: non sommare R due volte e non committare history
prima del successo. EOS hash resta distinto dal token di fine chat.
Prima sintetico e fault injection (EINTR/EOF/file mutato/OOM/stale/abort),
poi righe BF16 e proiezioni reali bounded tramite config A. Non mappare la
tabella PLE né usare storage GPU per essa. Confronta ID, dot/segno/gate,
z/Delta e history, oltre a token/chunk/reset, ai gate congelati.
Solo dopo CPU completo avanza con staging/kernel Metal privati; il protocollo
GPU pubblico e il commit globale restano incrementi del coordinatore.
```

## F — MoE floating point e packed

```text
SPEC_ID=F, nuovo branch impl/qwen38-f-v2 e worktree ../picoforge-f-v2.
Implementa create/destroy, work_size/create/reset/destroy e moe_forward,
collegando direttamente B scalar e C SiLU/sigmoid. Config copiata, pesi
borrowed, workspace con budget e nessuna allocazione per token.
Riusa route/grouping F: softmax FP32 su tutti gli esperti, top-k sulle
probabilità, tie ID minore e norm_topk_prob=true. Prova routed e shared
separati, shared gate, permutazione inversa, accumulo pesato e nessun residuo.
Dopo parità floating point passa IQ2_XXS/Q2_K/Q8_0 e slice bounded A+B.
Down legge X con Klogico640, pesi Kfisico768: niente padding obbligatorio
né lettura di 768 float da un input di 640. Verifica gruppi vuoti, T=0,
stride/alias/errori/nonfinite, chunk, route stabili e near-tie documentati.
Proiezioni al bound B e output/intermedi ai gate F e integration.md §6.
Dopo CPU packed verificato prova Metal privato usando B encode e i kernel C;
nessun decoder duplicato, commit/wait per esperto o modifica Makefile/header.
```

## Comandi disponibili nella base

Da ciascun nuovo worktree, Python esistente e percorsi espliciti:

```sh
Q38_PY=/path/to/picoforge/tools/venv/bin/python
make qwen38-test-assets PYTHON="$Q38_PY" \
  Q38_REFERENCE_ROOT=/path/to/picoforge \
  Q38_DWARFSTAR=/path/to/ds4-qwen38-baseline \
  Q38_SOURCE_DIR=/path/to/picoforge-qwen38-integrate/build/qwen38-f/upstream
make test-qwen38-cpu PYTHON="$Q38_PY"
make test-qwen38-gpu PYTHON="$Q38_PY"
```

La preparazione verifica/copia tre piccoli file upstream già presenti e
verifica/crea slice B; senza `Q38_SOURCE_DIR` scarica esplicitamente quei
sorgenti fissati, senza installare pacchetti. I test sono offline.
Target isolati: `test-qwen38-a`, `test-qwen38-{b,c,d,e,f}-cpu`,
`test-qwen38-{a,b,c}-gpu`, `test-qwen38-links`, `test-qwen38-contracts`.
Le regole dei harness v1 linkano le dipendenze v1: se una nuova composizione
richiede più `.c`, compila un harness separato nel build del tuo worktree e
consegna al coordinatore il comando esatto; non modificare la build condivisa.
