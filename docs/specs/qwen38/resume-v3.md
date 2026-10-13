# Incrementi dopo BASE_INTEGRATA_V2

Usare lo SHA completo della consegna del coordinatore al posto di
`<BASE_INTEGRATA_V2>`. Le consegne v2 sono già incluse: nessun cherry-pick
duplicato. `resume.md` rimane lo storico del giro precedente.

## Regole comuni per ogni incarico

```text
Repository /Users/cristiancirje/Desktop/Dev/picoforge.
MODALITA=ESEGUI, BASE_INTEGRATA_V2=<BASE_INTEGRATA_V2>.
Leggi istruzioni applicabili, docs/qwen38-q2-session-prompt.md,
docs/specs/qwen38/integration.md incluso §12, spec e consegna del componente,
docs/specs/qwen38/resume-v3.md e report integrato v2.
Crea branch/worktree nuovi dalla base indicata, preservando quelli esistenti.
Mantieni Qwen3.8-Flash-Next Q2, FP32 e tolleranze congelate. Niente dipendenze
nuove, conversione completa o checksum integrale del checkpoint. Riusa gli
asset esistenti in sola lettura; fixture bounded nel build del tuo worktree.
Usa solo i file assegnati. Header pubblici, build, lock e grafo sono del
coordinatore; proponi gli eventuali contratti condivisi prima di implementarli.
Lavora per incrementi committati e verificati, non fermarti alla spec.
Ogni GPU/workload pesante usa il lock comune; i target GPU lo acquisiscono già.
Nessuno skip vale come parità. Riporta intermedi, contributi, stato e output,
non soltanto il residuo. Consegna SHA, comandi, metriche e limiti precisi.
Nessun merge/push su main e nessuna modifica negli altri worktree.
```

## G0 — coordinatore, lifecycle GPU comune (prima integrazione)

**Stato:** consegnato come BASE_GPU_CONTRATTI, vedi integration.md §13.

Branch `coord/qwen38-gpu-lifecycle`, worktree `../picoforge-gpu-lifecycle`.
Ownership: `src/qwen38_gpu.h`, nuovi `src/qwen38_gpu.m`,
`tests/test_qwen38_gpu_lifecycle.m`, target/coordinamento e integration.md.
Gli header pubblici aggiuntivi restano del coordinatore.

Congelare un contratto concreto di sessione/transazione sopra
`Qwen38GpuCommand`: device, buffer trattenuti, scratch disgiunto, key/epoch,
start/count, stato IDLE/ENCODING/IN_FLIGHT/VALID/INVALID e ammissione T=0.
Il contesto possiede queue/command buffer; i componenti codificano soltanto.
Begin valida ownership/capacità e azzera una sola error word; submit chiude
l'encoder; completion legge errore Metal e bit numerici prima di pubblicare
posizione e ticket C/D/E. Errore dopo scritture invalida l'intera sessione:
nessun rollback implicito e nessuna history considerata valida parzialmente.
Reset richiede completion e nuovo epoch, quindi reset/replay di tutti gli stati.

Impedire riuso dello scratch e delle mappe Shared di F prima di completion,
destroy/reset mentre busy e alias con pesi/error word/stato, anche no-copy.
Testare fail-before-submit, errore kernel dopo scritture, lifetime dei buffer,
stale epoch, owner estraneo, doppio submit/finish e mancata pubblicazione.
Consegnare prima contratto + harness verificato come `BASE_GPU_CONTRATTI`.
Non chiedere a C/D/E/F di inventare quattro lifecycle incompatibili.

## C — adattatore HC/GDN al lifecycle

```text
SPEC_ID=C, branch impl/qwen38-c-v3, worktree ../picoforge-c-v3.
Conserva ownership C già assegnata, inclusi private header e harness.
Prima di BASE_GPU_CONTRATTI puoi ampliare i test privati di ammissione,
stato/epoch e chunk, usando la composizione C4 già verificata CPU/Metal.
Dopo acquisizione del contratto comune, collega conv/GDN, history e scratch
al ticket globale: nessun commit/wait per operatore o pubblicazione anticipata.
Verifica stesso comando, più chunk, reset/replay e fault dopo conv ma prima
di GDN/HC-add. Riusa i 18 intermedi e slice layer 0; non cambiare formule.
Consegna test numerici e di stato sul contratto comune, non una nuova API
pubblica privata di C. La transazione globale rimane del coordinatore.
```

## E — PLE Metal (può iniziare subito in privato)

```text
SPEC_ID=E, branch impl/qwen38-e-v3, worktree ../picoforge-e-v3.
Ownership E esistente più src/qwen38_ple.m, src/qwen38_ple.metal e
tests/test_qwen38_ngram.m; header privato qwen38_ngram_internal.h.
Implementa staging bounded delle sole righe lette dal reader BF16, B encode
Q8_0 key/value, RMS C, gate PLE, conv C dilatata e output Delta FP32.
La tabella resta disk-only; niente mapping/store GPU della tabella intera.
Niente decoder/matmul/conv duplicati. R+Delta appartiene al grafo, una volta.
Verifica gate/dot/segno, u/z/conv/Delta e history rispetto a CPU/oracle, prima
sintetico poi fixture reale v2, token/chunk/reset, EOS hash e fault I/O/GPU.
Nel harness privato il caller possiede submission/completion; conserva job,
lease e staging fino a completamento. Dopo BASE_GPU_CONTRATTI adatta il job
al ticket comune: commit solo dopo successo globale, abort senza history
pubblicata; errore dopo mutazione invalida sessione. Nessun ABI pubblico nuovo.
Consegna al coordinatore dipendenze e kernel da collegare nella metallib.
```

## D — selezione QSA GPU

```text
SPEC_ID=D, branch impl/qwen38-d-v3, worktree ../picoforge-d-v3.
Ownership D esistente. Parti dagli score GPU già disponibili: seleziona top-k
blocchi su device, tie ID minore, ordina gli ID finali temporalmente e aggiungi
il tail causale. Usa scratch bounded, non una matrice dense T×T.
Confronta ID bit-esatti a parità di score, tie, near-tie con margini e soglia
2051; verifica output rispetto all'oracle indipendente, incluso N=32768.
Componi B proiezioni -> prep C/D -> pool/score/selezione/attention -> B output,
senza readback per token per decidere gli ID. Le frequenze RoPE precomputate
al setup rimangono lecite; non reintrodurre pow Metal con diversa precisione.
Dopo BASE_GPU_CONTRATTI lega cache/ticket e pubblicazione next_pos al lifecycle
comune. Mantieni la possibilità di testare selezione privata indipendentemente.
```

## B/F — routing e prodotti esperti su GPU

B, branch `impl/qwen38-b-v3`, worktree `../picoforge-b-v3`: ownership B
esistente. Proporre al coordinatore l'estensione packed per ID esperto per
riga GPU (`expert_ids[P]`), validare range e layout, K logico640/fisico768,
offset/stride/alias/error word e limite workspace. Poi implementare nei file
B e confrontare ogni prodotto sullo stesso input al bound B. Nessun nuovo
decoder in F; l'header pubblico si modifica soltanto nel contratto coordinato.

```text
SPEC_ID=F, branch impl/qwen38-f-v3, worktree ../picoforge-f-v3.
Ownership F esistente. Implementa router F32 con B, softmax FP32 su tutti
i 512 esperti, top-k sulle probabilità con tie ID minore e rinormalizzazione,
grouping/permutazioni su GPU. Prima oracle su logits/probabilità identici,
tie/near-tie con margini, T=0, NaN/Inf, gruppi vuoti e chunk <=32.
Riusa il post-routing Metal v2; evita lettura CPU degli ID e submission/wait
per esperto. Per il prodotto indicizzato dipendi dall'incremento B concordato;
nel frattempo completa routing/grouping privati e test su route congelate.
Dopo BASE_GPU_CONTRATTI proteggi scratch/mappe fino a completion. Verifica
routed/shared separati e Y, intermedi locali e bound B; poi slice reali
0/3/47. Le prove v2 su questi layer sono CPU, non ancora GPU end-to-end.
```

## G1–G3 — coordinatore, grafo ordinario e logits

Ownership: nuovi `src/qwen38_forward.c`, `src/qwen38_forward.m`,
`tests/test_qwen38_forward.c/.m/.py`, `tools/oracle/qwen38_forward.py`;
gli eventuali header pubblici, main/CLI, Makefile e metallib sempre coordinatore.

1. **G1 CPU indipendente:** ordinare embedding → espansione HC → 48 layer
   dal catalogo A (36 GDN/12 QSA), PLE al layer 1, HC-attn e HC-FFN con MoE,
   mixer finale/norma/LM head non tied. Prima grafo piccolo non degenere e
   slice di layer singoli, senza hardcode al posto del catalogo e senza MTP.
   Tracciare embedding, input/output HC, contributi GDN/QSA, PLE Delta,
   routed/shared, residui e stato; poi hidden finale e logits FP32.
2. **G2 GPU dopo G0 + adattatori C/D/E/F:** collegare le stesse operazioni
   al lifecycle; confronto CPU/Metal e oracle indipendente a ogni confine.
   Confrontare proiezioni sullo stesso input e contributi separati. Prefill,
   decode, chunk e reset devono produrre stato/output coerenti ai gate §6.
   Un disaccordo del token scelto richiede margine/logits, non un allargamento
   automatico della tolleranza. Mantenere trace bounded, non `[1024,V]`.
3. **G3 checkpoint completo:** solo dopo parità intermedia e budget composto,
   prova ordinaria Q2 rispetto a DwarfStar fissato: primi prompt deterministici,
   logits dell'ultimo token, top-k/argmax e poi generazione. Pubblicare differenze
   numeriche e qualità separatamente dalla perdita Q2/BF16. Obiettivo 8K/chunk1024
   e 48–50 GiB da misurare (footprint, Metal, pressione/swap), non dedurre dalla
   somma dei cap dei harness. Nessuna ottimizzazione, MTP o benchmark prima dei gate.

Ordine di integrazione: **G0 → C/E adattatori + D selezione + B indicizzato/F
routing → G2 → G3**. E Metal privato, D selezione, F routing e G1 CPU possono
avanzare indipendentemente sulla v2. Ogni passaggio condiviso produce una base
con SHA esplicito e test combinati prima che le sessioni consumino la nuova ABI.
