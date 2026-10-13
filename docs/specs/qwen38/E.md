# E — N-gram: indirizzamento, letture BF16 e composizione PLE

Stato: **SPEC, proposta per il consolidamento**, 2026-10-12.
`BASE_CONCORDATA`: da determinare; nessuna implementazione autorizzata qui.
Base ispezionata: picoforge `7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`.
Branch di questa spec: `spec/qwen38-e`; worktree esterno
`/Users/cristiancirje/Desktop/Dev/picoforge-spec-e`.

Le formule contrassegnate **verificate** sono ricavate dalle fonti sotto;
API, budget e tolleranze sono **proposte**, non risultati numerici acquisiti.
I blocchi mancanti non sono sostituiti da stub per dichiarare parità.

## 1. Obiettivo e confini

Implementare in seguito il Per-Layer Embedding (PLE) del checkpoint Q2:
ID esatti dalla storia dei token, lettura selettiva della tabella originale
BF16, concatenazione delle righe, proiezioni key/value, gate per ramo,
normalizzazione, convoluzione depthwise causale dilatata e somma al residuo.
Il percorso iniziale è testo, una sessione, generazione ordinaria, prefill
massimo 1024 token per chunk e decode di un token, contesto iniziale 8192.

E possiede indirizzamento, I/O, staging, gate specifico PLE e orchestrazione
del blocco. **C possiede la primitiva comune di convoluzione**, le
normalizzazioni e le primitive elementari concordate; **B possiede i formati
e le proiezioni packed**, incluse Q8_0 necessarie qui. A risolve configurazione,
token e binding; il coordinatore inserisce il PLE nel grafo. Nessuna tabella
completa in Metal, mmap della tabella, warmup delle sue pagine, copia completa
o espansione del modello. BF16 n-gram non significa INT2, Q2_K o IQ2_XXS.

Esclusi da questo incremento: multimodalità, batch di sessioni ragged o con
padding, MTP, rollback arbitrario, prefix cache persistente e modifiche alla
ricetta Q2. Un EOS presente tra token validi è supportato; non è un reset
della sessione. Dimensioni derivate dai metadati, con rifiuto esplicito di
geometrie non supportate, senza costanti di checkpoint nel codice operativo.
Qwen3, safetensors, q4_row e q8_row restano regressioni separate.

## 2. Stato corrente, provenienza e codice riutilizzabile

### Fonti effettivamente ispezionate

| Fonte | Revisione e punti pertinenti |
|---|---|
| picoforge | Base sopra; `src/gguf.h`, `gguf.c`, `gguf_reader.c`, `gguf_plan.c`, `gpu_weights.m`, `ops.c`; inventario e verifica checkpoint nella baseline |
| DwarfStar | `fc80bd695da76ee14bcb86820bf900a5cdfda806`, checkout locale `../ds4-qwen38-baseline`, pulito; remoto `main` ancora a questa revisione nella verifica del 12 ottobre |
| [DwarfStar ds4.c][ds4] | `qwen4_ple_step` circa L68121; `qwen4_ref_ple` L68147; `qwen4_ngram_row/read` L58107–58242; `qwen4_graph_forward_tokens` L59479 e chiamata PLE prima di HC-attention; validazione metadati L7100–7139 |
| [Kernel PLE DwarfStar][metal] | `kernel_qwen4_ple_gate` L971–1043; `kernel_qwen4_ple_conv` L1059–1109 |
| [Conversione pack][pack] e [conversione GGUF][convert] | `NORM_FOLD_SUFFIXES`, `needs_norm_fold`, `encode_action(kind="norm_fold")`; `prod_bf16_f32`; binding PLE a `blk.1`, L523–529 del convertitore |
| [Tabelle native BF16][native] | `source_layout`, confronto buffer hash originali e concatenazione shard senza requantizzare le righe |
| [Configurazione Qwen originale][config] | `de4b8e4d43b917e7706784d8bb445c9af86a3540`; config letto dalla fonte, non dedotto dal nome del modello |
| [Transformers][hf] | `db2304e8694db3ecd6f39d353beba361688afe97`, `modeling_qwen4_exp.py`: RMSNorm L143–163, NGramEmbedding L1071–1171, PLELayer L1174–1246, DecoderLayer L1249–1279 |
| [Report Qwen §2.3][paper] | Contesto architetturale; per dettagli eseguibili prevalgono config, pesi e codice revisionato |

SHA-256 del file Transformers letto:
`f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa`.
Questa revisione del codice è distinta da quella dei pesi; non si assume che
il `transformers_version` del config identifichi un commit. L'ambiente esistente
`/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv` ha Torch `2.13.0` e
non contiene il modulo installato `transformers.models.qwen4_exp`.
Non è stato aggiornato né installato alcun pacchetto.

Letti integralmente `qwen38-q2-session-prompt.md`, `qwen38-q2-baseline.md`,
`qwen38-storage.md`, README, Makefile, DESIGN e BENCHMARKS; consultati ricerca
large-model, project-review, esperimento 30B e confronto 80–100B. Letto anche
`CLAUDE.md` ignorato nel checkout principale. La nuova direzione e la modalità
SPEC prevalgono sulle vecchie tappe. Nessun `AGENTS.md` aggiuntivo trovato
negli antenati o nei percorsi pertinenti; valgono le istruzioni ricevute.

Artifact: `antirez/qwen3.8-flash-next-gguf`, revisione
`d600fe1a43d2e1cdcadb85144ce3142f66f9eefe`, file
`models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf` nel checkout principale.
SHA-256 registrato dalla verifica completa precedente:
`b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
Questa sessione ha verificato solo `stat`: 147207127040 byte, inode 30631842,
mtime_ns 1791785892435960350, uguali al report; **non ha ricalcolato l'hash**.

### Riutilizzo e limiti

- `Gguf` possiede fd e mapping del solo header, massimo 128 MiB; `GgufTensor`
  espone shape fisica, tipo, offset assoluto e byte. Il PLE non esiste ancora.
- `GgufPlan.disk_table` è una copia del descrittore BF16; il planner esclude
  `per_layer_token_embd.weight` dalle viste. `gpu_weights_find` non restituisce
  un binding per questa tabella. Conservare e testare tale invariante.
- `GpuWeightStore` e `GgufBinding` possono fornire i piccoli pesi PLE residenti;
  un binding conserva un owner e non deve sopravvivere al plan. Non copiare
  plan per valore né memorizzare puntatori a metadata dopo `gguf_close`.
- `bf16_to_f32` e convenzioni row-major già presenti sono riutilizzabili dopo
  test di endianness e bit. `matmul_q` non decodifica Q8_0 GGUF; non riusarlo
  come se `q8_row` fosse equivalente. Le helper GGUF che chiamano `die()` non
  costituiscono un contratto recuperabile per errori I/O della sessione.
- DwarfStar ordina e deduplica richieste in partizioni di 4096 righe, ma la
  deduplicazione attraversa solo ciascuna partizione/thread e legge una riga
  per `pread`. Non è già l'accorpamento di intervalli proposto qui.
- I test upstream `test_qwen4_ngrams.c` e `test_qwen4_ngram_state.c` sono utili
  per casi limite e state failure, non bastano come oracle indipendente se
  il nostro algoritmo è tradotto dallo stesso codice.

## 3. Semantica matematica e layout verificati

### 3.1 Parametri e binding

Notazione: `D` hidden size, `S` rami HC, `N` ordine n-gram massimo,
`G` teste per ordine, `H=(N-1)G`, `d` dimensione riga, `P=H*d`, `C=S*D`,
`K` kernel convoluzione, `delta=N`, `L=(K-1)*delta` righe di storia.

| Parametro | Fonte | Valore di questo artifact |
|---|---|---:|
| D, S | `qwen4exp.embedding_length`, `qwen4exp.hyper_connection.count` | 2560, 4 |
| Layer PLE | `qwen4exp.ple.layers` | `[1]`, indice zero-based |
| N, G, K | `qwen4exp.ple.ngram_size`, `.heads_per_ngram`, `.conv_kernel` | 3, 8, 4 |
| H, d, P, C, L | derivati; `embedding_length_per_layer_input` e `.ple.row_dimension` devono coincidere | 16, 160, 2560, 10240, 9 |
| EOS hash | `qwen4exp.ple.eos_token_id` | 248044 |
| Vocab token | configurazione A | 248320 |
| Epsilon RMS | `qwen4exp.attention.layer_norm_rms_epsilon` | F32 ≈ 1e-6 |
| Righe fisiche | `qwen4exp.ple.row_count` | 320001536 |

Il config HF `ple_layer_ids=[2]` è one-based: Transformers verifica
`layer_idx+1`; coincide con GGUF `blk.1`. Il PLE precede il mix HC
dell'attention nel secondo decoder layer, non segue la MoE e non si applica
a tutti i layer. L'EOS per hash non va sostituito con lo stop del chat template.

| Tensore | Forma GGUF (dimensione 0 contigua) | Tipo | Interpretazione logica |
|---|---|---|---|
| `per_layer_token_embd.weight` | `[160,320001536]` | BF16, id 30 | tabella `[rows,d]`, 320 byte/riga |
| `blk.1.ple_key.weight` | `[2560,10240]` | Q8_0, id 8 | Wk `[C,P]` |
| `blk.1.ple_value.weight` | `[2560,2560]` | Q8_0, id 8 | Wv `[D,P]` |
| `blk.1.ple_norm_key.weight` | `[10240]` | F32 | gamma_k `[S,D]` |
| `blk.1.ple_norm_query.weight` | `[10240]` | F32 | gamma_q `[S,D]` |
| `blk.1.ple_norm_conv.weight` | `[10240]` | F32 | gamma_c `[S,D]` |
| `blk.1.ple_conv1d.weight` | `[4,10240]` | F32 | Wc `[C,K]`, tap più vecchio per primo |

La tabella inizia a byte **44806635520**, dimensione **102400491520**.
Offset di una riga `r`: `base + r*(2*d)`; tutti i calcoli controllati prima
della conversione a `off_t`. Le proiezioni hanno righe fisiche Q8_0 di
`(P/32)*34 = 2720` byte, non `P` byte. Non c'è padding 640/768 nel PLE;
quello appartiene ai down degli esperti di B/F.

### 3.2 ID: storia, EOS, hash, ordine delle teste

Sia `x_t` un token valido. Per `j>0`, `c_j(t)=x_(t-j)` solo se la posizione
esiste e non è a/prima dell'ultimo EOS **strettamente precedente a t**;
altrimenti `c_j(t)=eos`. `c_0(t)=x_t`, anche quando `x_t=eos`.
Quindi l'EOS corrente conserva il contesto precedente per il proprio hash;
dal token successivo quel contesto è tagliato. All'avvio i `N-1` token
precedenti sono EOS. Il numero di messaggi chat non entra nella formula.

Per ordine `n=2..N` e testa `g=0..G-1`:

```text
h = (n-2)*G + g
z_n(t) = XOR_{j=0..n-1} (uint64(c_j(t)) * multiplier[j])
row[t,h] = head_offset[h] + (z_n(t) mod head_vocab[h])
embedding[t,h*d+i] = BF16_to_F32(table[row[t,h],i])
```

Il prodotto è prima dell'XOR, il modulo è dopo l'intero XOR. Non si usano
hash della stringa, BPE intermedi, RoPE, somma dei prodotti o un rolling hash
approssimato. Le righe concatenate sono prima tutte le bigram, poi trigram,
con indice testa crescente; il sort I/O deve ripristinare questo ordine.
Storia CPU proposta newest-first: `prev[j-1]=x_(t-j)` prima del taglio EOS.
Avanzamento una volta per token consumato dal modello, non per layer o lettura.

Moltiplicatori reali: `[23703573157769,20109073645365,8052911324071]`.
Leggerli dai metadati, senza rigenerarli da seed 1234 o ricercare primi.
Offset e vocabolari di testa sono i vettori `.ple.head_offsets` e
`.ple.head_vocab_sizes` dell'inventario. Somma vocabolari **320001446**;
le **90** righe finali fino a 320001536 sono padding e non sono raggiunte
dall'hash. Non applicare il modulo sulla dimensione totale padded.

Transformers calcola su `int64`; nel checkpoint il prodotto massimo è
5886047582964040311 < INT64_MAX, quindi tutti i termini e il loro XOR
restano nonnegativi e concordano con `uint64` DwarfStar. **Proposta**: A/E
validano `multiplier[j]*(vocab_size-1) <= INT64_MAX`; configurazioni fuori
dominio sono rifiutate, non interpretate con modulo unsigned dopo overflow
che potrebbe differire da `torch.remainder` signed.

### 3.3 Lookup e precisione

Ogni parola BF16 little-endian `b` si espande con bit FP32 `uint32(b)<<16`,
tramite `memcpy`, senza aliasing UB o conversione numerica degli interi.
Zero con segno, subnormali e valori finiti restano esatti. NaN/Inf nelle
righe richieste sono errori di dati; il payload non viene pubblicato.
La verifica strutturale GGUF non garantisce finitezza né integrità del payload.

### 3.4 Gate, RMSNorm e composizione

Per ciascun token, input residuo `R_t[S,D]` dal layer precedente e lookup
`e_t[P]`. Definire, separatamente per ogni ramo:

```text
RMS(x,gamma) = gamma * x / sqrt(mean_D(x*x) + eps)
k_t = Wk e_t                       # [S,D], nessun bias
v_t = Wv e_t                       # [D], condiviso tra i rami
kn_t,s = RMS(k_t,s, gamma_k,s)
q_t,s  = RMS(R_t,s, gamma_q,s)
a_t,s  = dot(kn_t,s, q_t,s) / sqrt(D)
b_t,s  = sign(a_t,s) * sqrt(max(abs(a_t,s), 1e-6))
g_t,s  = sigmoid(b_t,s)
u_t,s  = g_t,s * v_t
z_t,s  = RMS(u_t,s, gamma_c,s)
w_t,c  = sum_{k=0..K-1} Wc[c,k] * z_(t-(K-1-k)*delta),c
Delta_t = u_t + SiLU(w_t)
R_out,t = R_t + Delta_t
```

`sign(0)=0`, dunque gate esattamente 0.5 per dot nullo. Il clamp 1e-6 della
radice firmata è una costante dell'operatore upstream, distinta dall'epsilon
RMS del config. Nessuna softmax tra rami, nessuna divisione per S, nessun
gate HC/injection aggiuntivo nella somma PLE. SiLU solo dopo la convoluzione;
la storia conserva `z`, non `u`, `w` o `SiLU(w)`.

**Norme già trasformate:** HF usa `1+weight` (zero-centered RMSNorm).
La catena di conversione DwarfStar applica `norm_fold`, arrotonda a BF16,
poi espande a F32 nel GGUF. `gamma` sopra è direttamente il F32 del checkpoint:
NON aggiungere 1 e NON recuperare il peso originale BF16 per il confronto Q2.
L'oracle che usa il modulo HF deve iniettare una normalizzazione con gamma
diretta oppure adattare `weight=gamma-1` verificando l'esatta ricostruzione
in F32. La politica va condivisa con A/C, non affidata a una deduzione dal dtype.

### 3.5 Convoluzione, chunk e reset

Storia logica FP32 `[L,C]` oldest-first, inizialmente zero. Il tap `K-1`
legge il token corrente; tap 0 legge `t-9` per questo modello; altri tap
leggono `t-6,t-3`. Il buffer deve comunque conservare gli ultimi **9 token
consecutivi**, non solo tre campioni distanti. Ring buffer ammesso, con
cursore e ordine logico espliciti; snapshot/test normalizzano oldest-first.

Prefill, decode e append valutano la stessa equazione. Un chunk usa la coda
del precedente, aggiorna storia e contatore solo sui token effettivamente
consumati; non aggiunge zeri ai suoi confini. Per K=1, L=0: niente accesso
a storia negativa. Il cambio di chunk non è reset. Un EOS modifica solo
l'indirizzamento futuro; **non azzera lo stato convoluzionale**. `reset`
esplicito ripristina EOS nella storia token, zero nella convoluzione e
posizione zero; invalida i job della vecchia generazione.

Proposta precisione iniziale: attivazioni, stato, prodotti e riduzioni FP32
in C/Metal, senza conversioni BF16 intermedie implicite. Oracle analitico
in FP64 più riferimento upstream FP32; l'esecuzione HF BF16 è un confronto
distinto, non la tolleranza di questo percorso. Il test di equivalenza per
chunk confronta anche l'intera storia, non soltanto l'ultimo output.

## 4. API proposte, ownership e gestione errori

Nomi da ratificare in `integration.md`; non sono dichiarazioni già esistenti.
Header pubblico comune materializzato solo dal coordinatore. Tipi opachi E
evitano di esportare layout interni e puntatori a buffer riciclabili.

```c
typedef struct Qwen38NgramStore Qwen38NgramStore;
typedef struct Qwen38NgramState Qwen38NgramState;
typedef struct Qwen38NgramJob Qwen38NgramJob;
typedef struct Qwen38PleWorkspace Qwen38PleWorkspace;
typedef struct Qwen38PleGpu Qwen38PleGpu;

typedef enum {
    Q38_NGRAM_OK, Q38_NGRAM_INVALID, Q38_NGRAM_OVERFLOW,
    Q38_NGRAM_NOMEM, Q38_NGRAM_BUSY, Q38_NGRAM_CANCELLED,
    Q38_NGRAM_IO, Q38_NGRAM_EOF, Q38_NGRAM_NONFINITE,
    Q38_NGRAM_STALE, Q38_NGRAM_GPU
} Qwen38NgramStatus;

typedef struct {
    Qwen38NgramStatus code;
    int os_errno;                 /* saved immediately; zero if not OS error */
    uint64_t file_offset, row;
    size_t requested_bytes, completed_bytes;
} Qwen38NgramError;

/* Qwen38NgramConfig and Qwen38PleWeights are shared contracts from A. */
Qwen38NgramStatus qwen38_ngram_open(const Gguf *gguf,
    const Qwen38NgramConfig *config, size_t max_tokens,
    size_t staging_budget, Qwen38NgramStore **out, Qwen38NgramError *err);
Qwen38NgramStatus qwen38_ngram_prepare(Qwen38NgramStore *store,
    const Qwen38NgramState *state, const int32_t *tokens, size_t count,
    uint64_t position, Qwen38NgramJob **out, Qwen38NgramError *err);
Qwen38NgramStatus qwen38_ngram_read(Qwen38NgramJob *job,
    Qwen38NgramError *err);
Qwen38NgramStatus qwen38_ngram_rows(const Qwen38NgramJob *job,
    const float **embedding, size_t *tokens, size_t *row_stride_floats);
Qwen38NgramStatus qwen38_ple_cpu(Qwen38NgramJob *job,
    const Qwen38PleWeights *weights, const float *residual,
    size_t residual_stride_floats, float *delta, size_t delta_stride_floats,
    Qwen38PleWorkspace *workspace, Qwen38NgramError *err);
Qwen38NgramStatus qwen38_ngram_commit(Qwen38NgramState *state,
    Qwen38NgramJob *job, Qwen38NgramError *err);
void qwen38_ngram_abort(Qwen38NgramJob *job);
```

Lifecycle richiesto oltre alle firme illustrate: `state_create/destroy`,
`workspace_create/destroy`, `state_reset`, `job_release`, `store_close`.
Le allocazioni falliscono senza oggetti parziali pubblicati; `*out=NULL` su
errore. Close/destroy restituiscono BUSY se ci sono job/dispatch in volo,
anziché liberare memoria ancora usata. `abort` invalida il risultato ma non
libera risorse in uso; `release` le ricicla solo dopo completamento.

### Config e span

`Qwen38NgramConfig` contiene vocab/eos, indice layer, N/G/d/P/D/S/K,
epsilon RMS e vettori multipliers/offsets/vocab_sizes. Lo store ne possiede
una copia immutabile; non prende in prestito byte dell'header. Per ora
un solo layer PLE; altri layout restituiscono unsupported tramite validazione
A prima di aprire lo store. I metadati devono rispettare lunghezze, positività,
prodotti sicuri e intervalli delle teste entro le righe fisiche; per il profilo
selezionato verificare offset contigui e padding dichiarato. Non hardcodare
un limite di riga 160 come sostituto di un budget in byte.

Input token C contigui `[T]`; rifiuto token negativi, >=vocab, count fuori
budget e posizione diversa dallo stato committato. T=0 è no-op senza I/O né
avanzamento, con puntatori null ammessi solo per span vuoti. Float contigui
sulla feature, stride per token in **elementi float**, almeno P o C; span
totali e overflow verificati. Output `delta[T,C]` separato dal residuo,
nessuna sovrapposizione tra output, input e stato. La somma `R+Delta`
appartiene al grafo e segue il contratto di C, una sola volta.

`Qwen38PleWeights`: binding tipizzati di A per Wk/Wv/gamma/Wc e owner della
vista; CPU accede a span residenti in sola lettura senza seconda copia,
Metal usa `buffer_id+offset` del medesimo store. Necessario un contratto CPU
di vista bounded da concordare col coordinatore: `Gguf` oggi non mappa i
payload. Non ricreare in E un loader generale. CPU può usare slice fixture
fornite da B prima di avere il binding integrato.

### FD, job e transazioni

Lo store copia `disk_table`, duplica con close-on-exec il fd read-only del
`Gguf`, registra dev/inode/size/mtime e configura budget; il reader originale
può poi chiudere. Il descrittore e il file devono restare immutabili fino
all'ultimo job. Controllo `fstat` all'apertura e prima dei batch I/O; un
cambiamento invalida store/cache. Non è un checksum né protegge da scritture
esterne che preservano tutti i metadati: immutabilità resta una precondizione.

Job: `FREE -> PREPARED -> READING -> READY -> COMPUTED -> COMMITTED`;
per GPU `READY -> IN_FLIGHT -> COMPUTED` solo dopo completion riuscita.
Errore/cancel porta ad `INVALID`, mai a READY. Token copiati o ID già
materializzati nel job; l'input caller serve solo durante `prepare`.
Ogni job porta owner store/state, epoch sessione, generazione slot,
posizione iniziale, T, storia token candidata e storia conv candidata.
I buffer restituiti da `rows` sono borrowed fino a release, non memorizzabili
oltre; accessor ed encoder verificano READY e generazione.

Prepare/read non avanzano lo stato della sessione. Il calcolo aggiorna una
copia della storia conv; commit pubblica **insieme** storia token, storia
conv e posizione, verificando owner/epoch/posizione e completamento.
Non si consente un secondo job dipendente non committato sulla stessa
sessione nella prima versione. L'eventuale lookahead I/O sarà incrementale,
senza consumare anticipatamente lo stato del modello.

Il coordinatore committa E solo quando l'intero chunk del grafo è riuscito.
Se C/D/F hanno già mutato stato e una fase fallisce, la sessione intera va
marcata non continuabile fino a reset/replay; il solo rollback E non basta.
Nessun output/logit del chunk fallito è valido. Snapshot globale e MTP sono
un incremento successivo, non una garanzia implicita di questa API.

### Letture bounded, deduplica e accorpamento

1. Validare tutte le righe e gli offset prima dell'I/O, dentro il tensore,
   il file e `OFF_MAX`; checked multiply/add anche per array e stride.
2. Pianificare al massimo `Tmax*H` richieste `(row,destination)` per job.
   Ordinare per riga, costruire un elenco unico e scatter inverso per tutte
   le occorrenze. Deduplicare globalmente nel job prima di eventuali worker.
3. Accorpare righe **consecutive** in intervalli, spezzati a un limite
   configurato di 64 KiB e su confini di riga. Per d=160: massimo 204 righe,
   65280 byte. Prima versione: niente letture dei buchi; unendo una riga non
   richiesta si introduce I/O amplificato da misurare separatamente.
4. Leggere intervalli nel buffer privato del job, poi fare scatter nelle
   righe contigue `[T,H,d]`. Nessuna allocazione proporzionale al row_count
   o al contesto totale. Prima versione sincrona, un reader C11/POSIX;
   concorrenza bounded solo dopo parità, senza introdurre GCD nel codice C.
5. Espandere BF16 in FP32, validare tutte le righe; pubblicare READY solo
   quando l'intero job è completo. Non esporre righe parzialmente aggiornate.
6. Cache opzionale disabilitata inizialmente; incremento successivo con
   massimo 8 MiB, chiave `(store_identity,table_identity,row)`, contenuto BF16
   immutabile e stato valid. Nessun puntatore GPU verso entry soggetta a
   eviction. Cache insert solo dopo una riga completa e validata; fallimento
   del batch invalida gli inserimenti pendenti. Confrontare cache on/off.

Un medesimo ID ripetuto deve leggere i byte una volta per job, anche se
diversi token/teste li richiedono. La deduplica non fonde attivazioni e non
salta gate o convoluzione per token ripetuti. I contatori distinguono byte
utili, byte richiesti, byte restituiti da pread, chiamate, retry, righe uniche,
cache hit, errori e high-water delle allocazioni; non sono traffico SSD fisico.

### Errori: niente righe stale

Per intervallo, loop `pread(fd,dst+done,len-done,offset+done)`:
`n>0` avanza solo di n; short read è progresso; `n<0 && errno==EINTR`
riprova senza perdere progresso, verificando cancel; `n==0` prima del termine
è EOF/troncamento; altro errore salva errno e invalida l'intero job.
Nessuna ripartenza dalla riga precedente, zero-fill di EOF, successo parziale
o riutilizzo del contenuto precedente di uno slot. Un backend di fault
injection resta privato ai test e usa lo stesso loop produttivo.

La validità non dipende dall'azzerare byte: accesso e submit sono vietati
senza READY. In debug poison dei buffer rilasciati e `valid_rows=0` ad ogni
riassegnazione. Anche un errore nell'ultima riga impedisce il submit di tutte
le righe del job. Errori permanenti I/O/dati invalidano il job e la richiesta;
riapertura/retry esplicito dopo diagnosi, senza fallback silenzioso. EOS non
rende valida una storia fallita. Reset invalida epoch/lease ma non riusa
slot IN_FLIGHT; attende o restituisce BUSY. Cache immutabile del modello può
sopravvivere a reset, ma non cambio file/modello/config.

### Interfaccia Metal

Glue Objective-C limitato a buffer, encoder, dipendenze e completion;
hash/I/O rimangono C11. Proposta `qwen38_ple_gpu_encode(gpu, job, weights,
R_span, delta_span, command_buffer, error)`, con span condiviso dal
coordinatore `{buffer,offset_bytes,length_bytes,row_stride_bytes}`.
Richiede READY, span sufficienti e allineati per FP32, owner coerenti;
nessun puntatore Objective-C nel contratto C del reader.

Encode usa proiezioni di B, gate E, RMS/conv di C e produce Delta senza
toccare R. Le risorse sono trattenute fino a completion, **e il contenuto
degli slot non viene riscritto** fino a quel momento: retainedReferences
da solo non impedisce un write CPU anticipato. Completion fallita invalida
job e richiesta. Store/weight view non chiudibili durante uso. Nella prima
versione usare il percorso command-buffer ordinario già validato dal repo;
Metal 4 unretained ownership richiederebbe un contratto aggiuntivo.

## 5. Dipendenze da consolidare

| Responsabile | Contratto necessario | Avanzamento E possibile prima |
|---|---|---|
| A | Config/vocab, EOS distinto da stop, layer 1, vettori hash, P vs D, binding e norma già folded | Hash/I/O sintetici; oracle su metadati congelati |
| B | Q8_0 decode e GEMV/GEMM `[T,P] * [O,P]^T`, con O=C per key e O=D per value, FP32 accumulo; BF16 widening comune senza nuovo decoder Q8 in E | Reader BF16 e gate su key/value già forniti; non dichiarare PLE completo |
| C | RMS con gamma diretta per gruppi D; conv FP32 depthwise `[T,C]`, W `[C,K]`, dilation=N, zero iniziale, storia L, SiLU e somma | Fixture gate e test del reader; composizione conv attende C |
| D/F | Nessuna primitiva importata direttamente; stato globale coerente dopo append/reset/failure | Test locali indipendenti |
| Coordinatore | Header, build, lifecycle del grafo, vista pesi CPU, commit/poison della sessione, lock GPU | Solo SPEC; nessun test GPU senza protocollo comune |

Richiesta precisa a C: supportare dilation configurabile e `L=(K-1)*delta`,
input e output separati, nessun reset automatico su EOS, restituzione della
storia candidata. Se C offre una conv senza attivazione, E compone SiLU da
C; se offre una variante con attivazione, fissarla esplicitamente a SiLU
una volta sola. Ring buffer/opzione fused sono dettagli da documentare
senza alterare ordine tap o formato snapshot. E non scrive un secondo kernel
convoluzione per aggirare una dipendenza mancante.

## 6. File posseduti e modifiche condivise richieste

In **questa sessione** unico file scritto e committato:
`docs/specs/qwen38/E.md`.

Proposta ownership futura E, da ratificare (non sono file creati qui):

| File | Contenuto |
|---|---|
| `src/qwen38_ngram.c` | Storia/ID, fd, planner I/O, loop pread, lease e stato |
| `src/qwen38_ngram_internal.h` | Strutture private E, hook fault injection; non API condivisa |
| `src/qwen38_ple.c` | Gate CPU e composizione chiamando B/C |
| `src/qwen38_ple_gpu.m` | Solo glue Metal, staging e completion |
| `src/qwen38_ple.metal` | Gate PLE e scatter/espansione se necessaria; niente conv/dequant Q8 duplicata |
| `tools/oracle/qwen38_ngram.py` | Oracle e manifest di provenienza; estrazione bounded opzionale |
| `tests/test_qwen38_ngram.c` | ID, I/O, fault injection e ownership ASan/UBSan |
| `tests/test_qwen38_ngram.py` | Esecuzione oracle, fixture e confronto C |
| `tests/test_qwen38_ple.m` | Parità Metal, lease in flight e composizione |
| `tests/fixtures/qwen38/ngram.json` | Golden piccoli e provenienza; niente pesi interi |

Coordinatore proprietario di Makefile, header pubblico Qwen38, `gguf.h`,
`picoforge.h`, entry point, `kernels.metal` se include aggregati, README,
report e `integration.md`. Richieste: dichiarazioni lifecycle/errori/span,
API vista CPU residenti, dipendenze B/C, target di test separati CPU/GPU,
link del nuovo `.metal`, inserimento PLE prima di HC-attention nel layer
indicato da A. Il Makefile attuale compila solo `src/kernels.metal` nella
metallib: aggiungere un file `.metal` non basta. Nessuna modifica a questi
file dalla sessione E senza riassegnazione esplicita.

## 7. Oracle indipendente e fixture

### Indipendenza e controllo già eseguito

Per gli ID usare **direttamente** i metodi del NGramEmbedding Transformers
fissato sopra: shift con `cummax` degli EOS e gather vettoriale, distinti
dal ciclo newest-first C/DwarfStar. Eseguire l'AST della classe revisionata
con Torch esistente, senza importare/aggiornare tutta Transformers. Evitare
il costruttore che allocherebbe la tabella; inizializzare solo metadata e
sostituire l'embedding con un catturatore degli indici. Nessun algoritmo hash
riscritto dentro il catturatore. Hash della classe sorgente estratta:
`874ee437373ae5ed3419da959f5163f1a5669549c66eccbdd7833e6869f87362`.

Questo controllo minimo CPU è stato eseguito in memoria in SPEC, senza GPU,
file fixture prodotti o letture di payload. Parametri sintetici N=3, G=2,
EOS=99, multiplier `[3,5,7]`, vocab testa `[11,13,17,19]`, offset
`[0,11,24,41]`, token `[1,2,99,3,4,99,99,5]`:

| t | token | ID attesi dalle quattro teste |
|---:|---:|---|
| 0 | 1 | 8, 22, 31, 43 |
| 1 | 2 | 3, 14, 38, 51 |
| 2 | 99 | 5, 16, 27, 48 |
| 3 | 3 | 2, 16, 25, 56 |
| 4 | 4 | 3, 14, 38, 51 |
| 5 | 99 | 9, 16, 31, 52 |
| 6 | 99 | 0, 14, 39, 41 |
| 7 | 5 | 7, 23, 27, 58 |

Con i metadata reali e token iniziale 1, il medesimo metodo ha restituito:

```text
16121432, 28938500, 59087997, 73487090,
81148277, 104500129, 120276032, 149373875,
176283436, 184305849, 216528839, 231080079,
257961536, 266068568, 289043455, 305959965
```

Sono golden di **indirizzamento**, non test di un'implementazione E esistente
né output del checkpoint. In ESEGUI salvare manifest e vettori con script
riproducibile, versione Torch e checksum fonte verificati prima di eseguire
l'estratto. Per chunk utilizzare il metodo upstream con cache minimale che
conserva l'input raw, verificata contro il riferimento full-sequence senza
cache; non usare tale shim come unico oracle del lifecycle.

Gate/composizione: esercitare il forward PLE upstream con `nn.Linear`,
RMS in FP32 e `torch.nn.functional.conv1d(groups=C,dilation=N)` su fixture
piccole. Il confronto conv principale è full-sequence con left padding
esplicito: non tradurre in Python il ring buffer C. Confrontare con un secondo
calcolo FP64 via matrici/dot e casi analitici; la sola traduzione di DwarfStar
CPU e Metal non vale come oracle indipendente. Adattamento norma folded
documentato e testato prima di passare pesi GGUF al modulo HF.

### Fixture richieste

- ID: inizio senza BOS esplicito, sequenze di 1/2/3 token, token 0 e V-1,
  token invalidi, EOS corrente/precedente/doppio ai confini chunk, falso EOS
  del template; chunk 1,2,3,7,9,10,255,256,1023,1024 con partizioni casuali.
  Storie di sessioni intercalate senza stato globale. Metadata malformati,
  overflow signed/unsigned, offset padded e numero teste incompatibile.
- BF16/I/O: bit di +/-0, min subnormal, min normal, +/-1, max finite,
  NaN/Inf; tutte le combinazioni BF16 finite confrontate bitwise su tabella
  sintetica piccola. Righe con indice codificato nei byte, incluse ultima e
  righe a cavallo pagina. Duplicati non adiacenti, richieste fuori ordine,
  intervalli consecutivi più lunghi di 64 KiB, buchi e scarti in cache.
- Fault injection: ogni punto di short read, anche byte dispari; EINTR prima
  e dopo progresso, EOF nell'ultimo byte, EIO/EBADF, fallimento allocazione,
  cancel, cambio file, overflow offset. Output precedentemente valido
  avvelenato; dopo errore rows/encode/commit devono fallire. Fallimento in
  ultima riga dopo altre complete non pubblica un batch parziale.
- Gate: D piccoli e D=2560, S=1/4, rami diversi per scoprire norm globale,
  dot zero, +/-tiny ai lati del clamp, saturazione +/-100, cancellation,
  gamma diretta vs errore `1+gamma`. Proiezioni non diagonali e one-hot per
  riconoscere trasposizioni/scatter; Wc asimmetrici per ordine tap/dilation.
- Conv/stato: impulso che appare ai ritardi 0/3/6/9, K=1, chunk più corto
  della storia, zeri iniziali, EOS senza reset conv, reset esplicito, append
  e chunk partition invariance. Sequenze sintetiche 8K,16K,32K a D ridotto.
- Ownership: chiusura reader GGUF dopo dup, chiusura store BUSY, job foreign,
  epoch vecchia dopo reset, encode precedente a READY, doppio commit,
  riuso slot con GPU in flight; completamento Metal fallito senza commit.

### Slice reali, dopo il consolidamento

Estrarre con pread e inventario, in sola lettura, solo le righe richieste da
una sequenza congelata (inclusi vettori sopra), le prime/ultime righe di ogni
range testa e campioni presso confini; massimo 2048 righe, 640 KiB raw.
Registrare ID, offset, byte e hash slice, artifact hash già verificato e
identità locale; non ripetere l'hash di 137 GiB per ogni test. L'oracle legge
quelle slice per indice logico, indipendentemente dal sort/accorpamento C.

Per PLE completo estrarre i **sei** tensori residenti Wk/Wv/tre norme/conv,
più le righe selezionate della tabella. Wk/Wv raw totali 34816000 byte;
norme+conv 286720 byte. Decodifica a blocchi tramite oracle B, senza duplicare
Q8_0 in E. FP32 completo delle due proiezioni costa 125 MiB: ammesso solo
nell'oracle bounded, oppure tile di output per ridurre picco. Nessuna slice
BF16 originale aggiuntiva necessaria per verificare il checkpoint Q2.

Per prova di composizione con attivazioni autentiche servirà una cattura del
residuo in ingresso al layer 1 e intermedi PLE da runtime revisionato, sotto
lock e fuori SPEC. DwarfStar è un confronto aggiuntivo dello stesso Q2; senza
queste catture non dichiarare parità end-to-end. Usare pesi originali BF16
per misurare perdita di compressione è un esperimento diverso.

## 8. Tolleranze proposte e criteri di accettazione

Congelare queste soglie prima dei test implementativi; sono budget motivati
da precisione e riduzioni, non valori misurati. Nessuna tolleranza viene
alzata soltanto per accettare un mismatch.

| Verifica | Criterio iniziale |
|---|---|
| ID, offset, ordine scatter, byte BF16, stato token | Esatto, 0 differenze |
| BF16 -> FP32 finito | Bitwise esatto anche +/-0 e subnormali; NaN/Inf rifiutati |
| Errori, lease, budget | Esatto: nessun READY/commit dopo fault, nessun accesso ASan/UBSan, limiti rispettati |
| Proiezioni Q8_0 | Soglie di B sul medesimo checkpoint, riportate separatamente dal gate |
| RMS C/Metal su fixture ben condizionate | `abs(a-b) <= 2e-5 + 2e-5*abs(ref)` per elemento |
| Gate scalar g C/Metal vs upstream FP32 | errore assoluto <= 1e-4; dot esattamente nullo deve dare 0.5 |
| Delta e storia z, blocco PLE | `abs(a-b) <= 2e-4 + 2e-4*abs(ref)` per elemento; relativo L2 <= 1e-4 quando norma ref >=1 |
| Token-by-token vs chunk, stesso backend | ID/storia token esatti; z/Delta/storia conv entro la soglia PLE, dopo ogni confine |

Le riduzioni di D=2560 termini e l'ordine FMA possono cambiare gli ultimi
bit FP32; la radice firmata amplifica errori vicino a zero. Nei casi costruiti
con forte cancellation misurare anche errore dot **prima** della radice;
confronto FP64 e somma assoluta dei prodotti identificano se serve cambiare
riduzione, precisione o formulazione stabile. Un'inversione del segno o una
violazione vicino al clamp non si ignora con una tolleranza globale più larga:
si porta una proposta al consolidamento. Test di u/z separati impediscono
che un grande residuo R nasconda l'errore di Delta. NaN inattesi falliscono.

Accettazione E richiede: oracle indipendente revisionato, tutti i casi CPU e
fault injection, sinteticità e slice Q2 distinte, Metal contro C dopo B/C,
chunk/append/reset e fault a metà grafo, allocazioni bounded, esclusione
tabella dalle viste, Qwen3 regressioni via coordinatore. Controlli locali
di primitive non dimostrano generazione completa né perdita rispetto a BF16.

## 9. Incrementi verificabili

1. **Oracle e golden**: fissare fonte Transformers e adattamento gamma,
   fixture hash/BF16/conv full-sequence. Zero payload completo. Uscita:
   golden manuali più metodi upstream concordi, manifest riproducibile.
2. **Hash C11 puro**: config validata, storia locale, ID e append/reset;
   nessuna I/O o GPU. Uscita: match esatto di tutte le partizioni e sanitizers.
3. **Reader C11 sincrono**: dup fd, buffer limitati, complete-read loop,
   BF16/scatter e fault injection; poi dedup globale/accorpamento. Uscita:
   medesimi byte e transizioni con planner semplice/ottimizzato, limite memoria.
4. **Gate C e composizione**: prima gate con key/value fixture; poi veri
   operatori B Q8_0 e C RMS/conv. Uscita: Delta e stato contro upstream,
   prima sintetici poi slice. Nessun doppione di conv o decoder.
5. **Metal bounded**: staging sincrono, ownership/completion, gate E,
   proiezioni B/conv C. Uscita: numerica e API validation sotto lock.
6. **Integrazione del coordinatore**: layer corretto, commit/poison globale,
   append/reset, regressioni Qwen3; oracle di intermedi/logits distinto.
7. **Solo dopo correttezza**: cache 8 MiB, due slot, I/O sovrapposto e pochi
   worker se utile. Misurare dedup/accorpamento/cache separatamente; nessuna
   promessa di accelerazione da semplice riduzione delle pread.

Ogni incremento entra in commit piccolo e controllabile; i limiti di memoria
sono precondizioni d'ammissione, non allocazioni tentate fino a fallimento.

## 10. Verifica, memoria e rischi

### Comandi e limiti della verifica in SPEC

Eseguiti: `git status --short`, `git rev-parse HEAD`, `git diff --cached
--stat`, controllo worktree, letture documentazione/sorgenti/inventario,
`stat` del checkpoint, fetch di config/sorgenti revisionati e piccolo oracle
CPU degli ID descritto in §7. Nessun build, test GPU, benchmark o lettura del
payload del modello. `make test-all` non è stato eseguito: include workload
GPU e non serve a validare una sola spec. Prima del commit: `git diff --check`
e verifica che lo staging contenga esclusivamente questo file.

In ESEGUI, **target da creare dal coordinatore**, non comandi oggi disponibili:

```sh
cd "$Q38_IMPL_WORKTREE"           # nuovo worktree dalla BASE_CONCORDATA
Q38_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
# I test invocano questo interprete tramite runner/variabile concordati,
# non creano un secondo venv né scrivono nel checkout principale.
"$Q38_PY" tests/test_qwen38_ngram.py --synthetic
make build/test_qwen38_ngram       # clang C11, warning del repo, ASan/UBSan
./build/test_qwen38_ngram
# Dopo lock comune e disponibilità effettiva di B/C:
make build/test_qwen38_ple
MTL_DEBUG_LAYER=1 ./build/test_qwen38_ple
```

Per slice il runner futuro accetta `--checkpoint` assoluto nel checkout
principale e `--output build/qwen38-ngram/<run-id>` locale, rifiutando overwrite;
limiti `--max-rows 2048` e `--max-read-bytes 67108864` includono proiezioni.
Comandi definitivi e nomi delle flag da implementare in incremento 1.
Build/test generali `make`, `make test-gguf`, `make test-gpu-weights`,
`make test-all` sono responsabilità d'integrazione del coordinatore, con
ambiente esistente e fixture isolate; nessun `make clean` nel checkout altrui.

**Lock aperto:** percorso, primitive e wrapper di sistema non sono stati
forniti. Non inventare un lock per worktree: il coordinatore deve fissare
in integration.md un unico lock condiviso. Ogni test GPU, anche sintetico,
e ogni carico pesante sul checkpoint lo acquisisce; benchmark con GPU
esclusiva. Fino ad allora restano eseguibili soltanto verifiche leggere CPU.

### Budget analitico proposto

Un token richiede al massimo `H*d*2=5120` byte utili di tabella; un chunk
1024 al massimo **5 MiB**, prima di dedup/cache. FP32 lookup **10 MiB**.
Non sono traffico fisico SSD: il page cache può amplificare o assorbire I/O.

Budget runtime E massimo **256 MiB** esclusi i pesi residenti già conteggiati
da storage e il residuo `[T,C]` del grafo, inclusi però tutti i suoi scratch:

| Voce, Tmax=1024 | Payload massimo proposto |
|---|---:|
| Due slot raw BF16 + FP32 lookup | 2*(5+10) = 30 MiB |
| ID + richieste `(row,dst)` 16 byte + mapping uint32, due slot | <= 0.75 MiB |
| key, u, z, Delta `[T,C]` FP32, un workspace condiviso | 4*40 = 160 MiB |
| value `[T,D]` FP32 | 10 MiB |
| Storia conv persistente e candidata, 2*9*10240*4 | 0.703125 MiB |
| Cache opzionale | 8 MiB massimo, 0 nel primo incremento |
| Buffer lettura intervalli | 64 KiB per reader, 1 reader iniziale |
| Totale payload previsto | < 210 MiB; resto fino a 256 per metadata, allineamenti e scratch B/C |

Nessuna query normalizzata `[T,C]` extra: RMS/gate lavorano per token/ramo
oppure scratch riusato. Se il contratto B/C richiede altro scratch, includerlo
nel conteggio prima di allocare; ridurre tile/chunk o proporre budget nuovo,
non nascondere memoria in un'altra categoria. R del grafo aggiunge 40 MiB
per chunk ma non va contato due volte. I sei pesi residenti PLE sono
35102720 byte (~33.48 MiB), parte dei 41.72 GiB delle viste, non una copia E.
Oracle slice: obiettivo <=512 MiB aggiuntivi oltre runtime Python/Torch,
FP64/tile espliciti; niente tabella o modello HF completo.

Il limite E non dimostra il picco globale 48–50 GiB: il coordinatore somma
allocazioni disgiunte, misura footprint/Metal/pressione/swap e ammette la
sessione rispetto ai limiti reali. L'OS file cache non è limitata dal nostro
staging; va osservata e non confusa con memoria malloc/Metal.

### Decisioni aperte e rischi concreti

1. Ratifica API/header/ownership e BASE_CONCORDATA; schema errori comune e
   transazione del grafo devono evitare commit parziali tra E/C/D/F.
2. C deve confermare gamma diretta, dilation=3 e history di nove token;
   B deve includere Q8_0 ordinario, senza aspettare soltanto i formati MoE.
3. Vista CPU dei piccoli pesi GGUF e tipi di span Metal da fissare; evitare
   doppia mappatura/copia e dangling binding dopo chiusura store.
4. La sensibilità del signed sqrt vicino a zero richiede test intermedi
   e potrebbe imporre riduzioni migliori; soglie ancora non misurate.
5. Il codice Transformers pinned fornisce oracle indipendente ma non è
   installato: estrazione minima revisionata prevista, nessuna nuova
   dipendenza. Se non basta, segnalare blocco prima di chiedere un upgrade.
6. I/O casuale, accorpamento scarso e cache di sistema possono dominare la
   latenza; nessun benchmark eseguito né beneficio prestazionale rivendicato.
7. MTP/rewind richiedono snapshot token+conv e dello stato degli altri
   componenti; rifiutare operazioni non supportate invece di ripristinare
   soltanto posizione o KV. Mask e multimodalità richiedono specifica futura.
8. Confermare lock comune prima di GPU/slice pesanti. Nessun push o merge
   da questa sessione; il commit contiene soltanto la spec E.

[ds4]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c
[metal]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/metal/qwen4.metal
[pack]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/qwen4_pack.py
[convert]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/qwen4_pack_to_qwen4exp.py
[native]: https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/gguf-tools/qwen4_native_ngrams.py
[config]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json
[hf]: https://github.com/huggingface/transformers/blob/db2304e8694db3ecd6f39d353beba361688afe97/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
[paper]: https://arxiv.org/html/2608.30320v1#S2.SS3

## 11. ESEGUI dalla base consolidata — hash e reader CPU

Base effettiva `c674b23bd1377b2ce0acef2d06fac8b160809446`, branch
`impl/qwen38-e-v1`, worktree `../picoforge-e-execute`. Le sezioni SPEC
precedenti sono conservate; prevalgono `integration.md` e gli header v1.
Recuperati soltanto i file oracle/fixture di `c1a219a` e codice/test CPU di
`4fbc52c`, senza reimportare la vecchia spec o file di altre sessioni.
Il primo recupero verificato è `9a34fb0`.

Implementato `qwen38_ngram_ids` con i tipi pubblici: array uint64,
storia int32 newest-first immutabile e candidata separata; batch vuoto copia
la storia. Checked bounds, dominio signed-int64 dei prodotti HF, intervalli
contigui, capacità e alias rifiutati prima di scrivere. Nessun limite di
ordine/head/token hardcoded nel runtime hash; i limiti dell'oracle restano
esplicitamente quelli delle fixture bounded. Righe oltre UINT32_MAX testate
separatamente, senza assumere il limite fisico del checkpoint nel contratto.

Reader ancora **privato**, pronto per essere incorporato nel futuro store/job:
usa solo `Qwen38Status`/`Qwen38Error`, fd duplicato read-only/CLOEXEC,
nessun mapping/cache, staging allocato al setup, dedup globale, accorpamento
di righe consecutive entro 64 KiB. CAPACITY distingue budget insufficiente
da OOM dell'allocatore. Lease valide solo dopo successo completo; ogni nuova
richiesta invalida la precedente, anche in caso di errore. OWNER per lease
estranee; STALE per file modificato/generazione invalida. Errori I/O riportano
errno, offset assoluto, riga e progresso dell'intervallo. Immutabilità del file
rimane precondizione: fstat non garantisce integrità crittografica.

Verifica ripetibile, Python esistente, nessun checkpoint aperto:

```sh
Q38_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
# Solo al primo setup, oppure copiare la piccola cache verificata dal vecchio worktree:
"$Q38_PY" tools/oracle/qwen38_ngram.py --fetch-source
"$Q38_PY" tests/test_qwen38_ngram.py -v
```

11 test passati, harness clang C11 con warning come errori,
`-ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer`.
ID bit-esatti contro i metodi Transformers originali fissati, EOS e chunk
1/2/3/7/9/10/255/256/1023/1024; BF16 bit-esatto contro PyTorch per tutti i
65.280 pattern finiti. Fixture fino a 128 KiB, boundary pagina, duplicati,
ultima riga, intervalli >64 KiB divisi; short read di 3 byte, EINTR e
EOF/EIO/EBADF a ogni byte di un batch di 32 byte (96 errori), troncamento
reale, NaN/Inf, fallimenti delle quattro allocazioni e limite esatto del
budget. Errori e successo successivo non riabilitano lease precedenti.
Il test controlla anche offset/errno/progresso, non soltanto il codice.
Nessuna prova di perdita Q2 rispetto a BF16 originale, GPU o prestazioni.

Store/state/job pubblico e history candidata richiedono ancora C1;
PLE completo C2, B Q8_0 e A. Le dichiarazioni non implementate non hanno
stub: questa consegna non certifica transazioni complete o generazione.

### Gate PLE normalizzato — incremento indipendente completato

Oracle `1109265`: `GateOracle` estrae via AST soltanto le tre assegnazioni
del gate originale `Qwen4ExpTextPLELayer.forward`, con corpi matematici
invariati; aggiunge esclusivamente catture per dot/g/u. Source hash completo
verificato prima di eseguire, hash delle espressioni nel manifest,
provenienza/licenza Apache-2.0 conservate nella piccola cache upstream.
Nessuna proiezione, RMS o convoluzione simulata nell'oracle.

`src/qwen38_ple.c` implementa l'entrypoint pubblico `qwen38_ple_gate`:
input key/query normalizzati `[T,S,D]`, value `[T,D]`; output g `[T,S]`,
u `[T,S,D]`. FP32, riduzione scalare con contrazione disabilitata nei test,
exp del negativo della radice per evitare overflow della sigmoid;
`dot==0` produce esattamente 0.5. È l'espressione fusa del gate PLE, senza
esportare un secondo operatore di attivazione C. Non alloca memoria,
non modifica input/stato. Stride/capacità in byte, feature contigua,
outer pitch ammesso, input/output disgiunti. Batch vuoto senza accessi.
Preflight di forme, stride, overflow, capacità, alias e finitezza; overflow
del dot produce NONFINITE e rende inutilizzabile l'intero output, anche
se alcune righe erano già calcolate. Nessuna promessa di rollback numerico.

Suite aggiornata: **15 test passati** ASan/UBSan, più harness release O2
con tutti i warning come errori. Fixture `[1,1,1]`, `[3,2,7]`, `[17,4,160]`,
`[3,4,2560]`, pitch aggiuntivo 0/1/7 float, guardie intatte,
parità token/chunk bitwise sul core, zero e near-zero con segno verificato,
cancellazione, saturazione, NaN/Inf e overflow prodotto/somma. Le fixture
analitiche verificano dot esatto; sulle fixture normalizzate si usa anche
un oracle FP64 e il bound analitico `gamma(2*D+4)*sum(abs(k*q))/sqrt(D)`.
Massimi osservati contro HF sulle fixture normalizzate: gate abs
**2.68220901e-7** (soglia 1e-4), dot abs **3.33786011e-6**;
u passa atol/rtol 2e-4 e relativo L2 1e-4. Questi dati non certificano
PLE Delta/z/history, non ancora calcolati, né input arbitrari mal condizionati.

Controlli eseguiti dalla radice del nuovo worktree:

```sh
"$Q38_PY" tests/test_qwen38_ngram.py -v
clang -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror \
  -ffp-contract=off -Isrc src/qwen38_ngram.c src/qwen38_ple.c \
  tests/test_qwen38_ngram.c -o build/qwen38-e/test_ngram_release
build/qwen38-e/test_ngram_release
make -j4
make PYTHON="$Q38_PY" test-qwen38-contracts test-gguf
```

Build generale e contratti/GGUF passati. `make` non cambia i flag globali:
il coordinatore deve conservare `-ffp-contract=off` quando collega i target
numerici E. Nessun header pubblico/Makefile modificato, nessun test GPU,
nessun payload del checkpoint letto, nessun benchmark; lock non necessario
per questi piccoli test CPU. `make test-all` e test GPU del grafo sono
rimandati all'integrazione dei componenti.

Per integrazione: acquisire i nuovi commit E in ordine sopra BASE_CONCORDATA,
non riacquisire `c1a219a`/`4fbc52c` né `340d026`. Gli entrypoint disponibili
sono hash pubblico, gate pubblico e reader privato; gli altri simboli del
contratto restano dichiarazioni, senza stub. Il coordinatore aggiungerà il
target della suite al Makefile. Restano C1 per history candidata e lifecycle
store/state/job, C2+B Q8_0+A per PLE completo, poi kernel/staging Metal,
commit globale, slice reali e integrazione del grafo. Nessun commit di altre
sessioni è stato importato o dipendenza dichiarata implicitamente soddisfatta.

## 12. ESEGUI v2 — store/job e PLE CPU composto

Base operativa `6a6b886e985185d7c9104eda888e6a406553aedf`;
branch `impl/qwen38-e-v2`, worktree esterno `../picoforge-e-v2`.
Applicati `resume.md` e `integration.md` §11. Nessun cherry-pick delle
consegne già integrate, nessuna modifica ai worktree precedenti.

### Implementazione e durata

Implementate tutte le API CPU di `qwen38_ngram.h`: store/state/job,
workspace e `qwen38_ple_cpu`, oltre a hash/gate già presenti. Lo store
copia config e array, duplica il fd tramite reader E, verifica il descrittore
BF16 e i confini; non mappa né precarica la tabella. Lo stato possiede
history token e backing C allineato. Il job possiede ID uint64 materializzati,
history token candidata e un secondo backing C: init/copy/commit usano
`qwen38_conv_state_*`, mai copie di handle opachi.

Baseline sincrona, **un solo job per store**, anche tra stati distinti.
Prepare/read non avanzano lo stato. READY richiede l'intero batch;
COMPUTED richiede Delta finita e history C completa. Commit ricontrolla
owner/epoch/posizione/file e pubblica history e posizione insieme. Abort
scarta soltanto la candidata. Store/state/workspace restano BUSY finché
referenziati: commit e abort non sostituiscono job_release. Release può
scartare anche un job non committato. Gli handle liberati non sono più
utilizzabili; nessuna promessa di rilevare un puntatore C dopo free.
Le righe sono view borrowed `[T,H*d]`, valide fino a release salvo abort,
errore o invalidazione dello store. T=0 è COMPUTED senza I/O, input o
avanzamento. File mutato → store STALE permanente, stato info non valido,
necessaria riapertura; un reset non recupera un file cambiato.

Workspace unico allocato al setup; nessuna allocazione dentro ple_cpu.
B esegue key/value da Q8_0 (anche F32 verificato); C RMS per gruppo D con
gamma diretta; E gate; C conv con SiLU, dilation=N e history raw oldest-first.
Output **Delta**, nessuna somma con R. Guadagni accettati come `[S,D]`,
vettore folded `[S*D]` o gain condiviso `[D]`; nessun +1 nel runtime.
Viste con stride byte verificati e buffer disgiunti; input/pesi borrowed
per la chiamata sincrona. Identità e immutabilità degli span dei pesi
rimangono responsabilità del chiamante, come nell'ABI B condiviso.
Errore strutturale prima del calcolo lascia il job READY; errore numerico
rende il job inutilizzabile. La sessione del grafo resta responsabilità
del coordinatore: il rollback locale E non recupera mutazioni C/D/F altrove.

### Evidenza numerica e fault

**17 test CPU passati, nessuno skip**, clang C11 ASan/UBSan con
`-fno-sanitize-recover=all -ffp-contract=off -fno-fast-math` e warning fatali.
Si conservano i test ID/BF16 esatti e ogni-byte EINTR/EOF/EIO/EBADF del reader.
Aggiunti: API pubbliche prima/dopo READY, doppio commit, owner estraneo,
BUSY, batch vuoto anche PLE, abort/retry, reset/replay, file troncato dopo
READY, allocazione fallita a ciascuno dei **15** punti di setup/job senza
risorse vive residue, capacità, dimensioni, nonfinite e alias R/Delta.

Oracle PLE: classi/metodi originali Transformers fissati, forward e conv
PyTorch invariati. Adapter della norma usa `_norm` upstream e applica il
gain GGUF già folded, evitando una sottrazione/riaddizione FP32 di 1.
Q8 sintetico costruito da pesi dyadic noti; reale decodificato a tile tramite
oracle indipendente B/DwarfStar, nessun decoder duplicato in E.

| Prova | Risultato |
|---|---|
| 23 token sintetici, F32/Q8_0, chunk 1/2/7/9/23, abort/retry e reset | max abs Delta `2.38418579e-7` |
| 32768 token sintetici, D=8/S=2, Q8_0, chunk 255/1024 | max abs Delta `7.15255737e-7` |
| Slice reale, D=2560/S=4, 2 token, chunk 1/2 | max abs Delta `1.19209290e-7`; relativo L2 `3.27002340e-7` |
| Proiezioni reali contro FP64, bound B per elemento | max rapporto errore/bound `0.00240521228` (<1) |
| Storia C | bit-esatta rispetto agli z nativi; entro atol/rtol 2e-4 e L2 1e-4 rispetto all'oracle completo |

Verificati anche key/query normalizzati, value, dot/segno, gate e z;
nella slice reale gli operatori C e gate E sono confrontati con oracle
alimentati dagli **stessi intermedi nativi**, oltre al PLE completo.
Le tolleranze §6 non sono state allargate. Due token reali non certificano
una conversazione reale: R è sintetico, non una cattura del layer precedente.
Nessuna misura della perdita Q2 contro pesi BF16 originali.

### Slice, memoria e riproduzione

Sotto lock comune: config e ruoli risolti da A sul GGUF esistente,
**35.102.720 byte** dei soli sei tensori PLE, 32 righe di embedding
(10.240 byte BF16), più 38 righe di controllo ai confini delle teste e
vicino ai boundary di pagina/intervallo (12.160 byte BF16).
Niente checksum completo o copia del checkpoint. Il manifest registra
artifact/revisione, stat, ID/offset, hash dei file, source/binary degli oracle
e del harness. Output solo in `build/qwen38-e/real-ple-final/`.

Contatore delle allocazioni E vive (non RSS, pesi borrowed esclusi):
1.029.958 byte con chunk1, 1.255.638 con chunk2. Include stato persistente,
candidata, reader, metadati e workspace. Per Tmax1024 a geometria reale il
calcolo del medesimo layout è **231.900.598 byte** (~221.2 MiB); non è una
misura di footprint del modello. Tutte le allocazioni aggregate E hanno
cap 256 MiB, anche con più stati/workspace; ogni create verifica inoltre
il proprio budget. Oracle: due matrici FP32 125 MiB, decoder a tile 512
righe, niente dequantizzazione dell'intero modello; budget addizionale
previsto entro 512 MiB oltre Python/Torch.

```sh
cd /Users/cristiancirje/Desktop/Dev/picoforge-e-v2
Q38_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
"$Q38_PY" tools/oracle/qwen38_ngram.py --fetch-source
"$Q38_PY" tests/test_qwen38_ngram.py -v
# Una volta, directory nuova (sorgenti DwarfStar esistenti, nessuna dipendenza):
"$Q38_PY" -c 'import sys; sys.path.insert(0,"tools/oracle"); from qwen38_packed import build_reference; build_reference("/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline", "build/qwen38-e/packed-reference")'
# Usare una directory --output nuova per ogni run, senza sovrascrivere evidenza.
"$Q38_PY" tools/qwen38_gpu_lock.py --timeout 60 -- "$Q38_PY" \
  tools/oracle/qwen38_ngram.py \
  --real-checkpoint /Users/cristiancirje/Desktop/Dev/picoforge/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf \
  --native build/qwen38-e/test_ngram \
  --packed-reference build/qwen38-e/packed-reference/reference \
  --output build/qwen38-e/real-ple-new
make -j4
make PYTHON="$Q38_PY" test-qwen38-contracts test-gguf
```

Passati build, 4 test contratti/lock e 7 GGUF; harness E anche O2 senza
sanitizer. Non eseguito make test-all, che coinvolge altri asset e GPU.
Nessun test/kernel GPU E in questa consegna; il lock sopra protegge il
workload CPU sul checkpoint, non indica uso GPU.

### Proposta concreta di integrazione condivisa

Il target `test-qwen38-e-cpu` continua a usare il runner Python E, che linka
ora anche `qwen38_conv.c`, `qwen38_base.c`, `qwen38_packed.c`,
`qwen38_config.c`, `qwen38_weights.c`, `gguf.c`, `gguf_reader.c`, `util.c`.
**Il coordinatore deve aggiungere `src/qwen38_conv.c` alla lista sorgenti
del target `build/qwen38-integration/test_links` in `tools/qwen38.mk`.**
La regola v1 non lo include; E non modifica il Makefile condiviso.
Verificato manualmente lo stesso harness con la dipendenza aggiunta:

```sh
clang -std=c11 -O1 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror \
  -ffp-contract=off -fno-fast-math -fsanitize=address,undefined \
  -fno-sanitize-recover=all -Isrc tests/test_qwen38_integration.c \
  src/qwen38_config.c src/qwen38_weights.c src/qwen38_packed.c \
  src/qwen38_base.c src/qwen38_conv.c src/qwen38_attention.c \
  src/qwen38_ngram.c src/qwen38_ple.c src/qwen38_moe.c \
  src/gguf_plan.c src/gguf.c src/gguf_reader.c src/util.c -lm \
  -o build/qwen38-e/test_links
build/qwen38-e/test_links \
  /Users/cristiancirje/Desktop/Dev/picoforge-qwen38-integrate/build/qwen38-a/small.gguf
```

PASS: collegamenti A→B→F, C→E e A→D/E preservati. Restano staging/kernel
Metal E, lifecycle GPU pubblico e commit globale; catture autentiche,
integrazione dei layer/logits e generazione del modello. Nessuna dipendenza
CPU mancante è sostituita da stub e nessun file altrui è stato modificato.

Manifest finale della slice reale (`build/qwen38-e/real-ple-final/manifest.json`),
SHA-256: `7867992a96b012551a83751c86a338cc4cf7639ed036603dc0dd97f000ae93ae`.

## Integrazione Metal G2 (2026-10-13)

`qwen38_ple_gpu.m`/`.metal` compongono B proiezioni packed, C RMS e conv,
gate signed-sqrt e Delta FP32. `qwen38_e_internal.h` contiene il contratto
privato di encode: viste dense, scratch/history disgiunti, preflight e
nessuna submission interna. Il bind esplicito dello stato E a G0 separa
la storia conv su GPU da quella CPU; prepare non clona una conv CPU obsoleta,
e il commit hash verifica epoch/posizione/completion della sessione.

Il reader e i suoi controlli di file/lease rimangono invariati. Le righe SSD
sono Shared bounded e restano vive fino a completion. Errori e reset sono
gestiti dal grafo globale. Test, misure, CLI e limiti in [G2.md](G2.md).
