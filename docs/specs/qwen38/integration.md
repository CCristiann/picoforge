# Qwen3.8 Q2 — contratto consolidato v1

Stato implementativo aggiornato: **BASE_INTEGRATA v1, §11**. Le sezioni
precedenti conservano contratto e provenienza del consolidamento.

2026-10-12 · MODALITA=CONSOLIDA · SPEC_ID=ALL.
Questa è la base normativa per ESEGUI. **Tutte le spec A–F sono acquisite**.
Le loro proposte restano conservate integralmente; in caso di differenza
prevalgono questo documento e gli header v1, non i prototipi nelle spec.
Il contratto è congelato prima dei test numerici, non certifica operatori
già implementati né generazione nativa. La direzione resta
**Qwen3.8-Flash-Next Q2, C11/Metal, M5 Pro 64 GiB, coding/ragionamento**.

## 1. Base, provenienza e perimetro

Branch coordinatore `coord/qwen38-contracts`, worktree esterno
`/Users/cristiancirje/Desktop/Dev/picoforge-qwen38-consolidate`.
Base precedente `7c0b2379b93c478a46b0d6e75c3bde91a57b1bce`.
Il commit di consolidamento che introduce questo file, con tutti gli header
e i sei documenti nel suo tree, è **BASE_CONCORDATA**. Lo SHA completo viene
consegnato dal coordinatore dopo il commit; non si inserisce un impossibile
hash autoreferenziale nel documento. Nessun merge/push su `main`.

| Spec | Branch sorgente | Commit sorgente acquisito |
|---|---|---|
| [A](A.md) | `spec/qwen38-a` | `c6ac2f0bc0d70f415b07ce8f46da2277f66f1e2b` |
| [B](B.md) | `spec/qwen38-b` | `82f38ce6b8c8ab16e99b08dd86e2c96f729312b4` |
| [C](C.md) | `spec/qwen38-c` | `2ba83ef9369d54fc170c83b1da23644fe3d8f51b` |
| [D](D.md) | `spec/qwen38-d` | `f788b1588f9e7fe168f9cfa5ef582cade0468277` |
| [E](E.md) | `spec/qwen38-e` | `340d02619285f14932a1aae665d0d5c4e4f09f98` |
| [F](F.md) | `spec/qwen38-f` | `dc4a82e0917d3a868a8b1175f87b77b233c1177a` |

Acquisiti mediante cherry-pick di commit contenenti soltanto il rispettivo
Markdown. Non acquisiti commit/working tree `*-oracle`, fixture in lavorazione
o implementazioni delle altre sessioni. A è byte-identica allo SHA comunicato
dall'utente. Le implementazioni successive non cambiano retroattivamente
questa provenienza.

Riferimenti immutabili: artifact HF `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe`,
checksum GGUF `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`;
DwarfStar `fc80bd695da76ee14bcb86820bf900a5cdfda806`; config/tokenizer originale
HF `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Non si ripete il checksum
di 137 GiB né si scarica un altro checkpoint.
Per gli oracle C–F fissare Transformers
`812a950d7037ba93bb624b154030e9068afdce4d`, modeling SHA-256
`f94401c10c1bbafd814cc174ba6f40dde1f40b35715ce1ae827ccf8ee3cd81fa`.
D/E hanno documentato un altro commit con gli stessi byte: è accettabile
conservarlo nel loro manifest verificando lo stesso hash, senza riscrivere
oracle già committati. Modulo upstream non installato nel venv: estrazione
AST allowlist revisionata, corpi matematici invariati e licenza conservata,
come specificato in C–F; nessuna nuova dipendenza o esecuzione dell'intero
modulo remoto. L'adattatore deve ancora superare i propri test.

## 2. Decisioni che chiudono le divergenze

| Tema | Decisione vincolante v1 |
|---|---|
| Errori A/B/C/D/E/F diversi | Un solo `Qwen38Status` e `Qwen38Error` in `qwen38_types.h`; nessun enum parallelo né alias con numerazione diversa |
| View rank 3/4, stride in float/byte | Rank 1–4, stride e capacità **in byte**, dimensioni in elementi. Eccezioni esplicite: array token/ID, capacità testuale e `decode_row` |
| B contro F, down 640/768 | `k_storage=768`, `k_logical=640`; B legge soltanto K logico da X. F passa una view con 640 colonne. Nessun obbligo di scratch padded né lettura di 768 float da 640 |
| Proiezioni non low-bit | B possiede anche GEMV/GEMM F32/F16/BF16/Q8_0. Nessun decoder/linear alternativo in C/D/E/F |
| HC: docstring pack/4 contro codice ×4 | Runtime sui pesi GGUF già convertiti: `SiLU(down(N)/H)`, media su H, `2 sigmoid(inject(N)/H)`. Nessuna riscalatura dei pesi al load |
| Norm e GDN | Gain GGUF diretto, nessun +1; `ssm_a` già `-exp(A_log)`; head GGUF tiled, associazione `h % Hk` |
| Q/gate QSA | `[T,H,2,D]`, interleaved per testa; non split globale Q/gate |
| QSA top-k | Budget in token, K divisibile per ratio; score non scalato, FP32, tie ID minore, blocchi selezionati serializzati in ordine temporale |
| Precisione | Attivazioni, cache QSA, storia e stato GDN FP32. Nessun half solo in prefill; FP64 solo negli oracle |
| PLE | Layer 1 zero-based; EOS hash 248044 distinto dallo stop chat; C possiede conv dilatata e norme; E produce Delta, il grafo somma R+Delta una sola volta |
| Routing MoE | Softmax FP32 su tutti gli esperti, top-k sulle probabilità, tie ID minore; `norm_topk_prob=true` dal profilo originale, non default Qwen3 |
| MTP | Inventariato e ancora compreso nelle viste storage esistenti; escluso dal grafo. Q4_K/MXFP4 non necessari al calcolo ordinario |
| Attach | Stesso `Gguf` vivo per describe→plan→attach; nuovo `GgufPlan.source` prova l'apertura. Non basta l'uguaglianza di shape/offset |
| Errori dopo mutazione | Nessun rollback implicito: invalidare tutta la sessione, output inutilizzabile, reset/replay |

HC è stato ricontrollato in `qwen4_ref_hc_mix` del checkout DwarfStar e nel
[GatedResidual ufficiale fissato](https://github.com/huggingface/transformers/blob/812a950d7037ba93bb624b154030e9068afdce4d/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).
Entrambi dividono la proiezione down e quella inject per H, e fanno la media
dei rami. Il commento contraddittorio del convertitore non cambia il runtime.
Questo risolve la formula; la parità numerica delle slice e il confronto
con i pesi originali rimangono verifiche distinte, non risultati del consolidamento.

## 3. Configurazione, layout e formule congelati

A è autorità di parsing e validazione, non delle costanti hardcoded nei kernel.
Profilo semantico iniziale `qwen38-q2-text-f32-v1`. Le dimensioni reali seguenti
sono controlli del checkpoint, non limiti delle fixture sintetiche:
E=2560, H=4 rami, rango HC=320, 48 trunk + 1 MTP;
36 GDN e 12 QSA (3,7,…,47). A registra origine GGUF o profilo verificato
per ogni campo non esportato, inclusi L2 eps=1e-6, SiLU, sigmoid,
`norm_topk_prob=true`, assenza bias e LM head non tied.
`Qwen38ConfigView`, `qwen38_config_layer` e `qwen38_config_ngram`
espongono dati const; gli array sono posseduti da A.

Matrici logiche W[out,in], GGUF `[in,out]`, asse 0 contiguo;
esperti `[Kphysical,N,experts]`, conv `[kernel,channels]`.
Attivazioni token-major, feature contigua. `Qwen38Tensor.storage` conserva
il descrittore fisico, `logical_shape` le dimensioni logiche nello stesso
ordine; assi inutilizzati=1. Ruoli `(layer, Qwen38Role)` risolti al setup,
nessuna ricerca per nome a ogni token. Ruoli globali con `QWEN38_GLOBAL_LAYER`.
MTP può essere descritto senza essere eseguibile; i ruoli extra MTP non
esposti dalla v1 restano classificati nel catalogo interno di A.

**Packed B.** Formati e formule di B §3, incluse tabelle IQ2_XXS e loro hash,
sono ratificati. Q2_K: blocco 256/84 byte, scale/min low/high nibble,
`w=(float(d)*scale)*q-float(dmin)*min`. IQ2_XXS: 256/66 byte,
griglia e segni ufficiali, non INT2. Q8_0: 32/34 byte, non q8_row.
Rank 1 è una riga; rank 2 una matrice; rank 3 una matrice per esperto.
Decode emette K fisico, matmul usa K logico: pad pesi finito nonzero non
influenza il prodotto. B non alloca scratch proporzionale a tutti gli esperti.
La conversione BF16 finita è widening bit-esatto; E può effettuare il semplice
`uint32(bits)<<16`/`memcpy` nel reader, senza introdurre un decoder packed.

**C.** Conv depthwise senza bias:
`u[t,c]=sum_j taps[c,j]*x[t-(K-1-j)*d,c]`, poi opzionalmente SiLU una volta.
Storia `[ (K-1)*d, C ]` oldest-first degli **input**; zero solo al reset.
GDN: Hk=16,Hv=48,Dk=Dv=128; stato canonico `[Hv,Dv,Dk]` (trasposto rispetto
all'oracle HF). Q/K: L2 con **somma** ed epsilon dentro radice;
Q riceve anche `1/sqrt(Dk)`. `h` usa key head `h%Hk`.
`log_decay=A*softplus(a+dt)`, `beta=sigmoid(b)`, `A<=0` già trasformata.
Sequenza: `Sd=exp(log_decay)*S`; `delta=beta*(v-Sd*k)`;
`S=Sd+delta*k^T`; `o=S*q`. Uscita: RMS per Dv, gain diretto e sigmoid(z).
Nessuna RoPE GDN. Log-decay deve essere finito non positivo; exp può
underflow a zero. Beta in [0,1]. Softplus/sigmoid stabili come C §3.4.

HC su R[T,H,E]: grouped RMS per E → N; lowrank/SiLU e sigmoid → G;
`mixed=mean_H(G*N)`. Inject calcolato sul medesimo N;
`R += inject[T,H] * block_out[T,E]`. Mixer finale senza inject.
La GPU potrà fondere norme/attivazioni solo verificandole contro C;
le primitive isolate C mantengono un solo contratto.

**D.** Config Hq=24,Hkv=2,D=256,Hi=4,Di=128,r=4,K=2048,Rrot=64,
theta=10^7. Q/K/IQ normalizzate per testa con gain diretto.
RoPE parziale su Q/K/IQ e chiavi indexer aggregate, abbinamento half-split,
frequenza `theta^(-2*i/Rrot)`. Testo: tre coordinate MRoPE uguali alla
posizione assoluta; niente YaRN/multimodale.
Pool **IKraw** di r token → RMS → RoPE alla posizione br.
Visibili solo `n=floor((p+1)/r)` blocchi completi;
`score[b]=sum_h max(dot(IQ[h],C[b]),0)`. Selezionare min(n,K/r) blocchi,
espanderli e aggiungere tail `[r*n,p]`. Nessun futuro/duplicato/sink.
Primo token sparso p=2051, massimo 2051 token selezionati nel profilo reale.
GQA `kv_head=floor(q_head/(Hq/Hkv))`, score `dot(Q,K)/sqrt(D)`;
softmax unico sull'insieme, online tiled con (max,sumexp,weighted_sum),
gate sigmoid dopo la divisione finale. D restituisce `[T,Hq,D]`;
il grafo appiattisce e chiama B per Wout (corregge il confine sintetico di C).
Cache FP32 K/V, IKraw e chiavi complete; nessuna eviction.

**E.** Ordine N=3, 8 teste per ordine, head_dim=160, 16 righe concatenate
bigram poi trigram. Vettori uint64 verbatim dai metadati; per testa:
`offset + (XOR_j (uint64(context[j])*multiplier[j])) % head_vocab`.
Prodotti <=INT64_MAX per equivalenza HF; 320001446 righe logiche,
320001536 fisiche, 90 padding mai generate. EOS corrente usa ancora la
storia precedente; solo i token successivi tagliano a EOS/posizioni anteriori.
Storia ID newest-first; EOS non azzera conv né GDN/KV.
Tabella BF16 disk-only: base 44806635520, 320 byte/riga, 102400491520 byte.
Dup fd read-only/CLOEXEC e copia metadati, `fstat` prima dei batch;
immutabilità file è precondizione, fstat non è un checksum.
Dedup globale per job, intervalli di sole righe consecutive <=64 KiB,
loop pread short-read/EINTR, EOF/errori invalidano l'intero job.
Nessun mmap tabella/cache iniziale; nessun READY parziale.
PLE: RMS key/query per ramo; `a=dot(kn,qn)/sqrt(E)`;
`g=sigmoid(sign(a)*sqrt(max(abs(a),1e-6)))`, sign(0)=0.
`u=g*value`; `z=RMS(u,gamma_conv)`;
`Delta=u+SiLU(conv(z))`, dilatazione=N, storia 9 token consecutivi.
Il grafo esegue R+Delta senza un'ulteriore iniezione HC.

**F.** Router F32, softmax stabile su 512 esperti, K=10; confrontare
probabilità FP32, tie ID minore, rinormalizzare i selezionati.
Per esperto `down(SiLU(gate(X))*up(X))`, I=640 logico;
shared I=640 è un parametro distinto, Q8_0 senza pad a 768,
moltiplicato per sigmoid del proprio gate scalare F32.
Riduzione routed in ordine slot 0..K−1, poi shared. Nessun residuo,
normalizzazione extra, token drop, pruning o softmax fra rami HC.

**Testo A.** Tokenizer qwen35/NFC e special mask HF di A §3.5,
template byte-esatto identificato dal suo hash; niente BPE duplicato.
Stop 248046 e 248044; nessun BOS aggiunto implicitamente.
Byte/stringhe possono contenere NUL. Decode UTF-8 dopo concatenazione
dei byte, carry streaming <=3. ID inutilizzati non producono testo.
`qwen38_text.h` usa un albero JSON tipizzato e ordinato per messaggi/tools,
anziché una seconda rappresentazione che perderebbe assente/null/ordine.
Il renderer valida lo schema ufficiale testuale; multimodalità unsupported.
Nessun parser JSON generale o Jinja interpreter da aggiungere all'engine.
A verifica i numeri/escape/float contro `apply_chat_template` ufficiale.

## 4. API materializzate e regole comuni

Header pubblici, tutti proprietà del coordinatore:

| Header | Contratto / implementatore |
|---|---|
| `src/qwen38_types.h` | Status, errori, view, key/epoch e shape condivise |
| `src/qwen38.h`, `src/qwen38_text.h` | Config, ruoli/binding, tokenizer/template — A |
| `src/qwen38_packed.h` | Layout, decode, matmul CPU e encode GPU — B; bridge storage — coordinatore |
| `src/qwen38_ops.h` | Conv, stato, GDN, HC, norme e attivazioni — C |
| `src/qwen38_attention.h` | Plan/create/begin/cpu/finish/reset/destroy — D |
| `src/qwen38_ngram.h` | Hash puro, store/job e PLE CPU — E |
| `src/qwen38_moe.h` | Route indipendente e forward via B — F |
| `src/qwen38_gpu.h` | Span GPU, encoder/device e parola di errore condivisa |

Sono dichiarazioni implementative, **non stub che restituiscono successo**.
La v1 materializza tutte le interfacce CPU tra sessioni e l'encode packed;
per C/D/E/F le prime verifiche Metal sono harness dei rispettivi operatori.
I loro entrypoint pubblici di composizione GPU saranno aggiunti dal
coordinatore dopo i gate CPU, con queste stesse shape/precisione e il
command context comune. Le sessioni possono implementare kernel e glue
privato assegnato senza inventare un secondo ABI pubblico. Questo è un
ordine di consegna, non una decisione aperta su formule/layout/stato.

View: per una view non vuota verificare checked
`sizeof(float)+sum_i((dim[i]-1)*stride_bytes[i]) <= capacity_bytes`.
Rank 1–4, ultimo asse stride=4, ogni stride esterno almeno l'estensione
dell'asse interno, multiplo di 4. D richiede contiguità delle dimensioni
interne testa/feature; pitch token ammesso. Nei batch vuoti solo T è zero;
non sottrarre 1 da dimensioni nulle. Niente broadcast/stride negativo o
cast di puntatori CPU in buffer GPU. Offset GPU+span verificati separatamente
contro `MTLBuffer.length`; poi la view contro span.bytes, misurati dall'offset.
Dimensioni/config uint32 dopo controlli, byte/offset size_t/uint64,
narrowing verso kernel solo dopo admission. Input/output disgiunti salvo
alias esatto espressamente ammesso da C §4; nessun alias parziale.

`T=0`: no letture token/attivazioni, no I/O né avanzamento; validare comunque
config/shape/stato. D begin(0) produce un ticket no-op da finish, output vuoto.
E prepare(0) produce job no-op già COMPUTED; read è no-op, commit non cambia
storia/posizione. L'hash puro copia prev→next anche per T=0 (nessun consumo).
Le chiamate testo output=NULL/cap=0 misurano e restituiscono OK senza
mutazione, inclusi decoder append/finish. CAPACITY riporta needed completo
e lascia lo stato streaming invariato. needed esclude NUL, capacità encode
in token, decode/render in byte. Nessun risultato parziale consumabile.

Errori: Q38_SHAPE per dimensioni incompatibili; STRIDE per layout view;
FORMAT per payload/formato malformato; UNSUPPORTED per variante riconosciuta
ma non eseguibile; OWNER per identità estranea; STALE per epoch/job vecchio;
BAD_STATE per sequenza di chiamate/posizione non valida; NONFINITE per
NaN/Inf/overflow aritmetico; IO/EOF con errno e progresso salvati.
CAPACITY riguarda spazio caller insufficiente, OOM allocazione fallita.
Diagnostica opzionale, code uguale al ritorno, nessun prompt/payload nei messaggi.
Create fallito azzera gli out e libera parziali. Validazione strutturale
prima di scrivere output/stato. Errori numerici dopo mutazione invalidano
output ed eventuale stato: non recuperare consumando valori parziali.
Il parser GGUF esistente può ancora terminare la CLI con `die()`;
le nuove API non propagano quel comportamento per I/O runtime.

Ownership: A copia config/tokenizer/descriptors, accessor borrowed const;
weights borrow plan solo dopo attach. `GgufPlan.source` non viene mai
dereferenziato dal planner/store; A confronta il testimone durante attach
con il medesimo Gguf ancora aperto e immutato. Dopo attach E duplica fd;
si può chiudere il reader, ma non riutilizzarlo per un nuovo attach.
Plan/store e binding C restano vivi fino all'ultimo loro utilizzatore;
retained MTLBuffer non mantiene vivo un puntatore C al catalogo.
`qwen38_weights_cpu_span` restituisce byte residenti read-only e bounded
dal medesimo mapping, senza seconda copia, solo per record del catalogo.
Per test CPU senza Metal usare slice bounded in memoria del chiamante,
`Q38PackedCpuView`; nessun loader completo alternativo in E/F.

C: query size include metadata/payload/alignment; init usa backing caller,
non alloca, copia shape/key e azzera stato. Handle non copiabile per valore;
export canonico per oracle. `conv_state_copy` copia solo in backing separato
con stessa key/shape, senza alias: serve alla history candidata di E.
D: create possiede cache/workspace; ticket in metadata riusabile, un solo
append in volo; destroy/reset BUSY non liberano né riusano risorse.
E: store copia array/config e possiede fd; un job riserva una state/epoch;
workspace e job hanno lifetime distinto, borrowed rows fino a release.
E reset/destroy/close BUSY se ancora referenziati. Candidate conv usa C,
nessun memcpy di handle opaco. Reader/hash possono essere provati senza
PLE; commit non vuoto richiede COMPUTED e history candidata completa.
F: config copiata, byte dei pesi borrowed; un workspace per esecuzione;
CPU sincrona, niente allocazioni per token. Route usa logits e scratch
probabilità forniti dal caller, quindi parte senza B o create dei pesi.

## 5. Transazione di sessione e GPU

Identità comune `(model, layer, epoch)`, model è un token stabile, non da
dereferenziare. Epoch non zero, cresce al reset globale; overflow rifiutato.
Stati validi all'inizio p=0; append richiede start_pos=next_pos, stesso modello,
config e generazione, capacità sufficiente, nessun uso in volo.
Il grafo possiede il contatore pubblicato della sessione e lo aggiorna
**solo dopo il successo di tutti i layer e del completamento GPU**.

Ordine: embedding replicato H volte; per ciascuno dei 48 layer:
eventuale PLE R+=Delta → HC-attn → GDN oppure QSA → proiezione B → combine;
HC-ffn → MoE → combine. Fine: output-HC senza inject → LM head.
Per CPU C avanza lo stato locale quando la primitiva finisce; è provvisorio
rispetto al grafo. D finish ed E commit sono rimandati al successo globale.
Se qualunque fase fallisce dopo mutazioni, invalidare **tutta** la sessione,
anche stati già avanzati; drain GPU, reset completo e replay richiesti.
Una preflight globale senza mutazioni fallita può lasciare valida la sessione.
Non promettere rollback C solo perché E conserva una history candidata.

Reset globale: drain, nuovo epoch, azzera S/history C, azzera lunghezze D,
EOS nella storia token E e zero nella history conv E; invalida lease/output.
EOS in una sequenza non invoca questo reset. Niente rewind/truncate/MTP,
fork o modifica di un prefisso senza replay.

GPU: device/compute_encoder borrowed in `Qwen38GpuCommand`, medesimo device
per tutte le risorse; componenti fanno encode, mai commit/wait per token o
esperto. Encoder con ordinamento/dependency standard tra dispatch; niente
async queue/untracked hazards nella baseline. Risorse retained fino a
completion; contenuti degli scratch/slot immutati fino all'ultimo lettore.
Il coordinatore azzera una parola uint32 allineata prima del command buffer;
kernel atomic-OR NONFINITE/BOUNDS/STATE, nessun reset da singolo componente.
A fine comando controllare **sia** errore Metal **sia** parola; encoding OK
non è risultato numerico OK. Nessun successivo comando che muti stato viene
ammesso prima dell'esito precedente nella baseline sincrona. Nello stesso
comando gli output a valle possono essere parziali: un errore invalida tutto.
La parola di errore è in storage Shared leggibile dalla CPU dopo completion.
I descrittori host (size_t, puntatori, bool) non vengono copiati come struct
MSL: il glue prepara argomenti kernel fixed-width, con offset/allineamenti
verificati dal harness. Nessun layout host/Metal implicito.

## 6. Tolleranze e gate di accettazione

Congelate prima dell'implementazione, non misure già superate. Non allargarle
per ottenere PASS. Le soglie delle spec restano valide con le precisazioni
seguenti; controlli di finitezza, guards e shape sono sempre obbligatori.

| Componente/stadio | Gate |
|---|---|
| A testo, metadati, binding; B decode; E ID/BF16; indici/storia copiata | Esatti, con distinzioni ±0 di B §8 e conversione bit-esatta |
| B prodotto | Contro r FP64: `abs(y-r) <= gamma(2*K+8)*sum(abs(w*x)) + 2*ulp32(float32(r))`, `u=2^-24`, `gamma(n)=n*u/(1-n*u)`; K logico, dominio finito senza overflow/underflow come B |
| C conv/attivazioni/RMS | err_inf <=1e-5*max(1,maxabs(ref)) per riga/head |
| C HC post-proiezione | stessa metrica, 2e-5 |
| C GDN T<=1025 / streaming 8K–32K | stessa metrica, 5e-5 / 2e-4, output **e stato** |
| HC/GDN composti con proiezioni B, T<=1025 / streaming 8K–32K | err_inf locale <=5e-4 / 1e-3 volte max(1,maxabs(ref)); contributo, residuo e stato controllati separatamente |
| D prep / score / attention+gate | atol/rtol: 2e-5/2e-5; 2e-4/2e-5; 3e-5/2e-4 su fixture D |
| E gate / PLE Delta e z/history | gate abs<=1e-4, dot zero→0.5; PLE atol=rtol=2e-4 e relativo L2<=1e-4 se norma ref>=1 |
| F route / attivazioni | atol=2e-6, rtol=2e-5; somma route entro2e-6 se normalizzata |
| F Y sintetico / slice | F §8: atol2e-5/rtol2e-4 e NRMSE<=1e-4; slice NRMSE<=2e-4 **e** max<=5e-4*max(1,RMS(ref)) |

Il bound B sostituisce quello alternativo più largo proposto da F per le
proiezioni. Le norme comuni sono testate secondo C; D/E possono aggiungere
gate di composizione, senza cambiare il contratto della primitiva C.
C mantiene il gate 1e-5 per le attivazioni, e i casi moderati consumati da F
devono soddisfare anche il criterio elementwise più stretto di F.
Token/chunk C bitwise solo a identico algoritmo/ordine di riduzione;
composizione/GEMM seguono i gate numerici e confrontano tutti i confini.

Layer composti: i gate HC/GDN aggiunti sopra sono obiettivi iniziali più larghi
di quelli core per includere proiezioni e prodotti di gate; **non** sono una
conseguenza garantita della somma delle tolleranze. Valgono su fixture finite
bounded e slice con input dichiarati; nessuna garanzia per dati arbitrari
mal condizionati. Sono necessari **insieme** ai gate locali: ogni proiezione B
contro FP64 sullo stesso input; ogni operatore C/D/E/F contro oracle alimentato
dalle stesse proiezioni; replay del layer completo contro oracle e confronto
di contributo, residuo e stato (il residuo grande non può nascondere un errore).
D deve inoltre conservare gli insiemi selezionati; F le route, E ID e storia.
I casi stress fuori scala richiedono bound analitici predefiniti nel manifest,
non tolleranze tarate dopo un fallimento. Le soglie qui non si trasferiscono
ai logits dell'intero backbone: quel gate appartiene al futuro incremento
del grafo, con evidenza delle 48 layer. La BASE non certifica qualità finale.

QSA/MoE: tie nativi su identici FP32 esatti; differenze HF nei tie non
promettono identità ID. Per near-tie registrare margine al confine e errore
score/logit, riprodurre su score FP32 identici; gap>2*errore è una verifica
di stabilità. Gap insufficiente non dà un PASS end-to-end: conservare caso,
maschera/route, decisione numerica esplicita prima dell'integrazione.
PLE signed-sqrt vicino a zero: misurare dot e segno prima del gate,
non nascondere un'inversione con la scala del residuo.

Ogni componente: oracle upstream indipendente + casi analitici → C11 con
ASan/UBSan → Metal con API validation → slice Q2 → composizione e stato.
Oracle e runtime devono usare gli stessi pesi Q2 dequantizzati; ciò non
misura la perdita Q2 contro BF16 originale. Catture di attivazioni autentiche
e confronto logits sono gate del coordinatore, non già disponibili.

## 7. Ownership esclusiva dei file

Ogni file elencato ha un solo proprietario. Gli header pubblici precedono
qualunque pattern di componente. Tutti gli altri file esistenti non elencati
restano al coordinatore; nessuna sessione li modifica per comodità.

| Owner | File / perimetro esclusivo |
|---|---|
| A | `src/qwen38_{config,weights,tokenizer,chat,unicode}.c`; **`src/tokenizer.c` assegnato ad A**, preservando default Qwen3; `src/qwen38_a_internal.h`; `tools/gen_qwen38_unicode.py`; `tools/oracle/qwen38_{config,text}.py`; `tools/synth/make_qwen38_a.py`; `tests/test_qwen38_{config,bindings,text}.c`, `tests/test_qwen38_a.py`; `tests/fixtures/qwen38/a/**`; `docs/specs/qwen38/A.md` |
| B | `src/qwen38_packed.c`, `src/qwen38_packed_gpu.m`, `src/qwen38_packed.metal`, `src/qwen38_iq2_grid.inc`, `src/qwen38_b_internal.h`; `tools/oracle/qwen38_packed.py`, `tools/oracle/qwen38_packed_reference.c`; `tests/test_qwen38_packed.{c,py}`, `tests/test_qwen38_packed_gpu.m`; `tests/fixtures/qwen38_packed/**`; `docs/specs/qwen38/B.md` |
| C | `src/qwen38_{conv,gdn,base}.c`, `src/qwen38_recurrent_gpu.m`, `src/qwen38_recurrent.metal`, `src/qwen38_c_internal.h`; `tools/oracle/qwen38_recurrent.py`; `tests/test_qwen38_recurrent.{c,m,py}`; `tests/fixtures/qwen38/c/**`; `docs/specs/qwen38/C.md` |
| D | `src/qwen38_attention.c`, `src/qwen38_attention_gpu.m`, `src/qwen38_attention.metal`, `src/qwen38_d_internal.h`; `tools/oracle/qwen38_attention.py`; `tests/test_qwen38_attention.{c,m,py}`; `tests/fixtures/qwen38/attention/**`; `docs/specs/qwen38/D.md` |
| E | `src/qwen38_ngram.c`, `src/qwen38_ngram_internal.h`, `src/qwen38_ple.c`, `src/qwen38_ple_gpu.m`, `src/qwen38_ple.metal`; `tools/oracle/qwen38_ngram.py`; `tests/test_qwen38_ngram.{c,py}`, `tests/test_qwen38_ple.m`; `tests/fixtures/qwen38/ngram.json`; `docs/specs/qwen38/E.md` |
| F | `src/qwen38_moe.c`, `src/qwen38_moe.m`, `src/qwen38_moe.metal`, `src/qwen38_f_internal.h`; `tools/oracle/qwen38_moe.py`; `tests/test_qwen38_moe.{c,m,py}`; `tests/fixtures/qwen38/f/**`; `docs/specs/qwen38/F.md` |
| Coordinatore | Tutti gli header pubblici Qwen38; `src/{picoforge,gguf}.h`, `src/gguf{,_reader,_plan}.c`, `src/gpu_weights.m`, `src/metal.m`, `src/kernels.metal`; Makefile, entrypoint/grafo, file Qwen3 salvo tokenizer.c assegnato; `tests/test_gguf_plan.c`, `tests/test_gpu_weights.m`, `tests/test_qwen38_gpu_lock.py`, `tools/qwen38_gpu_lock.py`; `integration.md`, README/DESIGN/BENCHMARKS, report aggregati |

Le parentesi graffe nella tabella sono espansioni di nomi esatti. File nuovi
non coperti richiedono assegnazione del coordinatore; output temporanei
`build/qwen38-<lettera>/` appartengono al worktree e sono ignorati.
Modifiche a un contratto richiedono proposta concreta al coordinatore,
che aggiorna header/documento/versione e comunica un commit additivo.
Non riscrivere storia o usare sei varianti dello stesso header.
A può dichiarare il raccordo BPE fra i propri `.c` nel privato
`qwen38_a_internal.h`; questo non richiede modifiche concorrenti a picoforge.h.

## 8. Dipendenze, incrementi autorizzati e ordine di integrazione

Tutte le sessioni possono ripartire da questa base con oracle/fixture CPU
già salvati separatamente e adattare soltanto il proprio codice.

| Sessione | Subito, indipendente | Dopo dipendenze reali |
|---|---|---|
| A | Config/validazioni, catalogo e binding, tokenizer semplice → template/tools con oracle ufficiale | Attach/byte-span con storage; regressioni tokenizer Qwen3 con coordinatore |
| B | Layout + decoder Q2_K e IQ2_XXS, F32/F16/BF16/Q8_0; matmul CPU FP32 e padding/stride | Encode Metal stesso contratto sotto lock; slice bounded e confronti con C/D/E/F |
| C | C0 oracle, C1 conv/storia dilatata, C2 norme/attivazioni/HC, C3 GDN prep/scan/gate | C4 proiezioni reali dopo A+B; C5 kernel Metal per stadio; C6 composizione col coordinatore |
| D | Oracle, RoPE/indexer/selettore/softmax su fixture proiettate; lifecycle cache CPU | Prep completo riusa RMS C2; slice/proiezioni dopo A+B; Metal conservativo dopo parità CPU |
| E | Oracle, hash puro, piano I/O e complete-read/fault injection, gate su key/query normalizzate | History tramite C1; PLE completo dopo C2+B Q8_0+A; staging/kernel Metal e transazione globale |
| F | Oracle e routing da logits, tie/stride/errori; grouping come permutazione di pair | MoE F32 dopo B scalar+C attivazioni; packed dopo IQ2_XXS/Q2_K/Q8_0; Metal senza duplicare decoder |

Priorità B per sbloccare gli altri: primo commit decoder Q2_K se già avviato;
poi **scalar/F32/F16/BF16 e Q8_0 prima di ottimizzare IQ2_XXS**;
completare IQ2_XXS corretto prima di MoE reale. Nessuna sessione aspetta una
GEMM ottimizzata per iniziare il core su fixture già proiettate.
Un'assenza di dipendenza blocca solo l'incremento composto, non autorizza stub
o una seconda implementazione dell'operatore altrui.

Ordine di integrazione del coordinatore, su branch dedicato:
1. A config/catalogo + B CPU scalar/packed + C conv/norme, ciascuno verificato.
2. C GDN/HC, D core QSA, E hash/reader/gate, F routing su fixture indipendenti.
3. B linear + A binding → GDN/HC, QSA, PLE e MoE CPU su slice, uno alla volta.
4. Kernel/glue Metal per stadio, header pubblici di composizione GPU e build
   della metallib aggiunti dal coordinatore con i rispettivi test verdi.
5. Grafo ordinario 8K/chunk1024/una sessione, commit/errori/reset/append,
   intermedi/logits e regressioni Qwen3; tokenizer/template A prima del testo.
6. Misure qualità/memoria/prestazioni; solo dopo 16K/32K, precisioni ridotte,
   fusioni, I/O concorrente e MTP. Nessuna prestazione implicita dai test core.

Il Makefile accetta ora `PYTHON=...` anche dai worktree. I target specifici
menzionati nelle spec A–F restano **da aggiungere insieme alle implementazioni**,
non target già funzionanti. Finché il relativo target non esiste, ogni
sessione compila il proprio harness CPU con clang C11, warning del progetto,
`-ffp-contract=off -fsanitize=address,undefined -Isrc`, e documenta il comando.
Non modificare Makefile in parallelo. Per nuovi file Metal compilare AIR
esplicita nel build locale, correttezza senza fast-math, e chiedere al
coordinatore il link nella metallib; non assumere inclusione automatica.
`make` oggi usa wildcard C/ObjC ma una sola sorgente Metal: per questo
compilare il harness isolato prima del wiring globale. Registry pipeline
attuale da 32 voci: contare i kernel e adeguare nel relativo incremento.

## 9. Lock effettivo, comandi e budget

**Un solo lock per tutti i worktree di questa repository**:
`/Users/cristiancirje/Desktop/Dev/picoforge/.git/qwen38-gpu.lock`.
Il wrapper risolve `git rev-parse --git-common-dir`, canonicalizza il path,
usa `fcntl.flock(LOCK_EX)` e conserva il fd attraverso exec del comando.
Mai unlink, rename o cancellazione per presunto PID stale: il kernel rilascia
il lock alla chiusura dell'ultimo fd. Timeout → exit75, exec fallito →126,
altrimenti stesso exit status del comando. Nested wrapper rifiutato.
È un lock cooperativo: processi lanciati fuori wrapper non vengono fermati.
Usarlo per **ogni** test GPU, DwarfStar, benchmark e workload pesante sul
checkpoint. Solo piccoli oracle CPU e test bounded possono procedere senza.
Nessun altro clone indipendente può usare un proprio lock e dirsi coordinato.

```sh
cd /Users/cristiancirje/Desktop/Dev/picoforge-qwen38-consolidate
Q38_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
"$Q38_PY" tools/qwen38_gpu_lock.py --print-path
make PYTHON="$Q38_PY" test-qwen38-contracts test-gguf
"$Q38_PY" tools/qwen38_gpu_lock.py --timeout 60 -- \
  make PYTHON="$Q38_PY" test-gpu-weights
```

Avvolgere la suite esterna una volta sola, comando in foreground;
niente daemon/background o chiusura del fd `PICOFORGE_GPU_LOCK_FD`.
Il wrapper usa exec: il PID indicato è quello del comando, senza processo
supervisore che possa morire lasciando libero il lock col figlio attivo.
Un processo che crea figli deve attenderli; non aggirare il lock eliminando
le variabili ereditate. Non è una sandbox contro processi non cooperanti.

Budget runtime iniziale (pesi esclusi dalle singole righe): A header128MiB,
catalogo4MiB, tokenizer64MiB, scratch testuale<=256MiB; C stato+scratch<=512MiB;
D cache444MiB a8K + scratch<=192MiB; E<=256MiB; F+B scratch<=32MiB con
microbatch<=32. B CPU non dequantizza un modello; workspace GPU conteggiato
nel componente chiamante, non nascosto. Conteggiare una volta R/scratch
condivisi. Pesi nelle viste esistenti 44795625472 byte inclusi MTP;
mapping virtuale non è picco fisico misurato. Non produrre logits `[1024,V]`
senza budget esplicito: baseline LM head sul token necessario.
Somma delle allocazioni disgiunte e misure footprint/Metal/pressione/swap
devono rispettare l'obiettivo48–50GiB e i limiti del device; questi massimi
componenti non dimostrano da soli il fit globale.
Non modificare wired-memory, SDK o modelli per far passare un test.
Riutilizzare il venv e checkpoint principali in sola lettura; fixture/slice
nel build locale con revisioni/hash e limiti delle rispettive spec.

## 10. Verifica del consolidamento e consegna

Eseguiti: controllo provenienza di tutti i commit e spec byte-identiche;
header singoli C11 e inclusione congiunta Objective-C con warning come errori;
test CPU lock su due worktree di una repository temporanea: contesa/timeout,
rilascio dopo uscita/crash, exec fallito, rifiuto nesting, inode stabile;
test GGUF con ASan/UBSan; test sintetico viste pesi sotto lock e Metal API
validation, inclusi byte CPU/GPU, owner estraneo, esclusione tabella e lifetime.
Nessun oracle delle sessioni importato; nessuna inferenza sul checkpoint.
`make test-all` non è il gate di questo commit di contratti: richiede asset
Qwen3 e lancia workload ulteriori; sarà eseguito al wiring dei componenti.
I target sopra sono ripetibili dal worktree senza nuovo venv.
Passati anche i 20 test delle utility della baseline (`test-qwen38-tools`)
e la build completa `make -j4` di C11, glue Objective-C e metallib esistenti.

Ripartenza sicura: creare un nuovo branch/worktree da BASE_CONCORDATA,
portare soltanto commit oracle propri già completi oppure committarli prima
nel worktree di origine. **Non fare reset/stash indiscriminato/rebase di
working tree sporchi e non cherry-pickare di nuovo la spec già acquisita.**
Ogni consegna ESEGUI riporta SHA, file modificati, comandi/esiti, copertura
sintetica o slice reale e dipendenze ancora necessarie. Il coordinatore
integra incrementi verificati; le sessioni non fanno merge/push su main.

## 11. BASE_INTEGRATA v1 — 2026-10-12

Questo aggiornamento sostituisce soltanto lo stato implementativo e i comandi
storici dei §8–10; formule, ABI, precisione, ownership ed errori dei §2–7
restano vincolanti. Branch `coord/qwen38-integrate`, worktree esterno
`/Users/cristiancirje/Desktop/Dev/picoforge-qwen38-integrate`, partito da
BASE_CONCORDATA `c674b23bd1377b2ce0acef2d06fac8b160809446`.
Il commit finale con questo aggiornamento è BASE_INTEGRATA; il coordinatore
ne restituisce lo SHA completo. Nessun merge/push su main.

Acquisiti con merge espliciti, senza conflitti, i seguenti head completati:

| Consegna | Branch | SHA acquisito |
|---|---|---|
| A | `impl/qwen38-a-v1` | `01b2333ba1ee284f9214758de97a8b1452b4781f` |
| B | `impl/qwen38-b-cpu` | `ee871bc3718898e0cee39e12fe6f678579ba9499` |
| C | `impl/qwen38-c` | `9b03f2554536745aed787371bcd2e54e3321f085` |
| D | `impl/qwen38-d-cpu` | `d9b148ae8d4d9b0c1efa12b746a76b3fb5bc491b` |
| E | `impl/qwen38-e-v1` | `b3321fb75266c7e44daa3a5b5b1a35a06fcfd222` |
| F | `impl/qwen38-f-v1` | `a2c78ba9002d75dcf636ec8b77b08ecd55e36cf7` |

Le modifiche delle sei consegne non hanno file in comune; nessun header
pubblico riscritto e nessun oracle non committato acquisito. Gli oracle già
contenuti nella storia di questi branch non vanno cherry-pickati di nuovo.
I rispettivi documenti A–F conservano le consegne originali: le vecchie
indicazioni “dipendenza assente” vanno lette insieme a questa sezione.

Disponibili: A config/catalogo/testo; B scalar e packed CPU/Metal in tutti i
sei formati; C conv/stato, RMS, attivazioni, HC, GDN e kernel per stadio;
D lifecycle e core privato su input preparati; E hash, reader privato e gate;
F routing e grouping. I 59 simboli pubblici implementati e le 23 dichiarazioni
ancora senza implementazione sono elencati nel report API. Non ci sono stub
che restituiscano successo per sostituire queste ultime.

**Build e verifica.** `tools/qwen38.mk` aggiunge target CPU/GPU espliciti.
La metallib comune collega ora AIR legacy, B e C; B/C usano
`-fno-fast-math -ffp-contract=off`, come gli oggetti C11 Qwen38. Registry Qwen3
invariato; pipeline B/C private. Harness nativi ASan/UBSan con errori fatali;
la dylib C caricata da Python resta senza sanitizer, affiancata dal harness.
I test ricevono sempre native harness, sorgente fissato e slice richieste;
`tools/qwen38_test.py` rende uno skip un errore di processo. I report B/C usano
directory nuove a ogni run, senza sovrascrivere l'evidenza precedente.

Correzioni d'integrazione: il test header controlla quelli pubblici, mentre
quelli privati vengono compilati nei harness; C rileva anche alias tra due
oggetti Metal no-copy sulla stessa memoria Shared, incluso l'error word.
Un test ha riprodotto il difetto prima della correzione e passa dopo.
La regressione shard Qwen3 ora usa `sys.executable` per i subprocess Python,
rispettando il venv scelto da Makefile anche nei worktree senza venv locale.
Alias esatto in-place C è ammesso usando lo stesso oggetto buffer e offset;
wrapper distinti con memoria sovrapposta sono rifiutati conservativamente.
Nessuna tolleranza è stata allargata.

Nuovi file del coordinatore: `tools/qwen38.mk`, `tools/qwen38_prepare_tests.py`,
`tools/qwen38_test.py`, `tests/test_qwen38_integration.c`,
`tests/test_qwen38_test.py`, `docs/specs/qwen38/resume.md` e
`bench/qwen38_integration_20261012/**`. Le correzioni locali in
`src/qwen38_recurrent_gpu.m`, `tests/test_qwen38_recurrent.m` e il raccordo
strict/sanitizer in `tools/synth/make_qwen38_a.py` sono manutenzione di
integrazione del coordinatore; ownership futura dei file resta C/A.

**Verificato:** 158 test CPU A–F (20/25/35/40/15/23), nessuno saltato;
harness ASan/UBSan, 4 test lock, controllo strict runner e test API collegate
A→B→F, C→E, A→D/E. B: 25 test Metal, 245 prodotti per backend, 137 slice
reali e 400164 byte packed complessivi. C Metal: primitive, errori/stato,
fixture lunghe fino a 32768 con head ridotte e T=1/65 a shape GDN reale;
max errore scalato 3.60608101e-5 entro i gate congelati. Il picco osservato
87130112 byte è allocazione Metal del harness, non footprint del modello.
A storage: piccolo GGUF sintetico, senza mapping completo del checkpoint.
Dettagli, comandi, regressioni ed evidenza nel
[report integrato](../../../bench/qwen38_integration_20261012/README.md).

**Ripartenza:** C4 può usare A+B per HC/GDN composto; D può implementare
`qwen38_attention_cpu` riusando RMS/attivazioni C; E può completare
store/state/job e PLE con C1/C2+B; F può implementare create/work/forward
con B+C, prima floating point poi packed. Prompt e ordine in
[resume.md](resume.md). Questi quattro incrementi CPU sono indipendenti
fra loro sulla BASE_INTEGRATA. Non attendere gli altri branch per iniziare.
Gli entrypoint pubblici GPU composti, nuovi link/target Metal D/E/F e grafo
restano al coordinatore; i test GPU privati possono avanzare dopo i gate CPU.

**Limiti:** nessun HC/GDN composto su proiezioni Q2 reali, prep pubblica D,
PLE completo o forward MoE nativo ancora certificati; nessun grafo 48 layer,
transazione globale GPU, logits Qwen3.8 o generazione nativa. Le prove sono
su primitive, core e collegamenti sintetici, più slice B; non misurano la
perdita Q2 rispetto al BF16 originale né prestazioni end-to-end. Il lock
rimane cooperativo e condiviso tramite il git-common-dir del repository.

## 12. BASE_INTEGRATA_V2 — 2026-10-12

Questo stato sostituisce disponibilità, limiti e comandi operativi del §11;
formule, ABI, precisione, tolleranze e ownership dei §2–7 rimangono invariati.
Base di partenza `6a6b886e985185d7c9104eda888e6a406553aedf`, nuovo branch
`coord/qwen38-integrate-v2`, worktree esterno `../picoforge-qwen38-integrate-v2`.
Il commit finale contenente questo aggiornamento è BASE_INTEGRATA_V2;
lo SHA completo viene restituito nella consegna. Nessun merge/push su main.

| Consegna acquisita | SHA completo |
|---|---|
| `impl/qwen38-c-v2` | `8a65af3beeb48eb046d0cbc95b522b3dba396959` |
| `impl/qwen38-d-v2` | `fcacfbcdef24f6502beb815caca1cf5a87378d0a` |
| `impl/qwen38-e-v2` | `4257c9c156d29dcf35acbcf5b7239122a917d963` |
| `impl/qwen38-f-v2` | `aa96e33fff92cd2f02064fde4adba187cd831d90` |

Quattro merge espliciti senza conflitti, file delle consegne disgiunti,
header pubblici invariati. Acquisiti solo commit completati; gli altri
worktree restano intatti. Il binario combinato definisce tutti gli **82
simboli pubblici** censiti nella v1: le 23 dichiarazioni CPU allora mancanti
sono implementate. Questo controllo di link non certifica un grafo GPU.

**Disponibilità.** C compone HC/GDN con A+B, 18 intermedi, stato conv/GDN,
residuo e prove CPU/Metal su fixture sintetica e pesi reali layer 0; controlli
reali separati 0/1/46. D offre `qwen38_attention_cpu` completo su proiezioni,
e stadi Metal prep/RoPE/pool/score/attention con selezione ancora CPU.
E offre store/state/job, history candidata e PLE CPU completo con Delta;
nessun PLE Metal consegnato. F offre create/work/forward CPU, route/grouping,
confronti sui layer reali 0/3/47 e post-routing Metal su fixture sintetiche
floating point/packed: routing e grouping restano CPU nel harness Metal.

**Wiring.** Oggetti C `src/nome.o` e Objective-C `src/nome.m.o` distinti:
`qwen38_moe.c` e `.m` non si sovrascrivono. Aggiornati anche i riferimenti
legacy `metal.m.o`/`gpu_weights.m.o`. La metallib comune collega AIR legacy,
B, C, D e F; pipeline Qwen38 private, registry Qwen3 invariato. Tutti i
kernel Qwen38 e sorgenti numerici usano FP32/no-fast-math/no-FMA contraction.
Harness ASan/UBSan fatali; dylib ctypes non sanitizzata affiancata dai harness.

La composizione C introduce dipendenze A/B/GGUF anche per D/E/F: i target
includono il gruppo CPU comune completo. Il harness E auto-compilato include
GDN e planner GGUF. Entrambi i percorsi F packed usano la stessa reference
indipendente configurabile; non è più richiesto duplicarla nella cartella F.
D CPU seleziona le due classi CPU, D Metal la classe Metal con tutti gli asset;
F Metal seleziona i quattro test post-routing supportati. I test CPU F restano
28 inclusa la slice reale, non vengono esclusi per far passare la suite GPU.
Il runner strict rifiuta qualsiasi skip nei test selezionati.

**Ownership di integrazione.** Coordinatore: `tools/qwen38_v2.mk`,
`tools/qwen38_prepare_compositions.py`, `tools/qwen38_compositions.py`,
`docs/specs/qwen38/resume-v3.md`, `bench/qwen38_integration_v2_20261012/**`,
oltre ai file build/contratti già assegnati. Le correzioni di collegamento in
`tests/test_qwen38_ngram.py` e di percorso reference in
`tests/test_qwen38_moe.py` sono manutenzione d'integrazione; ownership futura
resta E/F. Nuovi file D/F acquisiti restano assegnati ai rispettivi componenti.

**Preparazione ripetibile, esplicita e separata dai test.**

```sh
Q38_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
"$Q38_PY" tools/qwen38_gpu_lock.py --timeout 60 -- \
  make qwen38-test-assets PYTHON="$Q38_PY" \
  Q38_SOURCE_DIR=/Users/cristiancirje/Desktop/Dev/picoforge-qwen38-integrate/build/qwen38-f/upstream
make qwen38-composition-assets PYTHON="$Q38_PY"
make test-qwen38-cpu PYTHON="$Q38_PY"
make test-qwen38-gpu PYTHON="$Q38_PY"
make test-qwen38-compositions PYTHON="$Q38_PY"
```

`make test-qwen38` esegue in ordine CPU, GPU, composizioni. Non avvolgerlo
in un altro lock: i target pesanti lo acquisiscono già. `test-all` legacy
resta separato e va avvolto una volta. Preparazione online solo su richiesta
esplicita omettendo `Q38_SOURCE_DIR`; i test sono offline. I receipt delle
fixture C/D/F controllano hash di input/binary e dati; fixture parziali o
incompatibili falliscono senza sovrascrittura. Nessun nuovo checksum globale
del GGUF, mapping completo o copia della tabella PLE. C/D/E/F usano letture
limitate; la preparazione e le composizioni reali condividono il medesimo
lock nel git-common-dir. I run di verifica hanno directory report nuove.

**Evidenza e limiti.** Esiti, metriche, comandi, provenance e regressioni in
[report v2](../../../bench/qwen38_integration_v2_20261012/README.md).
169 test Python CPU A–F (20/25/35/44/17/28), oltre alle composizioni e ai gate
Metal separati; nessuno skip equivale a parità. Attivazioni delle slice reali
sono sintetiche: pesi reali non significano attivazioni catturate da prompt.
Restano da costruire lifecycle GPU comune, PLE Metal, selezione QSA/routing
MoE su device e grafo ordinario 48 layer con embedding/mixer finale/logits.
Nessuna generazione Qwen3.8 nativa, perdita Q2/BF16, misura di fit 48–50 GiB
o prestazione end-to-end è certificata dai harness componenti.

**Ordine successivo:** contratto/lifecycle comune G0; adattatori C/E e
selezione D, prodotto indicizzato B e routing F; grafo GPU e infine checkpoint
completo. E Metal privato, selezione D, routing F e grafo CPU bounded possono
iniziare subito. Ownership, dipendenze e incarichi concreti in
[resume-v3.md](resume-v3.md). Non ripartire dai prompt v2 storici.

## 13. BASE_GPU_CONTRATTI — G0, lifecycle GPU comune (2026-10-12)

Parte da BASE_INTEGRATA_V2 (`a84c249`), branch `coord/qwen38-gpu-lifecycle`,
worktree `../picoforge-gpu-lifecycle`. Il commit che introduce questa sezione
è BASE_GPU_CONTRATTI; lo SHA completo è nella consegna. Formule, ABI dei §2–7,
precisione, tolleranze e ownership restano invariati: le dichiarazioni aggiunte
a `src/qwen38_gpu.h` sono solo additive. Nessun merge/push su main.

**Contratto.** `Qwen38GpuSession` possiede coda, command buffer e parola
d'errore; i componenti ricevono soltanto il `Qwen38GpuCommand` restituito da
`qwen38_gpu_session_begin` e codificano, senza commit/wait. Sotto c'è
`Qwen38GpuLedger`, pura contabilità senza Metal, testabile da sola.

| Passo | Effetto |
|---|---|
| `begin(model,start,count)` | Ammette solo da IDLE/VALID con `start==next_pos` e capacità sufficiente (T=0 ammesso, non pubblica nulla); azzera **la** parola d'errore; apre l'encoder; ticket `(model,epoch,start,count)`. |
| `claim(ticket,span,READ/WRITE)` | Solo in ENCODING. Trattiene il buffer fino a fine comando; rifiuta ogni sovrapposizione con un claim WRITE, parola d'errore inclusa, anche tra viste no-copy dello stesso indirizzo; READ/READ ammesso (pesi condivisi). |
| `submit(ticket)` | Validato dal ledger **prima** del commit: un secondo submit non raggiunge mai Metal. Chiude l'encoder. |
| `abort(ticket, mutated)` | Solo prima di submit. `mutated=false` lascia la sessione com'era; `true` la invalida. Nessun commit. |
| `finish(ticket)` | Attende, legge errore Metal **e** parola d'errore; solo se entrambi puliti pubblica `next_pos`. Altrimenti INVALID e `Q38_GPU_ERROR` (detail: bit e testo Metal). Rilascia i claim solo qui. |
| `reset(model)` | Non con comando aperto (BUSY). Nuovo epoch (overflow rifiutato), `next_pos=0`: ogni ticket precedente diventa STALE. Unica uscita da INVALID. |
| `destroy` | BUSY lascia vivi sessione e risorse. |

Un errore dopo scritture GPU invalida l'intera sessione: nessun rollback
implicito, nessuna history parziale considerata valida. Per i componenti:
dopo `finish == Q38_OK` il grafo chiama i `finish(ticket, true)` di D/E e
avanza lo stato di C; in ogni altro caso `finish(..., false)` e replay dopo
`reset` con l'epoch di `qwen38_gpu_session_info`. Il grafo non pubblica
posizioni per conto proprio. IDLE = creata/resettata, VALID = ultimo comando
riuscito; un comando T=0 non cambia mai lo stato.

**Verifica.** `make test-qwen38-g0 PYTHON=<venv>` (lock comune, Metal API
validation, ASan/UBSan fatali, `-Werror`; incluso in `test-qwen38-gpu`).
11 gruppi: 6 sul ledger (posizioni/capacità, T=0, guasto prima del submit,
errore dopo le scritture, ticket/owner/doppio submit e finish, epoch) e 5
sulla sessione Metal (un kernel che scrive e poi segnala errore: le scritture
arrivano, nulla viene pubblicato, il replay dopo reset parte pulito; abort;
BUSY e stale; alias; lifetime dei buffer). Ogni regola è stata anche rotta di
proposito nell'implementazione: 19 mutanti su 20 scoperti dal test; l'unico
sopravvissuto è il limite (1) qui sotto. Il test ha anche trovato un errore
mio, `contents` chiesto a un buffer privato. Log in `bench/qwen38_gpu_lifecycle_20261012/test.log`.
`picoforge` e tutti gli harness dei componenti si ricompilano e collegano
con il nuovo header; `test-qwen38-contracts` passa.

**Limiti.** (1) Il ramo "errore Metal del command buffer" non è riprodotto: non
si innesca un guasto reale senza rischi per la macchina condivisa; la sua
conseguenza è coperta a livello ledger (`command_ok=false`), ma la riga che
calcola `ok` dal command buffer resta non esercitata (un mutante che la
sostituisce con `true` sopravvive). (2) Il rifiuto di un buffer di un altro
device non è testabile su un Mac con una sola GPU. (3) I claim non impediscono
a una CPU di scrivere in un buffer Shared mentre la GPU lavora: lo scratch
immutabile fino all'ultimo lettore resta un obbligo del chiamante. (4) Gli
alias tra buffer Private sono rilevati solo dentro lo stesso buffer; gli alias
via `MTLHeap` non sono modellati. (5) I componenti C/D/E/F **non** sono ancora
collegati alla sessione: G0 offre il contratto, non gli adattatori. (6) La suite
completa v2 (`make test-qwen38`: 169 test CPU A–F, gate GPU separati,
composizioni su pesi reali) è stata rieseguita sulla base G0, con asset
preparati dal checkpoint, senza skip e senza tolleranze allargate: tutto PASS
in circa 7 minuti, log in `bench/qwen38_gpu_lifecycle_20261012/full-suite.log`.
Non rieseguito: `make test-all` legacy sul Qwen3, che G0 non tocca (il motore
`picoforge` si ricollega con il nuovo oggetto).

**Ordine successivo:** adattatori C/E sul ticket comune, selezione D, prodotto
indicizzato B e routing F come in [resume-v3.md](resume-v3.md); poi G2.
