# Prompt per la nuova sessione: Qwen3.8-Flash-Next Q2

Lavora nella repository `/path/to/picoforge`.
Leggi questo documento come il passaggio di consegne della sessione precedente.
La ricerca è stata svolta l'11 ottobre 2026. Verifica lo stato attuale della
repository e dell'ambiente prima di agire; non assumere che siano immutati.

## Decisione e obiettivo

Ho scelto **Qwen3.8-Flash-Next Q2** come nuovo obiettivo di picoforge.
L'uso principale è **coding e ragionamento generale**, anche in italiano.
Il computer è un **MacBook Pro M5 Pro con 64 GiB di memoria unificata**.

Voglio eseguire e ottimizzare un modello sostanzialmente più grande di quello
che entra a precisione originale, attraverso compressione, gestione della
memoria, kernel Metal e un inference engine nostro in C11. Il riferimento
ingegneristico è il lavoro di antirez in `antirez/ds4`, chiamato DwarfStar.

Il risultato deve essere un engine nativo utilizzabile, con qualità e
prestazioni misurate. DwarfStar serve per baseline, studio e confronto;
il risultato finale non deve limitarsi a invocarlo come processo esterno.
Anche questo checkpoint è già eseguibile in DwarfStar: il nostro vantaggio
rispetto ai runtime esistenti va dimostrato, non dato per scontato.

Abbiamo valutato Qwen3-Coder-Next 80B Q4 e altri modelli, ma la scelta finale
è Qwen3.8 Q2. Non riaprire la selezione del modello senza una nuova evidenza
concreta di problemi di qualità, memoria o fattibilità. Il 30B esistente
rimane un riferimento per regressioni; l'80B è un eventuale confronto futuro.

## Istruzioni di lavoro

- Comunica in italiano, in modo concreto, con aggiornamenti brevi durante
  le attività lunghe. Non fermarti a un piano: esegui il lavoro autorizzato.
- Parti dalle istruzioni applicabili del repository e leggi i documenti
  sotto indicati. Le vecchie roadmap sul 30B non prevalgono su questa scelta.
- Preserva modifiche locali, file non tracciati, staging e lavoro delle altre
  sessioni. Non fare reset, clean, checkout distruttivi, stash indiscriminati
  o commit che inglobino cambiamenti altrui. Controlla nuovamente lo stato
  prima di modificare file condivisi.
- Delega solo se richiesto da me o dalle istruzioni del progetto.
- Usa `rg` con percorsi circoscritti; escludi dipendenze, build e worktree
  annidati. Non leggere o stampare `.env*`, chiavi o altri segreti.
- Riutilizza toolchain e ambienti presenti. Nessuna nuova dipendenza senza
  il mio OK. Il motore deve restare C11 + Metal, con Objective-C solo per
  l'interfaccia Metal; Python è ammesso per conversione, oracle e test.
- Non modificare limiti wired-memory, impostazioni di sistema o SDK per
  aggirare un problema senza prima spiegare la necessità e chiedere l'OK.
- Non eseguire chiamate API a pagamento, deploy o pubblicazioni implicite.
- Verifica le modifiche con i controlli proporzionati del progetto; distingui
  sempre misure locali, valori pubblicati, calcoli teorici e obiettivi.

## Documenti da leggere

1. `README.md`, `Makefile`, eventuali `AGENTS.md` applicabili.
2. `docs/large-model-research-2026-10-11.md`: ricerca principale, fonti,
   architettura, compressione, priorità Metal, piano e criteri di verifica.
3. `docs/DESIGN.md` e `docs/BENCHMARKS.md`: principi, oracle, convenzioni e
   protocollo sperimentale del progetto.
4. `docs/project-review-2026-10-11.md`: revisione dello stato precedente;
   la nuova direzione supera il precedente divieto di offload da SSD.
5. `docs/phase4-expert-experiment.md`: lavoro già svolto sul 30B da preservare;
   non è la nuova priorità di sviluppo.
6. `docs/80b-100b-model-comparison-2026-10-11.md`: contesto delle alternative,
   senza necessità di ripetere la ricerca.

Durante il passaggio di consegne HEAD era arrivato a
`7e3156a93503eb0785004993ece4d32278a7167d`, mentre un'altra sessione stava
aggiornando/registrando l'esperimento sugli esperti. È un riferimento
informativo, non un commit a cui tornare. Esegui `git status --short --branch`
e controlla HEAD/diff correnti. I documenti di ricerca e questo prompt
potrebbero essere ancora non tracciati.

## Stato reale: cosa esiste e cosa manca

Picoforge ha tokenizer, oracle NumPy, percorso CPU C, percorso GPU Metal,
quantizzazione, MoE e speculative decoding greedy. I modelli provati sono
Qwen3-0.6B e Qwen3-30B-A3B, più fixture sintetiche.

Nella ricerca precedente non sono stati scaricati i pesi Qwen3.8, installati
nuovi runtime o eseguiti benchmark di qualità/velocità con Qwen3.8. È stato
eseguito solo un piccolo probe locale delle capacità Metal, oltre a ricerca
online e lettura del codice. Il supporto Qwen3.8 non è implementato.

Limiti del codice precedentemente ispezionato, da confermare sullo stato attuale:

- `src/config.c` riconosce `qwen3` e `qwen3_moe`, non la nuova architettura.
- `src/gpu_forward.m` assume un solo checkpoint safetensors mappato e un solo
  grande MTLBuffer. Il percorso CPU accetta shard, quello GPU non ancora.
- Gli esperti MoE GPU supportano BF16 e `q8_row`; Q4 MoE e IQ2_XXS/Q2_K
  non sono implementati. I formati Q4 dense presenti non sono equivalenti.
- L'attention corrente ha vincoli di memoria threadgroup legati alla
  lunghezza del contesto; non basta aumentare il parametro fino a 32K.
- La UI avvia un processo per richiesta; una sessione residente richiederà
  lavoro dedicato in seguito. La UI non è la prima tappa.
- La speculative greedy esistente richiede uguaglianza dell'intera sequenza
  con il greedy ordinario. Mantieni questa garanzia quando applicabile.

## Hardware e budget

Un probe `MTLDevice` ha osservato:

| Proprietà | Valore |
|---|---:|
| GPU | Apple M5 Pro |
| Memoria fisica | 64,00 GiB |
| `recommendedMaxWorkingSetSize` | 51,84 GiB |
| `maxBufferLength` | 38,88 GiB |
| `maxThreadgroupMemoryLength` | 32 KiB |
| Memoria unificata | sì |
| macOS | 27.0.1, build 26A434 |
| SDK selezionato | macOS 26.5 |
| Compilatore Metal | 32023.883 |
| Spazio disco libero alla prima ispezione | circa 215 GiB |

Rimisura i valori rilevanti. Working set raccomandato, dimensione massima
del singolo buffer e RAM fisica sono limiti distinti. Più buffer risolvono
il limite per buffer, non aumentano la RAM.

Obiettivo iniziale: una sessione, **8K di contesto totale**, prefill in chunk
da **1.024 token**, picco applicativo indicativamente **48–50 GiB**, senza
decoding sostenuto dallo swap. Poi validare 16K e 32K. RSS da solo è
insufficiente: registra footprint fisico, allocazioni Metal, pressione
memoria e variazione dello swap. Buffer temporanei e staging I/O contano.

## Modello e checkpoint esatti

- Modello originale: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- Revisione ispezionata: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Architettura originale: `Qwen4ExpForConditionalGeneration`,
  `model_type=qwen4_exp`; il contenitore GGUF usa `qwen4exp`.
- Circa 125B di backbone, 6B attivi, più 51B di embedding n-gram e 4B MTP.
  Non confondere parametri totali, attivi e conteggio GGUF con padding.
- Checkpoint scelto: `antirez/qwen3.8-flash-next-gguf`, file
  **`Qwen3.8-Flash-Next-Q2.gguf`**.
- Revisione: `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe`.
- Fonte: https://huggingface.co/antirez/qwen3.8-flash-next-gguf/tree/d600fe1a43d2e1cdcadb85144ce3142f66f9eefe
- SHA-256 pubblicato:
  `b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`.
  È stato letto dai metadati, non verificato su un download locale.

Il file pesa **137,10 GiB**: **41,73 GiB di pesi main/MTP** da rendere
residenti e **95,37 GiB di tabelle n-gram originali BF16** da leggere
selettivamente da SSD. La memoria di cache e runtime si aggiunge.
Le tabelle n-gram non devono essere mappate interamente in Metal o precaricate.
Non serve scaricare anche il Q4 da 165,11 GiB, né l'intero originale BF16.

Prima del download controlla spazio reale, cache HF ed eventuali copie già
presenti. Usa un download riprendibile della sola revisione/file selezionati,
in `models/` o nella posizione già adottata dal progetto, evitando copie
complete duplicate e tenendo i pesi fuori da Git. Verifica l'hash dopo il
download e registra revisione, dimensioni e checksum. Non cancellare modelli
esistenti per fare spazio senza autorizzazione.

La ricetta Q2 usa IQ2_XXS calibrato per gate/up degli esperti routed e Q2_K
per down, con 640 input logici padded a 768. Altri tensori e MTP hanno
precisioni diverse. Leggi l'inventario reale dei tensori: Q2 non significa
due bit uniformi, e IQ2_XXS non equivale a INT2 nativo. Il contenitore include
già gli n-gram: nessun sidecar aggiuntivo per questo artifact.

La licenza dei pesi è Qwen Community License, distinta dalla MIT dell'engine.
Conserva provenienza e attribuzioni di eventuale codice riutilizzato.

## Riferimenti antirez e architettura

Repository: https://github.com/antirez/ds4
Commit studiato: `fc80bd695da76ee14bcb86820bf900a5cdfda806`.
Leggi a quella revisione, verificando eventuali fix successivi pertinenti:

- `docs/QWEN38_FLASH_NEXT.md`: esecuzione, budget e limiti.
- `docs/PERFORMANCE.md`, `docs/METAL.md`, `QA_BEFORE_RELEASES.md`.
- `ds4_metal.m`: viste no-copy multiple, mapping dei pesi e allocazioni.
- `metal/qwen4.metal`: operatori e kernel specifici del modello.
- `gguf-tools/README.md` e `gguf-tools/quality-testing/README.md`.
- `ds4_ssd.c` e `docs/SSD_STREAMING.md` come riferimenti per il futuro.

La documentazione riporta un piano di allocazione Q2 di circa **42,86 GiB**
a 8K/chunk 1024: non è un picco misurato sul nostro Mac. Le velocità pubblicate
su M5 Max non sono previsioni per M5 Pro.

Per Qwen3.8, DwarfStar legge gli n-gram da SSD ma non implementa attualmente
lo streaming degli esperti in quel percorso. Non usare il flag generico di
expert streaming dando per scontato che funzioni. La prima versione mantiene
residenti tutti gli esperti compressi.

La nuova architettura comprende Gated DeltaNet, causal convolution,
Qwen Sparse Attention con indexer, residui gated a quattro rami,
esperti routed/shared e lookup/gating/convoluzione n-gram. MTP viene dopo.
Le dimensioni vanno lette dalla configurazione/metadati, non hardcodate.
Il report architetturale è https://arxiv.org/html/2608.30320v1 ; il documento
locale di ricerca contiene i dettagli e i collegamenti alla configurazione.

## Ordine di lavoro

### 0. Baseline e fattibilità: prima attività concreta

Controlla stato del repository, toolchain, memoria e spazio. Usa DwarfStar
in una directory separata dalla sorgente di picoforge, a revisione registrata.
Verifica il suo build e le opzioni reali nella documentazione/help: non
inventare flag né fidarti degli snippet generici generati da Hugging Face.
Riutilizza strumenti presenti; eventuali nuove dipendenze richiedono OK.

Esegui Q2 sul Mac con una sessione, 8K, chunk 1024, inizialmente **senza MTP**.
Prepara prima un manifest riproducibile e salva comandi, revisioni, log,
risposte e metriche sotto le convenzioni `bench/` e `docs/`. Misura caricamento,
prefill, decode, tempo alla prima risposta, richiesta completa, memoria,
swap e I/O degli n-gram. Etichetta chiaramente le prime esecuzioni come pilot.

Verifica anche risposte di coding/ragionamento, non solo che generi testo.
Se emerge un problema bloccante, conserva la riproduzione minima e risolvilo
o spiegalo. Non dichiarare la scelta fallita per un errore di setup.

### 1. Storage e loader

Progetta viste dei pesi con buffer-id, offset, forma, formato e ownership
espliciti. Supporta più MTLBuffer no-copy con controlli di allineamento,
overflow, confini e durata delle mappature. Aggiungi un lettore GGUF
circoscritto all'architettura e ai formati effettivamente necessari,
mantenendo safetensors e i test Qwen3 esistenti. Non convertire il modello
intero in FP16 né creare copie complete solo per cambiare contenitore.

### 2. Oracle e operatori

Costruisci fixture sintetiche piccole e riferimenti indipendenti per
dequantizzazione, routing, ricorrenza, convoluzione, attention/indexer,
residui e n-gram. Usa slice/tensori/layer reali dove necessario senza
richiedere la residenza dell'intero BF16. Verifica tokenizer, template,
logits e transizioni di stato; non aggiungere il modello cambiando solo
una stringa nel parser.

### 3. Grafo completo C/Metal

Porta prima la generazione ordinaria corretta, con esperti residenti e
letture n-gram limitate. Deriva gli ID n-gram esatti dalla storia dei token,
deduplica/accorpa letture e usa staging bounded. Non riutilizzare righe
stale dopo errori I/O. Valida l'accumulo ricorrente inizialmente in FP32.

Usa attention tiled con softmax online, preservando selezione sparsa e
causalità di QSA; non sostituire silenziosamente l'algoritmo del modello.
Testa prefill a chunk, append, reset e continuazioni lunghe.

### 4. Ottimizzazione M5

Separa decode da prefill: SIMD GEMV low-bit per pochi token, grouped GEMM
Metal TensorOps/MPP quando le forme lo giustificano. Dequantizza in registri
o piccoli tile all'interno del calcolo. Profila fusioni gate/up/attivazione,
normalizzazione/residui, overhead dei dispatch, routing e buffer riutilizzabili.

Gli acceleratori dei core GPU M5 si usano tramite Metal/MPP; sono distinti
dall'Apple Neural Engine. Non introdurre un percorso ANE/Core ML.
L'SDK 26.5 osservato espone INT4 ma non tutte le nuove dichiarazioni host
INT2/FP4: ogni API va provata con probe di compilazione e test numerico sulla
toolchain reale. Nessun formato native INT2 sostituisce da solo IQ2_XXS.

### 5. Esperimenti successivi

Solo dopo una baseline funzionante: precisione mista calibrata per tensore,
contesto 16K/32K, sessioni residenti/prefix reuse, eventuale KV quantizzata,
poi MTP. Per MTP serve rollback completo di stato ricorrente, convoluzione,
indexer, n-gram e KV. Non copiare modalità speculative opportunistiche che
violano la parità greedy promessa dal progetto.

La ricerca su AWQ, GPTQv2, QuIP#, AQLM, KIVI, TurboQuant, LLM in a flash,
MoE-Infinity, FlashAttention e BaseRT M5 è già nel documento principale.
Non serve implementarle tutte: usa quelle giustificate da misure e qualità.
Pruning, modifica del top-k e distillazione sono esperimenti lossy separati.

## Qualità, test e criteri di successo

Mantieni distinti:

1. Correttezza dell'implementazione: stesso checkpoint, operatori/stato/logits
   confrontati con riferimenti indipendenti.
2. Danno da compressione: Q2 rispetto a una baseline a precisione superiore
   quando fattibile. Un oracle che dequantizza Q2 non misura la perdita BF16.
3. Utilità: problemi risolti nel tempo disponibile sul Mac.

Usa codice valutato con test eseguibili, bug fix su più file, matematica/logica
con risposte verificabili, istruzioni in italiano/inglese e contesti lunghi.
Se usi LiveCodeBench/EvalPlus, registra subset/versioni/scaffold e separa
calibrazione da valutazione. Un piccolo subset non produce un punteggio
ufficiale di leaderboard. NLL/perplexity da soli non provano capacità di coding.

Rispetta `docs/BENCHMARKS.md`: warmup e confini di timing appropriati al tipo
di misura, ordine randomizzato, 25 ripetizioni per decisioni prestazionali
quando praticabile, mediane e p10/p90, raw data e hash. Le ripetizioni dei
timing non sono nuovi esempi indipendenti per gli intervalli sulla qualità.
Separa cold load, prima esecuzione e regime caldo; niente altri carichi GPU
contemporanei. Non presentare RSS come memoria totale del modello.

Obiettivi iniziali proposti, non risultati acquisiti: 8K entro il budget
senza swap sostenuto; almeno 10 token/s in decode e prefill 8K entro 60 s,
escluso cold load; qualità utile sui task. Se non raggiunti, documenta il
collo di bottiglia prima di modificare ricetta o obiettivo.

Esegui `make` e i test pertinenti dopo modifiche al codice; `make test-all`
è il controllo completo del progetto e richiede le fixture/ambienti previsti.
Non reinstallare dipendenze alla cieca e non nascondere test non eseguibili.
Per parser binari, mapping e kernel numerici aggiungi verifiche mirate ai
rischi reali, oltre alla regressione dei modelli già supportati.

Aggiorna documentazione e risultati distinguendo baseline DwarfStar,
funzionalità native implementate e lavoro ancora pianificato. Il successo
non è dichiarare supporto Qwen3.8 dopo uno stub o un wrapper.

## Parti adesso

Inizia dall'ispezione e dalla baseline della fase 0; prepara un piano breve
ancorato ai file reali, quindi eseguilo. Procedi per incrementi verificabili
verso loader, oracle e grafo Metal quando fattibilità e qualità lo giustificano.
Non chiedermi di scegliere nuovamente il modello. Se incontri un limite
concreto o serve un'autorizzazione per nuove dipendenze, spiega esattamente
cosa hai verificato e quale decisione manca, continuando il lavoro indipendente.
