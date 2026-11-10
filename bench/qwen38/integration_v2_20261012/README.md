# Qwen3.8 Q2 — BASE_INTEGRATA_V2

2026-10-12, Apple M5 Pro, 64 GiB. Branch `coord/qwen38-integrate-v2`,
worktree `/path/to/picoforge-qwen38-integrate-v2`,
da `6a6b886e985185d7c9104eda888e6a406553aedf`.
Il commit che introduce questo report è BASE_INTEGRATA_V2; lo SHA completo
è nella consegna del coordinatore, senza hash autoreferenziale nel file.
Nessun merge/push su main. Direzione invariata: Qwen3.8-Flash-Next Q2, C11/Metal.

## Risultato verificato sulla base combinata

| Gate | Esito e copertura |
|---|---|
| Build | PASS: oggetti C/ObjC distinti, metallib legacy+B+C+D+F |
| ABI | 82 simboli pubblici definiti, 0 dichiarazioni prive d'implementazione |
| CPU A–F | PASS: **169 test**, ripartiti 20/25/35/44/17/28; zero skip |
| Contratti | Header C11+ObjC, 4 test lock, strict runner e collegamenti API PASS |
| A storage GPU | PASS sul GGUF sintetico, owner/lifetime/span |
| B CPU/Metal | 25 test per backend, 245 prodotti, sei formati, 137 slice reali |
| C primitive Metal | PASS, incluse history/stato, alias e GDN lungo fino a 32768 |
| C composto | PASS CPU/Metal sintetico e layer 0 reale, 18 intermedi e stato |
| D Metal | 10 test, 117 invocazioni, inclusa composizione QSA reale ibrida |
| E PLE CPU reale | PASS, due token a dimensione reale, chunk 1/2 e history |
| F CPU reale | PASS sui layer 0/3/47, routed/shared separati e bound proiezioni |
| F Metal | 4 test, 42 casi, 4280 confronti di stadio, 154 microbatch GPU |
| Qwen3 `make test-all` | PASS completo, zero skip; 55 run speculativi, shard e routing4096 |

Nessuna tolleranza allargata. I harness nativi usano ASan/UBSan fatali,
la dylib ctypes C è affiancata dal harness sanitizzato. Metal API validation
attiva sui test Qwen38; ogni test GPU e composizione reale usa il lock comune.
Build e inventario ABI sono controlli distinti dalla parità numerica.

I quattro head acquisiti con merge espliciti sono in [deliveries.json](deliveries.json).
Le consegne hanno file disgiunti e non cambiano gli header pubblici.
Main e i worktree delle consegne sono preservati; nessun lavoro non committato
di altre sessioni è stato acquisito.

## Evidenza numerica e perimetro

**C.** Controlli reali layer 0/1/46: 24 confronti locali. Composizione sintetica
T=65 e reale layer 0 T=9: CPU 3 test per fixture; CPU+Metal 4 per fixture
(i tre test CPU vengono rieseguiti nel runner Metal). Il test controlli è
separato. Proiezioni contro FP64 sul medesimo input, operatori contro upstream,
output/contributo/residuo e storia/stato contro la composizione indipendente.
Token singoli/chunk7/intero e reset/replay risultano bit-esatti sul medesimo
backend; non si richiede uguaglianza bit-a-bit tra CPU e GPU.

Sul layer reale: errore scalato composto massimo CPU **5.85658e-6**, Metal
**5.80236e-6**, gate **5e-4**. Massima frazione del bound B sulle proiezioni
CPU **0.0150928**, Metal **0.0148853**. Lettura packed **75654016 byte**,
workspace CPU **2721408**, stato/history **3268608**. Picco allocazioni Metal
del harness **83099648 byte**, incluse le viste pesi bounded; non è footprint
del modello completo. Primitive Metal lunghe: massimo errore scalato
**3.60608101e-5**, dentro i gate locali/lunghi congelati.
Vedere `c-*.json`, `recurrent-cpu.json` e `compositions.log`.

**D.** Layer QSA 3: **54750848 byte** letti, sotto cap64MiB, più probe limitati
dei layer27/47 inclusi nel conto. Sei proiezioni B: massimo errore assoluto
**3.51510e-7**, massima frazione bound **0.00094353**. Attenzione contro FP64:
**4.869873e-8** sia nel gate CPU sia nella composizione ibrida Metal.
Il percorso GPU usa prep C, RoPE/pool/score/attention D; proiezioni e selezione
sono CPU nel harness reale. Non è un layer QSA interamente GPU né il lifecycle
pubblico GPU. Vedere `attention-real.json`, `attention-gpu.json`.

**E.** PLE completo CPU con proiezioni Q8_0 e righe BF16 reali: **35102720 byte**
di proiezioni/gain/taps, 32 righe lookup (**10240 byte**) e probe di 38 righe
di confine (**12160 byte**). Delta max abs **1.1920929e-7**, errore relativo L2
**3.270024e-7**, massima frazione bound B **0.00240522**. Verificati ID/righe,
norme/gate/segno, u/z/Delta, storia raw e token/chunk. La suite sintetica
include fault I/O/OOM, abort, stale/lifetime e storia32768 con shape ridotte.
La tabella completa resta disk-only. Nessun PLE Metal. Vedere `ple-real.json`.

**F.** Slice reali layer0/3/47: **28354560 byte per layer**, un input sintetico
per layer, routing naturale su512 esperti e probe0/511 separati dai selezionati.
Output Y max abs **2.27973e-9**, NRMSE massimo **1.37930e-6**; massima frazione
bound B **0.00462106**. Metal: fixture sintetiche floating point e packed,
incluse down640/768; Y max abs **1.1920929e-6**, NRMSE **4.83618e-7**.
Routing/grouping sono CPU e congelati prima del comando. I confronti sulle
slice reali sono CPU; non vengono presentati come F GPU end-to-end.
Vedere `moe-{cpu,gpu,real}.json` e `moe-gpu-allocations.jsonl`.

**A/B e Qwen3.** I gate precedenti vengono rieseguiti sulla base combinata.
B conserva decode bit-esatto nel dominio dichiarato e i bound dei prodotti;
underflow/FTZ sono registrati separatamente. Report in `packed-*.json`,
log completi CPU/GPU e regressioni. Le slice B sono 400164 byte complessivi.

## Correzioni di integrazione

- Oggetti `.m.o` distinti dagli oggetti `.o`; aggiornati anche target legacy.
- Harness C/D/F e collegamenti API includono tutte le dipendenze C/A/B/GGUF.
  Il harness Python E aggiunge GDN e planner GGUF introdotti dalla composizione C.
- Entrambi i percorsi packed F rispettano `Q38_F_PACKED_REFERENCE` comune.
- AIR D/F nella metallib comune, con no-fast-math e FP contraction disabilitata.
  Nessuna modifica al registry delle pipeline Qwen3.
- Nuovi target espliciti per D/F Metal, fixture e composizioni. Receipt hash
  di input e file impediscono riuso silenzioso di preparazioni parziali/modificate.
- D CPU/Metal usano classi distinte; F Metal esegue i quattro test supportati,
  mentre CPU esegue tutti i28. Qualsiasi skip selezionato fa fallire il processo.

## Riproduzione

Prerequisiti già presenti: Xcode/Metal4, venv principale, checkout DwarfStar
fissato `fc80bd695da76ee14bcb86820bf900a5cdfda806`, sorgenti HF hash-pinned e
GGUF della fase0, già autenticato. Nessuna nuova dipendenza o download in questo run.

```sh
Q38_PY=/path/to/picoforge/tools/venv/bin/python
"$Q38_PY" tools/qwen38_gpu_lock.py --timeout 60 -- \
  make qwen38-test-assets PYTHON="$Q38_PY" \
  Q38_SOURCE_DIR=/path/to/picoforge-qwen38-integrate/build/qwen38-f/upstream
make qwen38-composition-assets PYTHON="$Q38_PY"
make -j4 picoforge PYTHON="$Q38_PY"
make test-qwen38-cpu PYTHON="$Q38_PY"
make test-qwen38-gpu PYTHON="$Q38_PY"
make test-qwen38-compositions PYTHON="$Q38_PY"
```

`make test-qwen38` equivale agli ultimi tre gate in sequenza. I target pesanti
acquisiscono già il lock: non annidare un wrapper esterno. Le fixture sono
preparate esplicitamente; i test falliscono se mancano dipendenze/asset.
I report numerici nuovi finiscono in `build/qwen38-integration/`.

Per `test-all`, nel nuovo worktree sono stati creati solo link locali ai
quattro modelli Qwen3 principali esistenti; nessuna riscrittura dei pesi.
Le tre opzioni `-o` impediscono rigenerazione dei quantizzati condivisi:

```sh
"$Q38_PY" tools/qwen38_gpu_lock.py --timeout 60 -- \
  env HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 UBSAN_OPTIONS=halt_on_error=1 \
  make test-all PYTHON="$Q38_PY" \
  -o models/Qwen3-0.6B-q8_row-embed -o models/Qwen3-0.6B-q4_g32 \
  -o models/Qwen3-0.6B-q4_g32-embed
```

`environment.json`, `source-sha256.json`, `binary-sha256.json`, manifest e
receipt conservano provenance e hash locali. Pesi, dump e binari restano
nel build ignorato; qui sono committati soltanto report/log/manifest.
I path e gli hash dei binari dipendono dal worktree/toolchain: sono evidenza
del run, non una promessa di build binaria identica su ogni macchina.

## Limiti e prossimi incrementi

Attivazioni sintetiche su pesi Q2 reali; nessuna cattura da prompt completo,
perdita Q2/BF16, logits/generazione nativa Qwen3.8 o prestazione end-to-end.
Il fit globale 48–50GiB non è dimostrato dai cap dei componenti.
Mancano lifecycle GPU comune, PLE Metal, selezione QSA e routing/grouping MoE
su device, grafo ordinario48layer e confronto di intermedi/logits. MTP escluso.

Incarichi con ownership e dipendenze in
[resume-v3.md](../../docs/specs/qwen38/resume-v3.md): prima G0 comune; poi
adattatori C/E, selezione D, prodotto indicizzato B e routing F; grafo CPU
bounded può iniziare subito, GPU completo e checkpoint intero seguono i gate.
