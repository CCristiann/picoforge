# B — esecuzione dalla base consolidata

2026-10-12. Base `c674b23bd1377b2ce0acef2d06fac8b160809446`;
branch `impl/qwen38-b-cpu`, worktree `/Users/cristiancirje/Desktop/Dev/picoforge-b-cpu`.
Il worktree precedente è preservato. Recuperati solo i tre commit oracle,
senza duplicare le spec né importare implementazioni delle altre sessioni.

## Implementato e verificato

`src/qwen38_packed.c` implementa gli header v1: layout rank 1–3, decoder dei
sei formati ordinari Q2_K/IQ2_XXS/Q8_0/F32/F16/BF16 e prodotto FP32 con stride
in byte. Legge soltanto K logico, non alloca scratch, non apre file e non
dequantizza matrici intere. I pesi possono avere base disallineata; input e
output FP32 richiedono allineamento naturale e non possono sovrapporsi.
Preflight strutturale prima delle scritture; errori numerici rendono invalido
l'output, anche se scritto in parte. M=0 valida la struttura senza leggere dati.

`src/qwen38_packed.metal` contiene un decoder diagnostico e un kernel lineare
con un accumulatore FP32 per elemento di output. `src/qwen38_packed_gpu.m`
implementa create/free/encode: verifica catalogo, device, layout, span, alias
e narrowing; fa solo encode nell'encoder borrowed. Nessun commit/wait nel
componente, nessuno scratch o cache di pesi. Il chiamante mantiene vivi
store, binding, context e risorse fino a completion, poi controlla errore
Metal e parola Shared. I kernel OR-ano i flag senza azzerarli.

La griglia IQ2 ha una sola copia di produzione, inclusa da C e Metal;
checksum e licenza sono in `src/qwen38_iq2_grid.inc`. L'oracle esegue ancora
DwarfStar originale esterno e non importa questa copia di produzione.
Non sono implementati INT2 nativo, q4_row/q8_row o i formati MTP Q4_K/MXFP4.
Questi ultimi sono riconosciuti ma restituiscono UNSUPPORTED per il calcolo.

## Evidenza

Report completo: `native.json`; provenienza delle slice: `native-slices.jsonl`.
I binari dei pesi/expected restano in build; nessun payload modello è in Git.
I report raw sono riferiti al commit codice registrato nel JSON; dopo i test
è stato corretto soltanto il docstring del runner, con AST eseguibile invariato.

- Harness C con ASan/UBSan fatale e warning come errori: passato.
- **25 test CPU + 25 test Metal**, zero skip, nessuna tolleranza allargata.
- Q2_K: tutte le posizioni e combinazioni scala/minimo/codice; IQ2_XXS:
  524.288 vettori griglia/segni/scala. Decode finito bit-esatto; conversioni
  F16/BF16 preservano anche il segno dello zero e i subnormali memorizzati.
- **245 casi di prodotto per backend**: 108 sintetici (M=1/2/7/32/33/128,
  N=1/7/33, K=1/255/257/640/2560, tre esperti, stride con spazi) e 137 reali.
  Zero, one-hot, cancellazioni e random. Padding 640/768 verificato anche
  con coda dei pesi non nulla e allocazione CPU di esattamente 640 float.
- Slice: Q2_K 27, IQ2_XXS 54, F32 8, F16 9, Q8_0 36, BF16 3;
  **400.164 byte di payload complessivi**, 56 nuove righe oltre alle 81
  precedenti. Include router, HC, shared, attention/SSM/indexer, output ed
  embedding; esclude la tabella n-gram di E e MTP.
- Metal API Validation, almeno due viste pesi per fixture, buffer/offset
  distinti, owner copiato/estraneo, stride/capacità/alias, M=0, guards,
  NONFINITE e BOUNDS asincroni, persistenza dei bit della parola errore.
- `make -j4`, `make test-qwen38-contracts` (4 test lock più header) e
  `make test-gguf` (7 test): passati. `make test-all` resta al wiring globale.

Prodotti confrontati con FP64 sugli stessi pesi Q2 decodificati, con il bound
congelato `gamma(2*K+8)*sum(abs(w*x)) + 2*ulp32(float32(ref))`.
Il JSON registra massimo assoluto, ULP, errore/somma, quota del bound,
percentile 99 e indice peggiore per caso. Non è una misura della perdita BF16.

Caso separato fuori dal dominio senza underflow: `2^-126 * 0.5` produce
`2^-127` in C e zero (FTZ) su Metal. Il test lo registra esplicitamente;
non si applica il bound ordinario a quel caso. NaN/Inf e overflow di prodotto
o accumulazione restituiscono NONFINITE/flag, non successo numerico.
Nessun benchmark o picco RSS/footprint è stato misurato. Non si dichiara
generazione Qwen3.8 né composizione completa con C/D/E/F.

## Riproduzione dal worktree

Riutilizzare Python e DwarfStar esistenti; nessuna installazione. La build
locale del riferimento è opzionale se si riusa il binario verificato del
worktree oracle preservato. Directory/report di output devono essere nuovi.

```sh
Q38_PY=/Users/cristiancirje/Desktop/Dev/picoforge/tools/venv/bin/python
Q38_DS4=/Users/cristiancirje/Desktop/Dev/ds4-qwen38-baseline
Q38_MODEL=/Users/cristiancirje/Desktop/Dev/picoforge/models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf
mkdir -p build/qwen38-b
"$Q38_PY" -B tools/oracle/qwen38_packed.py build "$Q38_DS4" build/qwen38-b/reference --sanitize
Q38_REF=build/qwen38-b/reference/reference
clang -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=all -Isrc tests/test_qwen38_packed.c src/qwen38_packed.c src/gguf.c src/gguf_reader.c src/util.c -lm -o build/qwen38-b/test_packed
build/qwen38-b/test_packed
xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -ffp-contract=off -Isrc -c src/qwen38_packed.metal -o build/qwen38-b/packed.air
xcrun -sdk macosx metallib build/qwen38-b/packed.air -o build/qwen38-b/packed.metallib
clang -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror -ffp-contract=off -fobjc-arc -Isrc tests/test_qwen38_packed_gpu.m src/qwen38_packed_gpu.m src/qwen38_packed.c src/gpu_weights.m src/gguf.c src/gguf_reader.c src/gguf_plan.c src/util.c -lm -framework Metal -framework Foundation -o build/qwen38-b/test_packed_gpu
for kind in 10 16 0 1 8 30; do
  "$Q38_PY" -B tools/oracle/qwen38_packed.py slices "$Q38_REF" "$Q38_MODEL" "build/qwen38-b/repro-$kind" --kind "$kind"
done
```

Passare a entrambe le suite le sei directory, con sei opzioni `--slices`:

```sh
"$Q38_PY" -B tests/test_qwen38_packed.py --reference "$Q38_REF" --native build/qwen38-b/test_packed --slices build/qwen38-b/repro-10 --slices build/qwen38-b/repro-16 --slices build/qwen38-b/repro-0 --slices build/qwen38-b/repro-1 --slices build/qwen38-b/repro-8 --slices build/qwen38-b/repro-30 --report build/qwen38-b/repro-cpu.json -v
"$Q38_PY" -B tools/qwen38_gpu_lock.py --timeout 60 -- env MTL_DEBUG_LAYER=1 "$Q38_PY" -B tests/test_qwen38_packed.py --reference "$Q38_REF" --native build/qwen38-b/test_packed_gpu --metal --slices build/qwen38-b/repro-10 --slices build/qwen38-b/repro-16 --slices build/qwen38-b/repro-0 --slices build/qwen38-b/repro-1 --slices build/qwen38-b/repro-8 --slices build/qwen38-b/repro-30 --report build/qwen38-b/repro-metal.json -v
make -j4 PYTHON="$Q38_PY"
make PYTHON="$Q38_PY" test-qwen38-contracts test-gguf
```

Attesi 25 test/OK per suite. Il lock è quello di `git-common-dir`, verificato
qui come `/Users/cristiancirje/Desktop/Dev/picoforge/.git/qwen38-gpu.lock`.
Il runner passa il fd ai figli e li attende; nessun wrapper annidato.

## Integrazione residua, proprietà del coordinatore

1. Aggiungere i target dei harness e la dipendenza di C/glue da header privato
   e griglia; collegare `packed.air` alla metallib principale conservando
   `-fno-fast-math -ffp-contract=off`. Sono due nuovi kernel, con pipeline
   privata B; nessuna voce aggiunta al registry legacy da 32 elementi.
   Il `make` corrente collega già C/ObjC via wildcard, ma **non** questo AIR.
2. Integrare binding A e le proiezioni reali nei test C/D/E/F. Per CPU usare
   `Q38PackedCpuView` borrowed; F passa down con K logico 640, non 768.
   Composizione, stato e logits richiedono i gate separati di integration.md.
3. Misurare prestazioni e fit globale dopo il grafo. Questo kernel elementare
   non è una GEMM tiled/TensorOps ottimizzata. Nessuna conversione half degli
   input e nessuno scratch proporzionale agli esperti sono introdotti.

Header pubblici, Makefile, entrypoint e file legacy non sono stati modificati.
