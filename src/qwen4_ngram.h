/* E: disk-only rows, then PLE through B and C. */
#ifndef PICOFORGE_QWEN4_NGRAM_H
#define PICOFORGE_QWEN4_NGRAM_H
#include "qwen4_packed.h"
#include "qwen4_ops.h"
typedef struct Qwen4NgramStore Qwen4NgramStore;
typedef struct Qwen4NgramState Qwen4NgramState;
typedef struct Qwen4NgramJob Qwen4NgramJob;
typedef struct Qwen4PleWorkspace Qwen4PleWorkspace;
typedef struct {
    QW4PackedCpuView key, value;
    Qwen4F32 gamma_key, gamma_query, gamma_conv, taps;
} Qwen4PleWeights;
/* Pure hash: prev/next have order-1 newest-first IDs; rows has T*head_count
 * uint64 elements. All lengths/capacities here count ELEMENTS, not bytes.
 * prev unchanged; no aliases; T=0 copies prev to next without reading tokens. */
Qwen4Status qwen4_ngram_ids(const Qwen4NgramConfig *, const int32_t *prev,
    size_t prev_count, const int32_t *tokens, size_t count,
    uint64_t *rows, size_t row_capacity, int32_t *next, size_t next_capacity,
    Qwen4Error *);
Qwen4Status qwen4_ngram_open(const Gguf *, const Qwen4NgramConfig *,
    size_t max_tokens, size_t staging_budget, Qwen4NgramStore **, Qwen4Error *);
Qwen4Status qwen4_ngram_close(Qwen4NgramStore *, Qwen4Error *);
Qwen4Status qwen4_ngram_state_create(Qwen4NgramStore *, Qwen4StateKey,
    Qwen4NgramState **, Qwen4Error *);
Qwen4StateInfo qwen4_ngram_state_info(const Qwen4NgramState *);
Qwen4Status qwen4_ngram_state_reset(Qwen4NgramState *, uint64_t epoch,
    Qwen4Error *);
Qwen4Status qwen4_ngram_state_destroy(Qwen4NgramState *, Qwen4Error *);
Qwen4Status qwen4_ngram_prepare(Qwen4NgramStore *, Qwen4NgramState *,
    const int32_t *tokens, size_t count, uint64_t position,
    Qwen4NgramJob **, Qwen4Error *);
Qwen4Status qwen4_ngram_read(Qwen4NgramJob *, Qwen4Error *);
Qwen4Status qwen4_ngram_rows(const Qwen4NgramJob *, Qwen4F32 *, Qwen4Error *);
Qwen4Status qwen4_ple_workspace_create(Qwen4NgramStore *, size_t byte_budget,
    Qwen4PleWorkspace **, Qwen4Error *);
Qwen4Status qwen4_ple_workspace_destroy(Qwen4PleWorkspace *, Qwen4Error *);
/* Core gate consumes NORMALIZED key/query [T,S,D], value [T,D].
 * Emits g[T,S] and u[T,S,D]; no RMS/conv duplication inside this primitive. */
Qwen4Status qwen4_ple_gate(Qwen4F32 key, Qwen4F32 query, Qwen4F32 value,
    Qwen4F32Mut g, Qwen4F32Mut u, Qwen4Error *);
/* Produces Delta [T,S,D]; R addition belongs to graph. Candidate conv only. */
Qwen4Status qwen4_ple_cpu(Qwen4NgramJob *, const Qwen4PleWeights *,
    Qwen4F32 residual, Qwen4F32Mut delta, Qwen4PleWorkspace *, Qwen4Error *);
Qwen4Status qwen4_ngram_commit(Qwen4NgramState *, Qwen4NgramJob *, Qwen4Error *);
Qwen4Status qwen4_ngram_abort(Qwen4NgramJob *, Qwen4Error *);
Qwen4Status qwen4_ngram_job_release(Qwen4NgramJob *, Qwen4Error *);
#endif
