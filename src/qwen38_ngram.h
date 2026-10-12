/* E: disk-only rows, then PLE through B and C. */
#ifndef PICOFORGE_QWEN38_NGRAM_H
#define PICOFORGE_QWEN38_NGRAM_H
#include "qwen38_packed.h"
#include "qwen38_ops.h"
typedef struct Qwen38NgramStore Qwen38NgramStore;
typedef struct Qwen38NgramState Qwen38NgramState;
typedef struct Qwen38NgramJob Qwen38NgramJob;
typedef struct Qwen38PleWorkspace Qwen38PleWorkspace;
typedef struct {
    Q38PackedCpuView key, value;
    Qwen38F32 gamma_key, gamma_query, gamma_conv, taps;
} Qwen38PleWeights;
/* Pure hash: prev/next have order-1 newest-first IDs; rows has T*head_count
 * uint64 elements. All lengths/capacities here count ELEMENTS, not bytes.
 * prev unchanged; no aliases; T=0 copies prev to next without reading tokens. */
Qwen38Status qwen38_ngram_ids(const Qwen38NgramConfig *, const int32_t *prev,
    size_t prev_count, const int32_t *tokens, size_t count,
    uint64_t *rows, size_t row_capacity, int32_t *next, size_t next_capacity,
    Qwen38Error *);
Qwen38Status qwen38_ngram_open(const Gguf *, const Qwen38NgramConfig *,
    size_t max_tokens, size_t staging_budget, Qwen38NgramStore **, Qwen38Error *);
Qwen38Status qwen38_ngram_close(Qwen38NgramStore *, Qwen38Error *);
Qwen38Status qwen38_ngram_state_create(Qwen38NgramStore *, Qwen38StateKey,
    Qwen38NgramState **, Qwen38Error *);
Qwen38StateInfo qwen38_ngram_state_info(const Qwen38NgramState *);
Qwen38Status qwen38_ngram_state_reset(Qwen38NgramState *, uint64_t epoch,
    Qwen38Error *);
Qwen38Status qwen38_ngram_state_destroy(Qwen38NgramState *, Qwen38Error *);
Qwen38Status qwen38_ngram_prepare(Qwen38NgramStore *, Qwen38NgramState *,
    const int32_t *tokens, size_t count, uint64_t position,
    Qwen38NgramJob **, Qwen38Error *);
Qwen38Status qwen38_ngram_read(Qwen38NgramJob *, Qwen38Error *);
Qwen38Status qwen38_ngram_rows(const Qwen38NgramJob *, Qwen38F32 *, Qwen38Error *);
Qwen38Status qwen38_ple_workspace_create(Qwen38NgramStore *, size_t byte_budget,
    Qwen38PleWorkspace **, Qwen38Error *);
Qwen38Status qwen38_ple_workspace_destroy(Qwen38PleWorkspace *, Qwen38Error *);
/* Core gate consumes NORMALIZED key/query [T,S,D], value [T,D].
 * Emits g[T,S] and u[T,S,D]; no RMS/conv duplication inside this primitive. */
Qwen38Status qwen38_ple_gate(Qwen38F32 key, Qwen38F32 query, Qwen38F32 value,
    Qwen38F32Mut g, Qwen38F32Mut u, Qwen38Error *);
/* Produces Delta [T,S,D]; R addition belongs to graph. Candidate conv only. */
Qwen38Status qwen38_ple_cpu(Qwen38NgramJob *, const Qwen38PleWeights *,
    Qwen38F32 residual, Qwen38F32Mut delta, Qwen38PleWorkspace *, Qwen38Error *);
Qwen38Status qwen38_ngram_commit(Qwen38NgramState *, Qwen38NgramJob *, Qwen38Error *);
Qwen38Status qwen38_ngram_abort(Qwen38NgramJob *, Qwen38Error *);
Qwen38Status qwen38_ngram_job_release(Qwen38NgramJob *, Qwen38Error *);
#endif
