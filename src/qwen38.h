/* A owns implementations; coordinator owns this shared contract. */
#ifndef PICOFORGE_QWEN38_H
#define PICOFORGE_QWEN38_H
#include "gguf.h"
#include "qwen38_types.h"
typedef struct Qwen38Config Qwen38Config;
typedef struct Qwen38Weights Qwen38Weights;
typedef enum { Q38_TRUNK, Q38_MTP, Q38_DISK_PLE } Qwen38Domain;
typedef enum { Q38_LAYER_GDN, Q38_LAYER_QSA } Qwen38LayerKind;
typedef enum {
    Q38_VECTOR, Q38_MATRIX, Q38_EXPERTS, Q38_CONV, Q38_GDN_TILED,
    Q38_Q_GATE_INTERLEAVED, Q38_DISK_ROWS
} Qwen38Layout;
typedef struct {
    uint32_t hidden_size, vocab_size, blocks, trunk_blocks, mtp_blocks;
    uint32_t max_positions, gdn_conv_kernel;
    Qwen38GdnShape gdn;
    Qwen38HcShape hc;
    Qwen38AttentionConfig attention; /* ratio overridden by the layer record */
    Qwen38MoeConfig moe;
    const char *semantic_profile;    /* borrowed, NUL terminated */
} Qwen38ConfigView;
typedef struct {
    Qwen38LayerKind kind;
    Qwen38Domain domain;
    uint32_t compress_ratio;
    bool has_ple;
} Qwen38Layer;
typedef struct {
    GgufTensor storage;
    uint64_t logical_shape[4], block_elems, block_bytes, row_bytes, plane_bytes;
    Qwen38Domain domain;
    Qwen38Layout layout;
    const GgufBinding *gpu;
} Qwen38Tensor;
/* Global roles use layer=QWEN38_GLOBAL_LAYER; other roles use zero-based layer. */
#define QWEN38_GLOBAL_LAYER UINT32_MAX
typedef enum {
    Q38_TOKEN_EMBD, Q38_LM_HEAD, Q38_OUTPUT_HC_NORM, Q38_OUTPUT_HC_DOWN,
    Q38_OUTPUT_HC_UP, Q38_PLE_TABLE,
    Q38_HC_ATTN_NORM, Q38_HC_ATTN_DOWN, Q38_HC_ATTN_UP, Q38_HC_ATTN_INJECT,
    Q38_HC_FFN_NORM, Q38_HC_FFN_DOWN, Q38_HC_FFN_UP, Q38_HC_FFN_INJECT,
    Q38_GDN_QKV, Q38_GDN_GATE, Q38_GDN_ALPHA, Q38_GDN_BETA, Q38_GDN_A,
    Q38_GDN_DT, Q38_GDN_CONV, Q38_GDN_NORM, Q38_GDN_OUT,
    Q38_ATTN_QG, Q38_ATTN_K, Q38_ATTN_V, Q38_ATTN_OUT,
    Q38_ATTN_Q_NORM, Q38_ATTN_K_NORM, Q38_INDEX_Q, Q38_INDEX_K,
    Q38_INDEX_Q_NORM, Q38_INDEX_K_NORM,
    Q38_MOE_ROUTER, Q38_MOE_GATE, Q38_MOE_UP, Q38_MOE_DOWN,
    Q38_SHARED_GATE, Q38_SHARED_UP, Q38_SHARED_DOWN, Q38_SHARED_ROUTER,
    Q38_PLE_KEY, Q38_PLE_VALUE, Q38_PLE_KEY_NORM, Q38_PLE_QUERY_NORM,
    Q38_PLE_CONV_NORM, Q38_PLE_CONV, Q38_ROLE_COUNT
} Qwen38Role;
Qwen38Status qwen38_config_load(const Gguf *, Qwen38Config **, Qwen38Error *);
void qwen38_config_free(Qwen38Config *);
const Qwen38ConfigView *qwen38_config_view(const Qwen38Config *);
Qwen38Status qwen38_config_layer(const Qwen38Config *, uint32_t,
    const Qwen38Layer **, Qwen38Error *);
Qwen38Status qwen38_config_ngram(const Qwen38Config *, uint32_t,
    const Qwen38NgramConfig **, Qwen38Error *);
Qwen38Status qwen38_weights_describe(const Gguf *, const Qwen38Config *,
    Qwen38Weights **, Qwen38Error *);
/* source must be the SAME live Gguf used for describe AND plan construction.
 * v1 adds this witness instead of inventing identity from equal descriptors. */
Qwen38Status qwen38_weights_attach(Qwen38Weights *, const Gguf *source,
    const GgufPlan *, Qwen38Error *);
Qwen38Status qwen38_weights_get(const Qwen38Weights *, uint32_t, Qwen38Role,
    const Qwen38Tensor **, Qwen38Error *);
void qwen38_weights_free(Qwen38Weights *);
#endif
