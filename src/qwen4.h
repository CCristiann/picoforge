/* The shared Qwen3.8 contract: config read from the GGUF metadata and the weight catalog (roles, spans). */
#ifndef PICOFORGE_QWEN4_H
#define PICOFORGE_QWEN4_H
#include "gguf.h"
#include "qwen4_types.h"
typedef struct Qwen4Config Qwen4Config;
typedef struct Qwen4Weights Qwen4Weights;
typedef enum { QW4_TRUNK, QW4_MTP, QW4_DISK_PLE } Qwen4Domain;
typedef enum { QW4_LAYER_GDN, QW4_LAYER_QSA } Qwen4LayerKind;
typedef enum {
    QW4_VECTOR, QW4_MATRIX, QW4_EXPERTS, QW4_CONV, QW4_GDN_TILED,
    QW4_Q_GATE_INTERLEAVED, QW4_DISK_ROWS
} Qwen4Layout;
typedef struct {
    uint32_t hidden_size, vocab_size, blocks, trunk_blocks, mtp_blocks;
    uint32_t max_positions, gdn_conv_kernel;
    Qwen4GdnShape gdn;
    Qwen4HcShape hc;
    Qwen4AttentionConfig attention; /* ratio overridden by the layer record */
    Qwen4MoeConfig moe;
    const char *semantic_profile;    /* borrowed, NUL terminated */
} Qwen4ConfigView;
typedef struct {
    Qwen4LayerKind kind;
    Qwen4Domain domain;
    uint32_t compress_ratio;
    bool has_ple;
} Qwen4Layer;
typedef struct {
    GgufTensor storage;
    uint64_t logical_shape[4], block_elems, block_bytes, row_bytes, plane_bytes;
    Qwen4Domain domain;
    Qwen4Layout layout;
    const GgufBinding *gpu;
} Qwen4Tensor;
/* Global roles use layer=QWEN4_GLOBAL_LAYER; other roles use zero-based layer. */
#define QWEN4_GLOBAL_LAYER UINT32_MAX
typedef enum {
    QW4_TOKEN_EMBD, QW4_LM_HEAD, QW4_OUTPUT_HC_NORM, QW4_OUTPUT_HC_DOWN,
    QW4_OUTPUT_HC_UP, QW4_PLE_TABLE,
    QW4_HC_ATTN_NORM, QW4_HC_ATTN_DOWN, QW4_HC_ATTN_UP, QW4_HC_ATTN_INJECT,
    QW4_HC_FFN_NORM, QW4_HC_FFN_DOWN, QW4_HC_FFN_UP, QW4_HC_FFN_INJECT,
    QW4_GDN_QKV, QW4_GDN_GATE, QW4_GDN_ALPHA, QW4_GDN_BETA, QW4_GDN_A,
    QW4_GDN_DT, QW4_GDN_CONV, QW4_GDN_NORM, QW4_GDN_OUT,
    QW4_ATTN_QG, QW4_ATTN_K, QW4_ATTN_V, QW4_ATTN_OUT,
    QW4_ATTN_Q_NORM, QW4_ATTN_K_NORM, QW4_INDEX_Q, QW4_INDEX_K,
    QW4_INDEX_Q_NORM, QW4_INDEX_K_NORM,
    QW4_MOE_ROUTER, QW4_MOE_GATE, QW4_MOE_UP, QW4_MOE_DOWN,
    QW4_SHARED_GATE, QW4_SHARED_UP, QW4_SHARED_DOWN, QW4_SHARED_ROUTER,
    QW4_PLE_KEY, QW4_PLE_VALUE, QW4_PLE_KEY_NORM, QW4_PLE_QUERY_NORM,
    QW4_PLE_CONV_NORM, QW4_PLE_CONV,
    QW4_MTP_EH, QW4_MTP_E_NORM, QW4_MTP_H_NORM,
    QW4_MTP_HEAD_NORM, QW4_MTP_HEAD_DOWN, QW4_MTP_HEAD_UP, QW4_ROLE_COUNT
} Qwen4Role;
Qwen4Status qwen4_config_load(const Gguf *, Qwen4Config **, Qwen4Error *);
void qwen4_config_free(Qwen4Config *);
const Qwen4ConfigView *qwen4_config_view(const Qwen4Config *);
Qwen4Status qwen4_config_layer(const Qwen4Config *, uint32_t,
    const Qwen4Layer **, Qwen4Error *);
Qwen4Status qwen4_config_ngram(const Qwen4Config *, uint32_t,
    const Qwen4NgramConfig **, Qwen4Error *);
Qwen4Status qwen4_weights_describe(const Gguf *, const Qwen4Config *,
    Qwen4Weights **, Qwen4Error *);
/* source must be the SAME live Gguf used for describe AND plan construction.
 * v1 adds this witness instead of inventing identity from equal descriptors. */
Qwen4Status qwen4_weights_attach(Qwen4Weights *, const Gguf *source,
    const GgufPlan *, Qwen4Error *);
Qwen4Status qwen4_weights_get(const Qwen4Weights *, uint32_t, Qwen4Role,
    const Qwen4Tensor **, Qwen4Error *);
void qwen4_weights_free(Qwen4Weights *);
#endif
