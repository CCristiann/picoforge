/* picoforge.h — shared types and helpers for the C engine.
 *
 * Phase 1. Grows one component at a time, like the Python oracle did.
 */
#ifndef PICOFORGE_H
#define PICOFORGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Every dimension the forward pass needs, all of it read from config.json
 * at load time (docs/DESIGN.md, principle 5). Point the engine at Qwen3-1.7B
 * and these numbers simply change; not one of them is compiled in. */
typedef struct {
    int   hidden_size;             /* width of the residual stream        */
    int   num_hidden_layers;       /* identical blocks, applied in order  */
    int   num_attention_heads;     /* query heads                         */
    int   num_key_value_heads;     /* fewer than Q heads -> GQA           */
    int   head_dim;                /* EXPLICIT in config, not hidden/heads */
    int   intermediate_size;       /* SwiGLU inner width                  */
    int   vocab_size;
    float rms_norm_eps;
    float rope_theta;              /* RoPE base frequency                 */
    int   max_position_embeddings;
    bool  tie_word_embeddings;     /* true -> LM head reuses the embeddings */
    int   bos_token_id;
    int   eos_token_id;
    char  torch_dtype[16];         /* storage dtype; we compute in fp32   */

    /* Sampling defaults and stop tokens, from generation_config.json. They
     * are the model's, not ours: Qwen3 ships 0.6 / 0.95 / 20 because that is
     * what it was tuned for, and inventing our own would mean benchmarking a
     * different model than the one that was released. */
    int   eos_ids[8];
    int   n_eos;
    float gen_temperature;
    float gen_top_p;
    int   gen_top_k;

    /* Mixture of experts: model_type "qwen3_moe" (Phase 4). A dense Qwen3 has
     * none of these keys and loads with num_experts = 0. */
    int   num_experts;             /* experts per MoE layer                 */
    int   num_experts_per_tok;     /* top-k: experts each token runs        */
    int   moe_intermediate_size;   /* SwiGLU inner width of ONE expert      */
    bool  norm_topk_prob;          /* renormalise the k weights to sum to 1 */
    int   decoder_sparse_step;     /* layer l is MoE if (l+1) % step == 0 ... */
    int   mlp_only_layers[64];     /* ... and l is not listed here           */
    int   n_mlp_only_layers;
} Qwen3Config;

enum { PF_MAX_TOPK = 64 };         /* bound on num_experts_per_tok            */

/* Fail loudly and immediately. Printf-style, prefixed, exit(1).
 * There is no error-recovery story in this engine on purpose: an
 * inference engine that limps on after a malformed model file produces
 * plausible garbage, which is the one failure mode we cannot detect. */
void  die(const char *fmt, ...);

/* Read a whole file into a NUL-terminated heap buffer. Caller frees. */
char *slurp(const char *path, size_t *len_out);

void  config_load(const char *model_dir, Qwen3Config *cfg);
void  config_print(const Qwen3Config *cfg);
bool  config_is_moe_layer(const Qwen3Config *cfg, int layer);

/* ------------------------------------------------------------------ json
 * Enough to walk a file whose shape you already know. Every one of these
 * dies rather than returning an error: a malformed model file is not a
 * condition an inference engine can sensibly continue from. */
const char *json_skip(const char *p);                 /* past one whole value  */
const char *json_expect(const char *p, char c);       /* assert next char      */
const char *json_string(const char *p, char *out, size_t cap);   /* no escapes */
const char *json_ints(const char *p, long *out, int cap, int *n_out);

/* ---------------------------------------------------------------- weights
 * safetensors: [u64 header length][JSON header][raw tensor bytes].
 * Deliberately dumb, and that is its point — a PyTorch .bin is a pickle,
 * and unpickling executes arbitrary code. Here there is nothing to execute.
 */
typedef enum { DT_BF16, DT_F32, DT_F16, DT_I8, DT_U8 } DType;

typedef struct {
    char        name[80];     /* longest in Qwen3-0.6B is 47 chars       */
    DType       dtype;
    int         ndim;
    long        shape[4];
    size_t      nelem;        /* product of shape                        */
    const void *data;         /* points INTO the mapping; never freed    */
} Tensor;

/* One file, or a sharded checkpoint (model.safetensors.index.json names the
 * file each tensor lives in). Every shard is mapped; tensors point into their
 * own shard's mapping. map / map_len are the first shard -- the whole model
 * when there is one file, which is what the GPU's single buffer binds. */
enum { ST_MAX_SHARDS = 64 };
typedef struct {
    void   *map;              /* the (first) file, mmapped read-only     */
    size_t  map_len;
    void   *maps[ST_MAX_SHARDS];
    size_t  map_lens[ST_MAX_SHARDS];
    int     n_maps;
    Tensor *tensors;          /* sorted by name, for binary search       */
    int     n_tensors;
} SafeTensors;

void          st_open(const char *model_dir, SafeTensors *st);
void          st_close(SafeTensors *st);
const Tensor *st_find(const SafeTensors *st, const char *name);  /* dies if absent */
const Tensor *st_try(const SafeTensors *st, const char *name);   /* NULL if absent */
void          st_summary(const SafeTensors *st);

/* bfloat16 IS the top half of an fp32: same sign bit, same 8 exponent bits,
 * 7 mantissa bits instead of 23. Widening is therefore a SHIFT, not a
 * conversion: append 16 zero bits and the value is bit-exactly the same
 * number. No table, no rounding, nothing lost. (Narrowing does lose — which
 * is why the checkpoint is the lossy artifact, never our load of it.) */
static inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    /* memcpy, not a pointer cast: casting uint32_t* to float* and
     * dereferencing violates strict aliasing and is undefined behaviour.
     * At -O2 the compiler emits zero instructions for this memcpy. */
    memcpy(&f, &bits, sizeof f);
    return f;
}

void tensor_to_f32(const Tensor *t, float *out);

/* ------------------------------------------------------------------- ops
 * The obvious, slow implementations. They are the reference the Metal
 * kernels get judged against, so none of them may be clever.
 * Weight arguments are bf16 straight out of the mapping, never copies. */
void  matmul(float *out, const float *x, const uint16_t *w, int n_in, int n_out);

/* A quantised weight matrix, [n_out, n_in], borrowed from the mapping like
 * every other weight. W[j, i] = d[j, i / group] * q[j, i].
 *
 * q is int8 for 8 bits. For 4 bits it is packed two codes per byte, LOW
 * nibble first, two's complement — not a choice but a measurement: it is
 * what TensorOps reads (tools/probe/int4_probe), and a file laid out any
 * other way would be silently misread by the GPU. */
typedef struct {
    int             bits;     /* 8 or 4                                    */
    int             group;    /* block length along n_in; n_in = one per row */
    const uint8_t  *q;
    const uint16_t *d;        /* bf16 scales, [n_out, n_in / group]        */
} QWeight;

/* The reference: dequantise each weight in the inner loop and multiply, the
 * obvious way. bf16 d times an integer q is exact in fp32, so this is the fp32
 * matmul over the dequantised matrix, digit for digit. */
void  matmul_q(float *out, const float *x, const QWeight *w, int n_in, int n_out);
void  dequant_row(float *out, const QWeight *w, int row, int n_in);   /* embedding lookup */
void  rmsnorm(float *out, const float *x, const uint16_t *w, int n, float eps);
void  softmax(float *x, int n);
void  rope_apply(float *x, int head_dim, int pos, float theta);
float silu(float z);

/* ----------------------------------------------------------------- model
 * Weights are borrowed pointers INTO the mapping — nothing here owns bytes,
 * so binding the whole model costs 311 lookups and zero copies.
 *
 * A Linear is one projection matrix in whichever form the checkpoint holds
 * it: bf16 (w set, q.bits 0) or quantised (w NULL). The file decides, tensor
 * by tensor -- X.weight means bf16, X.qweight + X.scales means quantised --
 * so nothing has to be told which format it is loading. */
typedef struct {
    const uint16_t *w;
    QWeight         q;
} Linear;

/* out = W x, in whichever form W is stored. */
void linear(float *out, const float *x, const Linear *l, int n_in, int n_out);
void set_q4_bf16_activations(bool on);   /* CPU mimics the GPU's Q4 narrowing */

/* One expert of a MoE layer: a narrow SwiGLU MLP of its own. */
typedef struct { Linear gate_proj, up_proj, down_proj; } Expert;

typedef struct {
    const uint16_t *input_ln, *q_norm, *k_norm, *post_attn_ln;
    Linear q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
    /* MoE layers only (NULL experts = dense): the router [num_experts, hidden]
     * and the experts. The dense projections above stay unbound. */
    Linear  router;
    Expert *experts;
} LayerWeights;

typedef struct {
    double          bytes;        /* weight bytes one decoded token reads     */
    Linear          embed;        /* [vocab, hidden]                          */
    Linear          head;         /* LM head: the embedding when tied         */
    const uint16_t *final_norm;
    LayerWeights   *layers;
    int             n_layers;
} Weights;

/* Every activation buffer the forward pass needs, allocated once.
 *
 * The K/V cache is the reason generation is affordable at all. Without it,
 * emitting token 500 means recomputing the keys and values of the 499 before
 * it, every time. With it, each step appends one row per layer and reads the
 * rest. Prefill and decode become the same code path: n tokens starting at
 * position pos, with n = prompt length once and n = 1 thereafter.
 *
 * It is not free. 28 layers x 1024 kv dims x 4 bytes x 2 tensors is 229 KB
 * per position, so 2048 tokens of context cost 469 MB — several times the
 * weights themselves. That arithmetic is why quantising the cache is a real
 * topic and not a micro-optimisation. */
typedef struct {
    int    max_seq;               /* capacity of the cache, in positions     */
    int    max_rows;              /* how many rows of logits fit             */
    float *x, *xb;                /* residual stream, and a scratch copy     */
    float *q;                     /* queries, (n, heads, head_dim)           */
    float *kcache, *vcache;       /* (layers, max_seq, kv_dim)               */
    float *att, *attout;          /* scores over the cache; merged heads     */
    float *hb, *hb2;              /* SwiGLU gate and up                      */
    float *router, *moe_out, *eb; /* MoE: expert probs, one token's sum, one expert's output */
    float *logits;                /* (max_rows, vocab)                       */
} RunState;

void weights_bind(const SafeTensors *st, const Qwen3Config *cfg, Weights *w);
void weights_free(Weights *w);
void state_alloc(RunState *s, const Qwen3Config *cfg, int max_seq, int max_rows);
void state_free(RunState *s);

/* Run n tokens whose first one sits at absolute position `pos`. Keys and
 * values for positions [0, pos) must already be in the cache; this appends
 * its own. Logits are written for rows [logits_from, n), row i landing at
 * logits + (i - logits_from) * vocab_size — generation wants only the last
 * row, and the LM head is 13% of the work at prompt length. */
void forward(const int *tokens, int n, int pos, int logits_from,
             const Weights *w, const Qwen3Config *cfg, RunState *s);

/* ------------------------------------------------------------- tokenizer
 * Byte-level BPE. The vocabulary starts from all 256 possible bytes, every
 * one of which really is present, so NOTHING is untokenisable: a binary
 * file, an unseen script, a novel emoji all encode — badly, perhaps, but
 * always. That is the whole point of going byte-level.
 *
 * The files spell tokens in a 256-character Unicode alphabet only because
 * JSON cannot hold raw control bytes. That is presentation. Inside this
 * struct every token is the byte string it actually stands for. */
/* An added token: matched in the raw text before the regex and before BPE,
 * because no merge rule can build one and none must ever be allowed to. */
typedef struct { int id; int len; unsigned char text[64]; } SpecialToken;

typedef struct {
    unsigned char *blob;        /* every token's bytes, concatenated        */
    int           *offset;      /* [n_tokens + 1] indices into blob         */
    int            n_tokens;
    int            n_merges;

    /* byte string -> id. Open addressing, slot holds id + 1 (0 = empty). */
    int           *str_slot;
    unsigned       str_mask;

    /* (left_id << 32 | right_id) -> merge rank. rank -1 marks an empty slot. */
    uint64_t      *merge_key;
    int           *merge_rank;
    unsigned       merge_mask;

    SpecialToken   specials[64];
    int            n_specials;
} Tokenizer;

/* Unicode general categories L and N, as sorted codepoint ranges. Generated
 * by tools/gen_unicode_tables.py; see src/unicode_tables.c. */
typedef struct { unsigned lo, hi; } CodepointRange;
extern const CodepointRange unicode_letters[];
extern const int unicode_letters_count;
extern const CodepointRange unicode_numbers[];
extern const int unicode_numbers_count;

/* NFC tables. b == 0 marks a singleton decomposition. Hangul is absent from
 * all three: its decomposition and composition are arithmetic. */
typedef struct { unsigned cp, a, b; } Decomposition;
extern const Decomposition unicode_decomp[];
extern const int unicode_decomp_count;

typedef struct { unsigned a, b, cp; } Composition;
extern const Composition unicode_compose[];
extern const int unicode_compose_count;

typedef struct { unsigned cp; unsigned char ccc; } CombiningClass;
extern const CombiningClass unicode_ccc[];
extern const int unicode_ccc_count;

/* NFC-normalise UTF-8. Returns the new byte length. */
int nfc_normalize(const char *in, int len, char *out, int cap);

void tokenizer_load(const char *model_dir, Tokenizer *t);
void tokenizer_free(Tokenizer *t);
void tokenizer_summary(const Tokenizer *t);
int  tokenizer_find(const Tokenizer *t, const unsigned char *b, int n); /* -1 absent */
int  tokenizer_rank(const Tokenizer *t, int left, int right);          /* -1 absent */
int  tokenizer_encode(const Tokenizer *t, const char *text, int len, int *out, int cap);
int  tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int cap);

/* ---------------------------------------------------------------- Metal
 * The GPU side, behind a C interface: no object pointer ever escapes
 * src/metal.m. Kernel indices are positional and match kernels.metal. */
typedef struct MetalContext MetalContext;
enum { MM_NAIVE = 0, MM_SIMDGROUP = 1, MM_TENSOROPS = 2 };

MetalContext *metal_init(const char *metallib_path);
void   metal_shutdown(MetalContext *ctx);
void   metal_info(const MetalContext *ctx);
bool   metal_has_kernel(const MetalContext *ctx, int which);

/* C[M,N] = A[M,K] * B[N,K]^T, with A fp32, B bf16 and C fp32. Returns the
 * GPU time in seconds, measured by the GPU rather than around the submit. */
double metal_dispatch_floor(MetalContext *ctx);

/* A matmul with its buffers allocated once, so a benchmark can repeat the
 * kernel without re-measuring the driver. */
typedef struct MetalMatmul MetalMatmul;
MetalMatmul *metal_matmul_prepare(MetalContext *ctx, int M, int N, int K);
void   metal_matmul_upload(MetalMatmul *mm, const float *A, const uint16_t *B);
double metal_matmul_run(MetalMatmul *mm, int which);
double metal_matmul_run_n(MetalMatmul *mm, int which, int count, double *wall_out);
double metal_matmul_run_tiled(MetalMatmul *mm, const char *kernel, int tile_m, int tile_n,
                              bool thread_scope, int count, double *wall_out);
void   metal_matmul_fill_c(MetalMatmul *mm, float value);   /* poison C before a check */
void   metal_matmul_download(MetalMatmul *mm, float *C);
void   metal_matmul_free(MetalMatmul *mm);

/* A quantised matmul, prepared once. `kernel` names the Metal function; A is
 * bf16 when bf16_act (Q4), fp32 otherwise. Pipelines come from the cache. */
typedef struct MetalQMatmul MetalQMatmul;
MetalQMatmul *metal_qmatmul_prepare(MetalContext *ctx, const char *kernel,
                                    int M, int N, int K, int G, int bits, bool bf16_act);
void   metal_qmatmul_upload(MetalQMatmul *mm, const void *A, const uint8_t *Q, const uint16_t *D);
double metal_qmatmul_run(MetalQMatmul *mm);
void   metal_qmatmul_download(MetalQMatmul *mm, float *C);
void   metal_qmatmul_free(MetalQMatmul *mm);

/* --quant-check: every quantised kernel against matmul_q on the CPU. */
bool   quant_check(MetalContext *ctx);

/* The measurement harness. Writes one CSV row per (kernel, shape). */
void   bench_matmul(MetalContext *ctx, const char *csv_path);
void   bench_qmatmul(MetalContext *ctx, const char *csv_path);   /* Phase 3 kernels */
void   bench_e2e(const char *model_dir, const char *csv_path);       /* whole forward passes */
void   bench_dispatch(MetalContext *ctx, const char *csv_path);     /* Phase 4: cost per dispatch */
void   bench_profile(const char *model_dir, const char *csv_path);  /* Phase 4: GPU time by op group */
void   bench_moe(const char *model_dir, const char *csv_path);      /* Phase 4: verify cost vs experts */
void   bench_moe_e2e(const char *model_dir, const char *csv_path);  /* Phase 4: MoE decode and verify */

/* Raw Metal objects, as void* so nothing else has to include Metal headers. */
void  *metal_device(MetalContext *ctx);
void  *metal_queue(MetalContext *ctx);
void  *metal_pipeline(MetalContext *ctx, const char *name);

/* --------------------------------------------------------- GPU forward
 * The whole model resident on the GPU. The weights are not uploaded: the
 * mmapped checkpoint is bound with no copy, so the GPU reads the same
 * physical pages the CPU does and "moving the model to the GPU" costs one
 * pointer. Every forward pass is a single command buffer. */
typedef struct GpuModel GpuModel;
GpuModel *gpu_model_create(MetalContext *ctx, const SafeTensors *st,
                           const Qwen3Config *cfg, int max_seq, int max_rows);
void gpu_model_free(GpuModel *g);
void gpu_set_matmul_kernel(GpuModel *g, int which);
void gpu_set_small_tile(GpuModel *g, bool on);   /* 8x32 TensorOps tiles for M <= 8, default on */
/* Routing trace, MoE models only. When on, every MoE layer of a pass of n <= 32
 * tokens records its top-k experts: trace[(layer * 32 + t) * k + j], most
 * probable first. Dense layers keep 0xFFFFFFFF. Valid until the next pass. */
void gpu_set_routing_trace(GpuModel *g, bool on);
const uint32_t *gpu_routing_trace(const GpuModel *g);
/* Returns the GPU seconds the pass took, measured by the GPU. */
double gpu_forward(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                   float *logits_out);
/* The same pass cut into one encoder per op group, GPU-timestamped; adds each
 * group's seconds into seconds[GPU_PROF_CLASSES] and computes no logits. */
enum { GPU_PROF_CLASSES = 8 };
double gpu_forward_profile(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                           double *seconds);
const char *gpu_prof_name(int cls);
/* One MoE block of `layer` with routing forced: token t runs experts[t*k..]. */
double gpu_moe_bench(GpuModel *g, int layer, int n, const int *experts, int *groups_out);

/* --ppl: perplexity of this model on a text file, paired against a reference
 * model on the same windows (KL, top-1 agreement). One CSV row appended. */
void eval_perplexity(const char *model_dir, const char *ref_dir, const char *corpus,
                     const char *csv_path);
double metal_matmul(MetalContext *ctx, int which,
                    float *C, const float *A, const uint16_t *B,
                    int M, int N, int K);

/* ------------------------------------------------------------- sampling */
int  sample(const float *logits, int vocab, float temperature, float top_p,
            int top_k, uint64_t *rng);
void chat_format(char *out, int cap, const char *user, bool thinking);
/* Generate up to max_new tokens. temperature 0 means greedy, which is the
 * only setting under which two implementations can be compared token for
 * token. If out_ids is non-NULL the generated ids are written there too. */
int  generate(const Tokenizer *tok, const Weights *w, const Qwen3Config *cfg,
              RunState *s, GpuModel *gpu, const char *prompt, int max_new,
              float temperature, float top_p, int top_k, uint64_t seed,
              int *out_ids, bool quiet);

/* ----------------------------------------------------------- speculation
 * Greedy speculative decoding on the GPU with a prompt-lookup drafter
 * (speculate.c). Emits exactly what greedy generate() emits; the stats say
 * how many passes that took. max_rows must be at least max_draft + 1. */
typedef struct {
    int    generated, passes, drafted, accepted;
    double decode_s;          /* wall time after prefill                    */
    double draft_s, verify_s; /* of which drafting, and the target's passes  */
    int    groups;            /* entries used in group_len                   */
    short  group_len[2048];   /* tokens emitted per loop step: the token the
                               * previous pass decided, plus the drafts this
                               * pass accepted -- what a UI colours as a unit */
} SpecStats;
/* drafter NULL: prompt lookup. Otherwise a draft model sharing the target's
 * vocabulary, with a context of at least max_seq. */
int  generate_speculative(const Tokenizer *tok, const Qwen3Config *cfg, GpuModel *gpu,
                          GpuModel *drafter, int max_seq, int max_rows, const char *prompt,
                          int max_new, int max_draft, int *out_ids, SpecStats *stats);

#endif /* PICOFORGE_H */
