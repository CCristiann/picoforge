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
 * at load time (CLAUDE.md principle #5). Point the engine at Qwen3-1.7B
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
} Qwen3Config;

/* Fail loudly and immediately. Printf-style, prefixed, exit(1).
 * There is no error-recovery story in this engine on purpose: an
 * inference engine that limps on after a malformed model file produces
 * plausible garbage, which is the one failure mode we cannot detect. */
void  die(const char *fmt, ...);

/* Read a whole file into a NUL-terminated heap buffer. Caller frees. */
char *slurp(const char *path, size_t *len_out);

void  config_load(const char *model_dir, Qwen3Config *cfg);
void  config_print(const Qwen3Config *cfg);

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
typedef enum { DT_BF16, DT_F32, DT_F16 } DType;

typedef struct {
    char        name[80];     /* longest in Qwen3-0.6B is 47 chars       */
    DType       dtype;
    int         ndim;
    long        shape[4];
    size_t      nelem;        /* product of shape                        */
    const void *data;         /* points INTO the mapping; never freed    */
} Tensor;

typedef struct {
    void   *map;              /* the whole file, mmapped read-only       */
    size_t  map_len;
    Tensor *tensors;
    int     n_tensors;
} SafeTensors;

void          st_open(const char *model_dir, SafeTensors *st);
void          st_close(SafeTensors *st);
const Tensor *st_find(const SafeTensors *st, const char *name);  /* dies if absent */
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
void  rmsnorm(float *out, const float *x, const uint16_t *w, int n, float eps);
void  softmax(float *x, int n);
void  rope_apply(float *x, int head_dim, int pos, float theta);
float silu(float z);

/* ----------------------------------------------------------------- model
 * Weights are borrowed pointers INTO the mapping — nothing here owns bytes,
 * so binding the whole model costs 311 lookups and zero copies. */
typedef struct {
    const uint16_t *input_ln, *q_proj, *k_proj, *v_proj;
    const uint16_t *q_norm, *k_norm, *o_proj;
    const uint16_t *post_attn_ln, *gate_proj, *up_proj, *down_proj;
} LayerWeights;

typedef struct {
    const uint16_t *embed;        /* [vocab, hidden]; tied, so also the LM head */
    const uint16_t *final_norm;
    LayerWeights   *layers;
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
void   metal_matmul_download(MetalMatmul *mm, float *C);
void   metal_matmul_free(MetalMatmul *mm);

/* The measurement harness. Writes one CSV row per (kernel, shape). */
void   bench_matmul(MetalContext *ctx, const char *csv_path);

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
void gpu_forward(GpuModel *g, const int *tokens, int n, int pos, int logits_from,
                 float *logits_out);
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

#endif /* PICOFORGE_H */
