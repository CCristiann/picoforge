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

/* Every activation buffer the forward pass needs, allocated once. */
typedef struct {
    int    seq;
    float *x, *xb;                /* residual stream, and a scratch copy     */
    float *q, *k, *v;             /* projections, laid out (seq, heads, dim) */
    float *att, *attout;          /* one head-row of scores; merged heads    */
    float *hb, *hb2;              /* SwiGLU gate and up                      */
    float *logits;                /* (seq, vocab)                            */
} RunState;

void weights_bind(const SafeTensors *st, const Qwen3Config *cfg, Weights *w);
void weights_free(Weights *w);
void state_alloc(RunState *s, const Qwen3Config *cfg, int seq);
void state_free(RunState *s);
void forward(const int *tokens, int seq, const Weights *w,
             const Qwen3Config *cfg, RunState *s);

/* ------------------------------------------------------------- tokenizer
 * Byte-level BPE. The vocabulary starts from all 256 possible bytes, every
 * one of which really is present, so NOTHING is untokenisable: a binary
 * file, an unseen script, a novel emoji all encode — badly, perhaps, but
 * always. That is the whole point of going byte-level.
 *
 * The files spell tokens in a 256-character Unicode alphabet only because
 * JSON cannot hold raw control bytes. That is presentation. Inside this
 * struct every token is the byte string it actually stands for. */
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
} Tokenizer;

void tokenizer_load(const char *model_dir, Tokenizer *t);
void tokenizer_free(Tokenizer *t);
void tokenizer_summary(const Tokenizer *t);
int  tokenizer_find(const Tokenizer *t, const unsigned char *b, int n); /* -1 absent */
int  tokenizer_rank(const Tokenizer *t, int left, int right);          /* -1 absent */

#endif /* PICOFORGE_H */
