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

#endif /* PICOFORGE_H */
