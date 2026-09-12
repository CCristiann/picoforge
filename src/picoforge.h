/* picoforge.h — shared types and helpers for the C engine.
 *
 * Phase 1. Grows one component at a time, like the Python oracle did.
 */
#ifndef PICOFORGE_H
#define PICOFORGE_H

#include <stdbool.h>
#include <stddef.h>

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

#endif /* PICOFORGE_H */
