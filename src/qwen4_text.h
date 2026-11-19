/* A: binary-safe tokenizer/template API. No implicit strlen or BOS. */
#ifndef PICOFORGE_QWEN4_TEXT_H
#define PICOFORGE_QWEN4_TEXT_H
#include "qwen4.h"
typedef struct Qwen4Tokenizer Qwen4Tokenizer;
typedef struct Qwen4Decoder Qwen4Decoder;
Qwen4Status qwen4_tokenizer_load(const Gguf *, const Qwen4Config *,
    Qwen4Tokenizer **, Qwen4Error *);
void qwen4_tokenizer_free(Qwen4Tokenizer *);
Qwen4Status qwen4_encode(const Qwen4Tokenizer *, Qwen4Bytes,
    int32_t *out, size_t token_capacity, size_t *needed, Qwen4Error *);
Qwen4Status qwen4_decode_bytes(const Qwen4Tokenizer *, const int32_t *,
    size_t count, bool skip_special, uint8_t *out, size_t byte_capacity,
    size_t *needed, Qwen4Error *);
Qwen4Status qwen4_decode_utf8(const Qwen4Tokenizer *, const int32_t *,
    size_t count, bool skip_special, uint8_t *out, size_t byte_capacity,
    size_t *needed, Qwen4Error *);
bool qwen4_token_is_stop(const Qwen4Tokenizer *, int32_t);
Qwen4Status qwen4_decoder_create(const Qwen4Tokenizer *, bool skip_special,
    Qwen4Decoder **, Qwen4Error *);
void qwen4_decoder_free(Qwen4Decoder *);
void qwen4_decoder_reset(Qwen4Decoder *);
Qwen4Status qwen4_decoder_append(Qwen4Decoder *, const int32_t *, size_t,
    uint8_t *, size_t byte_capacity, size_t *needed, Qwen4Error *);
Qwen4Status qwen4_decoder_finish(Qwen4Decoder *, uint8_t *, size_t byte_capacity,
    size_t *needed, Qwen4Error *);
/* Ordered, borrowed JSON tree. Finite numbers; no cycles, duplicate object
 * keys or unknown message schemas. NULL means absent, JSON_NULL explicit null. */
typedef struct Qwen4Json Qwen4Json;
typedef struct { Qwen4Bytes key; const Qwen4Json *value; } Qwen4JsonMember;
typedef enum {
    QW4_JSON_NULL, QW4_JSON_BOOL, QW4_JSON_INT, QW4_JSON_FLOAT,
    QW4_JSON_STRING, QW4_JSON_ARRAY, QW4_JSON_OBJECT
} Qwen4JsonKind;
struct Qwen4Json {
    Qwen4JsonKind kind;
    union {
        bool boolean;
        int64_t integer;
        double number;
        Qwen4Bytes string;
        struct { const Qwen4Json *items; size_t count; } array;
        struct { const Qwen4JsonMember *items; size_t count; } object;
    } as;
};
typedef enum { QW4_OMITTED, QW4_FALSE, QW4_TRUE } Qwen4Option;
typedef enum { QW4_EFFORT_OMITTED, QW4_EFFORT_LOW, QW4_EFFORT_MEDIUM,
    QW4_EFFORT_XHIGH } Qwen4Effort;
typedef struct {
    Qwen4Option enable_thinking, preserve_thinking;
    Qwen4Effort reasoning_effort;
    bool add_generation_prompt;
} Qwen4ChatOptions;
Qwen4Status qwen4_chat_render(const Qwen4Tokenizer *,
    const Qwen4Json *messages, const Qwen4Json *tools,
    Qwen4ChatOptions, uint8_t *out, size_t byte_capacity,
    size_t *needed, Qwen4Error *);
#endif
