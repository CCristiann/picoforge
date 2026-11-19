#include "qwen4_text_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef struct { Qwen4Bytes text; const int32_t *ids; size_t count; Qwen4Bytes decoded; } EncodeCase;
typedef struct { const int32_t *ids; size_t count; Qwen4Bytes keep, skip; } DecodeCase;
#include "text_cases.inc"
typedef struct {
    const Qwen4Json *messages, *tools; Qwen4ChatOptions options;
    Qwen4Bytes expected; bool error; const int32_t *ids; size_t count;
} ChatCase;
#include "chat_cases.inc"

static void chat_cases(Qwen4Tokenizer *t) {
    for (size_t i = 0; i < sizeof chats/sizeof *chats; ++i) {
        const ChatCase *test = chats+i; size_t needed; Qwen4Error e;
        Qwen4Status s = qwen4_chat_render(t,test->messages,test->tools,test->options,NULL,0,&needed,&e);
        if (test->error) { assert(s != QW4_OK && !needed); continue; }
        if (s || needed != test->expected.len) fprintf(stderr,"chat case %zu: status %d %s bytes %zu/%zu\n",i,s,e.detail,needed,test->expected.len);
        assert(s == QW4_OK && needed == test->expected.len);
        uint8_t *out = malloc(needed+1); assert(out); memset(out,0xA5,needed+1);
        if (needed) {
            assert(qwen4_chat_render(t,test->messages,test->tools,test->options,out,needed-1,&needed,&e) == QW4_CAPACITY);
            for (size_t j = 0; j <= needed; ++j) assert(out[j] == 0xA5);
        }
        assert(qwen4_chat_render(t,test->messages,test->tools,test->options,out,needed,&needed,&e) == QW4_OK);
        if (memcmp(out,test->expected.data,needed)) fprintf(stderr,"chat bytes mismatch %zu\n",i);
        assert(!memcmp(out,test->expected.data,needed) && out[needed] == 0xA5);
        int32_t *ids = malloc((test->count+1)*sizeof *ids); assert(ids); size_t n;
        assert(qwen4_encode(t,(Qwen4Bytes){out,needed},ids,test->count,&n,&e) == QW4_OK);
        assert(n == test->count && !memcmp(ids,test->ids,n*sizeof *ids));
        free(ids); free(out);
    }
    printf("chat OK: %zu HF fixtures, exact bytes and complete-prompt IDs\n",sizeof chats/sizeof *chats);
}
static void chat_rejected(Qwen4Tokenizer *t, const Qwen4Json *messages,
                          const Qwen4Json *tools, Qwen4Status wanted) {
    uint8_t bytes[16]; memset(bytes,0xA5,sizeof bytes); size_t n = 99; Qwen4Error e;
    Qwen4ChatOptions opts = {.enable_thinking=QW4_FALSE};
    assert(qwen4_chat_render(t,messages,tools,opts,bytes,sizeof bytes,&n,&e) == wanted);
    assert(e.code == wanted && !n);
    for (size_t i = 0; i < sizeof bytes; ++i) assert(bytes[i] == 0xA5);
}
#define BYTES(s) {(const uint8_t *)(s),sizeof(s)-1}
static void chat_errors(Qwen4Tokenizer *t) {
    Qwen4Json role = {.kind=QW4_JSON_STRING,.as.string=BYTES("user")};
    Qwen4Json text = {.kind=QW4_JSON_STRING,.as.string=BYTES("X")};
    Qwen4JsonMember fields[] = {{BYTES("role"),&role},{BYTES("content"),&text}};
    Qwen4Json message = {.kind=QW4_JSON_OBJECT,.as.object={fields,2}};
    Qwen4Json messages = {.kind=QW4_JSON_ARRAY,.as.array={&message,1}};
    /* The public byte span explicitly permits NULL for an empty string. */
    text.as.string = (Qwen4Bytes){NULL,0}; size_t empty_size;
    Qwen4ChatOptions empty_opts = {.enable_thinking=QW4_FALSE};
    assert(qwen4_chat_render(t,&messages,NULL,empty_opts,NULL,0,&empty_size,NULL) == QW4_OK);
    text.as.string = (Qwen4Bytes)BYTES("X");
    fields[1].key = fields[0].key;
    chat_rejected(t,&messages,NULL,QW4_FORMAT); fields[1].key = (Qwen4Bytes)BYTES("content");
    fields[1].value = &messages; chat_rejected(t,&messages,NULL,QW4_FORMAT); fields[1].value = &text;
    fields[1].key = (Qwen4Bytes)BYTES("unknown"); chat_rejected(t,&messages,NULL,QW4_UNSUPPORTED);
    fields[1].key = (Qwen4Bytes)BYTES("content");
    text.as.string = (Qwen4Bytes)BYTES("\xC0\x80"); chat_rejected(t,&messages,NULL,QW4_FORMAT);
    text = (Qwen4Json){.kind=QW4_JSON_FLOAT,.as.number=INFINITY};
    chat_rejected(t,&messages,NULL,QW4_NONFINITE);
    text = (Qwen4Json){.kind=QW4_JSON_STRING,.as.string=BYTES("image")};
    Qwen4JsonMember image_fields[] = {{BYTES("type"),&text}};
    Qwen4Json image = {.kind=QW4_JSON_OBJECT,.as.object={image_fields,1}};
    Qwen4Json parts = {.kind=QW4_JSON_ARRAY,.as.array={&image,1}};
    fields[1].value = &parts; chat_rejected(t,&messages,NULL,QW4_UNSUPPORTED);
    fields[1].value = &text;
    Qwen4Json nested[65];
    for (size_t i = 0; i < 64; ++i) nested[i] = (Qwen4Json){.kind=QW4_JSON_ARRAY,.as.array={nested+i+1,1}};
    nested[64] = text; chat_rejected(t,nested,NULL,QW4_RANGE);
    Qwen4Json invalid_tools = {.kind=QW4_JSON_OBJECT}; chat_rejected(t,&messages,&invalid_tools,QW4_TYPE);
}
#undef BYTES

static void decode(Qwen4Tokenizer *t, const int32_t *ids, size_t n, bool skip, Qwen4Bytes expected) {
    Qwen4Error e; size_t needed;
    assert(qwen4_decode_utf8(t,ids,n,skip,NULL,0,&needed,&e) == QW4_OK && needed == expected.len);
    uint8_t *out = malloc(needed+1); assert(out); memset(out,0xA5,needed+1);
    if (needed) {
        assert(qwen4_decode_utf8(t,ids,n,skip,out,needed-1,&needed,&e) == QW4_CAPACITY);
        for (size_t i = 0; i <= needed; ++i) assert(out[i] == 0xA5);
    }
    assert(qwen4_decode_utf8(t,ids,n,skip,out,needed,&needed,&e) == QW4_OK);
    assert(!memcmp(out,expected.data,needed) && out[needed] == 0xA5);
    Qwen4Decoder *d;
    assert(qwen4_decoder_create(t,skip,&d,&e) == QW4_OK);
    size_t pos = 0;
    for (size_t i = 0; i < n; ++i) {
        assert(qwen4_decoder_append(d,ids+i,1,NULL,0,&needed,&e) == QW4_OK);
        if (needed) assert(qwen4_decoder_append(d,ids+i,1,out+pos,needed-1,&needed,&e) == QW4_CAPACITY);
        assert(qwen4_decoder_append(d,ids+i,1,out+pos,needed,&needed,&e) == QW4_OK);
        pos += needed;
    }
    assert(qwen4_decoder_finish(d,NULL,0,&needed,&e) == QW4_OK);
    if (needed) assert(qwen4_decoder_finish(d,out+pos,needed-1,&needed,&e) == QW4_CAPACITY);
    assert(qwen4_decoder_finish(d,out+pos,needed,&needed,&e) == QW4_OK); pos += needed;
    assert(pos == expected.len && !memcmp(out,expected.data,pos));
    assert(qwen4_decoder_append(d,NULL,0,NULL,0,&needed,&e) == QW4_BAD_STATE);
    qwen4_decoder_reset(d);
    assert(qwen4_decoder_append(d,NULL,0,NULL,0,&needed,&e) == QW4_OK && !needed);
    qwen4_decoder_free(d); free(out);
}
int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1],"--nfc")) {
        size_t len; char *in = slurp(argv[2],&len); uint8_t *out; size_t n;
        assert(qw4a_nfc((Qwen4Bytes){(uint8_t *)in,len},&out,&n,NULL) == QW4_OK);
        FILE *f = fopen(argv[3],"wb"); assert(f);
        assert(fwrite(out,1,n,f) == n && !fclose(f)); free(in); free(out); return 0;
    }
    assert(argc == 2);
    Gguf g; gguf_open(argv[1],&g);
    Qwen4Config *c; Qwen4Tokenizer *t; Qwen4Error e;
    assert(qwen4_config_load(&g,&c,&e) == QW4_OK);
    for (size_t i = 0; i < g.metadata_count; ++i) {
        GgufMetadata *m = g.metadata+i;
        if (strcmp(m->name,"tokenizer.ggml.tokens") && strcmp(m->name,"tokenizer.ggml.merges") &&
            strcmp(m->name,"tokenizer.chat_template")) continue;
        uint32_t saved = m->type; m->type = 7; t = (void *)1;
        assert(qwen4_tokenizer_load(&g,c,&t,&e) == QW4_TYPE && !t); m->type = saved;
        size_t bytes = m->bytes; m->bytes = 0;
        assert(qwen4_tokenizer_load(&g,c,&t,&e) == QW4_FORMAT && !t); m->bytes = bytes;
    }
    Qwen4Status s = qwen4_tokenizer_load(&g,c,&t,&e);
    if (s) fprintf(stderr,"tokenizer load: %d %s %s\n",s,e.subject,e.detail);
    assert(s == QW4_OK);
    qwen4_config_free(c); gguf_close(&g); /* tokenizer owns every needed byte */
    for (size_t i = 0; i < sizeof encodes/sizeof *encodes; ++i) {
        const EncodeCase *test = encodes+i; size_t needed;
        assert(qwen4_encode(t,test->text,NULL,0,&needed,&e) == QW4_OK);
        int32_t *out = malloc((needed+1)*sizeof *out); assert(out);
        for (size_t j = 0; j <= needed; ++j) out[j] = -77;
        if (needed) {
            assert(qwen4_encode(t,test->text,out,needed-1,&needed,&e) == QW4_CAPACITY);
            for (size_t j = 0; j <= needed; ++j) assert(out[j] == -77);
        }
        assert(qwen4_encode(t,test->text,out,needed,&needed,&e) == QW4_OK);
        if (needed != test->count || memcmp(out,test->ids,needed*sizeof *out)) fprintf(stderr,"encode case %zu\n",i);
        assert(needed == test->count && !memcmp(out,test->ids,needed*sizeof *out) && out[needed] == -77);
        decode(t,out,needed,false,test->decoded); free(out);
    }
    for (size_t i = 0; i < sizeof decodes/sizeof *decodes; ++i) {
        decode(t,decodes[i].ids,decodes[i].count,false,decodes[i].keep);
        decode(t,decodes[i].ids,decodes[i].count,true,decodes[i].skip);
    }
    size_t needed; int32_t invalid = -1; uint8_t sentinel = 0xA5;
    assert(qwen4_decode_utf8(t,&invalid,1,false,&sentinel,1,&needed,&e) == QW4_RANGE && sentinel == 0xA5);
    const uint8_t malformed[] = {0xED,0xA0,0x80};
    assert(qwen4_encode(t,(Qwen4Bytes){malformed,3},NULL,0,&needed,&e) == QW4_FORMAT);
    assert(qwen4_token_is_stop(t,248044) && qwen4_token_is_stop(t,248046));
    assert(!qwen4_token_is_stop(t,248069));
    chat_cases(t);
    chat_errors(t);
    qwen4_tokenizer_free(t);
    printf("text OK: %zu HF encodes, %zu HF decode cases, streaming/capacity/lifetime\n",
        sizeof encodes/sizeof *encodes, sizeof decodes/sizeof *decodes);
    return 0;
}
