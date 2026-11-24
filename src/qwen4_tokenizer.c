/* The Qwen3.8 tokenizer, binary-safe, over the byte-level BPE of qwen4_bpe.c. */
#include "qwen4_text_internal.h"

static Qwen4Status strings(const Gguf *g, const char *key, Qwen4Bytes **out,
                            size_t *n, Qwen4Error *e) {
    Qwen4Bytes b; Qwen4Status s = qw4a_array(g, key, 8, &b, n, e);
    if (s) return s;
    if (*n > 1000000) return qw4a_error(e, QW4_RANGE, key, "tokenizer table limit");
    *out = calloc(*n ? *n : 1, sizeof **out);
    if (!*out) return qw4a_error(e, QW4_OOM, key, "table scratch");
    for (size_t i = 0; i < *n; ++i)
        if (!qw4a_string(&b, *out+i)) return qw4a_error(e, QW4_FORMAT, key, "string extent");
    return b.len ? qw4a_error(e, QW4_FORMAT, key, "array trailing bytes") : QW4_OK;
}
static bool equal(Qwen4Bytes b, const char *s) { return b.len == strlen(s) && !memcmp(b.data,s,b.len); }
static Qwen4Status string_profile(const Gguf *g, const char *key, const char *expected, Qwen4Error *e) {
    Qwen4Bytes b, value; Qwen4Status s = qw4a_meta(g, key, 8, &b, e);
    if (s) return s;
    if (!qw4a_string(&b, &value) || b.len) return qw4a_error(e, QW4_FORMAT, key, "string extent");
    if (!(expected ? equal(value,expected) : qw4a_template_matches(value)))
        return qw4a_error(e, QW4_UNSUPPORTED, key, "unrecognized text profile");
    return QW4_OK;
}
void qwen4_tokenizer_free(Qwen4Tokenizer *t) {
    if (t) { tokenizer_free(&t->bpe); free(t); }
}
static const char *const added_names[] = {
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>",
    "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>", "<think>", "</think>",
    "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
    "<tts_text_bos_single>", "<|audio_pad|>"
};
static Qwen4Status load_tables(const Gguf *g, Qwen4Tokenizer *t, Qwen4Error *e) {
    Qwen4Bytes *tokens = NULL, *merges = NULL, types; size_t nt = 0, nm = 0, nk = 0;
    Qwen4Status s = strings(g, "tokenizer.ggml.tokens", &tokens, &nt, e);
    if (!s) s = strings(g, "tokenizer.ggml.merges", &merges, &nm, e);
    if (!s) s = qw4a_array(g, "tokenizer.ggml.token_type", 5, &types, &nk, e);
    if (s) goto done;
    if (nt != t->vocab_size || nt != nk || types.len != nt*4) {
        s = qw4a_error(e, QW4_SHAPE, "tokens", "vocab/type array length"); goto done;
    }
    size_t base = 0;
    while (base < nt && qw4a_le(types.data+base*4,4) == 1) ++base;
    if (nt-base < 33) { s = qw4a_error(e, QW4_FORMAT, "tokens", "added-token set"); goto done; }
    for (size_t i = 0; i < 33; ++i) {
        uint64_t kind = qw4a_le(types.data+(base+i)*4,4);
        /* HF says six FIM/repository strings are NOT special despite GGUF CONTROL. */
        bool special = i < 14 || i >= 26;
        if (!equal(tokens[base+i], added_names[i]) || kind != (special || (i >= 16 && i <= 21) ? 3u : 4u)) {
            s = qw4a_error(e, QW4_UNSUPPORTED, "tokens", "added-token spelling/type profile"); goto done;
        }
        SpecialToken *a = t->bpe.specials+i;
        a->id = (int)(base+i); a->len = (int)tokens[base+i].len;
        memcpy(a->text, tokens[base+i].data, tokens[base+i].len); t->skip[i] = special;
    }
    t->bpe.n_specials = 33; t->pad = (int32_t)base; t->eos = (int32_t)base+2;
    for (size_t i = base+33; i < nt; ++i) {
        char pad[32]; snprintf(pad, sizeof pad, "[PAD%zu]", i);
        if (qw4a_le(types.data+i*4,4) != 5 || !equal(tokens[i],pad)) {
            s = qw4a_error(e, QW4_FORMAT, "tokens", "unused token profile"); goto done;
        }
    }
    s = qw4a_bpe_load(&t->bpe, tokens, base, merges, nm, e);
done:
    free(tokens); free(merges); return s;
}
Qwen4Status qwen4_tokenizer_load(const Gguf *g, const Qwen4Config *c,
                                  Qwen4Tokenizer **out, Qwen4Error *e) {
    if (out) *out = NULL;
    if (!g || !c || !out) return qw4a_error(e, QW4_INVALID_ARGUMENT, "tokenizer", "null argument");
    Qwen4Status s = string_profile(g,"tokenizer.ggml.model","gpt2",e);
    if (!s) s = string_profile(g,"tokenizer.ggml.pre","qwen35",e);
    if (!s) s = string_profile(g,"tokenizer.chat_template",NULL,e);
    if (s) return s;
    Qwen4Tokenizer *t = calloc(1,sizeof *t);
    if (!t) return qw4a_error(e,QW4_OOM,"tokenizer","allocation");
    t->vocab_size = c->view.vocab_size;
    s = load_tables(g,t,e);
    const char *keys[] = {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.bos_token_id"};
    for (size_t i = 0; !s && i < 3; ++i) {
        Qwen4Bytes b; s = qw4a_meta(g,keys[i],4,&b,e);
        if (!s && (b.len != 4 || qw4a_le(b.data,4) != (uint32_t)(i ? t->pad : t->eos)))
            s = qw4a_error(e,QW4_FORMAT,keys[i],"token ID profile");
    }
    if (!s) {
        Qwen4Bytes b; s = qw4a_meta(g,"tokenizer.ggml.add_bos_token",7,&b,e);
        if (!s && (b.len != 1 || b.data[0])) s = qw4a_error(e,QW4_UNSUPPORTED,"BOS","implicit BOS unsupported");
    }
    /* The PLE n-gram history resets on this token; no PLE, nothing to agree with. */
    if (!s && c->has_ple && c->ngram.eos_id != (uint32_t)t->pad) s = qw4a_error(e,QW4_SHAPE,"PLE EOS","tokenizer mismatch");
    if (s) { qwen4_tokenizer_free(t); return s; }
    *out = t; return qw4a_error(e,QW4_OK,NULL,NULL);
}
Qwen4Status qwen4_encode(const Qwen4Tokenizer *t, Qwen4Bytes text,
    int32_t *out, size_t cap, size_t *needed, Qwen4Error *e) {
    if (needed) *needed = 0;
    if (!t || !needed || (!out && cap) || (!text.data && text.len))
        return qw4a_error(e,QW4_INVALID_ARGUMENT,"encode","null argument");
    if (text.len > QW4A_TEXT_LIMIT) return qw4a_error(e,QW4_RANGE,"encode","4 MiB input limit");
    int *candidate = malloc((text.len*4+1)*sizeof *candidate);
    if (!candidate) return qw4a_error(e,QW4_OOM,"encode","output scratch");
    size_t pos = 0, count = 0; Qwen4Status s = QW4_OK;
    while (pos < text.len) {
        size_t hit = text.len; int which = -1, hit_len = 0;
        for (int i = 0; i < t->bpe.n_specials; ++i) {
            const SpecialToken *a = t->bpe.specials+i;
            for (size_t j = pos; (size_t)a->len <= text.len-j && j <= hit; ++j) {
                if (memcmp(text.data+j,a->text,(size_t)a->len)) continue;
                if (j < hit || (j == hit && a->len > hit_len)) { hit = j; which = i; hit_len = a->len; }
                break;
            }
        }
        uint8_t *norm; size_t n;
        s = qw4a_nfc((Qwen4Bytes){text.data+pos,hit-pos},&norm,&n,e);
        if (s) break;
        if (n > 2*QW4A_TEXT_LIMIT) { free(norm); s = qw4a_error(e,QW4_RANGE,"encode","normalized scratch budget"); break; }
        int *scratch = malloc((n ? n : 1)*3*sizeof *scratch);
        if (!scratch) { free(norm); s = qw4a_error(e,QW4_OOM,"encode","BPE scratch"); break; }
        int got = qw4a_bpe_segment(&t->bpe,(char *)norm,(int)n,scratch,scratch+n,scratch+2*n,
                                  candidate+count,qw4a_category);
        free(scratch); free(norm);
        if (got < 0) { s = qw4a_error(e,QW4_FORMAT,"encode","invalid BPE result"); break; }
        count += (size_t)got; pos = hit;
        if (which >= 0) { candidate[count++] = t->bpe.specials[which].id; pos += (size_t)hit_len; }
    }
    if (!s) {
        *needed = count;
        if (out && cap < count) s = qw4a_error(e,QW4_CAPACITY,"encode","token capacity");
        else if (out && count) memcpy(out,candidate,count*sizeof *out);
    }
    free(candidate);
    return s ? s : qw4a_error(e,QW4_OK,NULL,NULL);
}
bool qwen4_token_is_stop(const Qwen4Tokenizer *t, int32_t id) { return t && (id == t->eos || id == t->pad); }
static Qwen4Bytes token_bytes(const Qwen4Tokenizer *t, int32_t id, bool skip) {
    const Tokenizer *b = &t->bpe;
    if (id < b->n_tokens) return (Qwen4Bytes){b->blob+b->offset[id],(size_t)(b->offset[id+1]-b->offset[id])};
    int k = id-b->n_tokens;
    if (k < b->n_specials && !(skip && t->skip[k]))
        return (Qwen4Bytes){b->specials[k].text,(size_t)b->specials[k].len};
    return (Qwen4Bytes){NULL,0}; /* UNUSED IDs have no spelling at decode. */
}
Qwen4Status qwen4_decode_bytes(const Qwen4Tokenizer *t, const int32_t *ids, size_t count,
    bool skip, uint8_t *out, size_t cap, size_t *needed, Qwen4Error *e) {
    if (needed) *needed = 0;
    if (!t || !needed || (!ids && count) || (!out && cap))
        return qw4a_error(e,QW4_INVALID_ARGUMENT,"decode","null argument");
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= t->vocab_size)
            return qw4a_error(e,QW4_RANGE,"decode","token ID outside vocabulary");
        size_t n = token_bytes(t,ids[i],skip).len;
        if (n > SIZE_MAX-total) return qw4a_error(e,QW4_OVERFLOW,"decode","byte count");
        total += n;
    }
    *needed = total;
    if (out && cap < total) return qw4a_error(e,QW4_CAPACITY,"decode","byte capacity");
    if (out) for (size_t i = 0, pos = 0; i < count; ++i) {
        Qwen4Bytes b = token_bytes(t,ids[i],skip);
        if (b.len) memcpy(out+pos,b.data,b.len);
        pos += b.len;
    }
    return qw4a_error(e,QW4_OK,NULL,NULL);
}
struct Qwen4Decoder {
    const Qwen4Tokenizer *tokenizer; /* borrowed; tokenizer must outlive decoder */
    uint8_t carry[3];
    size_t carry_len;
    bool skip, finished;
};
Qwen4Status qwen4_decoder_create(const Qwen4Tokenizer *t, bool skip,
    Qwen4Decoder **out, Qwen4Error *e) {
    if (out) *out = NULL;
    if (!t || !out) return qw4a_error(e,QW4_INVALID_ARGUMENT,"decoder","null argument");
    Qwen4Decoder *d = calloc(1,sizeof *d);
    if (!d) return qw4a_error(e,QW4_OOM,"decoder","allocation");
    d->tokenizer = t; d->skip = skip; *out = d;
    return qw4a_error(e,QW4_OK,NULL,NULL);
}
void qwen4_decoder_free(Qwen4Decoder *d) { free(d); }
void qwen4_decoder_reset(Qwen4Decoder *d) { if (d) { d->carry_len = 0; d->finished = false; } }
/* Rust/tokenizers lossy UTF-8 consumes a valid prefix of an invalid sequence
 * once; it does not replace each continuation byte of that prefix. */
static size_t invalid_prefix(const uint8_t *p, size_t n) {
    unsigned c = p[0]; size_t want = c >= 194 && c <= 223 ? 2 :
        c >= 224 && c <= 239 ? 3 : c >= 240 && c <= 244 ? 4 : 1;
    if (n > 1 && ((c == 224 && p[1] < 160) || (c == 237 && p[1] >= 160) ||
                  (c == 240 && p[1] < 144) || (c == 244 && p[1] >= 144))) return 1;
    size_t k = 1;
    while (k < want && k < n && p[k] >= 128 && p[k] <= 191) ++k;
    return k;
}
static size_t utf8_render(const uint8_t *raw, size_t n, bool finish, uint8_t *out, size_t *used) {
    size_t pos = 0, total = 0;
    static const uint8_t replacement[] = {239,191,189};
    while (pos < n) {
        unsigned cp; int k = qw4a_utf8(raw+pos,n-pos,&cp);
        if (!k && !finish) break;
        size_t take = k > 0 ? (size_t)k : invalid_prefix(raw+pos,n-pos);
        size_t emit = k > 0 ? take : sizeof replacement;
        if (out) memcpy(out+total,k > 0 ? raw+pos : replacement,emit);
        pos += take; total += emit;
    }
    *used = pos; return total;
}
static Qwen4Status decoder_run(Qwen4Decoder *d, const int32_t *ids, size_t count, bool finish,
    uint8_t *out, size_t cap, size_t *needed, Qwen4Error *e) {
    if (needed) *needed = 0;
    if (!d || !needed || (!out && cap) || (!ids && count))
        return qw4a_error(e,QW4_INVALID_ARGUMENT,"decoder","null argument");
    if (d->finished) return qw4a_error(e,QW4_BAD_STATE,"decoder","already finished");
    size_t bytes; Qwen4Status s = qwen4_decode_bytes(d->tokenizer,ids,count,d->skip,NULL,0,&bytes,e);
    if (s) return s;
    if (bytes > 16*QW4A_TEXT_LIMIT-d->carry_len)
        return qw4a_error(e,QW4_RANGE,"decoder","64 MiB raw-byte limit");
    size_t total = bytes+d->carry_len;
    uint8_t *raw = malloc(total ? total : 1);
    if (!raw) return qw4a_error(e,QW4_OOM,"decoder","raw scratch");
    memcpy(raw,d->carry,d->carry_len);
    s = qwen4_decode_bytes(d->tokenizer,ids,count,d->skip,raw+d->carry_len,bytes,&bytes,e);
    if (s) { free(raw); return s; }
    size_t used; *needed = utf8_render(raw,total,finish,NULL,&used);
    if (out && cap < *needed) s = qw4a_error(e,QW4_CAPACITY,"decoder","UTF-8 capacity");
    else if (out) {
        utf8_render(raw,total,finish,out,&used);
        d->carry_len = total-used;
        if (d->carry_len) memcpy(d->carry,raw+used,d->carry_len);
        d->finished = finish;
    }
    free(raw); return s ? s : qw4a_error(e,QW4_OK,NULL,NULL);
}
Qwen4Status qwen4_decoder_append(Qwen4Decoder *d, const int32_t *ids, size_t count,
    uint8_t *out, size_t cap, size_t *needed, Qwen4Error *e) {
    return decoder_run(d,ids,count,false,out,cap,needed,e);
}
Qwen4Status qwen4_decoder_finish(Qwen4Decoder *d, uint8_t *out, size_t cap,
                                  size_t *needed, Qwen4Error *e) {
    return decoder_run(d,NULL,0,true,out,cap,needed,e);
}
Qwen4Status qwen4_decode_utf8(const Qwen4Tokenizer *t, const int32_t *ids, size_t count,
    bool skip, uint8_t *out, size_t cap, size_t *needed, Qwen4Error *e) {
    if (!t) { if (needed) *needed = 0; return qw4a_error(e,QW4_INVALID_ARGUMENT,"decode","null tokenizer"); }
    Qwen4Decoder d = {.tokenizer=t,.skip=skip};
    return decoder_run(&d,ids,count,true,out,cap,needed,e);
}
