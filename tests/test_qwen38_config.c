#include "qwen38.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static GgufMetadata *field(Gguf *g, const char *key) {
    for (size_t i = 0; i < g->metadata_count; ++i)
        if (!strcmp(g->metadata[i].name, key)) return g->metadata+i;
    abort();
}
static void rejected(Gguf *g, const char *key, uint64_t value, size_t relative,
                     size_t width, Qwen38Status expected) {
    GgufMetadata *m = field(g, key);
    unsigned char *p = (unsigned char *)g->header+m->offset+relative, old[8];
    assert(width <= sizeof old && relative+width <= m->bytes);
    memcpy(old, p, width);
    for (size_t i = 0; i < width; ++i) p[i] = (unsigned char)(value >> (8*i));
    Qwen38Config *c = (void *)1; Qwen38Error error;
    assert(qwen38_config_load(g, &c, &error) == expected && error.code == expected && !c);
    memcpy(p, old, width);
}
int main(int argc, char **argv) {
    assert(argc == 2);
    Gguf source; gguf_open(argv[1], &source);
    /* Copy only the header (not the mapping's payload tail) for fault injection. */
    Gguf g = source;
    g.header_size = source.data_offset;
    g.header = malloc(g.header_size);
    assert(g.header); memcpy(g.header, source.header, g.header_size);
    Qwen38Config *c; Qwen38Error error;
    assert(qwen38_config_load(&g, &c, &error) == Q38_OK);
    const Qwen38ConfigView *v = qwen38_config_view(c);
    assert(v && !strcmp(v->semantic_profile, "qwen38-q2-text-f32-v1"));
    assert(v->blocks == v->trunk_blocks+v->mtp_blocks);
    const Qwen38Layer *l = NULL; const Qwen38NgramConfig *n = NULL;
    size_t gdn = 0, qsa = 0, ple = 0;
    for (uint32_t i = 0; i < v->blocks; ++i) {
        assert(qwen38_config_layer(c, i, &l, &error) == Q38_OK);
        assert(l->domain == (i < v->trunk_blocks ? Q38_TRUNK : Q38_MTP));
        if (l->kind == Q38_LAYER_GDN) ++gdn; else ++qsa;
        if (l->has_ple) {
            assert(qwen38_config_ngram(c, i, &n, &error) == Q38_OK);
            assert(n->head_count*n->head_dim == v->hidden_size && n->layer == i); ++ple;
        } else assert(qwen38_config_ngram(c, i, &n, &error) == Q38_MISSING && !n);
    }
    assert(ple == 1 && gdn+qsa == v->blocks);
    assert(qwen38_config_layer(c, v->blocks, &l, &error) == Q38_RANGE && !l);
    printf("config OK: E=%u V=%u trunk=%u MTP=%u GDN=%zu QSA=%zu\n",
           v->hidden_size, v->vocab_size, v->trunk_blocks, v->mtp_blocks, gdn, qsa);
    rejected(&g, "qwen4exp.embedding_length", 0, 0, 4, Q38_RANGE);
    rejected(&g, "qwen4exp.attention.head_count_kv", 5, 0, 4, Q38_SHAPE);
    rejected(&g, "qwen4exp.vocab_size", 1, 0, 4, Q38_RANGE);
    rejected(&g, "qwen4exp.attention.layer_norm_rms_epsilon", 0x7fc00000, 0, 4, Q38_NONFINITE);
    rejected(&g, "qwen4exp.ple.layer_multipliers", UINT64_MAX, 12, 8, Q38_RANGE);
    rejected(&g, "qwen4exp.ple.head_offsets", 1, 12, 8, Q38_RANGE);
    rejected(&g, "qwen4exp.ple.row_count", 1, 0, 4, Q38_RANGE);
    rejected(&g, "ds4.qwen4.down.physical_input", 7, 0, 4, Q38_SHAPE);
    rejected(&g, "qwen4exp.attention.compress_ratios", 1, 12, 8, Q38_SHAPE);
    GgufMetadata *m = field(&g, "qwen4exp.embedding_length");
    uint32_t type = m->type; m->type = 7;
    Qwen38Config *bad = (void *)1;
    assert(qwen38_config_load(&g, &bad, &error) == Q38_TYPE && !bad); m->type = type;
    size_t bytes = m->bytes; m->bytes = g.header_size;
    assert(qwen38_config_load(&g, &bad, &error) == Q38_FORMAT && !bad); m->bytes = bytes;
    char name[128]; memcpy(name, m->name, sizeof name); m->name[0] = 'X';
    assert(qwen38_config_load(&g, &bad, &error) == Q38_MISSING && !bad); memcpy(m->name, name, sizeof name);
    assert(qwen38_config_load(NULL, &bad, NULL) == Q38_INVALID_ARGUMENT && !bad);
    /* Config remains usable after reader close: arrays and scalar fields owned. */
    free(g.header); gguf_close(&source);
    for (uint32_t i = 0; i < v->blocks; ++i)
        if (qwen38_config_ngram(c, i, &n, &error) == Q38_OK) assert(n->multipliers[0] > 0);
    qwen38_config_free(c); qwen38_config_free(NULL);
    puts("config malformed metadata and lifetime OK");
    return 0;
}
