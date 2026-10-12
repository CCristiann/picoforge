/* Bounded binary fixture transport; compiled with ASan/UBSan. CPU only. */
#include "qwen38_d_internal.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void read_items(void *p, size_t size, size_t n) {
    assert(fread(p, size, n, stdin) == n);
}
static void write_items(const void *p, size_t size, size_t n) {
    assert(fwrite(p, size, n, stdout) == n);
}
static float *read_floats(size_t n) {
    assert(n <= 4000000);
    float *p = malloc((n ? n : 1) * sizeof(float));
    assert(p); read_items(p, sizeof(float), n); return p;
}

static void lifecycle(void) {
    Qwen38AttentionConfig c = {4,2,8,2,1,8,4,8,4,1e-6f,10000};
    Qwen38AttentionPlan p;
    Qwen38Error e;
    Qwen38AttentionState *s = NULL, *other = NULL;
    Qwen38AttentionWorkspace *w = NULL, *ow = NULL;
    Qwen38StateKey key = {&c,3,1};
    assert(qwen38_attention_plan(&c, 32, 8, &p, &e) == Q38_OK);
    size_t budget = p.cache_bytes + p.workspace_bytes;
    assert(qwen38_attention_create(&c, &p, key, budget - 1, &s, &w, &e) == Q38_CAPACITY);
    assert(!s && !w && e.code == Q38_CAPACITY);
    assert(qwen38_attention_create(&c, &p, key, budget, &s, &w, &e) == Q38_OK);
    assert(qwen38_attention_create(&c, &p, key, budget, &other, &ow, &e) == Q38_OK);
    assert(qwen38_attention_destroy(s, ow, &e) == Q38_OWNER);
    Qwen38AttentionTicket *t = NULL, *unused = NULL;
    assert(qwen38_attention_begin(s, 1, 1, 1, &t, &e) == Q38_BAD_STATE && !t);
    assert(qwen38_attention_begin(s, 0, 1, 2, &t, &e) == Q38_STALE && !t);
    assert(qwen38_attention_begin(s, 0, 9, 1, &t, &e) == Q38_CAPACITY && !t);
    assert(qwen38_attention_begin(s, 0, 0, 1, &t, &e) == Q38_OK);
    assert(qwen38_attention_begin(s, 0, 0, 1, &unused, &e) == Q38_BUSY && !unused);
    assert(qwen38_attention_destroy(s, w, &e) == Q38_BUSY);
    assert(qwen38_attention_reset(s, 2, &e) == Q38_BUSY);
    assert(qwen38_attention_finish(t, true, &e) == Q38_OK);
    assert(qwen38_attention_state_info(s).next_pos == 0);
    assert(qwen38_attention_finish(t, true, &e) == Q38_STALE);
    assert(qwen38_attention_begin(s, 0, 1, 1, &t, &e) == Q38_OK);
    assert(qwen38_attention_finish(t, true, &e) == Q38_BAD_STATE);
    assert(qwen38_attention_finish(t, false, &e) == Q38_OK);
    assert(!qwen38_attention_state_info(s).valid);
    assert(qwen38_attention_begin(s, 0, 1, 1, &t, &e) == Q38_BAD_STATE);
    assert(qwen38_attention_reset(s, 1, &e) == Q38_STALE);
    assert(qwen38_attention_reset(s, UINT64_MAX, &e) == Q38_OK);
    assert(qwen38_attention_reset(s, 0, &e) == Q38_STALE);
    assert(qwen38_attention_destroy(s, w, &e) == Q38_OK);
    assert(qwen38_attention_destroy(other, ow, &e) == Q38_OK);
    c = (Qwen38AttentionConfig){24,2,256,4,1,128,4,2048,64,1e-6f,10000000};
    assert(qwen38_attention_plan(&c,8192,1024,&p,&e) == Q38_OK);
    assert(p.cache_bytes >= 37u*1024u*1024u && p.cache_bytes < 37u*1024u*1024u+4096);
    assert(p.workspace_bytes < 192u*1024u*1024u);
    Qwen38AttentionPlan malformed = p; malformed.workspace_bytes--;
    assert(qwen38_attention_create(&c,&malformed,key,SIZE_MAX,&s,&w,&e) == Q38_SHAPE);
    assert(!s && !w);
    c.heads = UINT32_MAX; c.kv_heads = 1; c.head_dim = UINT32_MAX;
    assert(qwen38_attention_plan(&c, UINT32_MAX, 1, &p, &e) == Q38_OVERFLOW);
}

static Qwen38F32 dense(float *data, uint32_t t, uint32_t h, uint32_t d, bool rank2) {
    Qwen38F32 v = {.data=data, .capacity_bytes=(size_t)t*h*d*sizeof(float), .rank=rank2 ? 2 : 3,
                   .dim={t,h,d,0}, .stride_bytes={(size_t)h*d*4,(size_t)d*4,4,0}};
    if (rank2) { v.dim[1]=d; v.dim[2]=0; v.stride_bytes[1]=4; v.stride_bytes[2]=0; }
    return v;
}

static void append_fixture(void) {
    Qwen38AttentionConfig c;
    read_items(&c, sizeof(c), 1);
    uint32_t count, chunks;
    read_items(&count, sizeof(count), 1); read_items(&chunks, sizeof(chunks), 1);
    assert(count && count <= 32768 && chunks && chunks <= 32768);
    uint32_t *ends = calloc(chunks, sizeof(uint32_t)); assert(ends);
    read_items(ends, sizeof(uint32_t), chunks); assert(ends[chunks-1] == count);
    size_t hd = (size_t)c.heads*c.head_dim, kd = (size_t)c.kv_heads*c.head_dim;
    size_t id = (size_t)c.index_heads*c.index_dim;
    float *q = read_floats(count*hd), *gate = read_floats(count*hd);
    float *k = read_floats(count*kd), *v = read_floats(count*kd);
    float *iq = read_floats(count*id), *ik = read_floats((size_t)count*c.index_dim);
    float *blocks = read_floats((size_t)(count/c.ratio)*c.index_dim);
    float *output = calloc(count*hd, sizeof(float)); assert(output);
    Qwen38AttentionPlan plan; Qwen38Error error;
    assert(qwen38_attention_plan(&c, count, count, &plan, &error) == Q38_OK);
    Qwen38AttentionState *s; Qwen38AttentionWorkspace *w;
    Qwen38StateKey key = {&c,3,1};
    assert(qwen38_attention_create(&c, &plan, key, plan.cache_bytes+plan.workspace_bytes, &s,&w,&error) == Q38_OK);
    Qwen38Status status = Q38_OK;
    uint32_t begin = 0;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        assert(ends[chunk] >= begin && ends[chunk] <= count);
        uint32_t t = ends[chunk]-begin, b = begin/c.ratio, nb = ends[chunk]/c.ratio-b;
        Q38DPrepared in = {dense(q+begin*hd,t,c.heads,c.head_dim,false),
            dense(gate+begin*hd,t,c.heads,c.head_dim,false), dense(k+begin*kd,t,c.kv_heads,c.head_dim,false),
            dense(v+begin*kd,t,c.kv_heads,c.head_dim,false), dense(iq+begin*id,t,c.index_heads,c.index_dim,false),
            dense(ik+(size_t)begin*c.index_dim,t,1,c.index_dim,true),
            dense(blocks+(size_t)b*c.index_dim,nb,1,c.index_dim,true)};
        /* Nontrivial input token pitch; padding must never enter a dot. */
        size_t input_pitch = hd+5;
        float *pq = malloc((t ? t : 1)*input_pitch*4); assert(pq);
        for (size_t j = 0; j < (t ? t : 1)*input_pitch; ++j) pq[j] = 9876;
        for (uint32_t row = 0; row < t; ++row) memcpy(pq+row*input_pitch,q+((size_t)begin+row)*hd,hd*4);
        in.q.data = pq; in.q.capacity_bytes = t*input_pitch*4; in.q.stride_bytes[0] = input_pitch*4;
        size_t pitch = hd+3;
        float *padded = malloc((t ? t : 1)*pitch*sizeof(float)); assert(padded);
        for (size_t j = 0; j < (t ? t : 1)*pitch; ++j) padded[j] = 12345;
        Qwen38F32Mut out = {padded,(size_t)t*pitch*4,3,{t,c.heads,c.head_dim,0},{pitch*4,(size_t)c.head_dim*4,4,0}};
        Qwen38AttentionTicket *ticket;
        assert(qwen38_attention_begin(s,begin,t,1,&ticket,&error) == Q38_OK);
        assert(qwen38_attention_state_info(s).next_pos == begin);
        if (t) {
            Q38DPrepared bad = in; bad.q.dim[2]++;
            assert(q38d_append_prepared(ticket,&bad,out,w,&error) == Q38_SHAPE);
            bad = in; bad.q.stride_bytes[0] = 4;
            assert(q38d_append_prepared(ticket,&bad,out,w,&error) == Q38_STRIDE);
            Qwen38F32Mut small = out; small.capacity_bytes = 0;
            assert(q38d_append_prepared(ticket,&in,small,w,&error) == Q38_CAPACITY);
            assert(q38d_append_prepared(ticket,&in,out,NULL,&error) == Q38_OWNER);
            small = out; small.data = pq; small.stride_bytes[0]=hd*4; small.capacity_bytes=t*hd*4;
            assert(q38d_append_prepared(ticket,&in,small,w,&error) == Q38_INVALID_ARGUMENT);
            assert(qwen38_attention_state_info(s).valid);
        }
        status = q38d_append_prepared(ticket,&in,out,w,&error);
        assert(error.code == status);
        if (status != Q38_OK) {
            assert(!qwen38_attention_state_info(s).valid);
            assert(qwen38_attention_finish(ticket,true,&error) == Q38_BAD_STATE);
            assert(qwen38_attention_finish(ticket,false,&error) == Q38_OK);
            assert(qwen38_attention_reset(s,2,&error) == Q38_OK);
            free(padded); free(pq); break;
        }
        assert(qwen38_attention_state_info(s).next_pos == begin);
        assert(qwen38_attention_finish(ticket,true,&error) == Q38_OK);
        for (uint32_t row = 0; row < t; ++row) {
            memcpy(output+((size_t)begin+row)*hd,padded+row*pitch,hd*4);
            for (size_t j = hd; j < pitch; ++j) assert(padded[row*pitch+j] == 12345);
            for (size_t j = hd; j < input_pitch; ++j) assert(pq[row*input_pitch+j] == 9876);
        }
        free(padded); free(pq); begin = ends[chunk];
    }
    write_items(&status,sizeof(status),1); write_items(output,sizeof(float),count*hd);
    if (status == Q38_OK) {
        assert(!memcmp(q38d_cache_keys(s),k,count*kd*4));
        assert(!memcmp(q38d_cache_raw_index(s),ik,(size_t)count*c.index_dim*4));
        assert(!memcmp(q38d_cache_blocks(s),blocks,(size_t)(count/c.ratio)*c.index_dim*4));
        /* Reset leaves old bytes resident but makes every old row invisible. */
        assert(qwen38_attention_reset(s,2,&error) == Q38_OK);
        Qwen38AttentionTicket *ticket;
        assert(qwen38_attention_begin(s,0,1,1,&ticket,&error) == Q38_STALE);
        assert(qwen38_attention_begin(s,0,1,2,&ticket,&error) == Q38_OK);
        Q38DPrepared first = {dense(q,1,c.heads,c.head_dim,false), dense(gate,1,c.heads,c.head_dim,false),
            dense(k,1,c.kv_heads,c.head_dim,false), dense(v,1,c.kv_heads,c.head_dim,false),
            dense(iq,1,c.index_heads,c.index_dim,false), dense(ik,1,1,c.index_dim,true),
            dense(blocks,1/c.ratio,1,c.index_dim,true)};
        float *one = calloc(hd,sizeof(float)); assert(one);
        Qwen38F32Mut out = {one,hd*4,3,{1,c.heads,c.head_dim,0},{hd*4,(size_t)c.head_dim*4,4,0}};
        assert(q38d_append_prepared(ticket,&first,out,w,&error) == Q38_OK);
        assert(!memcmp(one,output,hd*4)); free(one);
        assert(qwen38_attention_finish(ticket,true,&error) == Q38_OK);
        assert(qwen38_attention_state_info(s).next_pos == 1);
    }
    assert(qwen38_attention_destroy(s,w,&error) == Q38_OK);
    free(ends); free(q); free(gate); free(k); free(v); free(iq); free(ik); free(blocks); free(output);
    assert(getchar() == EOF);
}

int main(int argc, char **argv) {
    assert(argc == 2);
    if (!strcmp(argv[1], "lifecycle")) { lifecycle(); puts("lifecycle OK"); return 0; }
    if (!strcmp(argv[1], "append")) { append_fixture(); return 0; }
    uint32_t dims[4];
    read_items(dims, sizeof(uint32_t), 4);
    for (size_t i = 0; i < 4; ++i) assert(dims[i] <= 32768);
    Qwen38Status status;
    if (!strcmp(argv[1], "rope")) {
        float theta;
        read_items(&theta, sizeof(float), 1);
        size_t n = (size_t)dims[0] * dims[1];
        float *x = read_floats(n);
        status = q38d_rope(x, dims[0], dims[1], dims[2], dims[3], theta);
        write_items(&status, sizeof(status), 1); write_items(x, sizeof(float), n); free(x);
    } else if (!strcmp(argv[1], "pool")) {
        float *raw = read_floats((size_t)dims[0]*dims[1]);
        float *pooled = calloc(dims[1] ? dims[1] : 1,sizeof(float)); assert(pooled);
        status = q38d_pool(raw,dims[0],dims[1],pooled);
        write_items(&status,sizeof(status),1); write_items(pooled,sizeof(float),dims[1]);
        free(raw); free(pooled);
    } else if (!strcmp(argv[1], "scores")) {
        float *q = read_floats((size_t)dims[0] * dims[1]);
        float *k = read_floats((size_t)dims[2] * dims[1]);
        float *s = calloc(dims[2] ? dims[2] : 1, sizeof(float)); assert(s);
        status = q38d_scores(q, k, dims[0], dims[1], dims[2], s);
        write_items(&status, sizeof(status), 1); write_items(s, sizeof(float), dims[2]);
        free(q); free(k); free(s);
    } else if (!strcmp(argv[1], "attend")) {
        Qwen38AttentionConfig c = {.heads=dims[0], .kv_heads=dims[1], .head_dim=dims[2]};
        size_t hd = (size_t)c.heads * c.head_dim;
        size_t cache = (size_t)dims[3] * c.kv_heads * c.head_dim;
        float *q = read_floats(hd), *gate = read_floats(hd);
        float *k = read_floats(cache), *v = read_floats(cache);
        uint32_t n; read_items(&n, sizeof(n), 1); assert(n <= dims[3]);
        uint32_t *ids = calloc(n ? n : 1, sizeof(uint32_t)); assert(ids);
        read_items(ids, sizeof(uint32_t), n);
        float *out = calloc(hd, sizeof(float)), *scratch = calloc(c.head_dim + 16, sizeof(float));
        assert(out && scratch);
        status = q38d_attend(&c, q, gate, k, v, dims[3], ids, n, out, scratch);
        write_items(&status, sizeof(status), 1); write_items(out, sizeof(float), hd);
        free(q); free(gate); free(k); free(v); free(ids); free(out); free(scratch);
    } else {
        assert(!strcmp(argv[1], "select"));
        float *s = read_floats(dims[0]);
        uint32_t n = dims[0] < dims[1] ? dims[0] : dims[1];
        uint32_t *heap = calloc(n ? n : 1, sizeof(uint32_t));
        uint32_t *ids = calloc(n ? n : 1, sizeof(uint32_t)); assert(heap && ids);
        status = q38d_select(s, dims[0], dims[1], heap, ids);
        write_items(&status, sizeof(status), 1); write_items(ids, sizeof(uint32_t), n);
        free(s); free(heap); free(ids);
    }
    assert(getchar() == EOF);
    return 0;
}
