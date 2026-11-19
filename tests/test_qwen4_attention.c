/* Bounded binary fixture transport; compiled with ASan/UBSan. CPU only. */
#include "qwen4_attention_internal.h"
#include "qwen4.h"
#include "qwen4_packed.h"
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
    Qwen4AttentionConfig c = {4,2,8,2,1,8,4,8,4,1e-6f,10000};
    Qwen4AttentionPlan p;
    Qwen4Error e;
    Qwen4AttentionState *s = NULL, *other = NULL;
    Qwen4AttentionWorkspace *w = NULL, *ow = NULL;
    Qwen4StateKey key = {&c,3,1};
    assert(qwen4_attention_plan(&c, 32, 8, &p, &e) == QW4_OK);
    size_t budget = p.cache_bytes + p.workspace_bytes;
    assert(qwen4_attention_create(&c, &p, key, budget - 1, &s, &w, &e) == QW4_CAPACITY);
    assert(!s && !w && e.code == QW4_CAPACITY);
    assert(qwen4_attention_create(&c, &p, key, budget, &s, &w, &e) == QW4_OK);
    assert(qwen4_attention_create(&c, &p, key, budget, &other, &ow, &e) == QW4_OK);
    assert(qwen4_attention_destroy(s, ow, &e) == QW4_OWNER);
    Qwen4AttentionTicket *t = NULL, *unused = NULL;
    assert(qwen4_attention_begin(s, 1, 1, 1, &t, &e) == QW4_BAD_STATE && !t);
    assert(qwen4_attention_begin(s, 0, 1, 2, &t, &e) == QW4_STALE && !t);
    assert(qwen4_attention_begin(s, 0, 9, 1, &t, &e) == QW4_CAPACITY && !t);
    assert(qwen4_attention_begin(s, 0, 0, 1, &t, &e) == QW4_OK);
    assert(qwen4_attention_begin(s, 0, 0, 1, &unused, &e) == QW4_BUSY && !unused);
    assert(qwen4_attention_destroy(s, w, &e) == QW4_BUSY);
    assert(qwen4_attention_reset(s, 2, &e) == QW4_BUSY);
    assert(qwen4_attention_finish(t, true, &e) == QW4_OK);
    assert(qwen4_attention_state_info(s).next_pos == 0);
    assert(qwen4_attention_finish(t, true, &e) == QW4_STALE);
    assert(qwen4_attention_begin(s, 0, 1, 1, &t, &e) == QW4_OK);
    assert(qwen4_attention_finish(t, true, &e) == QW4_BAD_STATE);
    assert(qwen4_attention_finish(t, false, &e) == QW4_OK);
    assert(!qwen4_attention_state_info(s).valid);
    assert(qwen4_attention_begin(s, 0, 1, 1, &t, &e) == QW4_BAD_STATE);
    assert(qwen4_attention_reset(s, 1, &e) == QW4_STALE);
    assert(qwen4_attention_reset(s, UINT64_MAX, &e) == QW4_OK);
    assert(qwen4_attention_reset(s, 0, &e) == QW4_STALE);
    assert(qwen4_attention_destroy(s, w, &e) == QW4_OK);
    assert(qwen4_attention_destroy(other, ow, &e) == QW4_OK);
    c = (Qwen4AttentionConfig){24,2,256,4,1,128,4,2048,64,1e-6f,10000000};
    assert(qwen4_attention_plan(&c,8192,1024,&p,&e) == QW4_OK);
    assert(p.cache_bytes >= 37u*1024u*1024u && p.cache_bytes < 37u*1024u*1024u+4096);
    assert(p.workspace_bytes < 192u*1024u*1024u);
    Qwen4AttentionPlan malformed = p; malformed.workspace_bytes--;
    assert(qwen4_attention_create(&c,&malformed,key,SIZE_MAX,&s,&w,&e) == QW4_SHAPE);
    assert(!s && !w);
    c.heads = UINT32_MAX; c.kv_heads = 1; c.head_dim = UINT32_MAX;
    assert(qwen4_attention_plan(&c, UINT32_MAX, 1, &p, &e) == QW4_OVERFLOW);
}

static Qwen4F32 dense(float *data, uint32_t t, uint32_t h, uint32_t d, bool rank2) {
    Qwen4F32 v = {.data=data, .capacity_bytes=(size_t)t*h*d*sizeof(float), .rank=rank2 ? 2 : 3,
                   .dim={t,h,d,0}, .stride_bytes={(size_t)h*d*4,(size_t)d*4,4,0}};
    if (rank2) { v.dim[1]=d; v.dim[2]=0; v.stride_bytes[1]=4; v.stride_bytes[2]=0; }
    return v;
}

static void append_fixture(void) {
    Qwen4AttentionConfig c;
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
    Qwen4AttentionPlan plan; Qwen4Error error;
    assert(qwen4_attention_plan(&c, count, count, &plan, &error) == QW4_OK);
    Qwen4AttentionState *s; Qwen4AttentionWorkspace *w;
    Qwen4StateKey key = {&c,3,1};
    assert(qwen4_attention_create(&c, &plan, key, plan.cache_bytes+plan.workspace_bytes, &s,&w,&error) == QW4_OK);
    Qwen4Status status = QW4_OK;
    uint32_t begin = 0;
    for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
        assert(ends[chunk] >= begin && ends[chunk] <= count);
        uint32_t t = ends[chunk]-begin, b = begin/c.ratio, nb = ends[chunk]/c.ratio-b;
        QW4DPrepared in = {dense(q+begin*hd,t,c.heads,c.head_dim,false),
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
        Qwen4F32Mut out = {padded,(size_t)t*pitch*4,3,{t,c.heads,c.head_dim,0},{pitch*4,(size_t)c.head_dim*4,4,0}};
        Qwen4AttentionTicket *ticket;
        assert(qwen4_attention_begin(s,begin,t,1,&ticket,&error) == QW4_OK);
        assert(qwen4_attention_state_info(s).next_pos == begin);
        if (t) {
            QW4DPrepared bad = in; bad.q.dim[2]++;
            assert(qw4d_append_prepared(ticket,&bad,out,w,&error) == QW4_SHAPE);
            bad = in; bad.q.stride_bytes[0] = 4;
            assert(qw4d_append_prepared(ticket,&bad,out,w,&error) == QW4_STRIDE);
            Qwen4F32Mut small = out; small.capacity_bytes = 0;
            assert(qw4d_append_prepared(ticket,&in,small,w,&error) == QW4_CAPACITY);
            assert(qw4d_append_prepared(ticket,&in,out,NULL,&error) == QW4_OWNER);
            small = out; small.data = pq; small.stride_bytes[0]=hd*4; small.capacity_bytes=t*hd*4;
            assert(qw4d_append_prepared(ticket,&in,small,w,&error) == QW4_INVALID_ARGUMENT);
            assert(qwen4_attention_state_info(s).valid);
        }
        status = qw4d_append_prepared(ticket,&in,out,w,&error);
        assert(error.code == status);
        if (status != QW4_OK) {
            assert(!qwen4_attention_state_info(s).valid);
            assert(qwen4_attention_finish(ticket,true,&error) == QW4_BAD_STATE);
            assert(qwen4_attention_finish(ticket,false,&error) == QW4_OK);
            assert(qwen4_attention_reset(s,2,&error) == QW4_OK);
            free(padded); free(pq); break;
        }
        assert(qwen4_attention_state_info(s).next_pos == begin);
        assert(qwen4_attention_finish(ticket,true,&error) == QW4_OK);
        for (uint32_t row = 0; row < t; ++row) {
            memcpy(output+((size_t)begin+row)*hd,padded+row*pitch,hd*4);
            for (size_t j = hd; j < pitch; ++j) assert(padded[row*pitch+j] == 12345);
            for (size_t j = hd; j < input_pitch; ++j) assert(pq[row*input_pitch+j] == 9876);
        }
        free(padded); free(pq); begin = ends[chunk];
    }
    write_items(&status,sizeof(status),1); write_items(output,sizeof(float),count*hd);
    if (status == QW4_OK) {
        assert(!memcmp(qw4d_cache_keys(s),k,count*kd*4));
        assert(!memcmp(qw4d_cache_raw_index(s),ik,(size_t)count*c.index_dim*4));
        assert(!memcmp(qw4d_cache_blocks(s),blocks,(size_t)(count/c.ratio)*c.index_dim*4));
        /* Reset leaves old bytes resident but makes every old row invisible. */
        assert(qwen4_attention_reset(s,2,&error) == QW4_OK);
        Qwen4AttentionTicket *ticket;
        assert(qwen4_attention_begin(s,0,1,1,&ticket,&error) == QW4_STALE);
        assert(qwen4_attention_begin(s,0,1,2,&ticket,&error) == QW4_OK);
        QW4DPrepared first = {dense(q,1,c.heads,c.head_dim,false), dense(gate,1,c.heads,c.head_dim,false),
            dense(k,1,c.kv_heads,c.head_dim,false), dense(v,1,c.kv_heads,c.head_dim,false),
            dense(iq,1,c.index_heads,c.index_dim,false), dense(ik,1,1,c.index_dim,true),
            dense(blocks,1/c.ratio,1,c.index_dim,true)};
        float *one = calloc(hd,sizeof(float)); assert(one);
        Qwen4F32Mut out = {one,hd*4,3,{1,c.heads,c.head_dim,0},{hd*4,(size_t)c.head_dim*4,4,0}};
        assert(qw4d_append_prepared(ticket,&first,out,w,&error) == QW4_OK);
        assert(!memcmp(one,output,hd*4)); free(one);
        assert(qwen4_attention_finish(ticket,true,&error) == QW4_OK);
        assert(qwen4_attention_state_info(s).next_pos == 1);
    }
    assert(qwen4_attention_destroy(s,w,&error) == QW4_OK);
    free(ends); free(q); free(gate); free(k); free(v); free(iq); free(ik); free(blocks); free(output);
    assert(getchar() == EOF);
}

static void public_fixture(void) {
    Qwen4AttentionConfig c; uint32_t count,chunks;
    read_items(&c,sizeof(c),1); read_items(&count,4,1); read_items(&chunks,4,1);
    assert(count && count<=32768 && chunks && chunks<=32768);
    uint32_t *ends=calloc(chunks,4); assert(ends); read_items(ends,4,chunks);
    assert(ends[chunks-1]==count);
    size_t hd=(size_t)c.heads*c.head_dim,kd=(size_t)c.kv_heads*c.head_dim;
    size_t id=(size_t)c.index_heads*c.index_dim, bd=(size_t)(count/c.ratio)*c.index_dim;
    float *packed_qg=read_floats(count*hd*2), *k=read_floats(count*kd), *v=read_floats(count*kd);
    size_t qp=2*hd+5; float *qg=malloc(count*qp*4); assert(qg);
    for(size_t i=0;i<count*qp;++i) qg[i]=7654;
    for(uint32_t i=0;i<count;++i) memcpy(qg+i*qp,packed_qg+i*hd*2,hd*8);
    free(packed_qg);
    float *iq=read_floats(count*id), *ik=read_floats((size_t)count*c.index_dim);
    float *gq=read_floats(c.head_dim), *gk=read_floats(c.head_dim);
    float *giq=read_floats(c.index_dim), *gik=read_floats(c.index_dim);
    size_t pitch=hd+3;
    float *out=calloc(count*pitch,4), *trace=calloc(count*(2*hd+id),4); assert(out&&trace);
    Qwen4AttentionPlan plan; Qwen4Error e;
    assert(qwen4_attention_plan(&c,count,count,&plan,&e)==QW4_OK);
    Qwen4AttentionState *s; Qwen4AttentionWorkspace *w;
    assert(qwen4_attention_create(&c,&plan,(Qwen4StateKey){&c,3,1},
           plan.cache_bytes+plan.workspace_bytes,&s,&w,&e)==QW4_OK);
    Qwen4F32 gains[]={{gq,c.head_dim*4,1,{c.head_dim},{4}}, {gk,c.head_dim*4,1,{c.head_dim},{4}},
        {giq,c.index_dim*4,1,{c.index_dim},{4}}, {gik,c.index_dim*4,1,{c.index_dim},{4}}};
    Qwen4Status code=QW4_OK;
    uint32_t begin=0;
    for (uint32_t j=0;j<chunks;++j) {
        assert(ends[j]>=begin && ends[j]<=count); uint32_t n=ends[j]-begin;
        Qwen4AttentionInput in={.qg={qg+begin*qp,n*qp*4,4,{n,c.heads,2,c.head_dim},
            {qp*4,(size_t)c.head_dim*8,(size_t)c.head_dim*4,4}},
            .k=dense(k+begin*kd,n,c.kv_heads,c.head_dim,false), .v=dense(v+begin*kd,n,c.kv_heads,c.head_dim,false),
            .iq=dense(iq+begin*id,n,c.index_heads,c.index_dim,false), .ik=dense(ik+(size_t)begin*c.index_dim,n,1,c.index_dim,true),
            .gamma_q=gains[0],.gamma_k=gains[1],.gamma_iq=gains[2],.gamma_ik=gains[3]};
        Qwen4F32Mut y={out+begin*pitch,n*pitch*4,3,{n,c.heads,c.head_dim},{pitch*4,(size_t)c.head_dim*4,4}};
        Qwen4AttentionTicket *t;
        assert(qwen4_attention_begin(s,begin,n,1,&t,&e)==QW4_OK);
        Qwen4AttentionInput bad=in; bad.gamma_q.stride_bytes[0]=8;
        assert(qwen4_attention_cpu(t,&bad,y,w,&e)==QW4_STRIDE);
        bad=in; bad.qg.dim[2]=1;
        assert(qwen4_attention_cpu(t,&bad,y,w,&e)==QW4_SHAPE);
        if(n) {
            Qwen4F32Mut short_out=y; short_out.capacity_bytes=0;
            assert(qwen4_attention_cpu(t,&in,short_out,w,&e)==QW4_CAPACITY);
            assert(qwen4_attention_cpu(t,&in,y,NULL,&e)==QW4_OWNER);
            short_out=y; short_out.data=qg+begin*qp;
            assert(qwen4_attention_cpu(t,&in,short_out,w,&e)==QW4_INVALID_ARGUMENT);
        }
        assert(qwen4_attention_state_info(s).valid);
        code=qwen4_attention_cpu(t,&in,y,w,&e); assert(e.code==code);
        assert(qwen4_attention_state_info(s).next_pos==begin);
        if(code!=QW4_OK) {
            assert(!qwen4_attention_state_info(s).valid);
            assert(qwen4_attention_finish(t,false,&e)==QW4_OK);
            assert(qwen4_attention_reset(s,2,&e)==QW4_OK); break;
        }
        if(n) {
            const float *pq,*pg,*pi,*scores; const uint32_t *ids;
            qw4d_last_query(w,&pq,&pg,&pi,&scores,&ids);
            float *dst=trace+(ends[j]-1)*(2*hd+id);
            memcpy(dst,pq,hd*4); memcpy(dst+hd,pg,hd*4); memcpy(dst+2*hd,pi,id*4);
            assert(qwen4_attention_cpu(t,&in,y,w,&e)==QW4_BAD_STATE);
        }
        assert(qwen4_attention_finish(t,true,&e)==QW4_OK); begin=ends[j];
    }
    write_items(&code,sizeof(code),1);
    for(uint32_t i=0;i<count;++i) {
        write_items(out+i*pitch,4,hd);
        for(size_t g=hd;g<pitch;++g) assert(out[i*pitch+g]==0);
    }
    write_items(qw4d_cache_keys(s),4,count*kd);
    write_items(qw4d_cache_raw_index(s),4,(size_t)count*c.index_dim);
    write_items(qw4d_cache_blocks(s),4,bd);
    write_items(trace,4,count*(2*hd+id));
    const float *pq,*pg,*pi,*scores; const uint32_t *ids;
    qw4d_last_query(w,&pq,&pg,&pi,&scores,&ids);
    uint32_t blocks_count=count/c.ratio, selected=blocks_count<c.token_budget/c.ratio ? blocks_count : c.token_budget/c.ratio;
    write_items(scores,4,blocks_count); write_items(ids,4,(size_t)selected*c.ratio+count%c.ratio);
    if(code==QW4_OK) {
        assert(qwen4_attention_reset(s,2,&e)==QW4_OK);
        Qwen4AttentionInput first={.qg={qg,qp*4,4,{1,c.heads,2,c.head_dim},
            {qp*4,(size_t)c.head_dim*8,(size_t)c.head_dim*4,4}},
            .k=dense(k,1,c.kv_heads,c.head_dim,false), .v=dense(v,1,c.kv_heads,c.head_dim,false),
            .iq=dense(iq,1,c.index_heads,c.index_dim,false), .ik=dense(ik,1,1,c.index_dim,true),
            .gamma_q=gains[0],.gamma_k=gains[1],.gamma_iq=gains[2],.gamma_ik=gains[3]};
        float *replay=calloc(hd,4); assert(replay);
        Qwen4F32Mut y={replay,hd*4,3,{1,c.heads,c.head_dim},{hd*4,(size_t)c.head_dim*4,4}};
        Qwen4AttentionTicket *t;
        assert(qwen4_attention_begin(s,0,1,1,&t,&e)==QW4_STALE);
        assert(qwen4_attention_begin(s,0,1,2,&t,&e)==QW4_OK);
        assert(qwen4_attention_cpu(t,&first,y,w,&e)==QW4_OK);
        assert(!memcmp(replay,out,hd*4)); free(replay);
        assert(qwen4_attention_finish(t,true,&e)==QW4_OK);
        for(uint32_t i=0;i<count;++i) for(size_t d=2*hd;d<qp;++d) assert(qg[i*qp+d]==7654);
    }
    assert(qwen4_attention_destroy(s,w,&e)==QW4_OK);
    free(ends); free(qg); free(k); free(v); free(iq); free(ik); free(gq); free(gk); free(giq); free(gik);
    free(out); free(trace); assert(getchar()==EOF);
}

static void catalog(const char *path, uint32_t layer) {
    Gguf g; gguf_open(path,&g);
    Qwen4Config *cfg; Qwen4Weights *weights; Qwen4Error e;
    assert(qwen4_config_load(&g,&cfg,&e)==QW4_OK);
    const Qwen4ConfigView *v=qwen4_config_view(cfg); const Qwen4Layer *l;
    assert(qwen4_config_layer(cfg,layer,&l,&e)==QW4_OK && l->kind==QW4_LAYER_QSA && l->domain==QW4_TRUNK);
    assert(qwen4_weights_describe(&g,cfg,&weights,&e)==QW4_OK);
    Qwen4AttentionConfig c=v->attention; c.ratio=l->compress_ratio;
    printf("{\"hidden\":%u,\"config\":{\"width\":%u,\"heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,"
           "\"index_heads\":%u,\"index_dim\":%u,\"ratio\":%u,\"budget\":%u,\"rotary_dim\":%u,\"eps\":%.9g,\"theta\":%.9g},\"tensors\":[",
           v->hidden_size,v->hidden_size,c.heads,c.kv_heads,c.head_dim,c.index_heads,c.index_dim,
           c.ratio,c.token_budget,c.rotary_dim,(double)c.rms_eps,(double)c.rope_theta);
    Qwen4Role roles[]={QW4_ATTN_QG,QW4_ATTN_K,QW4_ATTN_V,QW4_INDEX_Q,QW4_INDEX_K,QW4_ATTN_OUT,
        QW4_ATTN_Q_NORM,QW4_ATTN_K_NORM,QW4_INDEX_Q_NORM,QW4_INDEX_K_NORM};
    for(size_t i=0;i<10;++i) {
        const Qwen4Tensor *t; assert(qwen4_weights_get(weights,layer,roles[i],&t,&e)==QW4_OK);
        const GgufTensor *s=&t->storage;
        printf("%s{\"name\":\"%s\",\"type\":%u,\"k\":%zu,\"rows\":%zu,\"offset\":%zu,\"bytes\":%zu}",
            i?",":"",s->name,s->type,s->shape[0],s->rank==1?1:s->shape[1],s->file_offset,s->bytes);
    }
    puts("]}"); qwen4_weights_free(weights); qwen4_config_free(cfg); gguf_close(&g);
}
static void linear_fixture(const uint32_t *d) {
    size_t m=d[0],rows=d[1],k=d[2];
    const GgufFormat *format=gguf_format(d[3]); assert(format && k%format->block==0);
    size_t bytes=k/format->block*format->bytes*rows; assert(bytes<=32u*1024u*1024u);
    unsigned char *raw=malloc(bytes); assert(raw); read_items(raw,1,bytes);
    float *x=read_floats(m*k),*y=calloc(m*rows,4); assert(y);
    GgufTensor t={.type=d[3],.rank=2,.shape={k,rows,1,1},.bytes=bytes};
    QW4PackedCpuView weight={.span={raw,bytes}}; Qwen4Error e;
    assert(qw4_packed_layout_init(&t,k,&weight.layout,&e)==QW4_OK);
    Qwen4F32 input={x,m*k*4,2,{m,k},{k*4,4}};
    Qwen4F32Mut out={y,m*rows*4,2,{m,rows},{rows*4,4}};
    Qwen4Status code=qw4_packed_matmul(&weight,0,input,out,&e);
    write_items(&code,sizeof(code),1); write_items(y,4,m*rows); free(raw); free(x); free(y);
}

int main(int argc, char **argv) {
    if(argc==4 && !strcmp(argv[1],"catalog")) { catalog(argv[2],(uint32_t)strtoul(argv[3],NULL,10)); return 0; }
    assert(argc == 2);
    if (!strcmp(argv[1], "lifecycle")) { lifecycle(); puts("lifecycle OK"); return 0; }
    if (!strcmp(argv[1], "append")) { append_fixture(); return 0; }
    if (!strcmp(argv[1], "public")) { public_fixture(); return 0; }
    uint32_t dims[4];
    read_items(dims, sizeof(uint32_t), 4);
    for (size_t i = 0; i < 4; ++i) assert(dims[i] <= 32768);
    if(!strcmp(argv[1],"linear")) { linear_fixture(dims); assert(getchar()==EOF); return 0; }
    Qwen4Status status;
    if (!strcmp(argv[1], "rope")) {
        float theta;
        read_items(&theta, sizeof(float), 1);
        size_t n = (size_t)dims[0] * dims[1];
        float *x = read_floats(n);
        status = qw4d_rope(x, dims[0], dims[1], dims[2], dims[3], theta);
        write_items(&status, sizeof(status), 1); write_items(x, sizeof(float), n); free(x);
    } else if (!strcmp(argv[1], "pool")) {
        float *raw = read_floats((size_t)dims[0]*dims[1]);
        float *pooled = calloc(dims[1] ? dims[1] : 1,sizeof(float)); assert(pooled);
        status = qw4d_pool(raw,dims[0],dims[1],pooled);
        write_items(&status,sizeof(status),1); write_items(pooled,sizeof(float),dims[1]);
        free(raw); free(pooled);
    } else if (!strcmp(argv[1], "scores")) {
        float *q = read_floats((size_t)dims[0] * dims[1]);
        float *k = read_floats((size_t)dims[2] * dims[1]);
        float *s = calloc(dims[2] ? dims[2] : 1, sizeof(float)); assert(s);
        status = qw4d_scores(q, k, dims[0], dims[1], dims[2], s);
        write_items(&status, sizeof(status), 1); write_items(s, sizeof(float), dims[2]);
        free(q); free(k); free(s);
    } else if (!strcmp(argv[1], "attend")) {
        Qwen4AttentionConfig c = {.heads=dims[0], .kv_heads=dims[1], .head_dim=dims[2]};
        size_t hd = (size_t)c.heads * c.head_dim;
        size_t cache = (size_t)dims[3] * c.kv_heads * c.head_dim;
        float *q = read_floats(hd), *gate = read_floats(hd);
        float *k = read_floats(cache), *v = read_floats(cache);
        uint32_t n; read_items(&n, sizeof(n), 1); assert(n <= dims[3]);
        uint32_t *ids = calloc(n ? n : 1, sizeof(uint32_t)); assert(ids);
        read_items(ids, sizeof(uint32_t), n);
        float *out = calloc(hd, sizeof(float)), *scratch = calloc(c.head_dim + 16, sizeof(float));
        assert(out && scratch);
        status = qw4d_attend(&c, q, gate, k, v, dims[3], ids, n, out, scratch);
        write_items(&status, sizeof(status), 1); write_items(out, sizeof(float), hd);
        free(q); free(gate); free(k); free(v); free(ids); free(out); free(scratch);
    } else {
        assert(!strcmp(argv[1], "select"));
        float *s = read_floats(dims[0]);
        uint32_t n = dims[0] < dims[1] ? dims[0] : dims[1];
        uint32_t *heap = calloc(n ? n : 1, sizeof(uint32_t));
        uint32_t *ids = calloc(n ? n : 1, sizeof(uint32_t)); assert(heap && ids);
        status = qw4d_select(s, dims[0], dims[1], heap, ids);
        write_items(&status, sizeof(status), 1); write_items(ids, sizeof(uint32_t), n);
        free(s); free(heap); free(ids);
    }
    assert(getchar() == EOF);
    return 0;
}
