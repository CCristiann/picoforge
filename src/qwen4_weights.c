/* Semantic catalog only. Packed arithmetic belongs to B; mapping to storage. */
#include "qwen4_text_internal.h"

struct Qwen4Weights {
    const Gguf *source;
    const void *header;
    int fd;
    size_t file_size, count;
    uint32_t blocks;
    Qwen4Tensor *items;
    size_t *roles;                 /* index+1, zero means absent */
    const GgufPlan *plan;
};
static bool same_tensor(const GgufTensor *a, const GgufTensor *b) {
    if (strcmp(a->name, b->name) || a->type != b->type || a->rank != b->rank ||
        a->file_offset != b->file_offset || a->bytes != b->bytes) return false;
    for (uint32_t i = 0; i < a->rank; ++i) if (a->shape[i] != b->shape[i]) return false;
    return true;
}
static Qwen4Status record(Qwen4Weights *w, const Gguf *g, uint32_t layer,
    Qwen4Role role, const char *suffix, uint32_t type, uint32_t rank,
    size_t k, size_t n, size_t experts, size_t logical_k,
    Qwen4Domain domain, Qwen4Layout layout, Qwen4Error *e) {
    char name[128];
    if (layer == QWEN4_GLOBAL_LAYER) snprintf(name, sizeof name, "%s", suffix);
    else snprintf(name, sizeof name, "blk.%u.%s", layer, suffix);
    const GgufTensor *t = NULL; size_t index = 0;
    for (size_t i = 0; i < g->tensor_count; ++i)
        if (!strcmp(g->tensors[i].name, name)) {
            if (t) return qw4a_error(e, QW4_FORMAT, name, "duplicate tensor");
            t = g->tensors+i; index = i;
        }
    if (!t) return qw4a_error(e, QW4_MISSING, name, "required tensor");
    if (t->type != type) return qw4a_error(e, QW4_FORMAT, name, "Q2 recipe format mismatch");
    size_t shape[4] = {k, n, experts, 1};
    if (t->rank != rank) return qw4a_error(e, QW4_SHAPE, name, "rank mismatch");
    for (uint32_t i = 0; i < rank; ++i)
        if (t->shape[i] != shape[i]) return qw4a_error(e, QW4_SHAPE, name, "physical shape mismatch");
    /* type is a schema constant verified above, so gguf_format cannot abort. */
    const GgufFormat *f = gguf_format(type);
    size_t row, plane, bytes;
    if (!k || k%f->block || !logical_k || logical_k > k)
        return qw4a_error(e, QW4_SHAPE, name, "packed row alignment");
    if (!qw4a_mul(k/f->block, f->bytes, &row) || !qw4a_mul(row, n, &plane) ||
        !qw4a_mul(plane, experts, &bytes))
        return qw4a_error(e, QW4_OVERFLOW, name, "tensor extent");
    if (bytes != t->bytes) return qw4a_error(e, QW4_FORMAT, name, "physical byte count");
    Qwen4Tensor *q = w->items+index;
    if (q->block_bytes) return qw4a_error(e, QW4_FORMAT, name, "duplicate role assignment");
    *q = (Qwen4Tensor){.storage=*t, .block_elems=f->block, .block_bytes=f->bytes,
        .row_bytes=row, .plane_bytes=plane, .domain=domain, .layout=layout};
    for (size_t i = 0; i < 4; ++i) q->logical_shape[i] = i < rank ? shape[i] : 1;
    q->logical_shape[0] = logical_k;
    if (role != QW4_ROLE_COUNT) {
        size_t slot = (layer == QWEN4_GLOBAL_LAYER ? w->blocks : layer)*QW4_ROLE_COUNT+(size_t)role;
        w->roles[slot] = index+1;
    }
    return QW4_OK;
}
static Qwen4Status catalog(const Gguf *g, const Qwen4Config *c, Qwen4Weights *w, Qwen4Error *e) {
    const Qwen4ConfigView *v = &c->view;
    size_t h = v->hc.streams, r = v->hc.low_rank, x = h*v->hidden_size, d = v->hidden_size;
    size_t f = v->moe.intermediate, fs = v->moe.shared_intermediate, ne = v->moe.experts;
    size_t q = (size_t)v->attention.heads*v->attention.head_dim;
    size_t kv = (size_t)v->attention.kv_heads*v->attention.head_dim;
    size_t gc = 2*v->gdn.key_heads*v->gdn.key_dim+c->inner;
    uint32_t layer = QWEN4_GLOBAL_LAYER; Qwen4Domain domain = QW4_TRUNK;
    Qwen4Status s;
#define ADD(role, name, ty, rank, k, n, ex, logk, layout) do { \
    s = record(w,g,layer,role,name,ty,rank,k,n,ex,logk,domain,layout,e); if(s) return s; } while(0)
#define VEC(role, name, k) ADD(role,name,0,1,k,1,1,k,QW4_VECTOR)
#define MAT(role, name, ty, k, n) ADD(role,name,ty,2,k,n,1,k,QW4_MATRIX)
#define HC(prefix, norm, down, up) \
    VEC(norm,prefix "_norm.weight",x); MAT(down,prefix "_down.weight",1,x,r); \
    MAT(up,prefix "_up.weight",1,r,x)
    MAT(QW4_TOKEN_EMBD,"token_embd.weight",30,d,v->vocab_size);
    MAT(QW4_LM_HEAD,"output.weight",8,d,v->vocab_size);
    HC("output_hc",QW4_OUTPUT_HC_NORM,QW4_OUTPUT_HC_DOWN,QW4_OUTPUT_HC_UP);
    if (c->has_ple) {                     /* the n-gram table, read from the SSD */
        domain = QW4_DISK_PLE;
        ADD(QW4_PLE_TABLE,"per_layer_token_embd.weight",30,2,c->ngram.head_dim,c->ngram.row_count,
            1,c->ngram.head_dim,QW4_DISK_ROWS);
    }
    for (layer = 0; layer < v->blocks; ++layer) {
        domain = c->layers[layer].domain;
        bool mtp = domain == QW4_MTP;
        HC("hc_attn",QW4_HC_ATTN_NORM,QW4_HC_ATTN_DOWN,QW4_HC_ATTN_UP);
        HC("hc_ffn",QW4_HC_FFN_NORM,QW4_HC_FFN_DOWN,QW4_HC_FFN_UP);
        MAT(QW4_HC_ATTN_INJECT,"hc_attn_inject.weight",1,x,h);
        MAT(QW4_HC_FFN_INJECT,"hc_ffn_inject.weight",1,x,h);
        MAT(QW4_MOE_ROUTER,"ffn_gate_inp.weight",0,d,ne);
        VEC(QW4_SHARED_ROUTER,"ffn_gate_inp_shexp.weight",d);
        MAT(QW4_SHARED_GATE,"ffn_gate_shexp.weight",8,d,fs);
        MAT(QW4_SHARED_UP,"ffn_up_shexp.weight",8,d,fs);
        MAT(QW4_SHARED_DOWN,"ffn_down_shexp.weight",8,fs,d);
        /* R5.3: the trunk's expert formats come from the config's profile (Q2 or Q4); the MTP layer is Q4 in both. */
        ADD(QW4_MOE_GATE,"ffn_gate_exps.weight",mtp?12:v->moe.trunk_type[0],3,d,f,ne,d,QW4_EXPERTS);
        ADD(QW4_MOE_UP,"ffn_up_exps.weight",mtp?12:v->moe.trunk_type[1],3,d,f,ne,d,QW4_EXPERTS);
        ADD(QW4_MOE_DOWN,"ffn_down_exps.weight",mtp?39:v->moe.trunk_type[2],3,mtp?f:v->moe.down_physical,d,ne,f,QW4_EXPERTS);
        if (c->layers[layer].kind == QW4_LAYER_GDN) {
            ADD(QW4_GDN_QKV,"attn_qkv.weight",8,2,d,gc,1,d,QW4_GDN_TILED);
            ADD(QW4_GDN_GATE,"attn_gate.weight",8,2,d,c->inner,1,d,QW4_GDN_TILED);
            ADD(QW4_GDN_OUT,"ssm_out.weight",8,2,c->inner,d,1,c->inner,QW4_GDN_TILED);
            MAT(QW4_GDN_ALPHA,"ssm_alpha.weight",0,d,v->gdn.value_heads);
            MAT(QW4_GDN_BETA,"ssm_beta.weight",0,d,v->gdn.value_heads);
            VEC(QW4_GDN_A,"ssm_a",v->gdn.value_heads);
            VEC(QW4_GDN_DT,"ssm_dt.bias",v->gdn.value_heads);
            VEC(QW4_GDN_NORM,"ssm_norm.weight",v->gdn.value_dim);
            ADD(QW4_GDN_CONV,"ssm_conv1d.weight",0,2,v->gdn_conv_kernel,gc,1,v->gdn_conv_kernel,QW4_CONV);
        } else {
            ADD(QW4_ATTN_QG,"attn_q.weight",8,2,d,2*q,1,d,QW4_Q_GATE_INTERLEAVED);
            MAT(QW4_ATTN_K,"attn_k.weight",8,d,kv); MAT(QW4_ATTN_V,"attn_v.weight",8,d,kv);
            MAT(QW4_ATTN_OUT,"attn_output.weight",8,q,d);
            VEC(QW4_ATTN_Q_NORM,"attn_q_norm.weight",v->attention.head_dim);
            VEC(QW4_ATTN_K_NORM,"attn_k_norm.weight",v->attention.head_dim);
            MAT(QW4_INDEX_Q,"indexer.q_proj.weight",8,d,(size_t)v->attention.index_heads*v->attention.index_dim);
            MAT(QW4_INDEX_K,"indexer.k_proj.weight",8,d,v->attention.index_dim);
            VEC(QW4_INDEX_Q_NORM,"indexer.q_norm.weight",v->attention.index_dim);
            VEC(QW4_INDEX_K_NORM,"indexer.k_norm.weight",v->attention.index_dim);
        }
        if (c->layers[layer].has_ple) {
            MAT(QW4_PLE_KEY,"ple_key.weight",8,d,x); MAT(QW4_PLE_VALUE,"ple_value.weight",8,d,d);
            VEC(QW4_PLE_KEY_NORM,"ple_norm_key.weight",x);
            VEC(QW4_PLE_QUERY_NORM,"ple_norm_query.weight",x);
            VEC(QW4_PLE_CONV_NORM,"ple_norm_conv.weight",x);
            ADD(QW4_PLE_CONV,"ple_conv1d.weight",0,2,c->ngram.conv_kernel,x,1,c->ngram.conv_kernel,QW4_CONV);
        }
        if (mtp) {
            MAT(QW4_MTP_EH,"nextn.eh_proj.weight",8,2*d,d);
            VEC(QW4_MTP_E_NORM,"nextn.enorm.weight",d); VEC(QW4_MTP_H_NORM,"nextn.hnorm.weight",x);
            HC("nextn.hc_head",QW4_MTP_HEAD_NORM,QW4_MTP_HEAD_DOWN,QW4_MTP_HEAD_UP);
        }
    }
#undef HC
#undef MAT
#undef VEC
#undef ADD
    return QW4_OK;
}
void qwen4_weights_free(Qwen4Weights *w) {
    if (!w) return;
    free(w->items); free(w->roles); free(w);
}
Qwen4Status qwen4_weights_describe(const Gguf *g, const Qwen4Config *c,
                                    Qwen4Weights **out, Qwen4Error *e) {
    if (out) *out = NULL;
    if (!g || !c || !out || !g->tensors || !g->tensor_count)
        return qw4a_error(e, QW4_INVALID_ARGUMENT, "catalog", "null/empty argument");
    size_t bytes, slots = ((size_t)c->view.blocks+1)*QW4_ROLE_COUNT;
    size_t metadata = sizeof *c+(size_t)c->view.blocks*sizeof(Qwen4Layer)+
        c->ngram.head_count*16+c->ngram.order*8+sizeof(Qwen4Weights);
    if (!qw4a_mul(g->tensor_count, sizeof(Qwen4Tensor), &bytes) ||
        bytes > 4*1024*1024 || metadata > 4*1024*1024-bytes ||
        slots*sizeof(size_t) > 4*1024*1024-bytes-metadata)
        return qw4a_error(e, QW4_RANGE, "catalog", "4 MiB config/catalog budget");
    size_t end = g->data_offset;
    for (size_t i = 0; i < g->tensor_count; ++i) {
        const GgufTensor *t = g->tensors+i;
        if (!memchr(t->name, 0, sizeof t->name) || !t->bytes || t->rank < 1 || t->rank > 4 ||
            t->file_offset < end || t->file_offset > g->file_size || t->bytes > g->file_size-t->file_offset)
            return qw4a_error(e, QW4_FORMAT, "catalog", "malformed/overlapping descriptor");
        end = t->file_offset+t->bytes;
    }
    Qwen4Weights *w = calloc(1, sizeof *w);
    if (!w) return qw4a_error(e, QW4_OOM, "catalog", "allocation");
    w->items = calloc(g->tensor_count, sizeof *w->items);
    w->roles = calloc(slots, sizeof *w->roles);
    if (!w->items || !w->roles) {
        qwen4_weights_free(w); return qw4a_error(e, QW4_OOM, "catalog", "allocation");
    }
    w->source = g; w->header = g->header; w->fd = g->fd;
    w->file_size = g->file_size; w->count = g->tensor_count; w->blocks = c->view.blocks;
    Qwen4Status s = catalog(g, c, w, e);
    for (size_t i = 0; !s && i < w->count; ++i)
        if (!w->items[i].block_bytes) s = qw4a_error(e, QW4_UNSUPPORTED, g->tensors[i].name, "unclassified tensor");
    if (s) { qwen4_weights_free(w); return s; }
    *out = w;
    return qw4a_error(e, QW4_OK, NULL, NULL);
}
Qwen4Status qwen4_weights_get(const Qwen4Weights *w, uint32_t layer, Qwen4Role role,
                               const Qwen4Tensor **out, Qwen4Error *e) {
    if (out) *out = NULL;
    if (!w || !out) return qw4a_error(e, QW4_INVALID_ARGUMENT, "weights", "null argument");
    if ((layer >= w->blocks && layer != QWEN4_GLOBAL_LAYER) || role < 0 || role >= QW4_ROLE_COUNT)
        return qw4a_error(e, QW4_RANGE, "weights", "layer/role index");
    size_t slot = (layer == QWEN4_GLOBAL_LAYER ? w->blocks : layer)*QW4_ROLE_COUNT+(size_t)role;
    size_t index = w->roles[slot];
    if (!index) return qw4a_error(e, QW4_MISSING, "weights", "role absent for this layer");
    *out = w->items+index-1;
    return qw4a_error(e, QW4_OK, NULL, NULL);
}
Qwen4Status qwen4_weights_attach(Qwen4Weights *w, const Gguf *g,
                                  const GgufPlan *p, Qwen4Error *e) {
    if (!w || !g || !p) return qw4a_error(e, QW4_INVALID_ARGUMENT, "attach", "null argument");
    if (w->plan) return qw4a_error(e, QW4_BUSY, "attach", "already attached");
    if (g != w->source || p->source != g || g->header != w->header || g->fd != w->fd ||
        g->fd < 0 || !g->header || g->file_size != w->file_size || p->file_size != w->file_size)
        return qw4a_error(e, QW4_OWNER, "attach", "same live reader required");
    /* The PLE table, when the model has one, stays on the SSD: never GPU-bound. */
    const Qwen4Tensor *disk = NULL;
    Qwen4Status s = qwen4_weights_get(w, QWEN4_GLOBAL_LAYER, QW4_PLE_TABLE, &disk, NULL);
    if (s != QW4_OK && s != QW4_MISSING) return qw4a_error(e, s, "attach", "PLE table lookup");
    if (g->tensor_count != w->count || !g->tensors || !p->weights || !p->views ||
        p->weight_count+p->streamed_count != w->count-(disk ? 1 : 0) || !p->view_count || p->view_count > p->weight_count)
        return qw4a_error(e, QW4_SHAPE, "attach", "catalog/plan sizes");
    if (!memchr(p->disk_table.name, 0, sizeof p->disk_table.name) ||
        (disk ? !same_tensor(&disk->storage, &p->disk_table) : p->disk_table.bytes != 0))
        return qw4a_error(e, QW4_FORMAT, "attach", "disk descriptor changed");
    for (size_t j = 0; j < p->view_count; ++j) {
        const GgufView *v = p->views+j;
        if (!v->bytes || v->bytes > SIZE_MAX-v->file_offset ||
            (disk && v->file_offset < disk->storage.file_offset+disk->storage.bytes &&
             disk->storage.file_offset < v->file_offset+v->bytes))
            return qw4a_error(e, QW4_RANGE, "attach", "view overflow or PLE overlap");
    }
    s = QW4_OK;
    /* Validate every association before publishing ANY borrowed plan pointer. */
    const GgufBinding **bindings = calloc(w->count, sizeof *bindings);
    if (!bindings) return qw4a_error(e, QW4_OOM, "attach", "association scratch");
    for (size_t i = 0; i < w->count && !s; ++i) {
        const GgufTensor *t = &w->items[i].storage;
        if (!memchr(g->tensors[i].name, 0, sizeof g->tensors[i].name) || !same_tensor(t, g->tensors+i)) {
            s = qw4a_error(e, QW4_STALE, "attach", "source catalog mutated"); break;
        }
        if (w->items[i].domain == QW4_DISK_PLE) continue;
        bool streamed = false;   /* R5.1b: read from the GGUF into the expert pool, never mapped */
        for (size_t j = 0; j < p->streamed_count && !streamed; ++j)
            streamed = !strcmp(t->name, p->streamed[j].name) && same_tensor(t, p->streamed+j);
        if (streamed) continue;
        for (size_t j = 0; j < p->weight_count; ++j) {
            const GgufBinding *b = p->weights+j;
            if (!memchr(b->tensor.name, 0, sizeof b->tensor.name)) {
                s = qw4a_error(e, QW4_FORMAT, "attach", "unterminated binding name"); break;
            }
            if (strcmp(t->name, b->tensor.name)) continue;
            if (bindings[i] || !same_tensor(t, &b->tensor)) {
                s = qw4a_error(e, QW4_FORMAT, t->name, "duplicate or changed binding"); break;
            }
            if (b->owner != p) { s = qw4a_error(e, QW4_OWNER, t->name, "binding owner"); break; }
            if (b->buffer_id >= p->view_count) { s = qw4a_error(e, QW4_RANGE, t->name, "buffer index"); break; }
            const GgufView *v = p->views+b->buffer_id;
            if (b->offset > v->bytes || t->bytes > v->bytes-b->offset ||
                v->file_offset > t->file_offset || b->offset != t->file_offset-v->file_offset) {
                s = qw4a_error(e, QW4_RANGE, t->name, "binding span"); break;
            }
            bindings[i] = b;
        }
        if (!s && !bindings[i]) s = qw4a_error(e, QW4_MISSING, t->name, "binding absent");
    }
    if (!s) {
        for (size_t i = 0; i < w->count; ++i) w->items[i].gpu = bindings[i];
        w->plan = p;
    }
    free(bindings);
    return s ? s : qw4a_error(e, QW4_OK, NULL, NULL);
}
