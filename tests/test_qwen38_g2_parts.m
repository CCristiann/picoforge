/* G2 adapter checks on the G1 graph fixture: each GPU composition against
 * its verified CPU counterpart, through the real G0 session lifecycle.
 *   test_qwen38_g2_parts GGUF METALLIB
 * Prints one JSON line per check; exits nonzero on any gate failure. */
#import <Metal/Metal.h>
#include "picoforge.h"
#include "qwen38_c_internal.h"
#include "qwen38_f_internal.h"
#include "qwen38_moe.h"
#include "qwen38_d_internal.h"
#include <assert.h>
#include <stdlib.h>

static id<MTLDevice> device;
static id<MTLLibrary> library;
static Gguf gguf;
static Qwen38Config *config;
static Qwen38Weights *weights;
static GpuWeightStore *store;
static Qwen38GpuSession *session;
static int failures;

static void ok(Qwen38Status rc, const Qwen38Error *e, const char *what) {
    if (rc == Q38_OK) return;
    fprintf(stderr, "FAIL %s: status %d %s: %s\n", what, rc, e->subject, e->detail);
    exit(1);
}
static id<MTLBuffer> shared(size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:bytes ? bytes : 4 options:MTLResourceStorageModeShared];
    assert(b); memset(b.contents, 0, b.length); return b;
}
static Qwen38GpuF32 gview(id<MTLBuffer> b, size_t offset, size_t rows, size_t cols) {
    return (Qwen38GpuF32){.span={(__bridge void *)b, offset, rows*cols*4}, .rank=2,
        .dim={rows, cols}, .stride_bytes={cols*4, 4}};
}
static Qwen38F32 cview(const float *p, size_t rows, size_t cols) {
    return (Qwen38F32){.data=p, .capacity_bytes=rows*cols*4, .rank=2, .dim={rows, cols}, .stride_bytes={cols*4, 4}};
}
static Qwen38F32Mut mview(float *p, size_t rows, size_t cols) {
    return (Qwen38F32Mut){.data=p, .capacity_bytes=rows*cols*4, .rank=2, .dim={rows, cols}, .stride_bytes={cols*4, 4}};
}
static const GgufBinding *binding(uint32_t layer, Qwen38Role role, Q38PackedCpuView *cpu) {
    const Qwen38Tensor *t; Qwen38Error e;
    ok(qwen38_weights_get(weights, layer, role, &t, &e), &e, "role");
    if (cpu) {
        ok(qwen38_weights_cpu_span(store, t->gpu, &cpu->span, &e), &e, "span");
        ok(q38_packed_layout_init(&t->storage, (size_t)t->logical_shape[0], &cpu->layout, &e), &e, "layout");
    }
    return t->gpu;
}
/* One command: begin, claim the given scratch buffers WRITE and all weight
 * views READ, let the caller encode, then submit and finish. */
typedef Qwen38Status (^Encode)(Qwen38GpuCommand);
static Qwen38Status command(size_t count, NSArray<id<MTLBuffer>> *scratch,
        NSArray<id<MTLBuffer>> *inputs, Encode encode, Qwen38Error *e) {
    Qwen38GpuTicket t; Qwen38GpuCommand cmd;
    Qwen38StateInfo info = qwen38_gpu_session_info(session);
    ok(qwen38_gpu_session_begin(session, config, info.next_pos, count, &t, &cmd, e), e, "begin");
    for (id<MTLBuffer> b in scratch)
        ok(qwen38_gpu_session_claim(session, &t, (Qwen38GpuSpan){(__bridge void *)b, 0, b.length}, Q38_GPU_USE_WRITE, e), e, "claim");
    for (id<MTLBuffer> b in inputs)
        ok(qwen38_gpu_session_claim(session, &t, (Qwen38GpuSpan){(__bridge void *)b, 0, b.length}, Q38_GPU_USE_READ, e), e, "claim inputs");
    const GgufPlan *plan = gpu_weights_plan(store);
    for (size_t i = 0; i < plan->weight_count; ++i) {
        const GgufBinding *w = &plan->weights[i];
        ok(qwen38_gpu_session_claim(session, &t, (Qwen38GpuSpan){gpu_weights_buffer(store, w), w->offset, w->tensor.bytes},
            Q38_GPU_USE_READ, e), e, "claim weights");
    }
    Qwen38Status code=encode(cmd);
    if (code) {
        ok(qwen38_gpu_session_abort(session,&t,false,e),e,"abort encode");
        return code;
    }
    ok(qwen38_gpu_session_submit(session, &t, e), e, "submit");
    return qwen38_gpu_session_finish(session, &t, e);
}
static double scaled(const float *a, const float *r, size_t n, bool *bitwise) {
    double err = 0, top = 1; *bitwise = true;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(a[i]) || !isfinite(r[i])) { *bitwise=false; return INFINITY; }
        err = fmax(err, fabs((double)a[i] - r[i])); top = fmax(top, fabs((double)r[i]));
        *bitwise &= memcmp(a + i, r + i, 4) == 0;
    }
    return err / top;
}
static void report(const char *name, uint32_t layer, double err, double gate, bool bitwise) {
    bool pass = err <= gate;
    printf("{\"check\":\"%s\",\"layer\":%u,\"err\":%.3e,\"gate\":%g,\"bitwise\":%s,\"pass\":%s}\n",
           name, layer, err, gate, bitwise ? "true" : "false", pass ? "true" : "false");
    failures += !pass;
}
static void compare(const char *name,uint32_t layer,const float *got,const float *ref,size_t n,double gate) {
    bool bitwise; double err=scaled(got,ref,n,&bitwise);
    report(name,layer,err,gate,bitwise);
}
static float *random_rows(size_t n, unsigned seed) {
    float *x = malloc(n * 4); srand(seed);
    for (size_t i = 0; i < n; ++i) x[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    return x;
}

/* F + B: device-routed MoE against qwen38_moe_forward, IDs compared exactly. */
static void check_moe(uint32_t layer, size_t t) {
    const Qwen38ConfigView *v = qwen38_config_view(config);
    const Qwen38MoeConfig *c = &v->moe;
    const Qwen38Role roles[] = {Q38_MOE_ROUTER, Q38_MOE_GATE, Q38_MOE_UP, Q38_MOE_DOWN,
        Q38_SHARED_GATE, Q38_SHARED_UP, Q38_SHARED_DOWN, Q38_SHARED_ROUTER};
    Q38PackedCpuView cpu[8]; const GgufBinding *gb[8]; Q38PackedLayout gl[8]; Qwen38Error e;
    for (size_t i = 0; i < 8; ++i) { gb[i] = binding(layer, roles[i], &cpu[i]); gl[i] = cpu[i].layout; }
    Qwen38MoeWeights w = {cpu[0], cpu[1], cpu[2], cpu[3], cpu[4], cpu[5], cpu[6], cpu[7]};
    Qwen38Moe *moe; Qwen38MoeWork *work; size_t bytes;
    ok(qwen38_moe_create(c, &w, &moe, &e), &e, "moe create");
    ok(qwen38_moe_work_size(moe, t, &bytes, &e), &e, "moe size");
    ok(qwen38_moe_work_create(moe, t, bytes, &work, &e), &e, "moe work");
    size_t h = c->hidden_size, k = c->top_k, p = t*k, i = c->intermediate, s = c->shared_intermediate, x_n = t*h;
    float *x = random_rows(x_n, 7 + layer), *y = malloc(x_n * 4), *rw = malloc(p * 4);
    uint32_t *ids = malloc(p * 4);
    Qwen38MoeRoutes routes = {{ids, p*4, t, k, k*4}, mview(rw, t, k)};
    ok(qwen38_moe_forward(moe, cview(x, t, h), mview(y, t, h), &routes, work, &e), &e, "moe cpu");
    id<MTLBuffer> bx = shared(x_n*4), by = shared(x_n*4), bl = shared(t*c->experts*4), bp = shared(t*c->experts*4),
        bw = shared(p*4), bg = shared(p*i*4), bu = shared(p*i*4), ba = shared(p*i*4), bd = shared(p*h*4),
        sg = shared(t*s*4), su = shared(t*s*4), sa = shared(t*s*4), sh = shared(t*h*4), sl = shared(t*4),
        ss = shared(t*4), br = shared(t*h*4), bi = shared(p*4), bm = shared(p*4);
    memcpy(bx.contents, x, x_n*4);
    Q38FRoutedWork rwk = {gview(bl,0,t,c->experts), gview(bp,0,t,c->experts), gview(bw,0,t,k),
        gview(bg,0,p,i), gview(bu,0,p,i), gview(ba,0,p,i), gview(bd,0,p,h), gview(sg,0,t,s), gview(su,0,t,s),
        gview(sa,0,t,s), gview(sh,0,t,h), gview(sl,0,t,1), gview(ss,0,t,1), gview(br,0,t,h),
        {(__bridge void *)bi, 0, p*4}, {(__bridge void *)bm, 0, p*4}};
    Q38FRouted *routed; Q38PackedGpu *b; Q38CGpu *ops;
    ok(q38f_routed_create((__bridge void *)device, (__bridge void *)library, &routed, &e), &e, "routed");
    ok(q38_packed_gpu_create((__bridge void *)device, (__bridge void *)library, &b, &e), &e, "B gpu");
    ok(q38c_gpu_create((__bridge void *)device, (__bridge void *)library, &ops, &e), &e, "C gpu");
    __block Qwen38Status rc = Q38_OK; __block Qwen38Error inner;
    const GgufBinding *const *gbp = gb; const Q38PackedLayout *glp = gl;
    ok(command(t, @[by, bl, bp, bw, bg, bu, ba, bd, sg, su, sa, sh, sl, ss, br, bi, bm], @[bx], ^(Qwen38GpuCommand cmd) {
        rc = q38f_routed_encode(routed, b, ops, cmd, c, store, gbp, glp, gview(bx,0,t,h), gview(by,0,t,h), &rwk, &inner);
        return rc;
    }, &e), &e, "moe command");
    ok(rc, &inner, "moe encode");
    bool same_ids = !memcmp(bi.contents, ids, p*4);
    report("moe_route_ids", layer, same_ids ? 0 : 1, 0, same_ids);
    compare("moe_route_weights",layer,bw.contents,rw,p,2e-6);
    compare("moe_y",layer,by.contents,y,x_n,5e-4);
    q38f_routed_destroy(routed); q38_packed_gpu_free(b); q38c_gpu_destroy(ops);
    qwen38_moe_work_destroy(work); qwen38_moe_destroy(moe); free(x); free(y); free(rw); free(ids);
}

static Qwen38GpuF32 g3(id<MTLBuffer> b, size_t rows, size_t a, size_t c) {
    return (Qwen38GpuF32){.span={(__bridge void *)b, 0, rows*a*c*4}, .rank=3, .dim={rows, a, c},
        .stride_bytes={a*c*4, c*4, 4}};
}
static Qwen38GpuF32 g1(id<MTLBuffer> b, size_t n) {
    return (Qwen38GpuF32){.span={(__bridge void *)b, 0, n*4}, .rank=1, .dim={n}, .stride_bytes={4}};
}
static void qsa_admission(Q38DGpu *dg,Q38CGpu *ops,const Qwen38AttentionConfig *c,
        const Q38DGpuLayer *l,NSArray *scratch,NSArray *inputs) {
    Qwen38Error e; uint32_t t=(uint32_t)l->qg.dim[0];
    ok(command(0,scratch,inputs,^(Qwen38GpuCommand cmd) {
        Q38DGpuLayer bad=*l;
#define REJECT(change,want) do { bad=*l; change; \
    assert(q38d_layer_encode(dg,ops,cmd,c,0,t,&bad,NULL)==(want)); } while (0)
        REJECT(bad.out.span.bytes=0,Q38_CAPACITY);
        REJECT(bad.v.span.bytes=0,Q38_CAPACITY);
        REJECT(bad.ik.rank=3,Q38_SHAPE);
        REJECT(bad.index_cache.stride_bytes[0]+=4,Q38_STRIDE);
        REJECT(bad.out=l->qn,Q38_INVALID_ARGUMENT);
        REJECT(bad.value_cache=l->key_cache,Q38_INVALID_ARGUMENT);
        REJECT(bad.out.span=cmd.error_word,Q38_CAPACITY);
        REJECT(bad.qg.span.offset_bytes++,Q38_STRIDE);
#undef REJECT
        assert(q38d_layer_encode(dg,ops,cmd,c,UINT32_MAX,t,l,NULL)==Q38_CAPACITY);
        Qwen38AttentionConfig wrong=*c; wrong.heads=0;
        assert(q38d_layer_encode(dg,ops,cmd,&wrong,0,t,l,NULL)==Q38_SHAPE);
        /* T=0 validates descriptors but records nothing, even after reset. */
        bad=*l;
        Qwen38GpuF32 *empty[]={&bad.qg,&bad.k,&bad.v,&bad.iq,&bad.ik,&bad.qn,&bad.qr,
            &bad.gate,&bad.kn,&bad.iqn,&bad.iqr,&bad.out};
        for (size_t i=0;i<sizeof empty/sizeof *empty;++i) { empty[i]->dim[0]=0; empty[i]->span.bytes=0; }
        return q38d_layer_encode(dg,ops,cmd,c,0,0,&bad,NULL);
    },&e),&e,"qsa preflight");
    assert(qwen38_gpu_session_info(session).next_pos==0);
    /* Even a late bad output must be rejected before C norms/copies run. */
    for (id<MTLBuffer> b in scratch) {
        const unsigned char *p=b.contents;
        for (size_t i=0;i<b.length;++i) assert(p[i]==0);
    }
}
/* D: full projected-input composition; CPU intermediates are observations
 * after completion, never inputs to the GPU selector or attention. */
static void check_qsa(uint32_t layer, size_t total, size_t chunk, uint32_t budget, bool ties) {
    const Qwen38Layer *kind; Qwen38Error e;
    ok(qwen38_config_layer(config, layer, &kind, &e), &e, "layer");
    Qwen38AttentionConfig c = qwen38_config_view(config)->attention;
    c.ratio = kind->compress_ratio;
    if (budget) c.token_budget=budget;
    size_t h = c.heads, j = c.kv_heads, d = c.head_dim, hi = c.index_heads, di = c.index_dim, cap = total;
    const Qwen38Role gr[] = {Q38_ATTN_Q_NORM, Q38_ATTN_K_NORM, Q38_INDEX_Q_NORM, Q38_INDEX_K_NORM};
    float *gamma[4]; size_t gn[4] = {d, d, di, di};
    for (size_t i = 0; i < 4; ++i) {
        Q38PackedCpuView v; binding(layer, gr[i], &v); gamma[i] = malloc(gn[i]*4);
        ok(q38_packed_decode_row(&v, 0, 0, gamma[i], gn[i], &e), &e, "gamma");
    }
    size_t w_qg = h*2*d, w_kv = j*d, w_iq = hi*di;
    float *qg = random_rows(total*w_qg, 11), *k = random_rows(total*w_kv, 12), *v = random_rows(total*w_kv, 13),
        *iq = random_rows(total*w_iq, 14), *ik = random_rows(total*di, 15), *out = malloc(total*h*d*4);
    if (ties) memset(iq,0,total*w_iq*4);
    float *baseline=malloc(total*h*d*4);
    Qwen38AttentionPlan plan; Qwen38AttentionState *state; Qwen38AttentionWorkspace *work;
    ok(qwen38_attention_plan(&c, (uint32_t)cap, (uint32_t)total, &plan, &e), &e, "plan");
    ok(qwen38_attention_create(&c, &plan, (Qwen38StateKey){config, layer, 1}, plan.cache_bytes + plan.workspace_bytes,
        &state, &work, &e), &e, "create");
    Q38DGpu *dg; Q38CGpu *ops;
    ok(q38d_gpu_create((__bridge void *)device, (__bridge void *)library, &dg, &e), &e, "D gpu");
    ok(q38c_gpu_create((__bridge void *)device, (__bridge void *)library, &ops, &e), &e, "C gpu");
    id<MTLBuffer> bq = shared(total*w_qg*4), bk = shared(total*w_kv*4), bv = shared(total*w_kv*4), bi = shared(total*w_iq*4),
        bik = shared(total*di*4), gq = shared(d*4), gk = shared(d*4), giq = shared(di*4), gik = shared(di*4),
        kc = shared(cap*w_kv*4), vc = shared(cap*w_kv*4), ic = shared(cap*di*4), bc = shared(cap/c.ratio*di*4 + 4),
        qn = shared(total*h*d*4), qr = shared(total*h*d*4), gt = shared(total*h*d*4), kn = shared(total*w_kv*4),
        iqn = shared(total*w_iq*4), iqr = shared(total*w_iq*4), pl = shared(di*4), pn = shared(di*4),
        sc = shared(cap/c.ratio*4 + 4), kp = shared(cap/c.ratio*4 + 4), ids = shared((c.token_budget + c.ratio)*4),
        bo = shared(total*h*d*4);
    memcpy(gq.contents, gamma[0], d*4); memcpy(gk.contents, gamma[1], d*4);
    memcpy(giq.contents, gamma[2], di*4); memcpy(gik.contents, gamma[3], di*4);
    NSArray *scratch=@[kc,vc,ic,bc,qn,qr,gt,kn,iqn,iqr,pl,pn,sc,kp,ids,bo];
    NSArray *inputs=@[bq,bk,bv,bi,bik,gq,gk,giq,gik];
    double errors[8]={0}; bool exact[8]={true,true,true,true,true,true,true,true},exact_ids=true,replay=true;
    for (uint64_t pass=0;pass<3;++pass) {
    ok(qwen38_gpu_session_reset(session,config,&e),&e,"qsa GPU reset");
    if (pass) ok(qwen38_attention_reset(state,pass+1,&e),&e,"qsa CPU reset");
    size_t step=pass==1 ? chunk : total; /* one-shot, chunks, reset+replay */
    for (size_t start=0,n=MIN(step,total); start<total; start+=n,n=MIN(step,total-start)) {
        Qwen38AttentionTicket *ticket;
        Qwen38AttentionInput in = {
            {qg + start*w_qg, n*w_qg*4, 4, {n, h, 2, d}, {w_qg*4, 2*d*4, d*4, 4}},
            {k + start*w_kv, n*w_kv*4, 3, {n, j, d}, {w_kv*4, d*4, 4}},
            {v + start*w_kv, n*w_kv*4, 3, {n, j, d}, {w_kv*4, d*4, 4}},
            {iq + start*w_iq, n*w_iq*4, 3, {n, hi, di}, {w_iq*4, di*4, 4}},
            cview(ik + start*di, n, di),
            {gamma[0], d*4, 1, {d}, {4}}, {gamma[1], d*4, 1, {d}, {4}},
            {gamma[2], di*4, 1, {di}, {4}}, {gamma[3], di*4, 1, {di}, {4}}};
        ok(qwen38_attention_begin(state, start, (uint32_t)n, pass+1, &ticket, &e), &e, "begin");
        Qwen38F32Mut o = {out + start*h*d, n*h*d*4, 3, {n, h, d}, {h*d*4, d*4, 4}};
        ok(qwen38_attention_cpu(ticket, &in, o, work, &e), &e, "attention cpu");
        memcpy(bq.contents, qg + start*w_qg, n*w_qg*4); memcpy(bk.contents, k + start*w_kv, n*w_kv*4);
        memcpy(bv.contents, v + start*w_kv, n*w_kv*4); memcpy(bi.contents, iq + start*w_iq, n*w_iq*4);
        memcpy(bik.contents, ik + start*di, n*di*4);
        Q38DGpuLayer l = {
            {.span={(__bridge void *)bq, 0, n*w_qg*4}, .rank=4, .dim={n, h, 2, d}, .stride_bytes={w_qg*4, 2*d*4, d*4, 4}},
            g3(bk, n, j, d), g3(bv, n, j, d), g3(bi, n, hi, di), gview(bik, 0, n, di),
            g1(gq, d), g1(gk, d), g1(giq, di), g1(gik, di),
            g3(kc, cap, j, d), g3(vc, cap, j, d), gview(ic, 0, cap, di), gview(bc, 0, cap/c.ratio, di),
            g3(qn, n, h, d), g3(qr, n, h, d), g3(gt, n, h, d), g3(kn, n, j, d), g3(iqn, n, hi, di), g3(iqr, n, hi, di),
            g3(pl, 1, 1, di), g3(pn, 1, 1, di), g1(sc, cap/c.ratio),
            {(__bridge void *)kp, 0, kp.length}, {(__bridge void *)ids, 0, ids.length}, g3(bo, n, h, d)};
        __block Qwen38Status rc = Q38_OK; __block Qwen38Error inner;
        size_t pos0 = start, count = n; Q38DGpuLayer *lp = &l;
        if (!pass && !start) qsa_admission(dg,ops,&c,lp,scratch,inputs);
        if (pass==2 && !start) {
            /* Nonfinite V is found on device after writes; G0 must withhold
             * publication and force reset, then stale caches are replayed. */
            float saved=((float *)bv.contents)[n*w_kv-1];
            ((float *)bv.contents)[n*w_kv-1]=NAN;
            Qwen38Status failure=command(n,scratch,inputs,^(Qwen38GpuCommand cmd) {
                return q38d_layer_encode(dg,ops,cmd,&c,0,(uint32_t)count,lp,&inner);
            },&e);
            assert(failure==Q38_GPU_ERROR);
            Qwen38StateInfo invalid=qwen38_gpu_session_info(session);
            assert(!invalid.valid && !invalid.busy && invalid.next_pos==0);
            ok(qwen38_gpu_session_reset(session,config,&e),&e,"qsa fault reset");
            ((float *)bv.contents)[n*w_kv-1]=saved;
        }
        assert(qwen38_gpu_session_info(session).next_pos==start);
        ok(command(n, scratch, inputs, ^(Qwen38GpuCommand cmd) {
            rc = q38d_layer_encode(dg, ops, cmd, &c, (uint32_t)pos0, (uint32_t)count, lp, &inner);
            assert(qwen38_gpu_session_info(session).next_pos==pos0);
            return rc;
        }, &e), &e, "qsa command");
        ok(rc, &inner, "qsa encode");
        assert(qwen38_gpu_session_info(session).next_pos==start+n);
        ok(qwen38_attention_finish(ticket, true, &e), &e, "finish");
        bool bitwise;
        const float *cq,*cg,*ci,*cs; const uint32_t *ci_ids;
        q38d_last_query(work,&cq,&cg,&ci,&cs,&ci_ids);
        size_t end=start+n,blocks=end/c.ratio,selected=MIN(blocks,c.token_budget/c.ratio)*c.ratio+end%c.ratio;
        const float *got[]={bo.contents,(float *)qr.contents+(n-1)*h*d,(float *)gt.contents+(n-1)*h*d,
            (float *)iqr.contents+(n-1)*hi*di,kc.contents,ic.contents,bc.contents,sc.contents};
        const float *ref[]={out+start*h*d,cq,cg,ci,q38d_cache_keys(state),q38d_cache_raw_index(state),q38d_cache_blocks(state),cs};
        size_t sizes[]={n*h*d,h*d,h*d,hi*di,end*j*d,end*di,blocks*di,blocks};
        for (size_t i=0;i<8;++i) {
            errors[i]=fmax(errors[i],scaled(got[i],ref[i],sizes[i],&bitwise)); exact[i]&=bitwise;
        }
        exact_ids &= !memcmp(ids.contents,ci_ids,selected*4);
        if (!pass) memcpy(baseline+start*h*d,bo.contents,n*h*d*4);
        else replay &= !memcmp(baseline+start*h*d,bo.contents,n*h*d*4);
    }
    }
    printf("{\"qsa_tokens\":%zu,\"chunk\":%zu,\"budget\":%u,\"ties\":%s}\n",total,chunk,c.token_budget,ties ? "true" : "false");
    const char *names[]={"qsa_out","qsa_q","qsa_gate","qsa_iq","qsa_keys","qsa_raw_index","qsa_blocks","qsa_scores"};
    const double gates[]={3e-5,2e-5,1e-5,2e-5,2e-5,0,2e-5,2e-5};
    for (size_t i=0;i<8;++i) report(names[i],layer,errors[i],gates[i],exact[i]);
    report("qsa_ids",layer,exact_ids ? 0 : 1,0,exact_ids);
    report("qsa_chunk_reset",layer,replay ? 0 : 1,0,replay);
    qwen38_attention_destroy(state, work, &e); q38d_gpu_destroy(dg); q38c_gpu_destroy(ops);
    for (size_t i = 0; i < 4; ++i) free(gamma[i]);
    free(qg); free(k); free(v); free(iq); free(ik); free(out); free(baseline);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s GGUF METALLIB\n", argv[0]); return 2; }
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        NSError *err = nil;
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&err];
        assert(device && library);
        Qwen38Error e;
        gguf_open(argv[1], &gguf);
        ok(qwen38_config_load(&gguf, &config, &e), &e, "config");
        ok(qwen38_weights_describe(&gguf, config, &weights, &e), &e, "describe");
        store = gpu_weights_open((__bridge void *)device, &gguf, 0);
        ok(qwen38_weights_attach(weights, &gguf, gpu_weights_plan(store), &e), &e, "attach");
        ok(qwen38_gpu_session_create((__bridge void *)device, config, 1u << 20, &session, &e), &e, "session");
        for (uint32_t layer = 0; layer < qwen38_config_view(config)->trunk_blocks; ++layer) {
            const Qwen38Layer *kind; Qwen38Error le;
            ok(qwen38_config_layer(config, layer, &kind, &le), &le, "kind");
            check_moe(layer, 24);
            if (kind->kind == Q38_LAYER_QSA) {
                check_qsa(layer,40,13,0,false);
                check_qsa(layer,17,1,0,true);
                check_qsa(layer,3,1,0,false); /* no completed blocks */
                check_qsa(layer,2053,1023,2048,false);
            }
        }
        ok(qwen38_gpu_session_destroy(session, &e), &e, "session destroy");
        gpu_weights_close(store); qwen38_weights_free(weights); qwen38_config_free(config); gguf_close(&gguf);
    }
    return failures ? 1 : 0;
}
