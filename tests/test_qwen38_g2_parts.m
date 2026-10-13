/* G2 adapter checks on the G1 graph fixture: each GPU composition against
 * its verified CPU counterpart, through the real G0 session lifecycle.
 *   test_qwen38_g2_parts GGUF METALLIB
 * Prints one JSON line per check; exits nonzero on any gate failure. */
#import <Metal/Metal.h>
#include "picoforge.h"
#include "qwen38_c_internal.h"
#include "qwen38_f_internal.h"
#include "qwen38_moe.h"
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
typedef void (^Encode)(Qwen38GpuCommand);
static Qwen38Status command(size_t count, NSArray<id<MTLBuffer>> *scratch, Encode encode, Qwen38Error *e) {
    Qwen38GpuTicket t; Qwen38GpuCommand cmd;
    Qwen38StateInfo info = qwen38_gpu_session_info(session);
    ok(qwen38_gpu_session_begin(session, config, info.next_pos, count, &t, &cmd, e), e, "begin");
    for (id<MTLBuffer> b in scratch)
        ok(qwen38_gpu_session_claim(session, &t, (Qwen38GpuSpan){(__bridge void *)b, 0, b.length}, Q38_GPU_USE_WRITE, e), e, "claim");
    const GgufPlan *plan = gpu_weights_plan(store);
    for (size_t i = 0; i < plan->weight_count; ++i) {
        const GgufBinding *w = &plan->weights[i];
        ok(qwen38_gpu_session_claim(session, &t, (Qwen38GpuSpan){gpu_weights_buffer(store, w), w->offset, w->tensor.bytes},
            Q38_GPU_USE_READ, e), e, "claim weights");
    }
    encode(cmd);
    ok(qwen38_gpu_session_submit(session, &t, e), e, "submit");
    return qwen38_gpu_session_finish(session, &t, e);
}
static double scaled(const float *a, const float *r, size_t n, bool *bitwise) {
    double err = 0, top = 1; *bitwise = true;
    for (size_t i = 0; i < n; ++i) {
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
    ok(command(t, @[by, bl, bp, bw, bg, bu, ba, bd, sg, su, sa, sh, sl, ss, br, bi, bm], ^(Qwen38GpuCommand cmd) {
        rc = q38f_routed_encode(routed, b, ops, cmd, c, store, gbp, glp, gview(bx,0,t,h), gview(by,0,t,h), &rwk, &inner);
    }, &e), &e, "moe command");
    ok(rc, &inner, "moe encode");
    bool same_ids = !memcmp(bi.contents, ids, p*4), bitwise;
    report("moe_route_ids", layer, same_ids ? 0 : 1, 0, same_ids);
    report("moe_route_weights", layer, scaled(bw.contents, rw, p, &bitwise), 2e-6, bitwise);
    report("moe_y", layer, scaled(by.contents, y, x_n, &bitwise), 5e-4, bitwise);
    q38f_routed_destroy(routed); q38_packed_gpu_free(b); q38c_gpu_destroy(ops);
    qwen38_moe_work_destroy(work); qwen38_moe_destroy(moe); free(x); free(y); free(rw); free(ids);
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
        for (uint32_t layer = 0; layer < qwen38_config_view(config)->trunk_blocks; ++layer) check_moe(layer, 24);
        ok(qwen38_gpu_session_destroy(session, &e), &e, "session destroy");
        gpu_weights_close(store); qwen38_weights_free(weights); qwen38_config_free(config); gguf_close(&gguf);
    }
    return failures ? 1 : 0;
}
