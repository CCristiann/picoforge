/* Private CPU plumbing for component C; public contracts stay in qwen38_ops.h. */
#ifndef PICOFORGE_QWEN38_C_INTERNAL_H
#define PICOFORGE_QWEN38_C_INTERNAL_H
#ifndef __METAL_VERSION__
#include "qwen38_ops.h"
#include "qwen38_gpu.h"
#include "qwen38.h"
#include "qwen38_packed.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define Q38C_TRY(call) do { Qwen38Status q38c_rc = (call); if (q38c_rc != Q38_OK) return q38c_rc; } while (0)
static inline Qwen38Status q38c_error(Qwen38Error *e, Qwen38Status code, const char *detail) {
    if (e) {
        memset(e, 0, sizeof(*e)); e->code = code;
        if (code != Q38_OK) {
            snprintf(e->subject, sizeof(e->subject), "qwen38/C");
            snprintf(e->detail, sizeof(e->detail), "%s", detail);
        }
    }
    return code;
}
static inline bool q38c_mul(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return false;
    *out = a * b; return true;
}
static inline bool q38c_add(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return false;
    *out = a + b; return true;
}
static inline Qwen38F32 q38c_ro(Qwen38F32Mut v) {
    Qwen38F32 r = {.data=v.data, .capacity_bytes=v.capacity_bytes, .rank=v.rank};
    memcpy(r.dim, v.dim, sizeof(r.dim)); memcpy(r.stride_bytes, v.stride_bytes, sizeof(r.stride_bytes));
    return r;
}
/* Layout validation does not inspect a CPU pointer; also used by Metal glue. */
static inline Qwen38Status q38c_layout(Qwen38F32 v, size_t *extent, Qwen38Error *e) {
    if (v.rank < 1 || v.rank > QWEN38_MAX_RANK) return q38c_error(e, Q38_SHAPE, "rank must be 1..4");
    size_t span = sizeof(float);
    for (size_t i = v.rank; i-- > 0;) {
        if (!v.dim[i] && i != 0) return q38c_error(e, Q38_SHAPE, "only the first axis may be empty");
        if (v.dim[i] > UINT32_MAX) return q38c_error(e, Q38_RANGE, "dimension exceeds uint32");
        size_t stride = v.stride_bytes[i], extra;
        if (stride % sizeof(float) || stride < span || (i == v.rank-1 && stride != sizeof(float)))
            return q38c_error(e, Q38_STRIDE, "noncontiguous feature or overlapping axes");
        if (v.dim[i] && (!q38c_mul(v.dim[i]-1, stride, &extra) || !q38c_add(span, extra, &span)))
            return q38c_error(e, Q38_OVERFLOW, "view span overflows");
    }
    *extent = v.dim[0] ? span : 0;
    return Q38_OK;
}
static inline Qwen38Status q38c_view(Qwen38F32 v, Qwen38Error *e) {
    size_t span;
    Q38C_TRY(q38c_layout(v, &span, e));
    if (!span) return Q38_OK;
    if (!v.data) return q38c_error(e, Q38_INVALID_ARGUMENT, "null nonempty view");
    if ((uintptr_t)v.data % _Alignof(float)) return q38c_error(e, Q38_STRIDE, "unaligned float pointer");
    if (span > UINTPTR_MAX - (uintptr_t)v.data) return q38c_error(e, Q38_OVERFLOW, "pointer span overflows");
    if (span > v.capacity_bytes) return q38c_error(e, Q38_CAPACITY, "view exceeds capacity");
    return Q38_OK;
}
/* The following indexing helpers are used only AFTER q38c_view succeeds. */
static inline size_t q38c_count(Qwen38F32 v) {
    size_t n = 1; for (size_t i = 0; i < v.rank; ++i) n *= v.dim[i]; return n;
}
static inline size_t q38c_offset(Qwen38F32 v, size_t index) {
    size_t offset = 0;
    for (size_t i = v.rank; i-- > 0;) {
        offset += (index % v.dim[i]) * (v.stride_bytes[i] / sizeof(float)); index /= v.dim[i];
    }
    return offset;
}
static inline float q38c_get(Qwen38F32 v, size_t index) { return v.data[q38c_offset(v, index)]; }
static inline size_t q38c_span(Qwen38F32 v) {
    if (!v.dim[0]) return 0;
    size_t n = sizeof(float);
    for (size_t i = 0; i < v.rank; ++i) n += (v.dim[i]-1) * v.stride_bytes[i];
    return n;
}
static inline bool q38c_overlap(const void *a, size_t na, const void *b, size_t nb) {
    return na && nb && (uintptr_t)a < (uintptr_t)b + nb && (uintptr_t)b < (uintptr_t)a + na;
}
static inline bool q38c_same(Qwen38F32 a, Qwen38F32 b) {
    if (a.rank != b.rank) return false;
    for (size_t i = 0; i < a.rank; ++i) if (a.dim[i] != b.dim[i]) return false;
    return true;
}
static inline Qwen38Status q38c_alias(Qwen38F32 out, Qwen38F32 in, bool exact, Qwen38Error *e) {
    if (!q38c_overlap(out.data, q38c_span(out), in.data, q38c_span(in))) return Q38_OK;
    if (exact && out.data == in.data && q38c_same(out, in)) {
        bool equal = true;
        for (size_t i = 0; i < out.rank; ++i) equal &= out.stride_bytes[i] == in.stride_bytes[i];
        if (equal) return Q38_OK;
    }
    return q38c_error(e, Q38_INVALID_ARGUMENT, "forbidden view alias");
}
static inline Qwen38Status q38c_finite(Qwen38F32 v, Qwen38Error *e) {
    for (size_t i = 0; i < q38c_count(v); ++i)
        if (!isfinite(q38c_get(v, i))) return q38c_error(e, Q38_NONFINITE, "nonfinite input");
    return Q38_OK;
}
static inline float q38c_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z = expf(x); return z / (1.0f + z);
}
static inline float q38c_silu(float x) { return x * q38c_sigmoid(x); }

static inline Qwen38Status q38c_dims(Qwen38F32 v, size_t rank, const size_t *dims, Qwen38Error *e) {
    Q38C_TRY(q38c_view(v, e));
    if (v.rank != rank) return q38c_error(e, Q38_SHAPE, "unexpected rank");
    for (size_t i=0; i<rank; ++i) if (v.dim[i] != dims[i]) return q38c_error(e, Q38_SHAPE, "unexpected shape");
    return Q38_OK;
}
typedef struct {
    const void *self;
    uint64_t magic;
    Qwen38StateInfo info;
    Qwen38Memory metadata, payload;
} Q38CState;
enum { Q38C_ALIGNMENT = 64 };
#define Q38C_CONV_MAGIC UINT64_C(0x513338434f4e5631)
#define Q38C_GDN_MAGIC UINT64_C(0x51333847444e3031)
static inline Qwen38Status q38c_size(size_t metadata, size_t elements, Qwen38StateSize *out, Qwen38Error *e) {
    if (!out) return q38c_error(e, Q38_INVALID_ARGUMENT, "null size output");
    *out = (Qwen38StateSize){0}; size_t bytes;
    if (!q38c_mul(elements, sizeof(float), &bytes) || !q38c_add(bytes, Q38C_ALIGNMENT-1, &bytes))
        return q38c_error(e, Q38_OVERFLOW, "state size overflows");
    out->payload_bytes = bytes & ~(size_t)(Q38C_ALIGNMENT-1);
    out->metadata_bytes = (metadata + Q38C_ALIGNMENT-1) & ~(size_t)(Q38C_ALIGNMENT-1);
    out->alignment = Q38C_ALIGNMENT;
    return q38c_error(e, Q38_OK, NULL);
}
static inline Qwen38Status q38c_backing(Qwen38Memory m, size_t need, Qwen38Error *e) {
    if (m.bytes < need) return q38c_error(e, Q38_CAPACITY, "state backing too short");
    if (m.bytes && !m.data) return q38c_error(e, Q38_INVALID_ARGUMENT, "null state backing");
    if ((uintptr_t)m.data % Q38C_ALIGNMENT) return q38c_error(e, Q38_STRIDE, "unaligned state backing");
    if (m.bytes > UINTPTR_MAX - (uintptr_t)m.data) return q38c_error(e, Q38_OVERFLOW, "backing span overflows");
    return Q38_OK;
}
static inline Qwen38Status q38c_backings(Qwen38StateKey key, Qwen38Memory m, Qwen38Memory p,
                                        Qwen38StateSize size, Qwen38Error *e) {
    if (!key.model || !key.epoch) return q38c_error(e, Q38_INVALID_ARGUMENT, "state requires model and nonzero epoch");
    Q38C_TRY(q38c_backing(m, size.metadata_bytes, e)); Q38C_TRY(q38c_backing(p, size.payload_bytes, e));
    if (q38c_overlap(m.data, m.bytes, p.data, p.bytes)) return q38c_error(e, Q38_INVALID_ARGUMENT, "state backings overlap");
    return Q38_OK;
}
static inline Qwen38Status q38c_state_check(const Q38CState *s, uint64_t magic, bool require_valid, Qwen38Error *e) {
    if (!s || s->self != s || s->magic != magic) return q38c_error(e, Q38_BAD_STATE, "invalid or copied handle");
    if (s->info.busy) return q38c_error(e, Q38_BUSY, "state in use");
    if (require_valid && !s->info.valid) return q38c_error(e, Q38_BAD_STATE, "state requires reset");
    return Q38_OK;
}
static inline Qwen38Status q38c_state_alias(const Q38CState *s, Qwen38F32 v, Qwen38Error *e) {
    size_t bytes = q38c_span(v);
    if (q38c_overlap(v.data, bytes, s->metadata.data, s->metadata.bytes) ||
        q38c_overlap(v.data, bytes, s->payload.data, s->payload.bytes))
        return q38c_error(e, Q38_INVALID_ARGUMENT, "view aliases state backing");
    return Q38_OK;
}
static inline Qwen38Status q38c_state_reset(Q38CState *s, uint64_t magic, uint64_t epoch, size_t bytes, Qwen38Error *e) {
    Q38C_TRY(q38c_state_check(s, magic, false, e));
    if (s->info.key.epoch == UINT64_MAX) return q38c_error(e, Q38_OVERFLOW, "epoch exhausted");
    if (epoch <= s->info.key.epoch) return q38c_error(e, Q38_STALE, "reset requires a newer epoch");
    if (bytes) memset(s->payload.data, 0, bytes);
    s->info.key.epoch=epoch; s->info.next_pos=0; s->info.valid=true;
    return q38c_error(e, Q38_OK, NULL);
}
/* C4 private composition. A resolves roles; B owns every decode and product.
 * Catalog/config, packed spans and all backings remain borrowed by the caller. */
typedef enum { Q38CW_HN, Q38CW_HD, Q38CW_HU, Q38CW_HI, Q38CW_QKV, Q38CW_Z,
    Q38CW_AP, Q38CW_BP, Q38CW_A, Q38CW_DT, Q38CW_CONV, Q38CW_GN, Q38CW_OUT, Q38CW_COUNT } Q38CWeight;
extern const Qwen38Role q38c_layer_roles[Q38CW_COUNT];
typedef struct {
    Qwen38ConfigView config;
    Qwen38ConvShape conv;
    const void *model;
    uint32_t layer;
    bool bound;
    size_t constant_bytes;
    Qwen38Tensor tensor[Q38CW_COUNT];
    Q38PackedCpuView weight[Q38CW_COUNT];
    Qwen38F32 constant[Q38CW_COUNT]; /* only HN/A/DT/CONV/GN, decoded by B */
    Qwen38Memory constant_memory;
} Q38CLayer;
Qwen38Status q38c_layer_plan(const Qwen38Config *, const Qwen38Weights *, uint32_t,
    Q38CLayer *, Qwen38Error *);
Qwen38Status q38c_layer_bind(Q38CLayer *, const Qwen38CpuSpan[Q38CW_COUNT], Qwen38Memory, Qwen38Error *);
typedef enum { Q38CT_N, Q38CT_LOW, Q38CT_ACT, Q38CT_HG, Q38CT_IL, Q38CT_INJECT,
    Q38CT_MIX, Q38CT_QKV, Q38CT_CONV, Q38CT_PREP, Q38CT_Z, Q38CT_ALPHA, Q38CT_BR,
    Q38CT_LOGG, Q38CT_BETA, Q38CT_SCAN, Q38CT_NORM, Q38CT_OUT, Q38CT_COUNT } Q38CStage;
extern const char *const q38c_stage_names[Q38CT_COUNT];
typedef struct {
    const Q38CLayer *owner;
    size_t capacity_tokens;
    Qwen38Memory memory;
    Qwen38F32Mut trace[Q38CT_COUNT]; /* flat [T,features], token pitch retained */
} Q38CWork;
Qwen38Status q38c_work_size(const Q38CLayer *, size_t tokens, size_t *, Qwen38Error *);
Qwen38Status q38c_work_init(const Q38CLayer *, size_t tokens, Qwen38Memory, Q38CWork *, Qwen38Error *);
Qwen38Status q38c_work_check(const Q38CLayer *, const Q38CWork *, Qwen38F32 residual, Qwen38Error *);
Qwen38Status q38c_hc_cpu(const Q38CLayer *, Q38CWork *, Qwen38F32 residual, Qwen38Error *);
static inline Qwen38F32Mut q38c_trace(const Q38CWork *w, Q38CStage stage, size_t tokens) {
    Qwen38F32Mut v=w->trace[stage]; v.dim[0]=tokens; return v;
}
static inline Qwen38F32Mut q38c_group(Qwen38F32Mut flat, size_t groups, size_t width) {
    flat.rank=3; flat.dim[1]=groups; flat.dim[2]=width;
    flat.stride_bytes[1]=width*4; flat.stride_bytes[2]=4; return flat;
}
typedef struct {
    const Q38CLayer *owner;
    Qwen38ConvState *conv;
    Qwen38GdnState *gdn;
    Qwen38StateKey key;
    bool valid, busy;
} Q38CBlockState;
Qwen38Status q38c_block_init(const Q38CLayer *, Qwen38ConvState *, Qwen38GdnState *, Q38CBlockState *, Qwen38Error *);
Qwen38Status q38c_block_reset(Q38CBlockState *, uint64_t epoch, Qwen38Error *);
Qwen38Status q38c_block_cpu(const Q38CLayer *, Q38CWork *, Qwen38F32Mut residual,
    Q38CBlockState *, uint64_t start, Qwen38Error *);

/* Private harness interface only; session lifecycle remains in the CPU API.
 * Caller owns command completion, error checking and all resource lifetimes. */
typedef struct Q38CGpu Q38CGpu;
/* Slots: ACT/HC_LOW/HC_INJECT x,out; RMS x,gain,out;
 * HC_READ normed,logits,mixed; HC_ADD residual,block,inject;
 * CONV x,taps,out,history[L,C] (raw oldest-first, mutable after success);
 * GDN_PREP qkv,a,b,A,dt,prepared,log_decay,beta; GDN_NORM o,gain,out,z;
 * GDN_SCAN prepared,log_decay,beta,S[Hv,Dv,Dk],out[T,Hv,Dv].
 * Raw GPU state has NO private epoch/position API: the harness owns it and
 * discards/resets it on any command error. Public lifecycle glue is pending. */
typedef enum { Q38C_ACT, Q38C_RMS, Q38C_HC_LOW, Q38C_HC_INJECT,
    Q38C_HC_READ, Q38C_HC_ADD, Q38C_CONV, Q38C_GDN_PREP, Q38C_GDN_NORM,
    Q38C_GDN_SCAN, Q38C_GPU_OPS } Q38CGpuOp;
typedef struct { Qwen38Activation activation; float eps; Qwen38HcShape hc;
    Qwen38ConvShape conv; Qwen38GdnShape gdn; } Q38CGpuSpec;
Qwen38Status q38c_gpu_create(void *device, void *library, Q38CGpu **, Qwen38Error *);
void q38c_gpu_destroy(Q38CGpu *);
Qwen38Status q38c_gpu_encode(Q38CGpu *, Qwen38GpuCommand, Q38CGpuOp,
    const Q38CGpuSpec *, const Qwen38GpuF32 *, size_t count, Qwen38Error *);
typedef uint32_t Q38CUint;
#else
typedef uint Q38CUint;
#endif
/* Fixed-width wire format, strides in FLOAT ELEMENTS after host admission.
 * Slot numbers and operation parameters are private to these kernels/glue. */
typedef struct { Q38CUint dim[4], stride[4], rank, count; } Q38CGpuView;
typedef struct { Q38CGpuView v[8]; Q38CUint p[6], read_mask; float eps; } Q38CGpuArgs;
#ifndef __METAL_VERSION__
_Static_assert(sizeof(Q38CGpuView)==40 && sizeof(Q38CGpuArgs)==352, "Metal argument ABI");
_Static_assert(offsetof(Q38CGpuArgs,p)==320 && offsetof(Q38CGpuArgs,eps)==348, "Metal offsets");
#endif
#endif
