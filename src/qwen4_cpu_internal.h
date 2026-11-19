/* Private plumbing of the CPU ops (norms, conv, Gated DeltaNet); public contracts stay in qwen4_ops.h. */
#ifndef PICOFORGE_QWEN4_CPU_INTERNAL_H
#define PICOFORGE_QWEN4_CPU_INTERNAL_H
#ifndef __METAL_VERSION__
#include "qwen4_ops.h"
#include "qwen4_gpu.h"
#include "qwen4.h"
#include "qwen4_packed.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define QW4C_TRY(call) do { Qwen4Status qw4c_rc = (call); if (qw4c_rc != QW4_OK) return qw4c_rc; } while (0)
static inline Qwen4Status qw4c_error(Qwen4Error *e, Qwen4Status code, const char *detail) {
    if (e) {
        memset(e, 0, sizeof(*e)); e->code = code;
        if (code != QW4_OK) {
            snprintf(e->subject, sizeof(e->subject), "qwen4/C");
            snprintf(e->detail, sizeof(e->detail), "%s", detail);
        }
    }
    return code;
}
static inline bool qw4c_mul(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return false;
    *out = a * b; return true;
}
static inline bool qw4c_add(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return false;
    *out = a + b; return true;
}
static inline Qwen4F32 qw4c_ro(Qwen4F32Mut v) {
    Qwen4F32 r = {.data=v.data, .capacity_bytes=v.capacity_bytes, .rank=v.rank};
    memcpy(r.dim, v.dim, sizeof(r.dim)); memcpy(r.stride_bytes, v.stride_bytes, sizeof(r.stride_bytes));
    return r;
}
/* Layout validation does not inspect a CPU pointer; also used by Metal glue. */
static inline Qwen4Status qw4c_layout(Qwen4F32 v, size_t *extent, Qwen4Error *e) {
    if (v.rank < 1 || v.rank > QWEN4_MAX_RANK) return qw4c_error(e, QW4_SHAPE, "rank must be 1..4");
    size_t span = sizeof(float);
    for (size_t i = v.rank; i-- > 0;) {
        if (!v.dim[i] && i != 0) return qw4c_error(e, QW4_SHAPE, "only the first axis may be empty");
        if (v.dim[i] > UINT32_MAX) return qw4c_error(e, QW4_RANGE, "dimension exceeds uint32");
        size_t stride = v.stride_bytes[i], extra;
        if (stride % sizeof(float) || stride < span || (i == v.rank-1 && stride != sizeof(float)))
            return qw4c_error(e, QW4_STRIDE, "noncontiguous feature or overlapping axes");
        if (v.dim[i] && (!qw4c_mul(v.dim[i]-1, stride, &extra) || !qw4c_add(span, extra, &span)))
            return qw4c_error(e, QW4_OVERFLOW, "view span overflows");
    }
    *extent = v.dim[0] ? span : 0;
    return QW4_OK;
}
static inline Qwen4Status qw4c_view(Qwen4F32 v, Qwen4Error *e) {
    size_t span;
    QW4C_TRY(qw4c_layout(v, &span, e));
    if (!span) return QW4_OK;
    if (!v.data) return qw4c_error(e, QW4_INVALID_ARGUMENT, "null nonempty view");
    if ((uintptr_t)v.data % _Alignof(float)) return qw4c_error(e, QW4_STRIDE, "unaligned float pointer");
    if (span > UINTPTR_MAX - (uintptr_t)v.data) return qw4c_error(e, QW4_OVERFLOW, "pointer span overflows");
    if (span > v.capacity_bytes) return qw4c_error(e, QW4_CAPACITY, "view exceeds capacity");
    return QW4_OK;
}
/* The following indexing helpers are used only AFTER qw4c_view succeeds. */
static inline size_t qw4c_count(Qwen4F32 v) {
    size_t n = 1; for (size_t i = 0; i < v.rank; ++i) n *= v.dim[i]; return n;
}
static inline size_t qw4c_offset(Qwen4F32 v, size_t index) {
    size_t offset = 0;
    for (size_t i = v.rank; i-- > 0;) {
        offset += (index % v.dim[i]) * (v.stride_bytes[i] / sizeof(float)); index /= v.dim[i];
    }
    return offset;
}
static inline float qw4c_get(Qwen4F32 v, size_t index) { return v.data[qw4c_offset(v, index)]; }
static inline size_t qw4c_span(Qwen4F32 v) {
    if (!v.dim[0]) return 0;
    size_t n = sizeof(float);
    for (size_t i = 0; i < v.rank; ++i) n += (v.dim[i]-1) * v.stride_bytes[i];
    return n;
}
static inline bool qw4c_overlap(const void *a, size_t na, const void *b, size_t nb) {
    return na && nb && (uintptr_t)a < (uintptr_t)b + nb && (uintptr_t)b < (uintptr_t)a + na;
}
static inline bool qw4c_same(Qwen4F32 a, Qwen4F32 b) {
    if (a.rank != b.rank) return false;
    for (size_t i = 0; i < a.rank; ++i) if (a.dim[i] != b.dim[i]) return false;
    return true;
}
static inline Qwen4Status qw4c_alias(Qwen4F32 out, Qwen4F32 in, bool exact, Qwen4Error *e) {
    if (!qw4c_overlap(out.data, qw4c_span(out), in.data, qw4c_span(in))) return QW4_OK;
    if (exact && out.data == in.data && qw4c_same(out, in)) {
        bool equal = true;
        for (size_t i = 0; i < out.rank; ++i) equal &= out.stride_bytes[i] == in.stride_bytes[i];
        if (equal) return QW4_OK;
    }
    return qw4c_error(e, QW4_INVALID_ARGUMENT, "forbidden view alias");
}
static inline Qwen4Status qw4c_finite(Qwen4F32 v, Qwen4Error *e) {
    for (size_t i = 0; i < qw4c_count(v); ++i)
        if (!isfinite(qw4c_get(v, i))) return qw4c_error(e, QW4_NONFINITE, "nonfinite input");
    return QW4_OK;
}
static inline float qw4c_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float z = expf(x); return z / (1.0f + z);
}
static inline float qw4c_silu(float x) { return x * qw4c_sigmoid(x); }

static inline Qwen4Status qw4c_dims(Qwen4F32 v, size_t rank, const size_t *dims, Qwen4Error *e) {
    QW4C_TRY(qw4c_view(v, e));
    if (v.rank != rank) return qw4c_error(e, QW4_SHAPE, "unexpected rank");
    for (size_t i=0; i<rank; ++i) if (v.dim[i] != dims[i]) return qw4c_error(e, QW4_SHAPE, "unexpected shape");
    return QW4_OK;
}
typedef struct {
    const void *self;
    uint64_t magic;
    Qwen4StateInfo info;
    Qwen4Memory metadata, payload;
} QW4CState;
enum { QW4C_ALIGNMENT = 64 };
#define QW4C_CONV_MAGIC UINT64_C(0x513338434f4e5631)
#define QW4C_GDN_MAGIC UINT64_C(0x51333847444e3031)
static inline Qwen4Status qw4c_size(size_t metadata, size_t elements, Qwen4StateSize *out, Qwen4Error *e) {
    if (!out) return qw4c_error(e, QW4_INVALID_ARGUMENT, "null size output");
    *out = (Qwen4StateSize){0}; size_t bytes;
    if (!qw4c_mul(elements, sizeof(float), &bytes) || !qw4c_add(bytes, QW4C_ALIGNMENT-1, &bytes))
        return qw4c_error(e, QW4_OVERFLOW, "state size overflows");
    out->payload_bytes = bytes & ~(size_t)(QW4C_ALIGNMENT-1);
    out->metadata_bytes = (metadata + QW4C_ALIGNMENT-1) & ~(size_t)(QW4C_ALIGNMENT-1);
    out->alignment = QW4C_ALIGNMENT;
    return qw4c_error(e, QW4_OK, NULL);
}
static inline Qwen4Status qw4c_backing(Qwen4Memory m, size_t need, Qwen4Error *e) {
    if (m.bytes < need) return qw4c_error(e, QW4_CAPACITY, "state backing too short");
    if (m.bytes && !m.data) return qw4c_error(e, QW4_INVALID_ARGUMENT, "null state backing");
    if ((uintptr_t)m.data % QW4C_ALIGNMENT) return qw4c_error(e, QW4_STRIDE, "unaligned state backing");
    if (m.bytes > UINTPTR_MAX - (uintptr_t)m.data) return qw4c_error(e, QW4_OVERFLOW, "backing span overflows");
    return QW4_OK;
}
static inline Qwen4Status qw4c_backings(Qwen4StateKey key, Qwen4Memory m, Qwen4Memory p,
                                        Qwen4StateSize size, Qwen4Error *e) {
    if (!key.model || !key.epoch) return qw4c_error(e, QW4_INVALID_ARGUMENT, "state requires model and nonzero epoch");
    QW4C_TRY(qw4c_backing(m, size.metadata_bytes, e)); QW4C_TRY(qw4c_backing(p, size.payload_bytes, e));
    if (qw4c_overlap(m.data, m.bytes, p.data, p.bytes)) return qw4c_error(e, QW4_INVALID_ARGUMENT, "state backings overlap");
    return QW4_OK;
}
static inline Qwen4Status qw4c_state_check(const QW4CState *s, uint64_t magic, bool require_valid, Qwen4Error *e) {
    if (!s || s->self != s || s->magic != magic) return qw4c_error(e, QW4_BAD_STATE, "invalid or copied handle");
    if (s->info.busy) return qw4c_error(e, QW4_BUSY, "state in use");
    if (require_valid && !s->info.valid) return qw4c_error(e, QW4_BAD_STATE, "state requires reset");
    return QW4_OK;
}
static inline Qwen4Status qw4c_state_alias(const QW4CState *s, Qwen4F32 v, Qwen4Error *e) {
    size_t bytes = qw4c_span(v);
    if (qw4c_overlap(v.data, bytes, s->metadata.data, s->metadata.bytes) ||
        qw4c_overlap(v.data, bytes, s->payload.data, s->payload.bytes))
        return qw4c_error(e, QW4_INVALID_ARGUMENT, "view aliases state backing");
    return QW4_OK;
}
static inline Qwen4Status qw4c_state_reset(QW4CState *s, uint64_t magic, uint64_t epoch, size_t bytes, Qwen4Error *e) {
    QW4C_TRY(qw4c_state_check(s, magic, false, e));
    if (s->info.key.epoch == UINT64_MAX) return qw4c_error(e, QW4_OVERFLOW, "epoch exhausted");
    if (epoch <= s->info.key.epoch) return qw4c_error(e, QW4_STALE, "reset requires a newer epoch");
    if (bytes) memset(s->payload.data, 0, bytes);
    s->info.key.epoch=epoch; s->info.next_pos=0; s->info.valid=true;
    return qw4c_error(e, QW4_OK, NULL);
}
/* Private layer composition. The text layer resolves roles; the packed layer owns every decode and product.
 * Catalog/config, packed spans and all backings remain borrowed by the caller. */
typedef enum { QW4CW_HN, QW4CW_HD, QW4CW_HU, QW4CW_HI, QW4CW_QKV, QW4CW_Z,
    QW4CW_AP, QW4CW_BP, QW4CW_A, QW4CW_DT, QW4CW_CONV, QW4CW_GN, QW4CW_OUT, QW4CW_COUNT } QW4CWeight;
extern const Qwen4Role qw4c_layer_roles[QW4CW_COUNT];
typedef struct {
    Qwen4ConfigView config;
    Qwen4ConvShape conv;
    const void *model;
    uint32_t layer;
    bool bound;
    size_t constant_bytes;
    Qwen4Tensor tensor[QW4CW_COUNT];
    QW4PackedCpuView weight[QW4CW_COUNT];
    Qwen4F32 constant[QW4CW_COUNT]; /* only HN/A/DT/CONV/GN, decoded by B */
    Qwen4Memory constant_memory;
} QW4CLayer;
Qwen4Status qw4c_layer_plan(const Qwen4Config *, const Qwen4Weights *, uint32_t,
    QW4CLayer *, Qwen4Error *);
Qwen4Status qw4c_layer_bind(QW4CLayer *, const Qwen4CpuSpan[QW4CW_COUNT], Qwen4Memory, Qwen4Error *);
typedef enum { QW4CT_N, QW4CT_LOW, QW4CT_ACT, QW4CT_HG, QW4CT_IL, QW4CT_INJECT,
    QW4CT_MIX, QW4CT_QKV, QW4CT_CONV, QW4CT_PREP, QW4CT_Z, QW4CT_ALPHA, QW4CT_BR,
    QW4CT_LOGG, QW4CT_BETA, QW4CT_SCAN, QW4CT_NORM, QW4CT_OUT, QW4CT_COUNT } QW4CStage;
extern const char *const qw4c_stage_names[QW4CT_COUNT];
typedef struct {
    const QW4CLayer *owner;
    size_t capacity_tokens;
    Qwen4Memory memory;
    Qwen4F32Mut trace[QW4CT_COUNT]; /* flat [T,features], token pitch retained */
} QW4CWork;
Qwen4Status qw4c_work_size(const QW4CLayer *, size_t tokens, size_t *, Qwen4Error *);
Qwen4Status qw4c_work_init(const QW4CLayer *, size_t tokens, Qwen4Memory, QW4CWork *, Qwen4Error *);
Qwen4Status qw4c_work_check(const QW4CLayer *, const QW4CWork *, Qwen4F32 residual, Qwen4Error *);
Qwen4Status qw4c_hc_cpu(const QW4CLayer *, QW4CWork *, Qwen4F32 residual, Qwen4Error *);
static inline Qwen4F32Mut qw4c_trace(const QW4CWork *w, QW4CStage stage, size_t tokens) {
    Qwen4F32Mut v=w->trace[stage]; v.dim[0]=tokens; return v;
}
static inline Qwen4F32Mut qw4c_group(Qwen4F32Mut flat, size_t groups, size_t width) {
    flat.rank=3; flat.dim[1]=groups; flat.dim[2]=width;
    flat.stride_bytes[1]=width*4; flat.stride_bytes[2]=4; return flat;
}
typedef struct {
    const QW4CLayer *owner;
    Qwen4ConvState *conv;
    Qwen4GdnState *gdn;
    Qwen4StateKey key;
    bool valid, busy;
} QW4CBlockState;
Qwen4Status qw4c_block_init(const QW4CLayer *, Qwen4ConvState *, Qwen4GdnState *, QW4CBlockState *, Qwen4Error *);
Qwen4Status qw4c_block_reset(QW4CBlockState *, uint64_t epoch, Qwen4Error *);
Qwen4Status qw4c_block_cpu(const QW4CLayer *, QW4CWork *, Qwen4F32Mut residual,
    QW4CBlockState *, uint64_t start, Qwen4Error *);

/* Private harness interface only; session lifecycle remains in the CPU API.
 * Caller owns command completion, error checking and all resource lifetimes. */
typedef struct QW4CGpu QW4CGpu;
/* Slots: ACT/HC_LOW/HC_INJECT x,out; RMS x,gain,out;
 * HC_READ normed,logits,mixed; HC_ADD residual,block,inject;
 * CONV x,taps,out,history[L,C] (raw oldest-first, mutable after success);
 * GDN_PREP qkv,a,b,A,dt,prepared,log_decay,beta; GDN_NORM o,gain,out,z;
 * GDN_SCAN prepared,log_decay,beta,S[Hv,Dv,Dk],out[T,Hv,Dv].
 * Raw GPU state has NO private epoch/position API: the harness owns it and
 * discards/resets it on any command error. Public lifecycle glue is pending. */
typedef enum { QW4C_ACT, QW4C_RMS, QW4C_HC_LOW, QW4C_HC_INJECT,
    QW4C_HC_READ, QW4C_HC_ADD, QW4C_CONV, QW4C_GDN_PREP, QW4C_GDN_NORM,
    QW4C_GDN_SCAN, QW4C_GPU_OPS } QW4CGpuOp;
typedef struct { Qwen4Activation activation; float eps; Qwen4HcShape hc;
    Qwen4ConvShape conv; Qwen4GdnShape gdn; } QW4CGpuSpec;
Qwen4Status qw4c_gpu_create(void *device, void *library, QW4CGpu **, Qwen4Error *);
void qw4c_gpu_destroy(QW4CGpu *);
Qwen4Status qw4c_gpu_encode(QW4CGpu *, Qwen4GpuCommand, QW4CGpuOp,
    const QW4CGpuSpec *, const Qwen4GpuF32 *, size_t count, Qwen4Error *);
/* Decode V: GDN_PREP (views pv[8], as its qw4c_gpu_encode) then GDN_SCAN (sv[5])
 * in one dispatch; the scan's PREP, LOGG and BETA views are neither read nor
 * written. QW4_UNSUPPORTED, nothing encoded: exact or fuse family, or a shape
 * outside the fast scan; the caller then encodes the two ops. */
Qwen4Status qw4c_gpu_encode_prep_scan(QW4CGpu *, Qwen4GpuCommand, const QW4CGpuSpec *,
    const Qwen4GpuF32 *pv, const Qwen4GpuF32 *sv, Qwen4Error *);
/* R1.6a: true = RMS and GDN_NORM run check + qw4c_rms (the oracle), false =
 * qw4c_rms_fast with the check folded in. qw4c_gpu_create sets it from
 * QW4_REF_KERNELS (family fuse); this override is for tests only. */
void qw4c_gpu_set_reference(QW4CGpu *, bool fuse);
/* C4 raw-buffer composition, deliberately private. On ANY encode failure,
 * discard the unsubmitted command. On completion error discard/reset both
 * histories and replay. No lifecycle/epoch/position ABI is introduced here. */
typedef struct {
    Qwen4GpuF32 residual, history, state;
    Qwen4GpuF32 trace[QW4CT_COUNT], constant[QW4CW_COUNT];
    /* R1.8 profile only (NULL otherwise: no extra dispatch, barrier or split):
     * called before each dispatch group of the block with the command it must
     * continue on (the hook may replace cmd->compute_encoder). */
    void (*mark)(void *ctx, Qwen4GpuCommand *cmd, const char *op, uint32_t type, uint64_t bytes);
    void *mark_ctx;
    /* R4 MTP verify only (NULL otherwise): the scan runs one row at a time and
     * snap is called after each of rows 0..snap_rows-1, with the state then
     * holding that row's result, so the caller can copy it. Same FP32 steps as
     * the one-dispatch scan: a row reads back the exact state the last wrote. */
    void (*snap)(void *ctx, Qwen4GpuCommand *cmd, size_t row);
    void *snap_ctx; size_t snap_rows;
} QW4CGpuWork;
Qwen4Status qw4c_block_encode(QW4CGpu *, QW4PackedGpu *, Qwen4GpuCommand,
    const QW4CLayer *, const GpuWeightStore *, const GgufBinding *const[QW4CW_COUNT],
    const QW4CGpuWork *, Qwen4Error *);
typedef uint32_t QW4CUint;
#else
typedef uint QW4CUint;
#endif
/* Fixed-width wire format, strides in FLOAT ELEMENTS after host admission.
 * Slot numbers and operation parameters are private to these kernels/glue. */
typedef struct { QW4CUint dim[4], stride[4], rank, count; } QW4CGpuView;
typedef struct { QW4CGpuView v[8]; QW4CUint p[6], read_mask; float eps; } QW4CGpuArgs;
/* qw4c_rms_fast: one threadgroup of this many threads per row (32-lane simdgroups). */
#define QW4C_RMS_THREADS 256u
#ifndef __METAL_VERSION__
_Static_assert(sizeof(QW4CGpuView)==40 && sizeof(QW4CGpuArgs)==352, "Metal argument ABI");
_Static_assert(offsetof(QW4CGpuArgs,p)==320 && offsetof(QW4CGpuArgs,eps)==348, "Metal offsets");
#endif
#endif
