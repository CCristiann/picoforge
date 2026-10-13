/* Bounded GGUF v3 decoding. This interface does not imply model inference. */
#ifndef PICOFORGE_GGUF_H
#define PICOFORGE_GGUF_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const unsigned char *data;
    size_t size, pos;
} GgufCursor;

size_t gguf_add(size_t a, size_t b);
size_t gguf_mul(size_t a, size_t b);
const unsigned char *gguf_bytes(GgufCursor *c, size_t n);
uint64_t gguf_uint(GgufCursor *c, size_t width);
const unsigned char *gguf_string(GgufCursor *c, size_t *length);
void gguf_name(GgufCursor *c, char *out, size_t capacity);
void gguf_skip_value(GgufCursor *c, uint32_t type);

typedef struct { uint32_t id; const char *name; size_t block, bytes; } GgufFormat;
typedef struct {
    char name[128];
    uint32_t type, rank;
    size_t shape[4], file_offset, bytes;
} GgufTensor;
typedef struct { char name[128]; uint32_t type; size_t offset, bytes; } GgufMetadata;
typedef struct {
    int fd;
    void *header;                    /* owns only a bounded, CPU-only mapping */
    size_t header_size, file_size, data_offset, tensor_count, metadata_count;
    GgufTensor *tensors;              /* owned, sorted by file offset */
    GgufMetadata *metadata;           /* owned; values borrow header bytes */
} Gguf;

const GgufFormat *gguf_format(uint32_t id);
void gguf_open(const char *path, Gguf *g);
void gguf_close(Gguf *g);
void gguf_inspect(const Gguf *g);

typedef struct { size_t file_offset, bytes; } GgufView;
typedef struct GgufPlan GgufPlan;
typedef struct {
    GgufTensor tensor;               /* copy: physical shape and packed format */
    size_t buffer_id, offset;        /* one tensor always fits in one view */
    const GgufPlan *owner;           /* borrowed; valid until the plan is freed */
} GgufBinding;
struct GgufPlan {
    const Gguf *source;              /* identity witness only; never dereferenced */
    GgufView *views;
    GgufBinding *weights;            /* plan owns descriptors; no header borrows */
    size_t view_count, weight_count, page_size, file_size;
    GgufTensor disk_table;           /* bytes == 0 when absent; never GPU-bound */
};
void gguf_plan(const Gguf *g, size_t page_size, size_t buffer_limit, GgufPlan *plan);
void gguf_plan_free(GgufPlan *plan);

typedef struct GpuWeightStore GpuWeightStore;
/* device is the borrowed Metal device from metal_device(). Zero cap means the
 * device limit. Command buffers must retain resources through completion. */
GpuWeightStore *gpu_weights_open(void *device, const Gguf *g, size_t buffer_cap);
const GgufPlan *gpu_weights_plan(const GpuWeightStore *store);
const GgufBinding *gpu_weights_find(const GpuWeightStore *store, const char *name);
void *gpu_weights_buffer(const GpuWeightStore *store, const GgufBinding *binding);
void gpu_weights_close(GpuWeightStore *store);
size_t gpu_weights_live_mapped_bytes(void); /* virtual map spans, NOT physical RAM */
#endif
