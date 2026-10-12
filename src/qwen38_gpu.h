/* C-facing Metal contract; no Objective-C types or implicit CPU pointers. */
#ifndef PICOFORGE_QWEN38_GPU_H
#define PICOFORGE_QWEN38_GPU_H
#include "qwen38_types.h"
typedef struct {
    void *buffer;                  /* borrowed MTLBuffer */
    size_t offset_bytes, bytes;     /* bytes is accessible length FROM offset */
} Qwen38GpuSpan;
typedef struct {
    Qwen38GpuSpan span;
    size_t rank, dim[QWEN38_MAX_RANK], stride_bytes[QWEN38_MAX_RANK];
} Qwen38GpuF32;
typedef struct {
    void *device, *compute_encoder; /* borrowed, same device as every resource */
    Qwen38GpuSpan error_word;       /* aligned uint32, zero before command */
} Qwen38GpuCommand;
enum {
    Q38_GPU_NONFINITE = 1u,
    Q38_GPU_BOUNDS = 2u,
    Q38_GPU_STATE = 4u
};
/* Host status means encoded, NOT completed. Kernel atomically ORs error bits.
 * Caller checks command-buffer error AND this word after completion. No
 * component commits/waits on the command buffer or resets this word itself. */
#endif
