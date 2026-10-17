/* R0.2 opt-in per-op GPU profile. This GPU samples timestamps only at encoder
 * (stage) boundaries, so profile mode gives every group its own compute
 * encoder. Only ordinary trunk appends are recorded, never MTP appends. */
#ifndef PICOFORGE_QWEN38_PROFILE_H
#define PICOFORGE_QWEN38_PROFILE_H
#include <stdio.h>
#include "qwen38_types.h"
typedef struct Q38Profile Q38Profile;
enum { Q38_PROFILE_NONE = 0x7fffffff, Q38_PROFILE_MIXED = 0x7ffffffe };  /* format sentinels */
/* device: borrowed id<MTLDevice>. Takes the new, empty csv file in all cases. */
Qwen38Status q38_profile_create(void *device, FILE *csv, Q38Profile **, Qwen38Error *);
/* Caller (CLI) labels the next trunk append: "prefill"/"decode", first position. */
void q38_profile_begin_append(Q38Profile *, const char *phase, uint64_t token);
/* Graph only: the next group's MTLComputePassDescriptor, owned by the profile
 * and valid until the next call; NULL when no sample slot can be provided. */
void *q38_profile_group(Q38Profile *, uint32_t layer, const char *op, uint32_t gguf_type,
    uint64_t weight_bytes);
/* Graph only, after completion: writes one row per group if ok, else drops them.
 * False if a timestamp is missing or the write fails. */
bool q38_profile_end_append(Q38Profile *, bool ok);
/* Flushes and closes the csv; false on a write error. NULL is accepted. */
bool q38_profile_close(Q38Profile *);
#endif
