/* E CPU hash and bounded reader. No model or GPU loader. */
#define _POSIX_C_SOURCE 200809L
#include "qwen38_ngram_internal.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static Qwen38Status hash_result(Qwen38Error *e, Qwen38Status code) {
    if (e) {
        *e=(Qwen38Error){.code=code};
        snprintf(e->subject,sizeof e->subject,"ngram IDs");
    }
    return code;
}

/* Check ranges before pointer arithmetic; no overlap between any arrays. */
static bool overlaps(const void *a, size_t n, const void *b, size_t m) {
    uintptr_t x=(uintptr_t)a, y=(uintptr_t)b;
    return n && m && (x <= y ? y-x < n : x-y < m);
}

Qwen38Status qwen38_ngram_ids(const Qwen38NgramConfig *c, const int32_t *prev,
    size_t prev_count, const int32_t *tokens, size_t count, uint64_t *out,
    size_t capacity, int32_t *next, size_t next_capacity, Qwen38Error *e) {
    Qwen38Status code=Q38_OK;
    if (!c || !c->multipliers || !c->head_offsets || !c->head_vocab_sizes ||
        !c->vocab_size || c->vocab_size > (uint32_t)INT32_MAX+1u ||
        c->eos_id >= c->vocab_size || !c->row_count) code=Q38_INVALID_ARGUMENT;
    else if (c->order < 2 || !c->heads_per_order || c->multiplier_count != c->order ||
        (uint64_t)(c->order-1u)*c->heads_per_order != c->head_count) code=Q38_SHAPE;
    if (code != Q38_OK) return hash_result(e,code);
    if (c->head_count > SIZE_MAX/8 || c->multiplier_count > SIZE_MAX/8 ||
        count > SIZE_MAX/4 || count > SIZE_MAX/8/c->head_count)
        return hash_result(e,Q38_OVERFLOW);
    uint64_t end=0;
    for (size_t j=0; j<c->order; j++) {
        uint64_t m=c->multipliers[j];
        if (!m) return hash_result(e,Q38_RANGE);
        if (m > INT64_MAX || (c->vocab_size > 1 && m > (uint64_t)INT64_MAX/(c->vocab_size-1u)))
            return hash_result(e,Q38_OVERFLOW);
    }
    for (size_t h=0; h<c->head_count; h++) {
        uint64_t size=c->head_vocab_sizes[h];
        if (c->head_offsets[h] != end || !size) return hash_result(e,Q38_RANGE);
        if (size > (uint64_t)INT64_MAX-end) return hash_result(e,Q38_OVERFLOW);
        end+=size;
        if (end > c->row_count) return hash_result(e,Q38_RANGE);
    }
    const size_t history=c->order-1u, total=count*c->head_count;
    if (prev_count != history) return hash_result(e,Q38_SHAPE);
    if (capacity < total || next_capacity < history) return hash_result(e,Q38_CAPACITY);
    if (!prev || !next || (count && (!tokens || !out))) return hash_result(e,Q38_INVALID_ARGUMENT);
    const void *ptr[]={prev,tokens,out,next,c->multipliers,c->head_offsets,c->head_vocab_sizes,c};
    const size_t bytes[]={history*4,count*4,total*8,history*4,c->order*8ull,
        c->head_count*8,c->head_count*8,sizeof *c};
    for (size_t i=0; i<8; i++) {
        if (bytes[i] > UINTPTR_MAX-(uintptr_t)ptr[i]) return hash_result(e,Q38_OVERFLOW);
        /* Metadata arrays may share storage, but never writable outputs. */
        for (size_t j=0; j<i; j++)
            if ((i<4 && j<4) || i==2 || i==3 || j==2 || j==3)
                if (overlaps(ptr[i],bytes[i],ptr[j],bytes[j])) return hash_result(e,Q38_INVALID_ARGUMENT);
    }
    for (size_t j=0; j<history; j++)
        if (prev[j]<0 || (uint32_t)prev[j]>=c->vocab_size) return hash_result(e,Q38_RANGE);
    for (size_t t=0; t<count; t++)
        if (tokens[t]<0 || (uint32_t)tokens[t]>=c->vocab_size) return hash_result(e,Q38_RANGE);
    /* Prefix addressing avoids allocation and keeps caller history immutable. */
    for (size_t t=0; t<count; t++) {
        uint64_t mixed=(uint64_t)tokens[t]*c->multipliers[0];
        bool cut=false;
        for (size_t j=1; j<c->order; j++) {
            uint32_t token=(uint32_t)(t>=j ? tokens[t-j] : prev[j-t-1]);
            cut=cut || token==c->eos_id;
            mixed^=(uint64_t)(cut ? c->eos_id : token)*c->multipliers[j];
            for (size_t g=0; g<c->heads_per_order; g++) {
                size_t h=(j-1)*c->heads_per_order+g;
                out[t*c->head_count+h]=c->head_offsets[h]+mixed%c->head_vocab_sizes[h];
            }
        }
    }
    for (size_t j=0; j<history; j++) next[j]=count>j ? tokens[count-j-1] : prev[j-count];
    return hash_result(e,Q38_OK);
}

typedef struct { uint64_t row; size_t destination; } Request;
struct Q38eReader {
    int fd;
    uint64_t base, rows, generation;
    size_t width, row_bytes, capacity, raw_bytes, count;
    bool ready;
    struct stat identity;
    Q38eAllocator allocator;
    Request *requests;
    unsigned char *raw;
    float *values;
    Q38eReadStats stats;
};

static void *default_allocate(void *context, size_t n) { (void)context; return malloc(n); }
static void default_release(void *context, void *p) { (void)context; free(p); }

void q38e_reader_close(Q38eReader *r) {
    if (!r) return;
    Q38eAllocator a = r->allocator;
    if (r->fd >= 0) (void)close(r->fd);
    if (r->requests) a.release(a.context,r->requests);
    if (r->raw) a.release(a.context,r->raw);
    if (r->values) a.release(a.context,r->values);
    a.release(a.context,r);
}

static Qwen38Status reader_open_impl(int fd, uint64_t base, uint64_t rows, size_t width,
    size_t capacity, size_t max_read, size_t budget, const Q38eAllocator *allocator,
    Q38eReader **out) {
    _Static_assert(sizeof(off_t) == 8 && sizeof(float) == 4, "requires 64-bit offsets/FP32");
    if (!out) return Q38_INVALID_ARGUMENT;
    *out = NULL;
    if (!width || !rows || !capacity || max_read > 65536 ||
        width > max_read/2 || base % 2) return Q38_INVALID_ARGUMENT;
    size_t row_bytes = width*2, raw_bytes = max_read/row_bytes*row_bytes;
    if (base > INT64_MAX || rows > ((uint64_t)INT64_MAX-base)/row_bytes)
        return Q38_OVERFLOW;
    if (capacity > SIZE_MAX/sizeof(Request) || capacity > SIZE_MAX/(width*sizeof(float)))
        return Q38_OVERFLOW;
    size_t plan_bytes = capacity*sizeof(Request), values_bytes = capacity*width*sizeof(float);
    size_t needed = sizeof(Q38eReader);
    const size_t parts[] = {plan_bytes, values_bytes, raw_bytes};
    for (size_t i=0; i<3; i++) {
        if (parts[i] > SIZE_MAX-needed) return Q38_OVERFLOW;
        needed += parts[i];
    }
    if (needed > budget) return Q38_CAPACITY;
    Q38eAllocator a = allocator ? *allocator : (Q38eAllocator){NULL,default_allocate,default_release};
    if (!a.allocate || !a.release) return Q38_INVALID_ARGUMENT;
    struct stat st;
    int flags = fcntl(fd,F_GETFL);
    if (flags < 0 || fstat(fd,&st) < 0) return Q38_IO;
    if ((flags & O_ACCMODE) != O_RDONLY || !S_ISREG(st.st_mode) || st.st_size < 0)
        return Q38_INVALID_ARGUMENT;
    if (base + rows*row_bytes > (uint64_t)st.st_size) return Q38_EOF;
    Q38eReader *r = a.allocate(a.context,sizeof *r);
    if (!r) return Q38_OOM;
    *r = (Q38eReader){.fd=-1,.base=base,.rows=rows,.width=width,.row_bytes=row_bytes,
        .capacity=capacity,.raw_bytes=raw_bytes,.identity=st,.allocator=a};
    r->stats.staging_bytes = needed;
    r->fd = fcntl(fd,F_DUPFD_CLOEXEC,0);
    if (r->fd < 0) { int saved=errno; q38e_reader_close(r); errno=saved; return Q38_IO; }
    r->requests = a.allocate(a.context,plan_bytes);
    r->values = a.allocate(a.context,values_bytes);
    r->raw = a.allocate(a.context,raw_bytes);
    if (!r->requests || !r->values || !r->raw) { q38e_reader_close(r); return Q38_OOM; }
    *out = r;
    return Q38_OK;
}

static Qwen38Status unchanged(const Q38eReader *r) {
    struct stat st;
    if (fstat(r->fd,&st) < 0) return Q38_IO;
    const struct stat *old = &r->identity;
    bool same = st.st_dev == old->st_dev && st.st_ino == old->st_ino && st.st_size == old->st_size;
#ifdef __APPLE__
    /* Darwin's strict POSIX profile exposes scalar seconds/nanoseconds. */
    same = same && st.st_mtime == old->st_mtime && st.st_mtimensec == old->st_mtimensec &&
        st.st_ctime == old->st_ctime && st.st_ctimensec == old->st_ctimensec;
#else
    same = same &&
        st.st_mtim.tv_sec == old->st_mtim.tv_sec && st.st_mtim.tv_nsec == old->st_mtim.tv_nsec &&
        st.st_ctim.tv_sec == old->st_ctim.tv_sec && st.st_ctim.tv_nsec == old->st_ctim.tv_nsec;
#endif
    return same ? Q38_OK : Q38_STALE;
}

static int request_order(const void *a, const void *b) {
    const Request *x=a, *y=b;
    return (x->row > y->row) - (x->row < y->row);
}

static ssize_t file_read(void *context, int fd, void *dst, size_t n, off_t off) {
    (void)context;
    return pread(fd,dst,n,off);
}

static Qwen38Status read_exact(Q38eReader *r, uint64_t offset, size_t bytes,
    Q38ePread fn, void *context) {
    r->stats.requested_bytes=bytes;
    r->stats.completed_bytes=0;
    while (r->stats.completed_bytes < bytes) {
        size_t done=r->stats.completed_bytes, left=bytes-done;
        r->stats.error_offset=offset+done;
        r->stats.calls++;
        ssize_t n=fn(context,r->fd,r->raw+done,left,(off_t)(offset+done));
        if (n < 0) {
            int saved=errno;
            if (saved == EINTR) continue;
            r->stats.os_errno=saved;
            return Q38_IO;
        }
        if (!n) return Q38_EOF;
        if ((size_t)n > left) { r->stats.os_errno=EIO; return Q38_IO; }
        r->stats.completed_bytes+=(size_t)n;
        r->stats.bytes_read+=(size_t)n;
    }
    return Q38_OK;
}

static Qwen38Status reader_read_impl(Q38eReader *r, const uint64_t *rows, size_t count,
    Q38ePread fn, void *context, Q38eLease *lease) {
    if (lease) *lease=(Q38eLease){0};
    if (!r) return Q38_INVALID_ARGUMENT;
    /* Even rejected requests invalidate all results of the previous job. */
    r->ready=false; r->count=0;
    r->stats=(Q38eReadStats){.staging_bytes=r->stats.staging_bytes,.error_offset=r->base};
    if (r->generation == UINT64_MAX) return Q38_OVERFLOW;
    r->generation++;
    if (!lease || (count && !rows)) return Q38_INVALID_ARGUMENT;
    if (count > r->capacity) return Q38_CAPACITY;
    for (size_t i=0; i<count; i++) if (rows[i] >= r->rows) return Q38_RANGE;
    Qwen38Status status=count ? unchanged(r) : Q38_OK;
    if (status != Q38_OK) { if (status == Q38_IO) r->stats.os_errno=errno; return status; }
    for (size_t i=0; i<count; i++) r->requests[i]=(Request){rows[i],i};
    qsort(r->requests,count,sizeof(Request),request_order);
    if (!fn) fn=file_read;
    for (size_t begin=0; begin<count;) {
        size_t end=begin+1;
        uint64_t first=r->requests[begin].row, last=first;
        /* Duplicates do not consume extra raw rows; gaps are never read. */
        while (end<count) {
            uint64_t next=r->requests[end].row;
            if (next > last+1 || next-first >= r->raw_bytes/r->row_bytes) break;
            last=next; end++;
        }
        size_t nrows=(size_t)(last-first+1);
        r->stats.unique_rows+=nrows;
        r->stats.intervals++;
        status=read_exact(r,r->base+first*r->row_bytes,nrows*r->row_bytes,fn,context);
        if (status != Q38_OK) return status;
        for (size_t i=begin; i<end; i++) {
            const unsigned char *raw=r->raw+(size_t)(r->requests[i].row-first)*r->row_bytes;
            float *dst=r->values+r->requests[i].destination*r->width;
            for (size_t j=0; j<r->width; j++) {
                uint32_t bits=((uint32_t)raw[2*j] | (uint32_t)raw[2*j+1]<<8)<<16;
                if ((bits & 0x7f800000u) == 0x7f800000u) return Q38_NONFINITE;
                memcpy(dst+j,&bits,sizeof bits);
            }
        }
        begin=end;
    }
    status=count ? unchanged(r) : Q38_OK;
    if (status != Q38_OK) { if (status == Q38_IO) r->stats.os_errno=errno; return status; }
    r->count=count; r->ready=true;
    *lease=(Q38eLease){r,r->generation};
    return Q38_OK;
}

static Qwen38Status reader_copy_impl(const Q38eReader *r, Q38eLease lease,
    float *dst, size_t capacity) {
    if (!r) return Q38_INVALID_ARGUMENT;
    if (lease.owner && lease.owner != r) return Q38_OWNER;
    if (!lease.owner || !r->ready || lease.generation != r->generation) return Q38_STALE;
    size_t n=r->count*r->width;
    if (n && !dst) return Q38_INVALID_ARGUMENT;
    if (capacity < n) return Q38_CAPACITY;
    Qwen38Status status=n ? unchanged(r) : Q38_OK;
    if (status != Q38_OK) return status;
    if (n) memcpy(dst,r->values,n*sizeof(float));
    return Q38_OK;
}

Q38eReadStats q38e_reader_stats(const Q38eReader *r) {
    return r ? r->stats : (Q38eReadStats){0};
}

static Qwen38Status reader_result(Qwen38Error *e, Qwen38Status code,
    const Q38eReader *r, int saved_errno) {
    if (e) {
        *e=(Qwen38Error){.code=code,.os_errno=code==Q38_IO ? saved_errno : 0};
        snprintf(e->subject,sizeof e->subject,"ngram BF16 reader");
        if (r && (code==Q38_IO || code==Q38_EOF)) {
            e->file_offset=r->stats.error_offset;
            e->row=(e->file_offset-r->base)/r->row_bytes;
            e->requested_bytes=r->stats.requested_bytes;
            e->completed_bytes=r->stats.completed_bytes;
        }
    }
    return code;
}

Qwen38Status q38e_reader_open(int fd, uint64_t base, uint64_t rows, size_t width,
    size_t capacity, size_t max_read, size_t budget, const Q38eAllocator *allocator,
    Q38eReader **out, Qwen38Error *e) {
    Qwen38Status code=reader_open_impl(fd,base,rows,width,capacity,max_read,budget,allocator,out);
    return reader_result(e,code,NULL,errno);
}

Qwen38Status q38e_reader_read(Q38eReader *r, const uint64_t *rows, size_t count,
    Q38ePread fn, void *context, Q38eLease *lease, Qwen38Error *e) {
    Qwen38Status code=reader_read_impl(r,rows,count,fn,context,lease);
    return reader_result(e,code,r,r ? r->stats.os_errno : 0);
}

Qwen38Status q38e_reader_copy(const Q38eReader *r, Q38eLease lease,
    float *dst, size_t capacity, Qwen38Error *e) {
    Qwen38Status code=reader_copy_impl(r,lease,dst,capacity);
    return reader_result(e,code,NULL,errno);
}

