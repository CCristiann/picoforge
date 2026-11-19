/* E CPU hash and bounded reader. No model or GPU loader. */
#define _POSIX_C_SOURCE 200809L
#include "qwen4_ngram_internal.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static Qwen4Status hash_result(Qwen4Error *e, Qwen4Status code) {
    if (e) {
        *e=(Qwen4Error){.code=code};
        snprintf(e->subject,sizeof e->subject,"ngram IDs");
    }
    return code;
}

/* Check ranges before pointer arithmetic; no overlap between any arrays. */
static bool overlaps(const void *a, size_t n, const void *b, size_t m) {
    uintptr_t x=(uintptr_t)a, y=(uintptr_t)b;
    return n && m && (x <= y ? y-x < n : x-y < m);
}

Qwen4Status qwen4_ngram_ids(const Qwen4NgramConfig *c, const int32_t *prev,
    size_t prev_count, const int32_t *tokens, size_t count, uint64_t *out,
    size_t capacity, int32_t *next, size_t next_capacity, Qwen4Error *e) {
    Qwen4Status code=QW4_OK;
    if (!c || !c->multipliers || !c->head_offsets || !c->head_vocab_sizes ||
        !c->vocab_size || c->vocab_size > (uint32_t)INT32_MAX+1u ||
        c->eos_id >= c->vocab_size || !c->row_count) code=QW4_INVALID_ARGUMENT;
    else if (c->order < 2 || !c->heads_per_order || c->multiplier_count != c->order ||
        (uint64_t)(c->order-1u)*c->heads_per_order != c->head_count) code=QW4_SHAPE;
    if (code != QW4_OK) return hash_result(e,code);
    if (c->head_count > SIZE_MAX/8 || c->multiplier_count > SIZE_MAX/8 ||
        count > SIZE_MAX/4 || count > SIZE_MAX/8/c->head_count)
        return hash_result(e,QW4_OVERFLOW);
    uint64_t end=0;
    for (size_t j=0; j<c->order; j++) {
        uint64_t m=c->multipliers[j];
        if (!m) return hash_result(e,QW4_RANGE);
        if (m > INT64_MAX || (c->vocab_size > 1 && m > (uint64_t)INT64_MAX/(c->vocab_size-1u)))
            return hash_result(e,QW4_OVERFLOW);
    }
    for (size_t h=0; h<c->head_count; h++) {
        uint64_t size=c->head_vocab_sizes[h];
        if (c->head_offsets[h] != end || !size) return hash_result(e,QW4_RANGE);
        if (size > (uint64_t)INT64_MAX-end) return hash_result(e,QW4_OVERFLOW);
        end+=size;
        if (end > c->row_count) return hash_result(e,QW4_RANGE);
    }
    const size_t history=c->order-1u, total=count*c->head_count;
    if (prev_count != history) return hash_result(e,QW4_SHAPE);
    if (capacity < total || next_capacity < history) return hash_result(e,QW4_CAPACITY);
    if (!prev || !next || (count && (!tokens || !out))) return hash_result(e,QW4_INVALID_ARGUMENT);
    const void *ptr[]={prev,tokens,out,next,c->multipliers,c->head_offsets,c->head_vocab_sizes,c};
    const size_t bytes[]={history*4,count*4,total*8,history*4,c->order*8ull,
        c->head_count*8,c->head_count*8,sizeof *c};
    for (size_t i=0; i<8; i++) {
        if (bytes[i] > UINTPTR_MAX-(uintptr_t)ptr[i]) return hash_result(e,QW4_OVERFLOW);
        /* Metadata arrays may share storage, but never writable outputs. */
        for (size_t j=0; j<i; j++)
            if ((i<4 && j<4) || i==2 || i==3 || j==2 || j==3)
                if (overlaps(ptr[i],bytes[i],ptr[j],bytes[j])) return hash_result(e,QW4_INVALID_ARGUMENT);
    }
    for (size_t j=0; j<history; j++)
        if (prev[j]<0 || (uint32_t)prev[j]>=c->vocab_size) return hash_result(e,QW4_RANGE);
    for (size_t t=0; t<count; t++)
        if (tokens[t]<0 || (uint32_t)tokens[t]>=c->vocab_size) return hash_result(e,QW4_RANGE);
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
    return hash_result(e,QW4_OK);
}

typedef struct { uint64_t row; size_t destination; } Request;
struct QW4eReader {
    int fd;
    uint64_t base, rows, generation;
    size_t width, row_bytes, capacity, raw_bytes, count;
    bool ready;
    struct stat identity;
    QW4eAllocator allocator;
    Request *requests;
    unsigned char *raw;
    float *values;
    QW4eReadStats stats;
};

static void *default_allocate(void *context, size_t n) { (void)context; return malloc(n); }
static void default_release(void *context, void *p) { (void)context; free(p); }

void qw4e_reader_close(QW4eReader *r) {
    if (!r) return;
    QW4eAllocator a = r->allocator;
    if (r->fd >= 0) (void)close(r->fd);
    if (r->requests) a.release(a.context,r->requests);
    if (r->raw) a.release(a.context,r->raw);
    if (r->values) a.release(a.context,r->values);
    a.release(a.context,r);
}

static Qwen4Status reader_open_impl(int fd, uint64_t base, uint64_t rows, size_t width,
    size_t capacity, size_t max_read, size_t budget, const QW4eAllocator *allocator,
    QW4eReader **out) {
    _Static_assert(sizeof(off_t) == 8 && sizeof(float) == 4, "requires 64-bit offsets/FP32");
    if (!out) return QW4_INVALID_ARGUMENT;
    *out = NULL;
    if (!width || !rows || !capacity || max_read > 65536 ||
        width > max_read/2 || base % 2) return QW4_INVALID_ARGUMENT;
    size_t row_bytes = width*2, raw_bytes = max_read/row_bytes*row_bytes;
    if (base > INT64_MAX || rows > ((uint64_t)INT64_MAX-base)/row_bytes)
        return QW4_OVERFLOW;
    if (capacity > SIZE_MAX/sizeof(Request) || capacity > SIZE_MAX/(width*sizeof(float)))
        return QW4_OVERFLOW;
    size_t plan_bytes = capacity*sizeof(Request), values_bytes = capacity*width*sizeof(float);
    size_t needed = sizeof(QW4eReader);
    const size_t parts[] = {plan_bytes, values_bytes, raw_bytes};
    for (size_t i=0; i<3; i++) {
        if (parts[i] > SIZE_MAX-needed) return QW4_OVERFLOW;
        needed += parts[i];
    }
    if (needed > budget) return QW4_CAPACITY;
    QW4eAllocator a = allocator ? *allocator : (QW4eAllocator){NULL,default_allocate,default_release};
    if (!a.allocate || !a.release) return QW4_INVALID_ARGUMENT;
    struct stat st;
    int flags = fcntl(fd,F_GETFL);
    if (flags < 0 || fstat(fd,&st) < 0) return QW4_IO;
    if ((flags & O_ACCMODE) != O_RDONLY || !S_ISREG(st.st_mode) || st.st_size < 0)
        return QW4_INVALID_ARGUMENT;
    if (base + rows*row_bytes > (uint64_t)st.st_size) return QW4_EOF;
    QW4eReader *r = a.allocate(a.context,sizeof *r);
    if (!r) return QW4_OOM;
    *r = (QW4eReader){.fd=-1,.base=base,.rows=rows,.width=width,.row_bytes=row_bytes,
        .capacity=capacity,.raw_bytes=raw_bytes,.identity=st,.allocator=a};
    r->stats.staging_bytes = needed;
    r->fd = fcntl(fd,F_DUPFD_CLOEXEC,0);
    if (r->fd < 0) { int saved=errno; qw4e_reader_close(r); errno=saved; return QW4_IO; }
    r->requests = a.allocate(a.context,plan_bytes);
    r->values = a.allocate(a.context,values_bytes);
    r->raw = a.allocate(a.context,raw_bytes);
    if (!r->requests || !r->values || !r->raw) { qw4e_reader_close(r); return QW4_OOM; }
    *out = r;
    return QW4_OK;
}

static Qwen4Status unchanged(const QW4eReader *r) {
    struct stat st;
    if (fstat(r->fd,&st) < 0) return QW4_IO;
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
    return same ? QW4_OK : QW4_STALE;
}

static int request_order(const void *a, const void *b) {
    const Request *x=a, *y=b;
    return (x->row > y->row) - (x->row < y->row);
}

static ssize_t file_read(void *context, int fd, void *dst, size_t n, off_t off) {
    (void)context;
    return pread(fd,dst,n,off);
}

static Qwen4Status read_exact(QW4eReader *r, uint64_t offset, size_t bytes,
    QW4ePread fn, void *context) {
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
            return QW4_IO;
        }
        if (!n) return QW4_EOF;
        if ((size_t)n > left) { r->stats.os_errno=EIO; return QW4_IO; }
        r->stats.completed_bytes+=(size_t)n;
        r->stats.bytes_read+=(size_t)n;
    }
    return QW4_OK;
}

static Qwen4Status reader_read_impl(QW4eReader *r, const uint64_t *rows, size_t count,
    QW4ePread fn, void *context, QW4eLease *lease) {
    if (lease) *lease=(QW4eLease){0};
    if (!r) return QW4_INVALID_ARGUMENT;
    /* Even rejected requests invalidate all results of the previous job. */
    r->ready=false; r->count=0;
    r->stats=(QW4eReadStats){.staging_bytes=r->stats.staging_bytes,.error_offset=r->base};
    if (r->generation == UINT64_MAX) return QW4_OVERFLOW;
    r->generation++;
    if (!lease || (count && !rows)) return QW4_INVALID_ARGUMENT;
    if (count > r->capacity) return QW4_CAPACITY;
    for (size_t i=0; i<count; i++) if (rows[i] >= r->rows) return QW4_RANGE;
    Qwen4Status status=count ? unchanged(r) : QW4_OK;
    if (status != QW4_OK) { if (status == QW4_IO) r->stats.os_errno=errno; return status; }
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
        if (status != QW4_OK) return status;
        for (size_t i=begin; i<end; i++) {
            const unsigned char *raw=r->raw+(size_t)(r->requests[i].row-first)*r->row_bytes;
            float *dst=r->values+r->requests[i].destination*r->width;
            for (size_t j=0; j<r->width; j++) {
                uint32_t bits=((uint32_t)raw[2*j] | (uint32_t)raw[2*j+1]<<8)<<16;
                if ((bits & 0x7f800000u) == 0x7f800000u) return QW4_NONFINITE;
                memcpy(dst+j,&bits,sizeof bits);
            }
        }
        begin=end;
    }
    status=count ? unchanged(r) : QW4_OK;
    if (status != QW4_OK) { if (status == QW4_IO) r->stats.os_errno=errno; return status; }
    r->count=count; r->ready=true;
    *lease=(QW4eLease){r,r->generation};
    return QW4_OK;
}

static Qwen4Status reader_copy_impl(const QW4eReader *r, QW4eLease lease,
    float *dst, size_t capacity) {
    if (!r) return QW4_INVALID_ARGUMENT;
    if (lease.owner && lease.owner != r) return QW4_OWNER;
    if (!lease.owner || !r->ready || lease.generation != r->generation) return QW4_STALE;
    size_t n=r->count*r->width;
    if (n && !dst) return QW4_INVALID_ARGUMENT;
    if (capacity < n) return QW4_CAPACITY;
    Qwen4Status status=n ? unchanged(r) : QW4_OK;
    if (status != QW4_OK) return status;
    if (n) memcpy(dst,r->values,n*sizeof(float));
    return QW4_OK;
}

QW4eReadStats qw4e_reader_stats(const QW4eReader *r) {
    return r ? r->stats : (QW4eReadStats){0};
}

static Qwen4Status reader_result(Qwen4Error *e, Qwen4Status code,
    const QW4eReader *r, int saved_errno) {
    if (e) {
        *e=(Qwen4Error){.code=code,.os_errno=code==QW4_IO ? saved_errno : 0};
        snprintf(e->subject,sizeof e->subject,"ngram BF16 reader");
        if (r && (code==QW4_IO || code==QW4_EOF)) {
            e->file_offset=r->stats.error_offset;
            e->row=(e->file_offset-r->base)/r->row_bytes;
            e->requested_bytes=r->stats.requested_bytes;
            e->completed_bytes=r->stats.completed_bytes;
        }
    }
    return code;
}

Qwen4Status qw4e_reader_open(int fd, uint64_t base, uint64_t rows, size_t width,
    size_t capacity, size_t max_read, size_t budget, const QW4eAllocator *allocator,
    QW4eReader **out, Qwen4Error *e) {
    Qwen4Status code=reader_open_impl(fd,base,rows,width,capacity,max_read,budget,allocator,out);
    return reader_result(e,code,NULL,errno);
}

Qwen4Status qw4e_reader_read(QW4eReader *r, const uint64_t *rows, size_t count,
    QW4ePread fn, void *context, QW4eLease *lease, Qwen4Error *e) {
    Qwen4Status code=reader_read_impl(r,rows,count,fn,context,lease);
    return reader_result(e,code,r,r ? r->stats.os_errno : 0);
}

Qwen4Status qw4e_reader_copy(const QW4eReader *r, QW4eLease lease,
    float *dst, size_t capacity, Qwen4Error *e) {
    Qwen4Status code=reader_copy_impl(r,lease,dst,capacity);
    return reader_result(e,code,NULL,errno);
}


Qwen4Status qw4e_error(Qwen4Error *e, Qwen4Status code, const char *detail) {
    if (e) {
        *e=(Qwen4Error){.code=code};
        snprintf(e->subject,sizeof e->subject,"ngram lifecycle");
        snprintf(e->detail,sizeof e->detail,"%s",detail ? detail : "");
    }
    return code;
}
Qwen4Status qw4e_allocate(Qwen4NgramStore *s, size_t bytes, void **out, Qwen4Error *e) {
    *out=NULL;
    if (bytes > 256u*1024u*1024u-s->used) return qw4e_error(e,QW4_CAPACITY,"E aggregate budget");
    if (!bytes) return QW4_OK;
    *out=s->allocator.allocate(s->allocator.context,bytes);
    if (!*out) return qw4e_error(e,QW4_OOM,"allocation");
    memset(*out,0,bytes); s->used+=bytes;
    if (s->used>s->peak) s->peak=s->used;
    return QW4_OK;
}
void qw4e_free(Qwen4NgramStore *s, void *p, size_t bytes) {
    if (p) { s->allocator.release(s->allocator.context,p); s->used-=bytes; }
}
static Qwen4Status conv_new(Qwen4NgramStore *s, Qwen4StateKey key, QW4eConv *c, Qwen4Error *e) {
    Qwen4StateSize z=s->conv_size;
    c->bytes=z.metadata_bytes+z.payload_bytes+z.alignment-1;
    Qwen4Status code=qw4e_allocate(s,c->bytes,&c->allocation,e);
    if (code) return code;
    uintptr_t aligned=((uintptr_t)c->allocation+z.alignment-1)&~(uintptr_t)(z.alignment-1);
    return qwen4_conv_state_init(&s->conv,key,(Qwen4Memory){(void *)aligned,z.metadata_bytes},
        (Qwen4Memory){(void *)(aligned+z.metadata_bytes),z.payload_bytes},&c->handle,e);
}
static void conv_free(Qwen4NgramStore *s, QW4eConv *c) {
    qw4e_free(s,c->allocation,c->bytes); *c=(QW4eConv){0};
}
Qwen4Status qw4e_ngram_open_test(const Gguf *g, const Qwen4NgramConfig *c,
    size_t max_tokens, size_t budget, const QW4eAllocator *alloc,
    Qwen4NgramStore **out, Qwen4Error *e) {
    if (out) *out=NULL;
    if (!out || !g || !c || !g->tensors ||
        !c->multipliers || !c->head_offsets || !c->head_vocab_sizes)
        return qw4e_error(e,QW4_INVALID_ARGUMENT,"store arguments");
    if (!max_tokens) return qw4e_error(e,QW4_RANGE,"zero token capacity");
    if (!c->head_dim || !c->hidden_size || !c->streams || !c->conv_kernel || c->order<2 || !c->head_count)
        return qw4e_error(e,QW4_SHAPE,"store dimensions");
    if (c->multiplier_count!=c->order || (uint64_t)(c->order-1)*c->heads_per_order!=c->head_count)
        return qw4e_error(e,QW4_SHAPE,"hash array lengths");
    if (max_tokens>UINT32_MAX || c->head_count>UINT32_MAX/c->head_dim ||
        c->streams>UINT32_MAX/c->hidden_size || max_tokens>SIZE_MAX/c->head_count/8 ||
        c->head_count>(SIZE_MAX-c->order)/2 || (c->order+2*c->head_count)>SIZE_MAX/8)
        return qw4e_error(e,QW4_OVERFLOW,"store geometry");
    if (!isfinite(c->rms_eps)) return qw4e_error(e,QW4_NONFINITE,"RMS epsilon");
    if (c->rms_eps<=0) return qw4e_error(e,QW4_RANGE,"RMS epsilon");
    const GgufTensor *table=NULL;
    for (size_t i=0;i<g->tensor_count;i++)
        if (!strncmp(g->tensors[i].name,"per_layer_token_embd.weight",sizeof g->tensors[i].name)) {
            if (table) return qw4e_error(e,QW4_FORMAT,"duplicate disk table");
            table=g->tensors+i;
        }
    if (!table) return qw4e_error(e,QW4_MISSING,"disk table");
    if (table->type!=30) return qw4e_error(e,QW4_TYPE,"disk table must be BF16");
    if (c->row_count>SIZE_MAX/c->head_dim/2) return qw4e_error(e,QW4_OVERFLOW,"disk table extent");
    if (table->rank!=2 || table->shape[0]!=c->head_dim || table->shape[1]!=c->row_count ||
        table->bytes!=c->row_count*c->head_dim*2)
        return qw4e_error(e,QW4_SHAPE,"disk table shape");
    size_t arrays=(c->order+2*c->head_count)*8;
    if (budget>256u*1024u*1024u) budget=256u*1024u*1024u;
    if (sizeof(Qwen4NgramStore)>budget || arrays>budget-sizeof(Qwen4NgramStore))
        return qw4e_error(e,QW4_CAPACITY,"store staging budget");
    QW4eAllocator a=alloc ? *alloc : (QW4eAllocator){NULL,default_allocate,default_release};
    if (!a.allocate || !a.release) return qw4e_error(e,QW4_INVALID_ARGUMENT,"allocator");
    Qwen4NgramStore *s=a.allocate(a.context,sizeof *s);
    if (!s) return qw4e_error(e,QW4_OOM,"store allocation");
    *s=(Qwen4NgramStore){.allocator=a,.config=*c,.max_tokens=max_tokens,.used=sizeof *s,
        .width=c->head_count*c->head_dim,.channels=(size_t)c->streams*c->hidden_size};
    s->conv=(Qwen4ConvShape){s->channels,c->conv_kernel,c->order,true};
    Qwen4Status code=qwen4_conv_state_bytes(&s->conv,&s->conv_size,e);
    if (!code && (s->conv_size.payload_bytes>256u*1024u*1024u ||
        s->conv_size.metadata_bytes>SIZE_MAX-s->conv_size.payload_bytes-s->conv_size.alignment))
        code=qw4e_error(e,QW4_CAPACITY,"conv backing budget");
    if (!code) code=qw4e_allocate(s,arrays,(void **)&s->arrays,e);
    if (!code) {
        memcpy(s->arrays,c->multipliers,c->order*8);
        memcpy(s->arrays+c->order,c->head_offsets,c->head_count*8);
        memcpy(s->arrays+c->order+c->head_count,c->head_vocab_sizes,c->head_count*8);
        s->config.multipliers=s->arrays; s->config.head_offsets=s->arrays+c->order;
        s->config.head_vocab_sizes=s->arrays+c->order+c->head_count;
        /* Validate hash metadata without allocating arbitrary history arrays. */
        code=qwen4_ngram_ids(&s->config,NULL,c->order-1,NULL,0,NULL,0,NULL,0,e);
        if (code==QW4_CAPACITY) code=QW4_OK; /* next capacity is intentionally zero */
    }
    if (!code) code=qw4e_reader_open(g->fd,table->file_offset,c->row_count,c->head_dim,
        max_tokens*c->head_count,65536,budget-s->used,&a,&s->reader,e);
    if (code) { qw4e_free(s,s->arrays,arrays); a.release(a.context,s); return code; }
    s->used+=qw4e_reader_stats(s->reader).staging_bytes; s->peak=s->used;
    *out=s; return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_ngram_open(const Gguf *g, const Qwen4NgramConfig *c, size_t n,
    size_t budget, Qwen4NgramStore **out, Qwen4Error *e) {
    return qw4e_ngram_open_test(g,c,n,budget,NULL,out,e);
}
Qwen4Status qwen4_ngram_close(Qwen4NgramStore *s, Qwen4Error *e) {
    if (!s) return qw4e_error(e,QW4_INVALID_ARGUMENT,"null store");
    if (s->job || s->states || s->workspaces) return qw4e_error(e,QW4_BUSY,"store referenced");
    qw4e_reader_close(s->reader);
    s->allocator.release(s->allocator.context,s->arrays);
    s->allocator.release(s->allocator.context,s);
    return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_ngram_state_create(Qwen4NgramStore *s, Qwen4StateKey key,
    Qwen4NgramState **out, Qwen4Error *e) {
    if (out) *out=NULL;
    if (!s || !out || !key.model || !key.epoch) return qw4e_error(e,QW4_INVALID_ARGUMENT,"state key");
    if (s->changed) return qw4e_error(e,QW4_STALE,"store requires reopen");
    if (key.layer!=s->config.layer) return qw4e_error(e,QW4_OWNER,"state layer");
    Qwen4NgramState *state=NULL;
    Qwen4Status code=qw4e_allocate(s,sizeof *state,(void **)&state,e);
    if (code) return code;
    state->store=s; state->info=(Qwen4StateInfo){.key=key,.valid=true};
    size_t bytes=(s->config.order-1u)*sizeof(int32_t);
    code=qw4e_allocate(s,bytes,(void **)&state->previous,e);
    if (!code) code=conv_new(s,key,&state->conv,e);
    if (code) {
        conv_free(s,&state->conv); qw4e_free(s,state->previous,bytes); qw4e_free(s,state,sizeof *state);
        return code;
    }
    for (size_t i=0;i<s->config.order-1u;i++) state->previous[i]=(int32_t)s->config.eos_id;
    s->states++; *out=state; return qw4e_error(e,QW4_OK,NULL);
}
Qwen4StateInfo qwen4_ngram_state_info(const Qwen4NgramState *s) {
    Qwen4StateInfo info=s ? s->info : (Qwen4StateInfo){0};
    if (s && s->store->changed) info.valid=false;
    return info;
}
Qwen4Status qwen4_ngram_state_reset(Qwen4NgramState *s, uint64_t epoch, Qwen4Error *e) {
    if (!s) return qw4e_error(e,QW4_INVALID_ARGUMENT,"null state");
    if (s->info.busy) return qw4e_error(e,QW4_BUSY,"state referenced");
    if (s->store->changed) return qw4e_error(e,QW4_STALE,"store requires reopen");
    Qwen4Status code=qwen4_conv_state_reset(s->conv.handle,epoch,e);
    if (code) return code;
    s->info.key.epoch=epoch; s->info.next_pos=0; s->info.valid=true;
    for (size_t i=0;i<s->store->config.order-1u;i++) s->previous[i]=(int32_t)s->store->config.eos_id;
    return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_ngram_state_destroy(Qwen4NgramState *state, Qwen4Error *e) {
    if (!state) return qw4e_error(e,QW4_INVALID_ARGUMENT,"null state");
    if (state->info.busy) return qw4e_error(e,QW4_BUSY,"state referenced");
    Qwen4NgramStore *s=state->store;
    conv_free(s,&state->conv); qw4e_free(s,state->previous,(s->config.order-1u)*sizeof(int32_t));
    qw4e_free(s,state,sizeof *state); s->states--;
    return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qw4e_job_check(const Qwen4NgramJob *j, Qwen4Error *e) {
    if (!j) return qw4e_error(e,QW4_INVALID_ARGUMENT,"null job");
    if (j->store->changed) return qw4e_error(e,QW4_STALE,"store requires reopen");
    if (j->store->job!=j || j->state->store!=j->store) return qw4e_error(e,QW4_OWNER,"job owner");
    Qwen4StateInfo info=j->state->info;
    if (info.key.epoch!=j->key.epoch) return qw4e_error(e,QW4_STALE,"job epoch");
    if (info.key.model!=j->key.model || info.key.layer!=j->key.layer)
        return qw4e_error(e,QW4_OWNER,"job key");
    if (!info.valid) return qw4e_error(e,QW4_BAD_STATE,"state requires reset");
    return QW4_OK;
}
static void job_free(Qwen4NgramJob *j) {
    Qwen4NgramStore *s=j->store;
    conv_free(s,&j->candidate);
    qw4e_free(s,j->ids,j->count*s->config.head_count*8);
    qw4e_free(s,j->next,(s->config.order-1u)*sizeof(int32_t));
    qw4e_free(s,j,sizeof *j);
}
Qwen4Status qwen4_ngram_prepare(Qwen4NgramStore *s, Qwen4NgramState *state,
    const int32_t *tokens, size_t count, uint64_t position, Qwen4NgramJob **out, Qwen4Error *e) {
    if (out) *out=NULL;
    if (!s || !state || !out) return qw4e_error(e,QW4_INVALID_ARGUMENT,"prepare arguments");
    if (s->changed) return qw4e_error(e,QW4_STALE,"store requires reopen");
    if (state->store!=s) return qw4e_error(e,QW4_OWNER,"foreign state");
    if (s->job || state->info.busy) return qw4e_error(e,QW4_BUSY,"job pending release");
    if (!state->info.valid || position!=state->info.next_pos) return qw4e_error(e,QW4_BAD_STATE,"append position");
    if (count>s->max_tokens) return qw4e_error(e,QW4_CAPACITY,"token capacity");
    if (count>UINT64_MAX-position) return qw4e_error(e,QW4_OVERFLOW,"append position overflow");
    Qwen4NgramJob *j=NULL;
    Qwen4Status code=qw4e_allocate(s,sizeof *j,(void **)&j,e);
    if (code) return code;
    *j=(Qwen4NgramJob){.store=s,.state=state,.key=state->info.key,.position=position,
        .count=count,.phase=count ? QW4E_PREPARED : QW4E_COMPUTED};
    code=qw4e_allocate(s,count*s->config.head_count*8,(void **)&j->ids,e);
    if (!code) code=qw4e_allocate(s,(s->config.order-1u)*sizeof(int32_t),(void **)&j->next,e);
    if (!code) code=qwen4_ngram_ids(&s->config,state->previous,s->config.order-1u,tokens,count,
        j->ids,count*s->config.head_count,j->next,s->config.order-1u,e);
    if (!code && count && !state->gpu_session) code=conv_new(s,j->key,&j->candidate,e);
    if (!code && count && !state->gpu_session) code=qwen4_conv_state_copy(j->candidate.handle,state->conv.handle,e);
    if (code) { job_free(j); return code; }
    s->job=j; state->info.busy=true; *out=j; return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_ngram_read(Qwen4NgramJob *j, Qwen4Error *e) {
    Qwen4Status code=qw4e_job_check(j,e); if (code) return code;
    if (!j->count && j->phase==QW4E_COMPUTED) return qw4e_error(e,QW4_OK,NULL);
    if (j->phase!=QW4E_PREPARED) return qw4e_error(e,QW4_BAD_STATE,"read requires PREPARED");
    code=qw4e_reader_read(j->store->reader,j->ids,j->count*j->store->config.head_count,
        j->store->read_fn,j->store->read_context,&j->lease,e);
    j->phase=code ? QW4E_ABORTED : QW4E_READY;
    if (code==QW4_STALE) j->store->changed=true;
    return code;
}
Qwen4Status qwen4_ngram_rows(const Qwen4NgramJob *j, Qwen4F32 *out, Qwen4Error *e) {
    if (out) *out=(Qwen4F32){0};
    Qwen4Status code=qw4e_job_check(j,e); if (code) return code;
    if (!out) return qw4e_error(e,QW4_INVALID_ARGUMENT,"null rows output");
    if (j->phase!=QW4E_READY && j->phase!=QW4E_COMPUTED && j->phase!=QW4E_COMMITTED)
        return qw4e_error(e,QW4_BAD_STATE,"rows unavailable");
    const QW4eReader *r=j->store->reader;
    if (j->count) {
        if (!r->ready || j->lease.owner!=r || j->lease.generation!=r->generation)
            return qw4e_error(e,QW4_STALE,"reader lease");
        code=unchanged(r);
        if (code==QW4_STALE) j->store->changed=true;
        if (code) return reader_result(e,code,NULL,errno);
    }
    size_t width=j->store->width;
    *out=(Qwen4F32){.data=j->count ? r->values : NULL,.rank=2,.dim={j->count,width},
        .stride_bytes={width*4,4},.capacity_bytes=j->count*width*4};
    return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_ngram_commit(Qwen4NgramState *state, Qwen4NgramJob *j, Qwen4Error *e) {
    if (!state || !j) return qw4e_error(e,QW4_INVALID_ARGUMENT,"commit arguments");
    if (j->state!=state) return qw4e_error(e,QW4_OWNER,"commit state");
    Qwen4Status code=qw4e_job_check(j,e); if (code) return code;
    if (j->phase!=QW4E_COMPUTED || state->info.next_pos!=j->position)
        return qw4e_error(e,QW4_BAD_STATE,"commit requires computed contiguous job");
    if (j->count) {
        Qwen4F32 rows;
        code=qwen4_ngram_rows(j,&rows,e);
        if (code) { j->phase=QW4E_ABORTED; return code; }
        if (!state->gpu_session) {
            Qwen4StateInfo candidate=qwen4_conv_state_info(j->candidate.handle);
            if (!candidate.valid || candidate.next_pos!=j->position+j->count)
                return qw4e_error(e,QW4_BAD_STATE,"candidate history incomplete");
            code=qwen4_conv_state_copy(state->conv.handle,j->candidate.handle,e);
            if (code) { state->info.valid=false; j->phase=QW4E_ABORTED; return code; }
        }
        memcpy(state->previous,j->next,(j->store->config.order-1u)*sizeof(int32_t));
        state->info.next_pos+=j->count;
    }
    j->phase=QW4E_COMMITTED; return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_ngram_abort(Qwen4NgramJob *j, Qwen4Error *e) {
    Qwen4Status code=qw4e_job_check(j,e); if (code) return code;
    if (j->phase==QW4E_COMMITTED) return qw4e_error(e,QW4_BAD_STATE,"already committed");
    j->phase=QW4E_ABORTED; j->lease=(QW4eLease){0};
    return qw4e_error(e,QW4_OK,NULL);
}
Qwen4Status qwen4_ngram_job_release(Qwen4NgramJob *j, Qwen4Error *e) {
    if (!j) return qw4e_error(e,QW4_INVALID_ARGUMENT,"null job");
    if (j->store->job!=j) return qw4e_error(e,QW4_OWNER,"foreign job");
    if (j->workspace) j->workspace->job=NULL;
    j->state->info.busy=false; j->store->job=NULL;
    job_free(j); return qw4e_error(e,QW4_OK,NULL);
}
