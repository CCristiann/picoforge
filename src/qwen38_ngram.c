/* E CPU hash and bounded reader. No model or GPU loader. */
#define _POSIX_C_SOURCE 200809L
#include "qwen38_ngram_internal.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
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


Qwen38Status q38e_error(Qwen38Error *e, Qwen38Status code, const char *detail) {
    if (e) {
        *e=(Qwen38Error){.code=code};
        snprintf(e->subject,sizeof e->subject,"ngram lifecycle");
        snprintf(e->detail,sizeof e->detail,"%s",detail ? detail : "");
    }
    return code;
}
Qwen38Status q38e_allocate(Qwen38NgramStore *s, size_t bytes, void **out, Qwen38Error *e) {
    *out=NULL;
    if (bytes > 256u*1024u*1024u-s->used) return q38e_error(e,Q38_CAPACITY,"E aggregate budget");
    if (!bytes) return Q38_OK;
    *out=s->allocator.allocate(s->allocator.context,bytes);
    if (!*out) return q38e_error(e,Q38_OOM,"allocation");
    memset(*out,0,bytes); s->used+=bytes;
    if (s->used>s->peak) s->peak=s->used;
    return Q38_OK;
}
void q38e_free(Qwen38NgramStore *s, void *p, size_t bytes) {
    if (p) { s->allocator.release(s->allocator.context,p); s->used-=bytes; }
}
static Qwen38Status conv_new(Qwen38NgramStore *s, Qwen38StateKey key, Q38eConv *c, Qwen38Error *e) {
    Qwen38StateSize z=s->conv_size;
    c->bytes=z.metadata_bytes+z.payload_bytes+z.alignment-1;
    Qwen38Status code=q38e_allocate(s,c->bytes,&c->allocation,e);
    if (code) return code;
    uintptr_t aligned=((uintptr_t)c->allocation+z.alignment-1)&~(uintptr_t)(z.alignment-1);
    return qwen38_conv_state_init(&s->conv,key,(Qwen38Memory){(void *)aligned,z.metadata_bytes},
        (Qwen38Memory){(void *)(aligned+z.metadata_bytes),z.payload_bytes},&c->handle,e);
}
static void conv_free(Qwen38NgramStore *s, Q38eConv *c) {
    q38e_free(s,c->allocation,c->bytes); *c=(Q38eConv){0};
}
Qwen38Status q38e_ngram_open_test(const Gguf *g, const Qwen38NgramConfig *c,
    size_t max_tokens, size_t budget, const Q38eAllocator *alloc,
    Qwen38NgramStore **out, Qwen38Error *e) {
    if (out) *out=NULL;
    if (!out || !g || !c || !g->tensors ||
        !c->multipliers || !c->head_offsets || !c->head_vocab_sizes)
        return q38e_error(e,Q38_INVALID_ARGUMENT,"store arguments");
    if (!max_tokens) return q38e_error(e,Q38_RANGE,"zero token capacity");
    if (!c->head_dim || !c->hidden_size || !c->streams || !c->conv_kernel || c->order<2 || !c->head_count)
        return q38e_error(e,Q38_SHAPE,"store dimensions");
    if (c->multiplier_count!=c->order || (uint64_t)(c->order-1)*c->heads_per_order!=c->head_count)
        return q38e_error(e,Q38_SHAPE,"hash array lengths");
    if (max_tokens>UINT32_MAX || c->head_count>UINT32_MAX/c->head_dim ||
        c->streams>UINT32_MAX/c->hidden_size || max_tokens>SIZE_MAX/c->head_count/8 ||
        c->head_count>(SIZE_MAX-c->order)/2 || (c->order+2*c->head_count)>SIZE_MAX/8)
        return q38e_error(e,Q38_OVERFLOW,"store geometry");
    if (!isfinite(c->rms_eps)) return q38e_error(e,Q38_NONFINITE,"RMS epsilon");
    if (c->rms_eps<=0) return q38e_error(e,Q38_RANGE,"RMS epsilon");
    const GgufTensor *table=NULL;
    for (size_t i=0;i<g->tensor_count;i++)
        if (!strncmp(g->tensors[i].name,"per_layer_token_embd.weight",sizeof g->tensors[i].name)) {
            if (table) return q38e_error(e,Q38_FORMAT,"duplicate disk table");
            table=g->tensors+i;
        }
    if (!table) return q38e_error(e,Q38_MISSING,"disk table");
    if (table->type!=30) return q38e_error(e,Q38_TYPE,"disk table must be BF16");
    if (c->row_count>SIZE_MAX/c->head_dim/2) return q38e_error(e,Q38_OVERFLOW,"disk table extent");
    if (table->rank!=2 || table->shape[0]!=c->head_dim || table->shape[1]!=c->row_count ||
        table->bytes!=c->row_count*c->head_dim*2)
        return q38e_error(e,Q38_SHAPE,"disk table shape");
    size_t arrays=(c->order+2*c->head_count)*8;
    if (budget>256u*1024u*1024u) budget=256u*1024u*1024u;
    if (sizeof(Qwen38NgramStore)>budget || arrays>budget-sizeof(Qwen38NgramStore))
        return q38e_error(e,Q38_CAPACITY,"store staging budget");
    Q38eAllocator a=alloc ? *alloc : (Q38eAllocator){NULL,default_allocate,default_release};
    if (!a.allocate || !a.release) return q38e_error(e,Q38_INVALID_ARGUMENT,"allocator");
    Qwen38NgramStore *s=a.allocate(a.context,sizeof *s);
    if (!s) return q38e_error(e,Q38_OOM,"store allocation");
    *s=(Qwen38NgramStore){.allocator=a,.config=*c,.max_tokens=max_tokens,.used=sizeof *s,
        .width=c->head_count*c->head_dim,.channels=(size_t)c->streams*c->hidden_size};
    s->conv=(Qwen38ConvShape){s->channels,c->conv_kernel,c->order,true};
    Qwen38Status code=qwen38_conv_state_bytes(&s->conv,&s->conv_size,e);
    if (!code && (s->conv_size.payload_bytes>256u*1024u*1024u ||
        s->conv_size.metadata_bytes>SIZE_MAX-s->conv_size.payload_bytes-s->conv_size.alignment))
        code=q38e_error(e,Q38_CAPACITY,"conv backing budget");
    if (!code) code=q38e_allocate(s,arrays,(void **)&s->arrays,e);
    if (!code) {
        memcpy(s->arrays,c->multipliers,c->order*8);
        memcpy(s->arrays+c->order,c->head_offsets,c->head_count*8);
        memcpy(s->arrays+c->order+c->head_count,c->head_vocab_sizes,c->head_count*8);
        s->config.multipliers=s->arrays; s->config.head_offsets=s->arrays+c->order;
        s->config.head_vocab_sizes=s->arrays+c->order+c->head_count;
        /* Validate hash metadata without allocating arbitrary history arrays. */
        code=qwen38_ngram_ids(&s->config,NULL,c->order-1,NULL,0,NULL,0,NULL,0,e);
        if (code==Q38_CAPACITY) code=Q38_OK; /* next capacity is intentionally zero */
    }
    if (!code) code=q38e_reader_open(g->fd,table->file_offset,c->row_count,c->head_dim,
        max_tokens*c->head_count,65536,budget-s->used,&a,&s->reader,e);
    if (code) { q38e_free(s,s->arrays,arrays); a.release(a.context,s); return code; }
    s->used+=q38e_reader_stats(s->reader).staging_bytes; s->peak=s->used;
    *out=s; return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_ngram_open(const Gguf *g, const Qwen38NgramConfig *c, size_t n,
    size_t budget, Qwen38NgramStore **out, Qwen38Error *e) {
    return q38e_ngram_open_test(g,c,n,budget,NULL,out,e);
}
Qwen38Status qwen38_ngram_close(Qwen38NgramStore *s, Qwen38Error *e) {
    if (!s) return q38e_error(e,Q38_INVALID_ARGUMENT,"null store");
    if (s->job || s->states || s->workspaces) return q38e_error(e,Q38_BUSY,"store referenced");
    q38e_reader_close(s->reader);
    s->allocator.release(s->allocator.context,s->arrays);
    s->allocator.release(s->allocator.context,s);
    return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_ngram_state_create(Qwen38NgramStore *s, Qwen38StateKey key,
    Qwen38NgramState **out, Qwen38Error *e) {
    if (out) *out=NULL;
    if (!s || !out || !key.model || !key.epoch) return q38e_error(e,Q38_INVALID_ARGUMENT,"state key");
    if (s->changed) return q38e_error(e,Q38_STALE,"store requires reopen");
    if (key.layer!=s->config.layer) return q38e_error(e,Q38_OWNER,"state layer");
    Qwen38NgramState *state=NULL;
    Qwen38Status code=q38e_allocate(s,sizeof *state,(void **)&state,e);
    if (code) return code;
    state->store=s; state->info=(Qwen38StateInfo){.key=key,.valid=true};
    size_t bytes=(s->config.order-1u)*sizeof(int32_t);
    code=q38e_allocate(s,bytes,(void **)&state->previous,e);
    if (!code) code=conv_new(s,key,&state->conv,e);
    if (code) {
        conv_free(s,&state->conv); q38e_free(s,state->previous,bytes); q38e_free(s,state,sizeof *state);
        return code;
    }
    for (size_t i=0;i<s->config.order-1u;i++) state->previous[i]=(int32_t)s->config.eos_id;
    s->states++; *out=state; return q38e_error(e,Q38_OK,NULL);
}
Qwen38StateInfo qwen38_ngram_state_info(const Qwen38NgramState *s) {
    Qwen38StateInfo info=s ? s->info : (Qwen38StateInfo){0};
    if (s && s->store->changed) info.valid=false;
    return info;
}
Qwen38Status qwen38_ngram_state_reset(Qwen38NgramState *s, uint64_t epoch, Qwen38Error *e) {
    if (!s) return q38e_error(e,Q38_INVALID_ARGUMENT,"null state");
    if (s->info.busy) return q38e_error(e,Q38_BUSY,"state referenced");
    if (s->store->changed) return q38e_error(e,Q38_STALE,"store requires reopen");
    Qwen38Status code=qwen38_conv_state_reset(s->conv.handle,epoch,e);
    if (code) return code;
    s->info.key.epoch=epoch; s->info.next_pos=0; s->info.valid=true;
    for (size_t i=0;i<s->store->config.order-1u;i++) s->previous[i]=(int32_t)s->store->config.eos_id;
    return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_ngram_state_destroy(Qwen38NgramState *state, Qwen38Error *e) {
    if (!state) return q38e_error(e,Q38_INVALID_ARGUMENT,"null state");
    if (state->info.busy) return q38e_error(e,Q38_BUSY,"state referenced");
    Qwen38NgramStore *s=state->store;
    conv_free(s,&state->conv); q38e_free(s,state->previous,(s->config.order-1u)*sizeof(int32_t));
    q38e_free(s,state,sizeof *state); s->states--;
    return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status q38e_job_check(const Qwen38NgramJob *j, Qwen38Error *e) {
    if (!j) return q38e_error(e,Q38_INVALID_ARGUMENT,"null job");
    if (j->store->changed) return q38e_error(e,Q38_STALE,"store requires reopen");
    if (j->store->job!=j || j->state->store!=j->store) return q38e_error(e,Q38_OWNER,"job owner");
    Qwen38StateInfo info=j->state->info;
    if (info.key.epoch!=j->key.epoch) return q38e_error(e,Q38_STALE,"job epoch");
    if (info.key.model!=j->key.model || info.key.layer!=j->key.layer)
        return q38e_error(e,Q38_OWNER,"job key");
    if (!info.valid) return q38e_error(e,Q38_BAD_STATE,"state requires reset");
    return Q38_OK;
}
static void job_free(Qwen38NgramJob *j) {
    Qwen38NgramStore *s=j->store;
    conv_free(s,&j->candidate);
    q38e_free(s,j->ids,j->count*s->config.head_count*8);
    q38e_free(s,j->next,(s->config.order-1u)*sizeof(int32_t));
    q38e_free(s,j,sizeof *j);
}
Qwen38Status qwen38_ngram_prepare(Qwen38NgramStore *s, Qwen38NgramState *state,
    const int32_t *tokens, size_t count, uint64_t position, Qwen38NgramJob **out, Qwen38Error *e) {
    if (out) *out=NULL;
    if (!s || !state || !out) return q38e_error(e,Q38_INVALID_ARGUMENT,"prepare arguments");
    if (s->changed) return q38e_error(e,Q38_STALE,"store requires reopen");
    if (state->store!=s) return q38e_error(e,Q38_OWNER,"foreign state");
    if (s->job || state->info.busy) return q38e_error(e,Q38_BUSY,"job pending release");
    if (!state->info.valid || position!=state->info.next_pos) return q38e_error(e,Q38_BAD_STATE,"append position");
    if (count>s->max_tokens) return q38e_error(e,Q38_CAPACITY,"token capacity");
    if (count>UINT64_MAX-position) return q38e_error(e,Q38_OVERFLOW,"append position overflow");
    Qwen38NgramJob *j=NULL;
    Qwen38Status code=q38e_allocate(s,sizeof *j,(void **)&j,e);
    if (code) return code;
    *j=(Qwen38NgramJob){.store=s,.state=state,.key=state->info.key,.position=position,
        .count=count,.phase=count ? Q38E_PREPARED : Q38E_COMPUTED};
    code=q38e_allocate(s,count*s->config.head_count*8,(void **)&j->ids,e);
    if (!code) code=q38e_allocate(s,(s->config.order-1u)*sizeof(int32_t),(void **)&j->next,e);
    if (!code) code=qwen38_ngram_ids(&s->config,state->previous,s->config.order-1u,tokens,count,
        j->ids,count*s->config.head_count,j->next,s->config.order-1u,e);
    if (!code && count && !state->gpu_session) code=conv_new(s,j->key,&j->candidate,e);
    if (!code && count && !state->gpu_session) code=qwen38_conv_state_copy(j->candidate.handle,state->conv.handle,e);
    if (code) { job_free(j); return code; }
    s->job=j; state->info.busy=true; *out=j; return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_ngram_read(Qwen38NgramJob *j, Qwen38Error *e) {
    Qwen38Status code=q38e_job_check(j,e); if (code) return code;
    if (!j->count && j->phase==Q38E_COMPUTED) return q38e_error(e,Q38_OK,NULL);
    if (j->phase!=Q38E_PREPARED) return q38e_error(e,Q38_BAD_STATE,"read requires PREPARED");
    code=q38e_reader_read(j->store->reader,j->ids,j->count*j->store->config.head_count,
        j->store->read_fn,j->store->read_context,&j->lease,e);
    j->phase=code ? Q38E_ABORTED : Q38E_READY;
    if (code==Q38_STALE) j->store->changed=true;
    return code;
}
Qwen38Status qwen38_ngram_rows(const Qwen38NgramJob *j, Qwen38F32 *out, Qwen38Error *e) {
    if (out) *out=(Qwen38F32){0};
    Qwen38Status code=q38e_job_check(j,e); if (code) return code;
    if (!out) return q38e_error(e,Q38_INVALID_ARGUMENT,"null rows output");
    if (j->phase!=Q38E_READY && j->phase!=Q38E_COMPUTED && j->phase!=Q38E_COMMITTED)
        return q38e_error(e,Q38_BAD_STATE,"rows unavailable");
    const Q38eReader *r=j->store->reader;
    if (j->count) {
        if (!r->ready || j->lease.owner!=r || j->lease.generation!=r->generation)
            return q38e_error(e,Q38_STALE,"reader lease");
        code=unchanged(r);
        if (code==Q38_STALE) j->store->changed=true;
        if (code) return reader_result(e,code,NULL,errno);
    }
    size_t width=j->store->width;
    *out=(Qwen38F32){.data=j->count ? r->values : NULL,.rank=2,.dim={j->count,width},
        .stride_bytes={width*4,4},.capacity_bytes=j->count*width*4};
    return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_ngram_commit(Qwen38NgramState *state, Qwen38NgramJob *j, Qwen38Error *e) {
    if (!state || !j) return q38e_error(e,Q38_INVALID_ARGUMENT,"commit arguments");
    if (j->state!=state) return q38e_error(e,Q38_OWNER,"commit state");
    Qwen38Status code=q38e_job_check(j,e); if (code) return code;
    if (j->phase!=Q38E_COMPUTED || state->info.next_pos!=j->position)
        return q38e_error(e,Q38_BAD_STATE,"commit requires computed contiguous job");
    if (j->count) {
        Qwen38F32 rows;
        code=qwen38_ngram_rows(j,&rows,e);
        if (code) { j->phase=Q38E_ABORTED; return code; }
        if (!state->gpu_session) {
            Qwen38StateInfo candidate=qwen38_conv_state_info(j->candidate.handle);
            if (!candidate.valid || candidate.next_pos!=j->position+j->count)
                return q38e_error(e,Q38_BAD_STATE,"candidate history incomplete");
            code=qwen38_conv_state_copy(state->conv.handle,j->candidate.handle,e);
            if (code) { state->info.valid=false; j->phase=Q38E_ABORTED; return code; }
        }
        memcpy(state->previous,j->next,(j->store->config.order-1u)*sizeof(int32_t));
        state->info.next_pos+=j->count;
    }
    j->phase=Q38E_COMMITTED; return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_ngram_abort(Qwen38NgramJob *j, Qwen38Error *e) {
    Qwen38Status code=q38e_job_check(j,e); if (code) return code;
    if (j->phase==Q38E_COMMITTED) return q38e_error(e,Q38_BAD_STATE,"already committed");
    j->phase=Q38E_ABORTED; j->lease=(Q38eLease){0};
    return q38e_error(e,Q38_OK,NULL);
}
Qwen38Status qwen38_ngram_job_release(Qwen38NgramJob *j, Qwen38Error *e) {
    if (!j) return q38e_error(e,Q38_INVALID_ARGUMENT,"null job");
    if (j->store->job!=j) return q38e_error(e,Q38_OWNER,"foreign job");
    if (j->workspace) j->workspace->job=NULL;
    j->state->info.busy=false; j->store->job=NULL;
    job_free(j); return q38e_error(e,Q38_OK,NULL);
}
