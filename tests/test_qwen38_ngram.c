#define _POSIX_C_SOURCE 200809L
#include "qwen38_ngram_internal.h"
#include "qwen38.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint16_t bf16_value(size_t row, size_t col) {
    const uint16_t edge[]={0,0x8000,1,0x8001,0x007f,0x0080,0x3f80,0xbf80,0x7f7f};
    return col < sizeof edge/sizeof *edge ? edge[col] : (uint16_t)(0x3000+(row*17+col)%4096);
}

static int table_file(char *path, size_t rows, size_t width, size_t base) {
    int fd=mkstemp(path);
    assert(fd>=0);
    FILE *stream=fdopen(fd,"wb");
    assert(stream && fseek(stream,(long)base,SEEK_SET)==0);
    for (size_t r=0; r<rows; r++) for (size_t c=0; c<width; c++) {
        uint16_t b=bf16_value(r,c);
        unsigned char bytes[]={(unsigned char)b,(unsigned char)(b>>8)};
        assert(fwrite(bytes,1,2,stream)==2);
    }
    assert(fclose(stream)==0);
    fd=open(path,O_RDONLY);
    assert(fd>=0);
    return fd;
}

static void check_rows(Q38eReader *r, Q38eLease lease, const uint64_t *rows,
    size_t count, size_t width) {
    float *out=malloc(count*width*sizeof(float));
    assert(out && q38e_reader_copy(r,lease,out,count*width,NULL)==Q38_OK);
    for (size_t i=0; i<count; i++) for (size_t c=0; c<width; c++) {
        uint32_t bits;
        memcpy(&bits,out+i*width+c,4);
        assert(bits == (uint32_t)bf16_value((size_t)rows[i],c)<<16);
    }
    free(out);
}

static void reader_basic(void) {
    char path[]="build/qwen38-e/rows-XXXXXX";
    int fd=table_file(path,300,160,4094);
    Q38eReader *r=NULL;
    assert(q38e_reader_open(fd,4094,300,160,1024,65536,1<<20,NULL,&r,NULL)==Q38_OK);
    assert(close(fd)==0); /* Reader owns a duplicated fd, not this handle. */
    uint64_t rows[]={9,5,6,5,9,7,299};
    Q38eLease lease;
    assert(q38e_reader_read(r,rows,7,NULL,NULL,&lease,NULL)==Q38_OK);
    check_rows(r,lease,rows,7,160);
    Q38eReadStats stats=q38e_reader_stats(r);
    assert(stats.unique_rows==5 && stats.calls==3 && stats.intervals==3 && stats.bytes_read==1600);
    assert(stats.staging_bytes <= 1<<20);
    float sentinel=123.0f;
    assert(q38e_reader_copy(r,lease,&sentinel,1,NULL)==Q38_CAPACITY && sentinel==123.0f);
    Q38eLease old=lease;
    assert(q38e_reader_read(r,NULL,0,NULL,NULL,&lease,NULL)==Q38_OK);
    assert(q38e_reader_copy(r,lease,NULL,0,NULL)==Q38_OK);
    assert(q38e_reader_copy(r,old,&sentinel,1,NULL)==Q38_STALE);
    assert(q38e_reader_read(r,rows,SIZE_MAX,NULL,NULL,&lease,NULL)==Q38_CAPACITY);
    uint64_t invalid=300;
    assert(q38e_reader_read(r,&invalid,1,NULL,NULL,&lease,NULL)==Q38_RANGE);
    assert(q38e_reader_stats(r).calls==0);
    uint64_t consecutive[300];
    for (size_t i=0; i<300; i++) consecutive[i]=299-i;
    assert(q38e_reader_read(r,consecutive,300,NULL,NULL,&lease,NULL)==Q38_OK);
    check_rows(r,lease,consecutive,300,160);
    stats=q38e_reader_stats(r);
    assert(stats.calls==2 && stats.bytes_read==96000 && stats.unique_rows==300);
    q38e_reader_close(r);
    assert(unlink(path)==0);
}

typedef struct { size_t done, cap, fail_at, interrupt_at; int error; bool interrupted; } Fault;
static ssize_t fault_read(void *context,int fd,void *dst,size_t size,off_t offset) {
    Fault *f=context;
    if (!f->interrupted && f->done==f->interrupt_at) {
        f->interrupted=true; errno=EINTR; return -1;
    }
    if (f->done==f->fail_at) { errno=f->error; return f->error ? -1 : 0; }
    if (size > f->cap) size=f->cap;
    if (size > f->fail_at-f->done) size=f->fail_at-f->done;
    if (!f->interrupted && f->interrupt_at>f->done && size>f->interrupt_at-f->done)
        size=f->interrupt_at-f->done;
    ssize_t n=pread(fd,dst,size,offset);
    if (n>0) f->done+=(size_t)n;
    return n;
}

static void reader_faults(void) {
    char path[]="build/qwen38-e/fault-XXXXXX";
    int fd=table_file(path,8,4,0);
    Q38eReader *r=NULL;
    assert(q38e_reader_open(fd,0,8,4,8,16,4096,NULL,&r,NULL)==Q38_OK);
    uint64_t rows[]={2,1,2,4,5}; /* two intervals, 32 unique bytes */
    Q38eLease lease,old; Qwen38Error diagnostic;
    for (size_t point=0; point<32; point++) {
        Fault f={.cap=3,.fail_at=SIZE_MAX,.interrupt_at=point};
        assert(q38e_reader_read(r,rows,5,fault_read,&f,&lease,NULL)==Q38_OK);
        check_rows(r,lease,rows,5,4);
        assert(f.interrupted && f.done==32 && q38e_reader_stats(r).bytes_read==32);
        for (int kind=0; kind<3; kind++) {
            assert(q38e_reader_read(r,rows,5,NULL,NULL,&old,NULL)==Q38_OK);
            f=(Fault){.cap=3,.fail_at=point,.interrupt_at=0,.error=kind==1?EIO:kind==2?EBADF:0};
            assert(q38e_reader_read(r,rows,5,fault_read,&f,&lease,&diagnostic)==(kind?Q38_IO:Q38_EOF));
            float out[20]; for (size_t i=0;i<20;i++) out[i]=123.0f;
            assert(q38e_reader_copy(r,lease,out,20,NULL)==Q38_STALE);
            assert(q38e_reader_copy(r,old,out,20,NULL)==Q38_STALE);
            for (size_t i=0;i<20;i++) assert(out[i]==123.0f);
            assert(q38e_reader_stats(r).bytes_read==point);
            assert(q38e_reader_stats(r).os_errno==f.error);
            assert(diagnostic.code==(kind?Q38_IO:Q38_EOF));
            assert(diagnostic.os_errno==f.error && diagnostic.requested_bytes==16);
            assert(diagnostic.completed_bytes==point%16);
            assert(diagnostic.file_offset==(point<16 ? 8+point : 32+point-16));
            assert(diagnostic.row==diagnostic.file_offset/8);
        }
    }
    q38e_reader_close(r); close(fd); assert(unlink(path)==0);
}

typedef struct { size_t calls, fail, live, bytes; void *ptr[4]; size_t sizes[4]; } Allocation;
static void *test_allocate(void *context,size_t n) {
    Allocation *a=context;
    size_t i=a->calls++;
    assert(i<4);
    if (i==a->fail) return NULL;
    void *p=malloc(n);
    assert(p);
    a->ptr[i]=p; a->sizes[i]=n; a->live++; a->bytes+=n;
    return p;
}
static void test_release(void *context,void *p) {
    Allocation *a=context;
    for (size_t i=0;i<4;i++) if (a->ptr[i]==p) {
        a->ptr[i]=NULL; a->live--; free(p); return;
    }
    assert(0 && "unknown allocation");
}

static void reader_limits(void) {
    char path[]="build/qwen38-e/limits-XXXXXX";
    int fd=table_file(path,8,4,0);
    Q38eReader *r=NULL;
    assert(q38e_reader_open(fd,0,8,4,8,16,1,NULL,&r,NULL)==Q38_CAPACITY && !r);
    assert(q38e_reader_open(fd,0,8,4,SIZE_MAX,16,SIZE_MAX,NULL,&r,NULL)==Q38_OVERFLOW && !r);
    assert(q38e_reader_open(fd,0,UINT64_MAX,4,8,16,4096,NULL,&r,NULL)==Q38_OVERFLOW);
    assert(q38e_reader_open(fd,(uint64_t)INT64_MAX+1,8,4,8,16,4096,NULL,&r,NULL)==Q38_OVERFLOW);
    assert(q38e_reader_open(fd,64,1,4,8,16,4096,NULL,&r,NULL)==Q38_EOF);
    assert(q38e_reader_open(fd,0,0,4,8,16,4096,NULL,&r,NULL)==Q38_INVALID_ARGUMENT);
    assert(q38e_reader_open(fd,1,8,4,8,16,4096,NULL,&r,NULL)==Q38_INVALID_ARGUMENT);
    assert(q38e_reader_open(fd,0,8,SIZE_MAX,8,16,4096,NULL,&r,NULL)==Q38_INVALID_ARGUMENT);
    assert(q38e_reader_open(fd,0,8,4,8,7,4096,NULL,&r,NULL)==Q38_INVALID_ARGUMENT);
    assert(q38e_reader_open(fd,0,8,4,8,65537,4096,NULL,&r,NULL)==Q38_INVALID_ARGUMENT);
    assert(q38e_reader_open(-1,0,8,4,8,16,4096,NULL,&r,NULL)==Q38_IO);
    for (size_t fail=0;fail<5;fail++) {
        Allocation a={.fail=fail};
        Q38eAllocator allocator={&a,test_allocate,test_release};
        Qwen38Status st=q38e_reader_open(fd,0,8,4,8,16,4096,&allocator,&r,NULL);
        if (fail<4) { assert(st==Q38_OOM && !r && a.live==0); }
        else {
            assert(st==Q38_OK && a.live==4);
            size_t needed=q38e_reader_stats(r).staging_bytes;
            assert(needed==a.bytes && needed<=4096);
            uint64_t row=0; Q38eLease lease;
            for (int i=0;i<100;i++) assert(q38e_reader_read(r,&row,1,NULL,NULL,&lease,NULL)==Q38_OK);
            assert(a.calls==4); /* no application allocations during repeated reads */
            q38e_reader_close(r); assert(a.live==0);
            assert(q38e_reader_open(fd,0,8,4,8,16,needed-1,NULL,&r,NULL)==Q38_CAPACITY);
            assert(q38e_reader_open(fd,0,8,4,8,16,needed,NULL,&r,NULL)==Q38_OK);
            q38e_reader_close(r);
        }
    }
    int writable=open(path,O_RDWR);
    assert(writable>=0);
    assert(q38e_reader_open(writable,0,8,4,8,16,4096,NULL,&r,NULL)==Q38_INVALID_ARGUMENT);
    assert(q38e_reader_open(fd,0,8,4,8,16,4096,NULL,&r,NULL)==Q38_OK);
    Q38eLease lease; uint64_t row=7; float out[4]={123,123,123,123};
    assert(q38e_reader_read(r,&row,1,NULL,NULL,&lease,NULL)==Q38_OK);
    assert(ftruncate(writable,63)==0);
    assert(q38e_reader_copy(r,lease,out,4,NULL)==Q38_STALE && out[0]==123);
    assert(q38e_reader_read(r,&row,1,NULL,NULL,&lease,NULL)==Q38_STALE);
    assert(q38e_reader_copy(r,lease,out,4,NULL)==Q38_STALE);
    q38e_reader_close(r);
    close(writable); close(fd); assert(unlink(path)==0);
}

static ssize_t corrupt_read(void *context,int fd,void *dst,size_t n,off_t offset) {
    (void)context;
    ssize_t got=pread(fd,dst,n,offset);
    if (got>=2 && offset>=32) {
        unsigned char *b=dst;
        b[got-2]=0xc0; b[got-1]=0x7f; /* NaN only in the final interval */
    }
    return got;
}

typedef struct { int writable; bool truncated; } Truncate;
static ssize_t truncate_read(void *context,int fd,void *dst,size_t n,off_t offset) {
    Truncate *t=context;
    if (!t->truncated) { assert(ftruncate(t->writable,63)==0); t->truncated=true; }
    return pread(fd,dst,n,offset);
}

static void reader_stale(void) {
    char path[]="build/qwen38-e/stale-XXXXXX";
    int fd=table_file(path,8,4,0);
    Q38eReader *r=NULL,*other=NULL;
    assert(q38e_reader_open(fd,0,8,4,8,16,4096,NULL,&r,NULL)==Q38_OK);
    assert(q38e_reader_open(fd,0,8,4,8,16,4096,NULL,&other,NULL)==Q38_OK);
    uint64_t rows[]={0,1,6,7}; Q38eLease old,newer;
    float out[16]; for (size_t i=0;i<16;i++) out[i]=321.0f;
    assert(q38e_reader_read(r,rows,4,NULL,NULL,&old,NULL)==Q38_OK);
    assert(q38e_reader_copy(other,old,out,16,NULL)==Q38_OWNER);
    assert(q38e_reader_read(r,rows,4,corrupt_read,NULL,&newer,NULL)==Q38_NONFINITE);
    assert(q38e_reader_copy(r,old,out,16,NULL)==Q38_STALE);
    assert(q38e_reader_copy(r,newer,out,16,NULL)==Q38_STALE);
    for (size_t i=0;i<16;i++) assert(out[i]==321.0f);
    assert(q38e_reader_read(r,rows,4,NULL,NULL,&newer,NULL)==Q38_OK);
    check_rows(r,newer,rows,4,4);
    assert(q38e_reader_copy(r,old,out,16,NULL)==Q38_STALE);
    assert(q38e_reader_read(r,NULL,1,NULL,NULL,&newer,NULL)==Q38_INVALID_ARGUMENT);
    assert(q38e_reader_copy(r,newer,out,16,NULL)==Q38_STALE);
    int writable=open(path,O_RDWR); assert(writable>=0);
    Truncate t={.writable=writable};
    assert(q38e_reader_read(r,rows+3,1,truncate_read,&t,&newer,NULL)==Q38_EOF);
    assert(q38e_reader_stats(r).bytes_read==7);
    assert(q38e_reader_copy(r,newer,out,16,NULL)==Q38_STALE);
    q38e_reader_close(r); q38e_reader_close(other);
    close(fd); close(writable); assert(unlink(path)==0);
}

static int lookup_cli(const char *path) {
    uint64_t base,rows;
    size_t width,count,capacity,max_read,budget;
    assert(scanf("%"SCNu64" %"SCNu64" %zu %zu %zu %zu %zu",
        &base,&rows,&width,&count,&capacity,&max_read,&budget)==7);
    assert(count<=65536 && width<=1024);
    uint64_t *ids=malloc((count+1)*sizeof *ids);
    float *out=malloc((count*width+1)*sizeof *out);
    assert(ids && out);
    for (size_t i=0;i<count;i++) assert(scanf("%"SCNu64,ids+i)==1);
    int fd=open(path,O_RDONLY); assert(fd>=0);
    Q38eReader *r=NULL;
    Qwen38Status status=q38e_reader_open(fd,base,rows,width,capacity,max_read,budget,NULL,&r,NULL);
    close(fd);
    Q38eLease lease={0};
    if (status==Q38_OK) status=q38e_reader_read(r,ids,count,NULL,NULL,&lease,NULL);
    if (status==Q38_OK) status=q38e_reader_copy(r,lease,out,count*width,NULL);
    printf("status %d\n",(int)status);
    if (status==Q38_OK) for (size_t i=0;i<count*width;i++) {
        uint32_t bits; memcpy(&bits,out+i,4); printf("%08"PRIx32" ",bits);
    }
    putchar('\n');
    q38e_reader_close(r); free(ids); free(out);
    return 0;
}

static Qwen38NgramConfig small_config(void) {
    static const uint64_t mult[]={3,5,7}, offsets[]={0,11,24,41}, sizes[]={11,13,17,19};
    return (Qwen38NgramConfig){.order=3,.heads_per_order=2,.vocab_size=100,.eos_id=99,
        .row_count=64,.multipliers=mult,.head_offsets=offsets,.head_vocab_sizes=sizes,
        .multiplier_count=3,.head_count=4};
}

static void hash_errors(void) {
    Qwen38NgramConfig c=small_config(); Qwen38Error e;
    int32_t prev[]={99,99}, next[]={-7,-7}; uint64_t out[8];
    memset(out,0xaa,sizeof out);
    const int32_t tokens[]={1,-1};
    assert(qwen38_ngram_ids(&c,prev,2,tokens,2,out,8,next,2,&e)==Q38_RANGE);
    assert(e.code==Q38_RANGE && prev[0]==99 && next[0]==-7 && out[0]==UINT64_C(0xaaaaaaaaaaaaaaaa));
    assert(qwen38_ngram_ids(&c,prev,2,tokens,SIZE_MAX,out,8,next,2,&e)==Q38_OVERFLOW);
    assert(qwen38_ngram_ids(&c,prev,2,tokens,1,out,3,next,2,&e)==Q38_CAPACITY);
    assert(qwen38_ngram_ids(&c,prev,2,NULL,0,NULL,0,next,2,&e)==Q38_OK);
    assert(next[0]==99 && next[1]==99 && e.code==Q38_OK && !e.os_errno);
    assert(qwen38_ngram_ids(&c,prev,1,tokens,1,out,8,next,2,&e)==Q38_SHAPE);
    assert(qwen38_ngram_ids(&c,prev,2,tokens,1,out,8,next,1,&e)==Q38_CAPACITY);
    assert(qwen38_ngram_ids(&c,prev,2,tokens,1,out,8,prev,2,&e)==Q38_INVALID_ARGUMENT);
    uint64_t mult[]={UINT64_MAX,5,7}; c.multipliers=mult;
    assert(qwen38_ngram_ids(&c,prev,2,NULL,0,NULL,0,next,2,&e)==Q38_OVERFLOW);
    c=small_config(); c.head_count=3;
    assert(qwen38_ngram_ids(&c,prev,2,NULL,0,NULL,0,next,2,&e)==Q38_SHAPE);
    c=small_config(); c.row_count=59;
    assert(qwen38_ngram_ids(&c,prev,2,NULL,0,NULL,0,next,2,&e)==Q38_RANGE);
    assert(qwen38_ngram_ids(NULL,prev,2,NULL,0,NULL,0,next,2,&e)==Q38_INVALID_ARGUMENT);
    /* More than UINT32_MAX rows: contract arrays/outputs really are uint64. */
    uint64_t bigoff[]={0,UINT64_C(4294967311)}, bigsize[]={UINT64_C(4294967311),11};
    c=small_config(); c.heads_per_order=1; c.head_count=2;
    c.row_count=UINT64_C(4294967322); c.head_offsets=bigoff; c.head_vocab_sizes=bigsize;
    assert(qwen38_ngram_ids(&c,prev,2,tokens,1,out,8,next,2,&e)==Q38_OK);
    assert(out[1]>=bigoff[1]);
}

/* Tiny text protocol for Python differential tests, not engine API. */
static int hash_cli(void) {
    Qwen38NgramConfig c={0};
    uint64_t mult[16], offsets[960], sizes[960];
    size_t count,chunk;
    assert(scanf("%"SCNu32" %"SCNu32" %"SCNu32" %"SCNu32" %"SCNu64,
        &c.order,&c.heads_per_order,&c.vocab_size,&c.eos_id,&c.row_count)==5);
    assert(c.order>=2 && c.order<=16 && c.heads_per_order<=64);
    size_t heads=(size_t)(c.order-1u)*c.heads_per_order;
    c.multipliers=mult; c.head_offsets=offsets; c.head_vocab_sizes=sizes;
    c.multiplier_count=c.order; c.head_count=heads;
    for (uint32_t i=0;i<c.order;i++) assert(scanf("%"SCNu64,mult+i)==1);
    for (size_t i=0;i<heads;i++) assert(scanf("%"SCNu64,offsets+i)==1);
    for (size_t i=0;i<heads;i++) assert(scanf("%"SCNu64,sizes+i)==1);
    assert(scanf("%zu %zu",&count,&chunk)==2);
    assert(count && count<=32768 && chunk && chunk<=count);
    int32_t *tokens=malloc(count*sizeof *tokens), prev[16], next[16];
    uint64_t *out=malloc(count*heads*sizeof *out);
    assert(tokens && out);
    for (size_t i=0;i<count;i++) assert(scanf("%"SCNd32,tokens+i)==1);
    for (uint32_t j=0;j<c.order-1u;j++) prev[j]=(int32_t)c.eos_id;
    for (size_t off=0;off<count;off+=chunk) {
        size_t n=count-off<chunk ? count-off : chunk;
        Qwen38Status status=qwen38_ngram_ids(&c,prev,c.order-1u,tokens+off,n,
            out+off*heads,n*heads,next,c.order-1u,NULL);
        if (status!=Q38_OK) { free(tokens); free(out); return 2; }
        memcpy(prev,next,(c.order-1u)*sizeof *prev);
    }
    for (size_t i=0;i<count*heads;i++) printf("%"PRIu64" ",out[i]);
    putchar('\n');
    for (uint32_t j=0;j<c.order-1u;j++) printf("%"PRId32" ",prev[j]);
    putchar('\n');
    free(tokens); free(out);
    return 0;
}

static Qwen38F32Mut gate_view(float *data, size_t rank, size_t t, size_t s, size_t d, size_t pad) {
    size_t pitch=rank==3 ? s*(d+pad)+pad : s+pad;
    return (Qwen38F32Mut){.data=data,.capacity_bytes=t*pitch*4,.rank=rank,
        .dim={t,s,d,0},.stride_bytes={pitch*4,rank==3 ? (d+pad)*4 : 4,4,0}};
}
static Qwen38F32 gate_const(Qwen38F32Mut v) {
    Qwen38F32 r={.data=v.data,.rank=v.rank,.capacity_bytes=v.capacity_bytes};
    memcpy(r.dim,v.dim,sizeof r.dim); memcpy(r.stride_bytes,v.stride_bytes,sizeof r.stride_bytes);
    return r;
}

static void gate_errors(void) {
    float k[4]={1,2,3,4}, q[4]={1,2,3,4}, v[2]={-2,3}, g[2]={-7,-7}, u[4]={-7,-7,-7,-7};
    Qwen38F32 key=gate_const(gate_view(k,3,1,2,2,0)), query=gate_const(gate_view(q,3,1,2,2,0));
    Qwen38F32 value=gate_const(gate_view(v,2,1,2,1,0));
    Qwen38F32Mut gate=gate_view(g,2,1,2,1,0), output=gate_view(u,3,1,2,2,0);
    Qwen38Error e;
    Qwen38F32 bad=key; bad.rank=2;
    assert(qwen38_ple_gate(bad,query,value,gate,output,&e)==Q38_SHAPE);
    bad=key; bad.dim[1]=0;
    assert(qwen38_ple_gate(bad,bad,value,gate,output,&e)==Q38_SHAPE);
    bad=key; bad.stride_bytes[1]=4;
    assert(qwen38_ple_gate(bad,query,value,gate,output,&e)==Q38_STRIDE);
    bad=key; bad.stride_bytes[0]=17;
    assert(qwen38_ple_gate(bad,query,value,gate,output,&e)==Q38_STRIDE);
    bad=key; bad.capacity_bytes=15;
    assert(qwen38_ple_gate(bad,query,value,gate,output,&e)==Q38_CAPACITY);
    bad=key; bad.data=(const float *)((const unsigned char *)k+1);
    assert(qwen38_ple_gate(bad,query,value,gate,output,&e)==Q38_INVALID_ARGUMENT);
    bad=key; bad.data=NULL;
    assert(qwen38_ple_gate(bad,query,value,gate,output,&e)==Q38_INVALID_ARGUMENT);
    Qwen38F32Mut alias=output; alias.data=k+1;
    assert(qwen38_ple_gate(key,query,value,gate,alias,&e)==Q38_INVALID_ARGUMENT);
    alias=output; alias.data=g;
    assert(qwen38_ple_gate(key,query,value,gate,alias,&e)==Q38_INVALID_ARGUMENT);
    assert(qwen38_ple_gate(key,key,value,gate,output,&e)==Q38_INVALID_ARGUMENT);
    for (int i=0;i<3;i++) {
        float *p=i==0 ? k : i==1 ? q : v;
        float saved=p[0]; p[0]=i==1 ? INFINITY : NAN;
        assert(qwen38_ple_gate(key,query,value,gate,output,&e)==Q38_NONFINITE);
        assert(e.code==Q38_NONFINITE && g[0]==-7 && u[3]==-7); p[0]=saved;
    }
    k[0]=FLT_MAX; q[0]=2;
    assert(qwen38_ple_gate(key,query,value,gate,output,&e)==Q38_NONFINITE);
    k[0]=1; k[1]=1; q[0]=FLT_MAX; q[1]=FLT_MAX;
    assert(qwen38_ple_gate(key,query,value,gate,output,&e)==Q38_NONFINITE);
    k[1]=2; q[0]=1; q[1]=2;
    assert(qwen38_ple_gate(key,query,value,gate,output,&e)==Q38_OK);
    assert(e.code==Q38_OK && !e.os_errno && g[0]>0.5f && g[0]<=1.0f);
    /* Empty T still validates shape/stride; pointers are not dereferenced. */
    key.dim[0]=query.dim[0]=value.dim[0]=gate.dim[0]=output.dim[0]=0;
    key.data=query.data=value.data=NULL; gate.data=output.data=NULL;
    key.capacity_bytes=query.capacity_bytes=value.capacity_bytes=gate.capacity_bytes=output.capacity_bytes=0;
    assert(qwen38_ple_gate(key,query,value,gate,output,&e)==Q38_OK);
    key.stride_bytes[2]=8;
    assert(qwen38_ple_gate(key,query,value,gate,output,&e)==Q38_STRIDE);
}

static size_t gate_index(Qwen38F32Mut v, size_t t, size_t s, size_t d) {
    return (t*v.stride_bytes[0]+s*v.stride_bytes[1])/4+d;
}

static int gate_cli(void) {
    size_t t,s,d,pad;
    assert(scanf("%zu %zu %zu %zu",&t,&s,&d,&pad)==4);
    assert(t<=17 && s && s<=8 && d && d<=4096 && pad<=17);
    Qwen38F32Mut views[]={gate_view(NULL,3,t,s,d,pad),gate_view(NULL,3,t,s,d,pad),
        gate_view(NULL,2,t,d,1,pad),gate_view(NULL,2,t,s,1,pad),gate_view(NULL,3,t,s,d,pad)};
    for (size_t i=0;i<5;i++) {
        views[i].data=malloc(views[i].capacity_bytes+4); assert(views[i].data);
        for (size_t j=0;j<=views[i].capacity_bytes/4;j++) views[i].data[j]=987.25f;
    }
    for (size_t i=0;i<3;i++) for (size_t ti=0;ti<t;ti++)
        for (size_t si=0;si<(i==2 ? 1 : s);si++) for (size_t di=0;di<d;di++)
            assert(scanf("%f",views[i].data+gate_index(views[i],ti,si,di))==1);
    Qwen38Error e;
    Qwen38Status status=qwen38_ple_gate(gate_const(views[0]),gate_const(views[1]),
        gate_const(views[2]),views[3],views[4],&e);
    printf("status %d\n",(int)status);
    if (status==Q38_OK) {
        for (size_t ti=0;ti<t;ti++) for (size_t si=0;si<s;si++)
            printf("%.9g ",(double)q38e_ple_scaled_dot(views[0].data+gate_index(views[0],ti,si,0),
                views[1].data+gate_index(views[1],ti,si,0),d));
        putchar('\n');
        for (size_t ti=0;ti<t;ti++) for (size_t si=0;si<s;si++)
            printf("%.9g ",(double)views[3].data[gate_index(views[3],ti,si,0)]);
        putchar('\n');
        for (size_t ti=0;ti<t;ti++) for (size_t si=0;si<s;si++) for (size_t di=0;di<d;di++)
            printf("%.9g ",(double)views[4].data[gate_index(views[4],ti,si,di)]);
        putchar('\n');
    }
    for (size_t i=0;i<5;i++) {
        Qwen38F32Mut v=views[i];
        for (size_t j=0;j<=v.capacity_bytes/4;j++) {
            size_t offset=j*4, ti=offset/v.stride_bytes[0], rem=offset%v.stride_bytes[0];
            bool live=ti<t && (v.rank==2 ? rem/4<v.dim[1] :
                rem/v.stride_bytes[1]<s && rem%v.stride_bytes[1]/4<d);
            if (!live) assert(v.data[j]==987.25f);
        }
        free(v.data);
    }
    return 0;
}

typedef struct { size_t calls, fail, live; } Oom;
static void *oom_alloc(void *context, size_t n) {
    Oom *o=context;
    if (o->calls++==o->fail) return NULL;
    void *p=malloc(n); assert(p); o->live++; return p;
}
static void oom_free(void *context, void *p) {
    Oom *o=context; assert(o->live); o->live--; free(p);
}
static void lifecycle_oom(Gguf *g, Qwen38NgramConfig *c) {
    size_t failures=0;
    for (size_t fail=0;fail<20;fail++) {
        Oom o={.fail=fail}; Q38eAllocator a={&o,oom_alloc,oom_free};
        Qwen38NgramStore *s=NULL; Qwen38NgramState *state=NULL;
        Qwen38PleWorkspace *w=NULL; Qwen38NgramJob *j=NULL; Qwen38Error e;
        Qwen38Status code=q38e_ngram_open_test(g,c,4,1<<20,&a,&s,&e);
        if (!code) code=qwen38_ngram_state_create(s,(Qwen38StateKey){g,c->layer,1},&state,&e);
        if (!code) code=qwen38_ple_workspace_create(s,1<<20,&w,&e);
        int32_t token=1;
        if (!code) code=qwen38_ngram_prepare(s,state,&token,1,0,&j,&e);
        assert(code==Q38_OK || code==Q38_OOM); if (code) { failures++; assert(e.code==code); }
        if (j) assert(qwen38_ngram_job_release(j,&e)==Q38_OK);
        if (w) assert(qwen38_ple_workspace_destroy(w,&e)==Q38_OK);
        if (state) assert(qwen38_ngram_state_destroy(state,&e)==Q38_OK);
        if (s) assert(qwen38_ngram_close(s,&e)==Q38_OK);
        assert(o.live==0);
    }
    assert(failures==15);
}

static void lifecycle_tests(void) {
    char path[]="build/qwen38-e/lifecycle-XXXXXX";
    int fd=table_file(path,64,8,0);
    GgufTensor table={.name="per_layer_token_embd.weight",.type=30,.rank=2,.shape={8,64},.bytes=1024};
    Gguf gg={.fd=fd,.tensors=&table,.tensor_count=1};
    Qwen38NgramConfig c=small_config(); c.head_dim=8; c.hidden_size=8; c.streams=2;
    c.conv_kernel=4; c.rms_eps=1e-6f;
    lifecycle_oom(&gg,&c);
    Qwen38NgramStore *store=NULL; Qwen38NgramState *state=NULL,*other=NULL;
    Qwen38Error e; Qwen38NgramJob *job=NULL,*blocked=NULL; Qwen38F32 rows;
    Qwen38NgramConfig invalid=c; invalid.rms_eps=NAN;
    assert(qwen38_ngram_open(&gg,&invalid,32,1<<20,&store,&e)==Q38_NONFINITE && !store);
    invalid=c; invalid.hidden_size=0;
    assert(qwen38_ngram_open(&gg,&invalid,32,1<<20,&store,&e)==Q38_SHAPE && !store);
    assert(qwen38_ngram_open(&gg,&c,32,1,&store,&e)==Q38_CAPACITY && !store);
    assert(qwen38_ngram_open(&gg,&c,32,1<<20,&store,&e)==Q38_OK);
    assert(close(fd)==0); /* store owns its fd; no Gguf borrow */
    assert(qwen38_ngram_state_create(store,(Qwen38StateKey){&gg,0,1},&state,&e)==Q38_OK);
    assert(qwen38_ngram_state_create(store,(Qwen38StateKey){&gg,0,1},&other,&e)==Q38_OK);
    assert(qwen38_ngram_close(store,&e)==Q38_BUSY);
    assert(qwen38_ngram_prepare(store,state,NULL,0,1,&job,&e)==Q38_BAD_STATE && !job);
    assert(qwen38_ngram_prepare(store,state,NULL,0,0,&job,&e)==Q38_OK);
    assert(qwen38_ngram_state_reset(state,2,&e)==Q38_BUSY);
    assert(qwen38_ngram_state_destroy(state,&e)==Q38_BUSY);
    assert(qwen38_ngram_prepare(store,other,NULL,0,0,&blocked,&e)==Q38_BUSY && !blocked);
    assert(qwen38_ngram_read(job,&e)==Q38_OK);
    assert(qwen38_ngram_rows(job,&rows,&e)==Q38_OK && rows.dim[0]==0 && !rows.data);
    assert(q38e_reader_stats(store->reader).calls==0);
    assert(qwen38_ngram_commit(other,job,&e)==Q38_OWNER);
    assert(qwen38_ngram_commit(state,job,&e)==Q38_OK);
    assert(qwen38_ngram_commit(state,job,&e)==Q38_BAD_STATE);
    assert(qwen38_ngram_abort(job,&e)==Q38_BAD_STATE);
    assert(qwen38_ngram_job_release(job,&e)==Q38_OK);
    assert(!qwen38_ngram_state_info(state).next_pos);
    int32_t tokens[]={1,2,99,4};
    for (int kind=0;kind<4;kind++) {
        assert(qwen38_ngram_prepare(store,state,tokens,4,0,&job,&e)==Q38_OK);
        assert(qwen38_ngram_rows(job,&rows,&e)==Q38_BAD_STATE && !rows.data);
        Fault f={.cap=3,.fail_at=kind==3 ? SIZE_MAX : 3,.interrupt_at=0,.error=kind==1 ? EIO : kind==2 ? EBADF : 0};
        store->read_fn=fault_read; store->read_context=&f;
        assert(qwen38_ngram_read(job,&e)==(kind==3 ? Q38_OK : kind ? Q38_IO : Q38_EOF));
        assert(f.interrupted);
        if (kind==3) assert(qwen38_ngram_rows(job,&rows,&e)==Q38_OK && rows.dim[0]==4 && rows.dim[1]==32);
        else assert(qwen38_ngram_rows(job,&rows,&e)==Q38_BAD_STATE && !rows.data);
        assert(qwen38_ngram_commit(state,job,&e)==Q38_BAD_STATE); /* READY is not COMPUTED */
        assert(qwen38_ngram_abort(job,&e)==Q38_OK);
        assert(qwen38_ngram_rows(job,&rows,&e)==Q38_BAD_STATE && !rows.data);
        assert(qwen38_ngram_job_release(job,&e)==Q38_OK);
        assert(!state->info.next_pos && state->previous[0]==99);
    }
    store->read_fn=NULL; store->read_context=NULL;
    assert(qwen38_ngram_state_reset(state,2,&e)==Q38_OK);
    assert(qwen38_ngram_state_reset(state,2,&e)==Q38_STALE);
    assert(qwen38_ngram_prepare(store,state,tokens,4,0,&job,&e)==Q38_OK);
    assert(qwen38_ngram_read(job,&e)==Q38_OK);
    int changed=open(path,O_RDWR); assert(changed>=0 && ftruncate(changed,1023)==0); close(changed);
    assert(qwen38_ngram_rows(job,&rows,&e)==Q38_STALE && !rows.data);
    assert(!qwen38_ngram_state_info(state).valid);
    assert(qwen38_ngram_commit(state,job,&e)==Q38_STALE);
    assert(qwen38_ngram_job_release(job,&e)==Q38_OK);
    assert(qwen38_ngram_prepare(store,state,tokens,4,0,&job,&e)==Q38_STALE && !job);
    assert(qwen38_ngram_state_destroy(state,&e)==Q38_OK);
    assert(qwen38_ngram_state_destroy(other,&e)==Q38_OK);
    assert(qwen38_ngram_close(store,&e)==Q38_OK); assert(unlink(path)==0);
}


static void *read_fixture(const char *dir, const char *name, size_t bytes) {
    char path[1024]; assert(snprintf(path,sizeof path,"%s/%s",dir,name)>0);
    assert(bytes<=128u*1024u*1024u);
    FILE *f=fopen(path,"rb"); assert(f);
    void *p=malloc(bytes ? bytes : 1); assert(p && fread(p,1,bytes,f)==bytes);
    assert(fgetc(f)==EOF); assert(!fclose(f)); return p;
}
static void emit_fixture(const char *dir, const char *name, const float *p, size_t n, bool append) {
    char path[1024]; assert(snprintf(path,sizeof path,"%s/%s",dir,name)>0);
    FILE *f=fopen(path,append ? "ab" : "wb"); assert(f);
    assert(fwrite(p,sizeof *p,n,f)==n && !fclose(f));
}
static Q38PackedCpuView fixture_matrix(const char *dir, const char *name, size_t k, size_t n, uint32_t type) {
    size_t bytes=k*n*(type==0 ? 4 : 0);
    if (type==8) { assert(k%32==0); bytes=k/32*n*34; }
    GgufTensor tensor={.type=type,.rank=2,.shape={k,n},.bytes=bytes};
    Q38PackedCpuView view={.span={read_fixture(dir,name,bytes),bytes}};
    assert(q38_packed_layout_init(&tensor,k,&view.layout,NULL)==Q38_OK); return view;
}
static int ple_cli(const char *dir, size_t chunk, const char *checkpoint) {
    char path[1024]; assert(snprintf(path,sizeof path,"%s/config.txt",dir)>0);
    FILE *f=fopen(path,"r"); assert(f);
    Qwen38NgramConfig c={0}; uint64_t mult[16], offsets[960], sizes[960];
    size_t total; uint32_t type;
    assert(fscanf(f,"%"SCNu32" %"SCNu32" %"SCNu32" %"SCNu32" %"SCNu64,
        &c.order,&c.heads_per_order,&c.vocab_size,&c.eos_id,&c.row_count)==5);
    assert(fscanf(f,"%"SCNu32" %"SCNu32" %"SCNu32" %"SCNu32" %f %zu %"SCNu32,
        &c.head_dim,&c.hidden_size,&c.streams,&c.conv_kernel,&c.rms_eps,&total,&type)==7);
    assert(c.order>=2 && c.order<=16 && c.heads_per_order && c.heads_per_order<=64);
    c.head_count=(c.order-1u)*c.heads_per_order; c.multiplier_count=c.order;
    c.multipliers=mult; c.head_offsets=offsets; c.head_vocab_sizes=sizes;
    for (size_t i=0;i<c.order;i++) assert(fscanf(f,"%"SCNu64,mult+i)==1);
    for (size_t i=0;i<c.head_count;i++) assert(fscanf(f,"%"SCNu64,offsets+i)==1);
    for (size_t i=0;i<c.head_count;i++) assert(fscanf(f,"%"SCNu64,sizes+i)==1);
    assert(!fclose(f) && total && total<=32768 && chunk && chunk<=1024);
    assert(c.hidden_size<=2560 && c.streams<=4 && c.head_dim<=160 && c.conv_kernel<=4);
    size_t channels=(size_t)c.streams*c.hidden_size, width=c.head_count*c.head_dim;
    int32_t *tokens=read_fixture(dir,"tokens.bin",total*4);
    float *residual=read_fixture(dir,"residual.bin",total*channels*4), *delta=malloc(total*channels*4);
    assert(delta);
    Qwen38PleWeights w={.key=fixture_matrix(dir,"key.bin",width,channels,type),
        .value=fixture_matrix(dir,"value.bin",width,c.hidden_size,type)};
    Qwen38F32 *gains[]={&w.gamma_key,&w.gamma_query,&w.gamma_conv};
    const char *names[]={"gain_key.bin","gain_query.bin","gain_conv.bin"};
    for (size_t i=0;i<3;i++) *gains[i]=gate_const(gate_view(read_fixture(dir,names[i],channels*4),2,c.streams,c.hidden_size,1,0));
    w.taps=gate_const(gate_view(read_fixture(dir,"taps.bin",channels*c.conv_kernel*4),2,channels,c.conv_kernel,1,0));
    assert(snprintf(path,sizeof path,"%s/table.bin",dir)>0);
    GgufTensor table={.name="per_layer_token_embd.weight",.type=30,.rank=2,
        .shape={c.head_dim,(size_t)c.row_count},.bytes=(size_t)c.row_count*c.head_dim*2};
    Gguf g={.fd=checkpoint ? -1 : open(path,O_RDONLY),.tensors=&table,.tensor_count=1}; assert(checkpoint || g.fd>=0);
    Qwen38Config *real_config=NULL;
    if (checkpoint) {
        gguf_open(checkpoint,&g);
        assert(qwen38_config_load(&g,&real_config,NULL)==Q38_OK);
        const Qwen38NgramConfig *nc=NULL;
        for (uint32_t layer=0;layer<qwen38_config_view(real_config)->trunk_blocks;layer++)
            if (qwen38_config_ngram(real_config,layer,&nc,NULL)==Q38_OK) break;
        assert(nc && nc->hidden_size==c.hidden_size && nc->head_count==c.head_count && nc->head_dim==c.head_dim);
        c=*nc;
    }
    Qwen38NgramStore *store=NULL; Qwen38NgramState *state=NULL; Qwen38PleWorkspace *work=NULL;
    Qwen38Error e;
    assert(qwen38_ngram_open(&g,&c,chunk,32<<20,&store,&e)==Q38_OK);
    if (checkpoint) { gguf_close(&g); qwen38_config_free(real_config); c=store->config; }
    else close(g.fd);
    assert(qwen38_ngram_state_create(store,(Qwen38StateKey){&g,c.layer,1},&state,&e)==Q38_OK);
    assert(qwen38_ple_workspace_create(store,224<<20,&work,&e)==Q38_OK);
    Qwen38NgramJob *empty=NULL;
    assert(qwen38_ngram_prepare(store,state,NULL,0,0,&empty,&e)==Q38_OK);
    assert(qwen38_ngram_read(empty,&e)==Q38_OK);
    assert(qwen38_ple_cpu(empty,&w,gate_const(gate_view(NULL,3,0,c.streams,c.hidden_size,0)),
        gate_view(NULL,3,0,c.streams,c.hidden_size,0),work,&e)==Q38_OK);
    assert(qwen38_ngram_commit(state,empty,&e)==Q38_OK && !state->info.next_pos);
    assert(qwen38_ngram_job_release(empty,&e)==Q38_OK);
    assert(q38e_reader_stats(store->reader).calls==0);
    float *history=calloc((c.conv_kernel-1u)*c.order*channels+1,4); assert(history);
    size_t history_rows=(c.conv_kernel-1u)*c.order;
    if (!checkpoint) {
        Qwen38NgramJob *bad=NULL;
        assert(qwen38_ngram_prepare(store,state,tokens,1,0,&bad,&e)==Q38_OK);
        assert(qwen38_ngram_read(bad,&e)==Q38_OK);
        Qwen38F32 r=gate_const(gate_view(residual,3,1,c.streams,c.hidden_size,0));
        Qwen38F32Mut y=gate_view(delta,3,1,c.streams,c.hidden_size,0);
        assert(qwen38_ple_cpu(bad,&w,r,gate_view(residual,3,1,c.streams,c.hidden_size,0),work,&e)==Q38_INVALID_ARGUMENT);
        assert(bad->phase==Q38E_READY && !work->job);
        float saved=residual[0]; residual[0]=NAN;
        assert(qwen38_ple_cpu(bad,&w,r,y,work,&e)==Q38_NONFINITE);
        residual[0]=saved;
        assert(qwen38_ngram_commit(state,bad,&e)==Q38_BAD_STATE && state->info.next_pos==0);
        assert(qwen38_ngram_job_release(bad,&e)==Q38_OK);
    }
    /* First pass aborts every candidate, retries, then commits. Second pass
     * resets and replays; both must produce identical Delta and raw history. */
    for (size_t pass=0;pass<(checkpoint ? 1u : 2u);pass++) {
        if (pass) assert(qwen38_ngram_state_reset(state,2,&e)==Q38_OK);
        for (size_t start=0;start<total;start+=chunk) {
            size_t n=total-start<chunk ? total-start : chunk;
            for (int attempt=0;attempt<(pass || checkpoint ? 1 : 2);attempt++) {
                Qwen38NgramJob *job=NULL; Qwen38F32 rows;
                assert(qwen38_ngram_prepare(store,state,tokens+start,n,start,&job,&e)==Q38_OK);
                assert(qwen38_ngram_read(job,&e)==Q38_OK && qwen38_ngram_rows(job,&rows,&e)==Q38_OK);
                Qwen38F32 r=gate_const(gate_view(residual+start*channels,3,n,c.streams,c.hidden_size,0));
                Qwen38F32Mut y=gate_view(delta+start*channels,3,n,c.streams,c.hidden_size,0);
                assert(qwen38_ple_cpu(job,&w,r,y,work,&e)==Q38_OK);
                assert(qwen38_ple_workspace_destroy(work,&e)==Q38_BUSY);
                if (!pass && !attempt && !checkpoint) {
                    assert(qwen38_ngram_abort(job,&e)==Q38_OK && state->info.next_pos==start);
                } else {
                    assert(qwen38_ngram_commit(state,job,&e)==Q38_OK);
                    if (!pass) {
                        emit_fixture(dir,"delta.out",y.data,n*channels,start!=0);
                        emit_fixture(dir,"z.out",work->z,n*channels,start!=0);
                        emit_fixture(dir,"u.out",work->u,n*channels,start!=0);
                        emit_fixture(dir,"gate.out",work->gate,n*c.streams,start!=0);
                        emit_fixture(dir,"key_norm.out",work->key,n*channels,start!=0);
                        emit_fixture(dir,"query_norm.out",work->query,n*channels,start!=0);
                        emit_fixture(dir,"value.out",work->value,n*c.hidden_size,start!=0);
                        for (size_t ti=0;ti<n;ti++) for (size_t si=0;si<c.streams;si++)
                            work->conv[ti*c.streams+si]=q38e_ple_scaled_dot(work->key+(ti*c.streams+si)*c.hidden_size,
                                work->query+(ti*c.streams+si)*c.hidden_size,c.hidden_size);
                        emit_fixture(dir,"dot.out",work->conv,n*c.streams,start!=0);
                        emit_fixture(dir,"rows.out",rows.data,n*width,start!=0);
                        assert(q38_packed_matmul(&w.key,0,rows,gate_view(work->conv,2,n,channels,1,0),&e)==Q38_OK);
                        emit_fixture(dir,"key_proj.out",work->conv,n*channels,start!=0);
                        assert(q38_packed_matmul(&w.value,0,rows,gate_view(work->conv,2,n,c.hidden_size,1,0),&e)==Q38_OK);
                        emit_fixture(dir,"value_proj.out",work->conv,n*c.hidden_size,start!=0);
                    } else emit_fixture(dir,"replay.out",y.data,n*channels,start!=0);
                    assert(qwen38_conv_state_export(state->conv.handle,gate_view(history,2,history_rows,channels,1,0),&e)==Q38_OK);
                    emit_fixture(dir,pass ? "replay_history.out" : "history.out",history,history_rows*channels,start!=0);
                }
                assert(qwen38_ngram_job_release(job,&e)==Q38_OK);
            }
        }
    }
    printf("PASS: PLE total=%zu chunk=%zu peak_bytes=%zu\n",total,chunk,store->peak);
    assert(qwen38_ple_workspace_destroy(work,&e)==Q38_OK);
    assert(qwen38_ngram_state_destroy(state,&e)==Q38_OK && qwen38_ngram_close(store,&e)==Q38_OK);
    free((void *)w.key.span.data); free((void *)w.value.span.data);
    for (size_t i=0;i<3;i++) free((void *)gains[i]->data);
    free((void *)w.taps.data); free(tokens); free(residual); free(delta); free(history);
    return 0;
}

static int extract_real(const char *checkpoint, const char *dir) {
    Gguf g; gguf_open(checkpoint,&g);
    Qwen38Config *config=NULL; Qwen38Weights *weights=NULL; const Qwen38NgramConfig *c=NULL;
    assert(qwen38_config_load(&g,&config,NULL)==Q38_OK);
    for (uint32_t layer=0;layer<qwen38_config_view(config)->trunk_blocks;layer++)
        if (qwen38_config_ngram(config,layer,&c,NULL)==Q38_OK) break;
    assert(c && c->order<=16 && c->head_count<=960);
    assert(qwen38_weights_describe(&g,config,&weights,NULL)==Q38_OK);
    const Qwen38Tensor *table=NULL;
    assert(qwen38_weights_get(weights,QWEN38_GLOBAL_LAYER,Q38_PLE_TABLE,&table,NULL)==Q38_OK);
    printf("{\"layer\":%u,\"table_offset\":%zu,\"row_bytes\":%llu,\"files\":[",
        c->layer,table->storage.file_offset,(unsigned long long)table->row_bytes);
    const Qwen38Role roles[]={Q38_PLE_KEY,Q38_PLE_VALUE,Q38_PLE_KEY_NORM,Q38_PLE_QUERY_NORM,Q38_PLE_CONV_NORM,Q38_PLE_CONV};
    const char *names[]={"key.bin","value.bin","gain_key.bin","gain_query.bin","gain_conv.bin","taps.bin"};
    size_t total=0;
    for (size_t i=0;i<6;i++) {
        const Qwen38Tensor *tensor=NULL;
        assert(qwen38_weights_get(weights,c->layer,roles[i],&tensor,NULL)==Q38_OK);
        size_t bytes=tensor->storage.bytes; total+=bytes; assert(total<=36u*1024u*1024u);
        unsigned char *raw=malloc(bytes); assert(raw);
        size_t done=0;
        while (done<bytes) {
            ssize_t got=pread(g.fd,raw+done,bytes-done,(off_t)(tensor->storage.file_offset+done));
            if (got<0 && errno==EINTR) continue;
            assert(got>0); done+=(size_t)got;
        }
        char path[1024]; assert(snprintf(path,sizeof path,"%s/%s",dir,names[i])>0);
        FILE *f=fopen(path,"wb"); assert(f && fwrite(raw,1,bytes,f)==bytes && !fclose(f)); free(raw);
        printf("%s{\"name\":\"%s\",\"offset\":%zu,\"bytes\":%zu,\"type\":%u}",i ? "," : "",
            names[i],tensor->storage.file_offset,bytes,tensor->storage.type);
    }
    puts("]}");
    char path[1024]; assert(snprintf(path,sizeof path,"%s/config.txt",dir)>0);
    FILE *f=fopen(path,"w"); assert(f);
    fprintf(f,"%u %u %u %u %llu %u %u %u %u %.9g 2 8\n",c->order,c->heads_per_order,c->vocab_size,
        c->eos_id,(unsigned long long)c->row_count,c->head_dim,c->hidden_size,c->streams,c->conv_kernel,(double)c->rms_eps);
    for (size_t i=0;i<c->order;i++) fprintf(f,"%llu ",(unsigned long long)c->multipliers[i]);
    for (size_t i=0;i<c->head_count;i++) fprintf(f,"%llu ",(unsigned long long)c->head_offsets[i]);
    for (size_t i=0;i<c->head_count;i++) fprintf(f,"%llu ",(unsigned long long)c->head_vocab_sizes[i]);
    assert(!fclose(f));
    qwen38_weights_free(weights); qwen38_config_free(config); gguf_close(&g); return 0;
}

int main(int argc, char **argv) {
    if (argc==4 && !strcmp(argv[1],"--extract-real")) return extract_real(argv[2],argv[3]);
    if ((argc==4 || argc==5) && !strcmp(argv[1],"--ple")) return ple_cli(argv[2],(size_t)strtoul(argv[3],NULL,10),argc==5 ? argv[4] : NULL);
    if (argc == 2 && !strcmp(argv[1], "--gate")) return gate_cli();
    if (argc == 2 && !strcmp(argv[1], "--hash")) return hash_cli();
    if (argc == 3 && !strcmp(argv[1], "--lookup")) return lookup_cli(argv[2]);
    hash_errors();
    reader_basic();
    reader_faults();
    reader_limits();
    reader_stale();
    gate_errors();
    lifecycle_tests();
    puts("PASS: native hash, bounded BF16 I/O faults, normalized PLE gate contracts");
    return 0;
}
