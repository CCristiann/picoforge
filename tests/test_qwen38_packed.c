/* CPU safety harness and bounded binary differential-test transport. */
#include "qwen38_packed.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void expect(Qwen38Status actual, Qwen38Status wanted, const Qwen38Error *err) {
    if (actual != wanted || err->code != wanted) {
        fprintf(stderr, "packed status %d, expected %d: %s\n", actual, wanted, err->detail);
        abort();
    }
}
static void q2_checks(void) {
    unsigned char raw[1 + 84 * 12] = {0}; /* Unaligned weight base is intentional. */
    for (size_t b = 0; b < 12; b++) {
        unsigned char *p = raw + 1 + b * 84;
        memset(p, 0x23, 16); memset(p + 16, 0xe4, 64);
        p[81] = 0x3c; p[83] = 0x38; /* d=1, dmin=.5 */
    }
    GgufTensor tensor = {.type=10, .rank=3, .shape={768, 2, 2}, .bytes=sizeof raw - 1};
    Q38PackedCpuView view = {.span={raw + 1, sizeof raw - 1}};
    Qwen38Error err;
    expect(q38_packed_layout_init(&tensor, 640, &view.layout, &err), Q38_OK, &err);
    assert(view.layout.row_bytes == 252 && view.layout.expert_bytes == 504);
    float out[769];
    for (size_t i=0; i<769; i++) out[i] = 123;
    expect(q38_packed_decode_row(&view, 1, 1, out, 768, &err), Q38_OK, &err);
    const float expected[] = {-1, 2, 5, 8};
    for (size_t i=0; i<768; i++) assert(out[i] == expected[i % 128 / 32]);
    assert(out[768] == 123);
    out[0] = 123;
    expect(q38_packed_decode_row(&view, 2, 0, out, 768, &err), Q38_RANGE, &err);
    expect(q38_packed_decode_row(&view, 0, 2, out, 768, &err), Q38_RANGE, &err);
    expect(q38_packed_decode_row(&view, 0, 0, out, 767, &err), Q38_CAPACITY, &err);
    view.span.bytes--;
    expect(q38_packed_decode_row(&view, 0, 0, out, 768, &err), Q38_CAPACITY, &err);
    view.span.bytes++;
    view.layout.row_bytes++;
    expect(q38_packed_decode_row(&view, 0, 0, out, 768, &err), Q38_FORMAT, &err);
    view.layout.row_bytes--;
    expect(q38_packed_decode_row(&view, 0, 0, NULL, 768, &err), Q38_INVALID_ARGUMENT, &err);
    expect(q38_packed_decode_row(&view, 0, 0, (float *)(void *)raw, 768, &err), Q38_INVALID_ARGUMENT, &err);
    assert(out[0] == 123); /* All structural errors precede every output write. */
    raw[1 + 81] = 0x7c; /* +Inf half scale. */
    expect(q38_packed_decode_row(&view, 0, 0, out, 768, &err), Q38_NONFINITE, &err);
    Q38PackedLayout bad;
    tensor.rank = 4;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_SHAPE, &err);
    assert(!bad.tensor_bytes);
    tensor.rank = 3; tensor.shape[0] = 640;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_FORMAT, &err);
    tensor.shape[0] = 768; tensor.shape[1] = SIZE_MAX;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_OVERFLOW, &err);
    tensor.type = 12;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_OVERFLOW, &err);
    tensor.type = UINT32_MAX;
    expect(q38_packed_layout_init(&tensor, 640, &bad, &err), Q38_FORMAT, &err);
    puts("PASS: Q2_K layout, decode, bounds, padding and error preflight");
}

static size_t argument(const char *text) {
    assert(*text && strspn(text, "0123456789") == strlen(text));
    errno = 0;
    unsigned long long value = strtoull(text, NULL, 10);
    assert(!errno && value <= SIZE_MAX);
    return (size_t)value;
}
static void matmul_checks(void) {
    unsigned char packed[252 * 4] = {0};
    for (size_t b=0; b<12; b++) {
        memset(packed + b*84, 1, 16); memset(packed + b*84 + 16, 0x55, 64);
        packed[b*84 + 81] = 0x3c;
    }
    GgufTensor tensor = {.type=10, .rank=3, .shape={768,2,2}, .bytes=sizeof packed};
    Q38PackedCpuView w = {.span={packed,sizeof packed}};
    Qwen38Error err;
    expect(q38_packed_layout_init(&tensor, 640, &w.layout, &err), Q38_OK, &err);
    float *input = malloc(640 * sizeof(float)); /* ASan redzone starts at x[640]. */
    assert(input);
    for (size_t k=0; k<640; k++) input[k]=1;
    float output[4] = {NAN,NAN,123,123};
    Qwen38F32 x = {input,640*4,2,{1,640,1,1},{640*4,4,0,0}};
    Qwen38F32Mut y = {output,8,2,{1,2,1,1},{8,4,0,0}};
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_OK,&err);
    assert(output[0]==640 && output[1]==640 && output[2]==123);
    for (size_t row=0; row<4; row++) memset(packed+row*252+2*84+16+32,0xff,32);
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_OK,&err);
    assert(output[0]==640 && output[1]==640);
    output[0]=123;
    x.capacity_bytes--;
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_CAPACITY,&err);
    x.capacity_bytes++; x.stride_bytes[0]--;
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_STRIDE,&err);
    x.stride_bytes[0]++; y.dim[0]=2;
    y.capacity_bytes=16;
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_SHAPE,&err);
    y.dim[0]=1; y.data=input;
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_INVALID_ARGUMENT,&err);
    y.data=output;
    assert(output[0]==123);
    input[639]=INFINITY;
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_NONFINITE,&err);
    x.data=NULL; y.data=NULL; x.dim[0]=y.dim[0]=0; x.capacity_bytes=y.capacity_bytes=0;
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_OK,&err);
    x.stride_bytes[1]=8;
    expect(q38_packed_matmul(&w,1,x,y,&err), Q38_STRIDE,&err);
    free(input);
    /* Finite operands whose product overflows must report NONFINITE. */
    float weight=0x1p127f, large=4, result=0;
    tensor=(GgufTensor){.type=0,.rank=1,.shape={1},.bytes=4};
    expect(q38_packed_layout_init(&tensor,1,&w.layout,&err),Q38_OK,&err);
    w.span=(Qwen38CpuSpan){(const unsigned char *)&weight,4};
    x=(Qwen38F32){&large,4,2,{1,1,1,1},{4,4,0,0}};
    y=(Qwen38F32Mut){&result,4,2,{1,1,1,1},{4,4,0,0}};
    expect(q38_packed_matmul(&w,0,x,y,&err),Q38_NONFINITE,&err);
    puts("PASS: FP32 matmul, exact logical input allocation, padding and error preflight");
}

static int matrix_transport(int argc, char **argv) {
    assert(argc == 9); /* matmul TYPE Kphysical Klogical N experts expert M */
    size_t type=argument(argv[2]), kp=argument(argv[3]), k=argument(argv[4]);
    size_t n=argument(argv[5]), experts=argument(argv[6]), expert=argument(argv[7]), m=argument(argv[8]);
    const size_t limit=32u*1024u*1024u;
    assert(type==0 || type==1 || type==8 || type==10 || type==12 || type==16 || type==30 || type==39);
    assert(kp && k && k<=kp && n && experts && expert<experts && m<=128);
    assert(kp<=limit/4 && n<=limit/4/kp && experts<=limit/4/kp/n && m<=limit/4/k);
    const GgufFormat *format=gguf_format((uint32_t)type);
    assert(kp%format->block==0);
    size_t bytes=kp/format->block*format->bytes*n*experts;
    size_t xs=k+3, ys=n+3;
    assert(m<=limit/4/xs && m<=limit/4/ys);
    unsigned char *raw=malloc(bytes);
    float *input=malloc((m ? m*xs : 1)*4), *output=malloc((m ? m*ys : 1)*4);
    assert(raw && input && output && fread(raw,1,bytes,stdin)==bytes);
    for(size_t row=0;row<m;row++) {
        assert(fread(input+row*xs,4,k,stdin)==k);
        for(size_t j=k;j<xs;j++) input[row*xs+j]=NAN;
        for(size_t j=0;j<ys;j++) output[row*ys+j]=j<n ? NAN : 123;
    }
    assert(fgetc(stdin)==EOF && !ferror(stdin));
    GgufTensor tensor={.type=(uint32_t)type,.rank=3,.shape={kp,n,experts},.bytes=bytes};
    Q38PackedCpuView w={.span={raw,bytes}};
    Qwen38Error err;
    Qwen38Status status=q38_packed_layout_init(&tensor,k,&w.layout,&err);
    Qwen38F32 x={input,m*xs*4,2,{m,k,1,1},{xs*4,4,0,0}};
    Qwen38F32Mut y={output,m*ys*4,2,{m,n,1,1},{ys*4,4,0,0}};
    if(!status) status=q38_packed_matmul(&w,expert,x,y,&err);
    if(status) fprintf(stderr,"qwen38 native: %d %s\n",status,err.detail);
    else for(size_t row=0;row<m;row++) {
        for(size_t j=n;j<ys;j++) assert(output[row*ys+j]==123);
        assert(fwrite(output+row*ys,4,n,stdout)==n);
    }
    free(raw); free(input); free(output);
    return (int)status;
}
static int transport(int argc, char **argv) {
    assert(argc == 4);
    size_t type = argument(argv[1]), columns = argument(argv[2]), rows = argument(argv[3]);
    const size_t limit = 32u * 1024u * 1024u;
    assert(type <= UINT32_MAX && columns && rows && columns <= limit / 4 && rows <= limit / 4 / columns);
    /* Registry only called for these known executable formats. */
    assert(type == 0 || type == 1 || type == 8 || type == 10 || type == 12 || type == 16 || type == 30 || type == 39);
    const GgufFormat *format = gguf_format((uint32_t)type);
    assert(columns % format->block == 0);
    size_t bytes = columns / format->block * format->bytes * rows;
    assert(bytes <= limit);
    unsigned char *raw = malloc(bytes);
    float *out = malloc(columns * rows * sizeof *out);
    assert(raw && out);
    assert(fread(raw, 1, bytes, stdin) == bytes && fgetc(stdin) == EOF && !ferror(stdin));
    GgufTensor tensor = {.type=(uint32_t)type, .rank=2, .shape={columns,rows}, .bytes=bytes};
    Q38PackedCpuView view = {.span={raw,bytes}};
    Qwen38Error err;
    Qwen38Status status = q38_packed_layout_init(&tensor, columns, &view.layout, &err);
    for (size_t row=0; status == Q38_OK && row<rows; row++)
        status = q38_packed_decode_row(&view, 0, row, out + row * columns, columns, &err);
    if (status != Q38_OK) fprintf(stderr, "qwen38 native: %d %s\n", status, err.detail);
    else assert(fwrite(out, sizeof *out, columns * rows, stdout) == columns * rows);
    free(raw); free(out);
    return (int)status;
}
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1],"matmul")) return matrix_transport(argc,argv);
    if (argc > 1) return transport(argc, argv);
    q2_checks();
    matmul_checks();
    return 0;
}
