/* Ordinary greedy Qwen3.8 generation using A's tokenizer/template and G2. */
#import <Metal/Metal.h>
#include "picoforge.h"
#include "qwen38_cli.h"
#include "qwen38_forward_gpu.h"
#include "qwen38_text.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
static void check(Qwen38Status rc,const Qwen38Error *e) {
    if (rc) die("Qwen3.8 status %d %s: %s",rc,e->subject,e->detail);
}
static uint32_t number(const char *s,uint32_t low,uint32_t high,const char *what) {
    char *end; errno=0; unsigned long n=strtoul(s,&end,10);
    if (errno || s==end || *end || n<low || n>high) die("%s must be in [%u,%u]",what,low,high);
    return (uint32_t)n;
}
static void usage(void) {
    fprintf(stderr,"usage: picoforge --qwen38 FILE (--chat TEXT | --complete TEXT | --tokens ID,ID,...)\n"
        "       [--max-new N] [--context N] [--chunk N] [--thinking]\n"
        "       [--logits FILE.f32] [--token-ids FILE.json] [--metallib FILE]\n"
        "Greedy FP32 generation; defaults: context 8192, chunk 1024, max-new 32.\n");
}
static Qwen38Bytes bytes(const char *s) { return (Qwen38Bytes){(const uint8_t *)s,strlen(s)}; }
static uint8_t *chat(Qwen38Tokenizer *tok,const char *text,bool thinking,size_t *n) {
    Qwen38Json role={.kind=Q38_JSON_STRING,.as.string=bytes("user")};
    Qwen38Json content={.kind=Q38_JSON_STRING,.as.string=bytes(text)};
    Qwen38JsonMember fields[]={{bytes("role"),&role},{bytes("content"),&content}};
    Qwen38Json message={.kind=Q38_JSON_OBJECT,.as.object={fields,2}};
    Qwen38Json messages={.kind=Q38_JSON_ARRAY,.as.array={&message,1}};
    Qwen38ChatOptions options={.enable_thinking=thinking ? Q38_TRUE : Q38_FALSE,.add_generation_prompt=true};
    Qwen38Error e; check(qwen38_chat_render(tok,&messages,NULL,options,NULL,0,n,&e),&e);
    uint8_t *out=malloc(*n ? *n : 1); if (!out) die("chat allocation");
    check(qwen38_chat_render(tok,&messages,NULL,options,out,*n,n,&e),&e); return out;
}
static void emit(Qwen38Decoder *d,const int32_t *id) {
    Qwen38Error e; size_t n;
    if (id) check(qwen38_decoder_append(d,id,1,NULL,0,&n,&e),&e);
    else check(qwen38_decoder_finish(d,NULL,0,&n,&e),&e);
    uint8_t *out=malloc(n ? n : 1); if (!out) die("decoder allocation");
    if (id) check(qwen38_decoder_append(d,id,1,out,n,&n,&e),&e);
    else check(qwen38_decoder_finish(d,out,n,&n,&e),&e);
    if (n && fwrite(out,1,n,stdout)!=n) die("stdout write failed");
    fflush(stdout); free(out);
}
static void ids_json(FILE *file,const int32_t *ids,size_t n) {
    fputc('[',file);
    for (size_t i=0;i<n;i++) fprintf(file,"%s%d",i ? "," : "",ids[i]);
    fputc(']',file);
}
int qwen38_cli(int argc,char **argv) {
    if (argc==3 && !strcmp(argv[2],"--help")) { usage(); return 0; }
    if (argc<5) { usage(); return 2; }
    const char *input=NULL,*mode=NULL,*logits_path=NULL,*ids_path=NULL,*metallib="picoforge.metallib";
    uint32_t capacity=8192,chunk=1024,max_new=32; bool thinking=false;
    for (int i=3;i<argc;i++) {
        const char *opt=argv[i];
        if (!strcmp(opt,"--thinking")) { thinking=true; continue; }
        if (i+1==argc) die("missing value for %s",opt);
        const char *value=argv[++i];
        if (!strcmp(opt,"--chat") || !strcmp(opt,"--complete") || !strcmp(opt,"--tokens")) {
            if (mode) die("choose one prompt mode"); mode=opt; input=value;
        } else if (!strcmp(opt,"--max-new")) max_new=number(value,0,32768,opt);
        else if (!strcmp(opt,"--context")) capacity=number(value,1,32768,opt);
        else if (!strcmp(opt,"--chunk")) chunk=number(value,1,1024,opt);
        else if (!strcmp(opt,"--logits")) logits_path=value;
        else if (!strcmp(opt,"--token-ids")) ids_path=value;
        else if (!strcmp(opt,"--metallib")) metallib=value;
        else die("unknown Qwen3.8 option %s",opt);
    }
    if (!mode) die("a prompt is required");
    if (thinking && strcmp(mode,"--chat")) die("--thinking requires --chat");
    chunk=MIN(chunk,capacity);
    @autoreleasepool {
        Gguf g; gguf_open(argv[2],&g); Qwen38Config *config; Qwen38Weights *weights; Qwen38Tokenizer *tok;
        Qwen38Error e; check(qwen38_config_load(&g,&config,&e),&e);
        check(qwen38_weights_describe(&g,config,&weights,&e),&e);
        check(qwen38_tokenizer_load(&g,config,&tok,&e),&e);
        const Qwen38ConfigView *c=qwen38_config_view(config);
        if (capacity>c->max_positions) die("context exceeds model maximum");
        int32_t *tokens=malloc((size_t)capacity*4),*generated=malloc(MAX((size_t)max_new*4,4));
        if (!tokens || !generated) die("token allocation");
        size_t count=0;
        if (!strcmp(mode,"--tokens")) {
            const char *p=input;
            while (*p) {
                char *end; errno=0; long id=strtol(p,&end,10);
                if (errno || end==p || id<0 || (unsigned long)id>=c->vocab_size || (*end && *end!=',') || count==capacity)
                    die("invalid token list or context exceeded");
                tokens[count++]=(int32_t)id;
                if (*end==',' && !end[1]) die("trailing token delimiter");
                p=*end ? end+1 : end;
            }
        } else {
            size_t n=strlen(input); uint8_t *rendered=NULL; Qwen38Bytes prompt=bytes(input);
            if (!strcmp(mode,"--chat")) { rendered=chat(tok,input,thinking,&n); prompt=(Qwen38Bytes){rendered,n}; }
            check(qwen38_encode(tok,prompt,tokens,capacity,&count,&e),&e); free(rendered);
        }
        if (!count || max_new>capacity-count) die("prompt and requested generation exceed context, or prompt is empty");
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if (!device) die("Metal device unavailable");
        NSError *error=nil; id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(metallib)] error:&error];
        if (!lib) die("cannot load %s: %s",metallib,error.localizedDescription.UTF8String);
        GpuWeightStore *store=gpu_weights_open((__bridge void *)device,&g,0);
        check(qwen38_weights_attach(weights,&g,gpu_weights_plan(store),&e),&e);
        Qwen38GpuForward *graph;
        check(qwen38_gpu_forward_create(&g,config,weights,store,(__bridge void *)device,(__bridge void *)lib,chunk,capacity,1,&graph,&e),&e);
        fprintf(stderr,"Qwen3.8: %u layers, %zu prompt tokens, context %u, chunk %u; graph buffers %.1f MiB\n",
            c->trunk_blocks,count,capacity,chunk,(double)qwen38_gpu_forward_bytes(graph)/1048576.0);
        float *logits=malloc((size_t)c->vocab_size*4); if (!logits) die("logits allocation");
        Qwen38F32Mut y={.data=logits,.capacity_bytes=(size_t)c->vocab_size*4,.rank=2,.dim={1,c->vocab_size},.stride_bytes={(size_t)c->vocab_size*4,4}};
        for (size_t start=0;start<count;start+=chunk) {
            size_t n=MIN((size_t)chunk,count-start); y.dim[0]=start+n==count ? 1 : 0;
            check(qwen38_gpu_forward(graph,tokens+start,n,c->trunk_blocks,y,NULL,NULL,&e),&e);
        }
        y.dim[0]=1; FILE *logfile=logits_path ? fopen(logits_path,"wb") : NULL;
        if (logits_path && !logfile) die("cannot open logits output");
        Qwen38Decoder *decoder; check(qwen38_decoder_create(tok,true,&decoder,&e),&e);
        size_t got=0;
        for (uint32_t step=0;step<MAX(max_new,1u);step++) {
            if (logfile && fwrite(logits,4,c->vocab_size,logfile)!=c->vocab_size) die("logits write failed");
            if (!max_new) break;
            int32_t id=0;
            for (uint32_t j=0;j<c->vocab_size;j++) {
                if (!isfinite(logits[j])) die("nonfinite logit after successful append");
                if (logits[j]>logits[id]) id=(int32_t)j;
            }
            generated[got++]=id;
            if (qwen38_token_is_stop(tok,id)) break;
            emit(decoder,&id);
            if (step+1<max_new) check(qwen38_gpu_forward(graph,&id,1,c->trunk_blocks,y,NULL,NULL,&e),&e);
        }
        emit(decoder,NULL); fputc('\n',stdout);
        if (logfile && fclose(logfile)) die("logits close failed");
        if (ids_path) {
            FILE *out=fopen(ids_path,"w"); if (!out) die("cannot open token IDs output");
            fputs("{\"prompt_ids\":",out); ids_json(out,tokens,count); fputs(",\"generated_ids\":",out); ids_json(out,generated,got);
            fprintf(out,",\"context\":%u,\"chunk\":%u,\"greedy\":true}\n",capacity,chunk);
            if (fclose(out)) die("token IDs write failed");
        }
        fprintf(stderr,"Qwen3.8: generated %zu tokens%s\n",got,got && qwen38_token_is_stop(tok,generated[got-1]) ? " (stop)" : "");
        fprintf(stderr,"Qwen3.8 memory: {\"metal_allocated_bytes\":%zu,\"metal_recommended_bytes\":%zu,\"graph_buffer_bytes\":%zu}\n",
            (size_t)device.currentAllocatedSize,(size_t)device.recommendedMaxWorkingSetSize,qwen38_gpu_forward_bytes(graph));
        qwen38_decoder_free(decoder); qwen38_gpu_forward_destroy(graph); qwen38_tokenizer_free(tok);
        qwen38_weights_free(weights); gpu_weights_close(store); qwen38_config_free(config); gguf_close(&g);
        free(tokens); free(generated); free(logits);
    }
    return 0;
}
