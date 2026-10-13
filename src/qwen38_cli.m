/* Ordinary greedy Qwen3.8 generation using A's tokenizer/template and G2. */
#import <Metal/Metal.h>
#include "picoforge.h"
#include "qwen38_cli.h"
#include "qwen38_forward_gpu.h"
#include "qwen38_text.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
static volatile sig_atomic_t stopped;
static void stop_generation(int signum) { (void)signum; stopped=1; }
static double now(void) {
    struct timespec t; if (clock_gettime(CLOCK_MONOTONIC,&t)) die("clock_gettime");
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}
static char *read_prompt(const char *path) {
    FILE *f=fopen(path,"rb"); if (!f) die("cannot open prompt file");
    /* Bounded text input, also for a pipe. Embedded NUL is rejected explicitly. */
    size_t limit=4u<<20; char *s=malloc(limit+1); if (!s) die("prompt allocation");
    size_t n=fread(s,1,limit+1,f);
    if (ferror(f) || n>limit || memchr(s,0,n)) die("prompt file must be NUL-free text <=4 MiB");
    fclose(f); s[n]=0; return s;
}
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
        "       [--chat-file FILE | --complete-file FILE] [--encode-only]\n"
        "       [--repeat N] [--warmup N] [--replay-chunk N] [--runs-jsonl FILE]\n"
        "       [--session] (resident local worker protocol)\n"
        "       [--stop-after N] (SIGUSR1 also stops cooperatively)\n"
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
/* Local worker protocol: G <max-new> <messages> followed by U/A + hex lines.
 * The parent owns the conversation. We render the complete typed message tree
 * exactly once, then reset/replay it; no token-prefix reuse is claimed. */
static void hex(FILE *f,const uint8_t *p,size_t n) {
    for (size_t i=0;i<n;i++) fprintf(f,"%02x",p[i]);
}
static int nibble(char c) { return c>='0' && c<='9' ? c-'0' : c>='a' && c<='f' ? c-'a'+10 : -1; }
static int session_request(Qwen38Tokenizer *tok,int32_t *tokens,uint32_t capacity,size_t *count,uint32_t *max_new) {
    char header[128];
    if (!fgets(header,sizeof header,stdin)) return 0;
    if (!strcmp(header,"Q\n")) return 0;
    if (!strcmp(header,"R\n")) return 2;
    unsigned messages; char extra;
    if (sscanf(header,"G %u %u %c",max_new,&messages,&extra)!=2 || !messages || messages>128 || !*max_new || *max_new>capacity)
        die("invalid session header");
    Qwen38Json items[128],roles[128],contents[128]; Qwen38JsonMember fields[128][2];
    uint8_t *owned[128]={0}; size_t total=0; char *line=malloc((8u<<20)+4);
    if (!line) die("session input allocation");
    for (unsigned i=0;i<messages;i++) {
        if (!fgets(line,(8u<<20)+4,stdin)) die("incomplete session request");
        size_t n=strlen(line);
        if (n<3 || line[n-1]!='\n' || line[1]!=' ' || (line[0]!='U' && line[0]!='A') || (n-3)%2)
            die("invalid session message");
        size_t bytes_n=(n-3)/2; total+=bytes_n;
        if (total>(4u<<20)) die("session messages exceed 4 MiB");
        owned[i]=malloc(bytes_n ? bytes_n : 1); if (!owned[i]) die("message allocation");
        for (size_t j=0;j<bytes_n;j++) {
            int hi=nibble(line[2+2*j]),lo=nibble(line[3+2*j]);
            if (hi<0 || lo<0) die("invalid message hex"); owned[i][j]=(uint8_t)(hi*16+lo);
        }
        roles[i]=(Qwen38Json){.kind=Q38_JSON_STRING,.as.string=bytes(line[0]=='U' ? "user":"assistant")};
        contents[i]=(Qwen38Json){.kind=Q38_JSON_STRING,.as.string={owned[i],bytes_n}};
        fields[i][0]=(Qwen38JsonMember){bytes("role"),roles+i};
        fields[i][1]=(Qwen38JsonMember){bytes("content"),contents+i};
        items[i]=(Qwen38Json){.kind=Q38_JSON_OBJECT,.as.object={fields[i],2}};
    }
    free(line);
    Qwen38Json tree={.kind=Q38_JSON_ARRAY,.as.array={items,messages}};
    Qwen38ChatOptions options={.enable_thinking=Q38_FALSE,.add_generation_prompt=true};
    Qwen38Error e; size_t needed=0; uint8_t *rendered=NULL;
    Qwen38Status rc=qwen38_chat_render(tok,&tree,NULL,options,NULL,0,&needed,&e);
    if (!rc) {
        rendered=malloc(needed ? needed : 1); if (!rendered) die("render allocation");
        rc=qwen38_chat_render(tok,&tree,NULL,options,rendered,needed,&needed,&e);
    }
    if (!rc) rc=qwen38_encode(tok,(Qwen38Bytes){rendered,needed},tokens,capacity,count,&e);
    free(rendered); for (unsigned i=0;i<messages;i++) free(owned[i]);
    if (rc || !*count || *max_new>capacity-*count) {
        fputs("{\"event\":\"error\",\"code\":\"context_or_message\",\"message\":\"Conversazione troppo lunga o messaggi non validi. Riduci il limite di risposta o avvia una nuova conversazione.\"}\n",stdout);
        fflush(stdout); return -1;
    }
    return 1;
}

static void emit(Qwen38Decoder *d,const int32_t *id,bool session) {
    Qwen38Error e; size_t n;
    if (id) check(qwen38_decoder_append(d,id,1,NULL,0,&n,&e),&e);
    else check(qwen38_decoder_finish(d,NULL,0,&n,&e),&e);
    uint8_t *out=malloc(n ? n : 1); if (!out) die("decoder allocation");
    if (id) check(qwen38_decoder_append(d,id,1,out,n,&n,&e),&e);
    else check(qwen38_decoder_finish(d,out,n,&n,&e),&e);
    if (session && n) {
        fputs("{\"event\":\"token\",\"hex\":\"",stdout); hex(stdout,out,n); fputs("\"}\n",stdout);
    } else if (!session && n && fwrite(out,1,n,stdout)!=n) die("stdout write failed");
    fflush(stdout); free(out);
}
static void ids_json(FILE *file,const int32_t *ids,size_t n) {
    fputc('[',file);
    for (size_t i=0;i<n;i++) fprintf(file,"%s%d",i ? "," : "",ids[i]);
    fputc(']',file);
}
static void accumulate(Qwen38GpuForwardStats *total,Qwen38GpuForward *graph) {
    Qwen38GpuForwardStats s=qwen38_gpu_forward_stats(graph);
    total->prepare_s+=s.prepare_s; total->encode_s+=s.encode_s;
    total->wait_s+=s.wait_s; total->gpu_s+=s.gpu_s;
    total->ngram_calls+=s.ngram_calls; total->ngram_bytes+=s.ngram_bytes; total->ngram_rows+=s.ngram_rows;
}
static void stats_json(FILE *file,const char *name,Qwen38GpuForwardStats s) {
    fprintf(file,",\"%s\":{\"prepare_s\":%.9f,\"encode_s\":%.9f,\"wait_s\":%.9f,\"gpu_s\":%.9f,"
        "\"ngram_calls\":%zu,\"ngram_bytes\":%zu,\"ngram_rows\":%zu}",name,
        s.prepare_s,s.encode_s,s.wait_s,s.gpu_s,s.ngram_calls,s.ngram_bytes,s.ngram_rows);
}
int qwen38_cli(int argc,char **argv) {
    double entered=now();
    if (argc==3 && !strcmp(argv[2],"--help")) { usage(); return 0; }
    if (argc<4) { usage(); return 2; }
    const char *input=NULL,*mode=NULL,*logits_path=NULL,*ids_path=NULL,*metallib="picoforge.metallib";
    const char *runs_path=NULL; char *owned_input=NULL;
    uint32_t capacity=8192,chunk=1024,max_new=32,repeat=1,warmup=0,replay_chunk=0,stop_after=UINT32_MAX;
    bool thinking=false,encode_only=false,session=false;
    for (int i=3;i<argc;i++) {
        const char *opt=argv[i];
        if (!strcmp(opt,"--session")) { if (mode) die("choose one mode"); session=true; mode="session"; continue; }
        if (!strcmp(opt,"--thinking")) { thinking=true; continue; }
        if (!strcmp(opt,"--encode-only")) { encode_only=true; continue; }
        if (i+1==argc) die("missing value for %s",opt);
        const char *value=argv[++i];
        if (!strcmp(opt,"--chat") || !strcmp(opt,"--complete") || !strcmp(opt,"--tokens") ||
            !strcmp(opt,"--chat-file") || !strcmp(opt,"--complete-file")) {
            if (mode) die("choose one prompt mode"); mode=opt; input=value;
        } else if (!strcmp(opt,"--max-new")) max_new=number(value,0,32768,opt);
        else if (!strcmp(opt,"--context")) capacity=number(value,1,32768,opt);
        else if (!strcmp(opt,"--chunk")) chunk=number(value,1,1024,opt);
        else if (!strcmp(opt,"--logits")) logits_path=value;
        else if (!strcmp(opt,"--token-ids")) ids_path=value;
        else if (!strcmp(opt,"--metallib")) metallib=value;
        else if (!strcmp(opt,"--repeat")) repeat=number(value,1,1000,opt);
        else if (!strcmp(opt,"--warmup")) warmup=number(value,0,1000,opt);
        else if (!strcmp(opt,"--replay-chunk")) replay_chunk=number(value,1,1024,opt);
        else if (!strcmp(opt,"--stop-after")) stop_after=number(value,0,32768,opt);
        else if (!strcmp(opt,"--runs-jsonl")) runs_path=value;
        else die("unknown Qwen3.8 option %s",opt);
    }
    if (!mode) die("a prompt is required");
    if (session && (runs_path || logits_path || ids_path || encode_only || repeat!=1 || warmup || replay_chunk)) die("session owns its outputs and repetitions");
    if (!strcmp(mode,"--chat-file") || !strcmp(mode,"--complete-file")) {
        owned_input=read_prompt(input); input=owned_input;
        mode=!strcmp(mode,"--chat-file") ? "--chat" : "--complete";
    }
    if (thinking && strcmp(mode,"--chat")) die("--thinking requires --chat");
    chunk=MIN(chunk,capacity);
    if (replay_chunk>chunk) die("replay chunk exceeds allocated chunk");
    FILE *runs=runs_path ? fopen(runs_path,"wx") : NULL;
    if (runs_path && !runs) die("runs output must be a new file");
    struct sigaction action={0}; action.sa_handler=stop_generation; action.sa_flags=SA_RESTART; sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1,&action,NULL)) die("cannot install stop handler");
    @autoreleasepool {
        Gguf g; gguf_open(argv[2],&g); Qwen38Config *config; Qwen38Weights *weights; Qwen38Tokenizer *tok;
        Qwen38Error e; check(qwen38_config_load(&g,&config,&e),&e);
        check(qwen38_weights_describe(&g,config,&weights,&e),&e);
        check(qwen38_tokenizer_load(&g,config,&tok,&e),&e);
        const Qwen38ConfigView *c=qwen38_config_view(config);
        if (capacity>c->max_positions) die("context exceeds model maximum");
        int32_t *tokens=malloc((size_t)capacity*4),*generated=malloc((size_t)capacity*4);
        if (!tokens || !generated) die("token allocation");
        size_t count=0;
        if (session) { /* Request arrives after the model is ready. */ }
        else if (!strcmp(mode,"--tokens")) {
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
        if (!session && (!count || max_new>capacity-count)) die("prompt and requested generation exceed context, or prompt is empty");
        if (encode_only) {
            fputs("{\"prompt_ids\":",stdout); ids_json(stdout,tokens,count);
            fprintf(stdout,",\"prompt_tokens\":%zu,\"context\":%u}\n",count,capacity);
            qwen38_tokenizer_free(tok); qwen38_weights_free(weights); qwen38_config_free(config); gguf_close(&g);
            free(tokens); free(generated); free(owned_input); if (runs) fclose(runs); return 0;
        }
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
        double load_s=now()-entered;
        Qwen38Decoder *decoder; check(qwen38_decoder_create(tok,true,&decoder,&e),&e);
        if (session) { runs=stdout; fprintf(stdout,"{\"event\":\"ready\",\"load_s\":%.9f,\"context\":%u}\n",load_s,capacity); fflush(stdout); }
        for (uint32_t repetition=0;session || repetition<repeat+warmup;repetition++) {
        if (session) {
            int request=session_request(tok,tokens,capacity,&count,&max_new);
            if (!request) break;
            if (request<0) continue;
            if (request==2) {
                check(qwen38_gpu_forward_reset(graph,&e),&e); qwen38_decoder_reset(decoder);
                fputs("{\"event\":\"reset\"}\n",stdout); fflush(stdout); continue;
            }
        }
        double request_start=now(),reset_s=0;
        if (session || repetition) {
            check(qwen38_gpu_forward_reset(graph,&e),&e); qwen38_decoder_reset(decoder);
            reset_s=now()-request_start;
        }
        stopped=0;
        if (session) { fprintf(stdout,"{\"event\":\"started\",\"prompt_tokens\":%zu}\n",count); fflush(stdout); }
        uint32_t current_chunk=repetition && replay_chunk ? replay_chunk : chunk;
        Qwen38GpuForwardStats prefill_stats={0},decode_stats={0};
        double prefill_start=now();
        for (size_t start=0;start<count && !stopped;start+=current_chunk) {
            size_t n=MIN((size_t)current_chunk,count-start); y.dim[0]=start+n==count ? 1 : 0;
            check(qwen38_gpu_forward(graph,tokens+start,n,c->trunk_blocks,y,NULL,NULL,&e),&e);
            accumulate(&prefill_stats,graph);
        }
        double prefill_end=now(),first_token=0,decode_start=0;
        y.dim[0]=1; FILE *logfile=logits_path ? fopen(logits_path,"wb") : NULL;
        if (logits_path && !logfile) die("cannot open logits output");
        size_t got=0; const char *reason=stopped ? "cancelled" : "length";
        for (uint32_t step=0;step<MAX(max_new,1u) && !stopped;step++) {
            if (got==stop_after) { reason="cancelled"; break; }
            if (logfile && fwrite(logits,4,c->vocab_size,logfile)!=c->vocab_size) die("logits write failed");
            if (!max_new) break;
            int32_t id=0;
            for (uint32_t j=0;j<c->vocab_size;j++) {
                if (!isfinite(logits[j])) die("nonfinite logit after successful append");
                if (logits[j]>logits[id]) id=(int32_t)j;
            }
            generated[got++]=id;
            if (qwen38_token_is_stop(tok,id)) reason="stop";
            else emit(decoder,&id,session);
            if (first_token==0) { first_token=now(); decode_start=first_token; }
            if (!strcmp(reason,"stop")) break;
            if (got==stop_after) { reason="cancelled"; break; }
            if (step+1<max_new) {
                check(qwen38_gpu_forward(graph,&id,1,c->trunk_blocks,y,NULL,NULL,&e),&e);
                accumulate(&decode_stats,graph);
            }
        }
        double generation_end=now(); if (stopped) reason="cancelled";
        emit(decoder,NULL,session); if (!session) fputc('\n',stdout);
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
        if (runs) {
            fprintf(runs,"{\"event\":\"done\",\"repetition\":%u,\"warmup\":%s,\"load_s\":%.9f,\"reset_s\":%.9f,"
                "\"prefill_s\":%.9f,\"decode_s\":%.9f,\"request_s\":%.9f,\"ttft_request_s\":",
                repetition,repetition<warmup ? "true":"false",load_s,reset_s,prefill_end-prefill_start,
                first_token>0 ? generation_end-decode_start : 0,generation_end-request_start);
            if (first_token>0) fprintf(runs,"%.9f",first_token-request_start); else fputs("null",runs);
            fprintf(runs,",\"first_token_from_entry_s\":");
            if (first_token>0 && !repetition) fprintf(runs,"%.9f",first_token-entered); else fputs("null",runs);
            fprintf(runs,",\"prompt_tokens\":%zu,\"generated_tokens\":%zu,\"decode_tokens\":%zu,"
                "\"context\":%u,\"chunk\":%u,\"stop_reason\":\"%s\",\"position\":%llu,"
                "\"metal_allocated_bytes\":%zu,\"metal_recommended_bytes\":%zu,\"graph_buffer_bytes\":%zu,\"generated_ids\":",
                count,got,got ? got-1 : 0,capacity,current_chunk,reason,
                (unsigned long long)qwen38_gpu_forward_info(graph).next_pos,
                (size_t)device.currentAllocatedSize,(size_t)device.recommendedMaxWorkingSetSize,qwen38_gpu_forward_bytes(graph));
            ids_json(runs,generated,got);
            stats_json(runs,"prefill_profile",prefill_stats); stats_json(runs,"decode_profile",decode_stats);
            fputs("}\n",runs); fflush(runs);
        }
        }
        qwen38_decoder_free(decoder); qwen38_gpu_forward_destroy(graph); qwen38_tokenizer_free(tok);
        qwen38_weights_free(weights); gpu_weights_close(store); qwen38_config_free(config); gguf_close(&g);
        free(tokens); free(generated); free(logits);
    }
    free(owned_input); if (runs && runs!=stdout && fclose(runs)) die("runs close failed");
    return 0;
}
