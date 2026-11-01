/* Ordinary greedy Qwen3.8 generation using A's tokenizer/template and G2. */
#import <Metal/Metal.h>
#include "picoforge.h"
#include "qwen38_cli.h"
#include "qwen38_run.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
volatile sig_atomic_t q38_stopped;
static void stop_generation(int signum) { (void)signum; q38_stopped=1; }
double q38_now(void) {
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
void q38_check(Qwen38Status rc,const Qwen38Error *e) {
    if (rc) die("Qwen3.8 status %d %s: %s",rc,e->subject,e->detail);
}
static uint32_t number(const char *s,uint32_t low,uint32_t high,const char *what) {
    char *end; errno=0; unsigned long n=strtoul(s,&end,10);
    if (errno || s==end || *end || n<low || n>high) die("%s must be in [%u,%u]",what,low,high);
    return (uint32_t)n;
}
static void usage(void) {
    fprintf(stderr,"usage: picoforge --qwen38 FILE [--chat TEXT | --complete TEXT | --tokens ID,ID,...]\n"
        "       (no prompt: interactive chat; /help inside)\n"
        "   or: picoforge [-m FILE] [-p TEXT] [-c CONTEXT] [-n MAX_NEW] [--mtp] [--thinking] ...\n"
        "       (FILE defaults to $PICOFORGE_MODEL; -p is --chat, no -p is the chat)\n"
        "       [--max-new N] [--context N] [--chunk N] [--thinking]\n"
        "       [--logits FILE.f32] [--token-ids FILE.json] [--metallib FILE]\n"
        "       [--chat-file FILE | --complete-file FILE] [--encode-only]\n"
        "       [--repeat N] [--warmup N] [--replay-chunk N] [--runs-jsonl FILE]\n"
        "       [--session] (resident local worker protocol)\n"
        "       [--stop-after N] (SIGUSR1 also stops cooperatively)\n"
        "       [--mtp] (one-token predictor, exact greedy verification)\n"
        "       [--profile FILE.csv] (per-op GPU time of trunk appends; MTP appends are not profiled)\n"
        "       [--teacher-forced N] (--tokens only: prefill all but the last N IDs, append those one by one)\n"
        "Greedy FP32 generation; defaults: context 8192, chunk 1024, max-new 32.\n");
}
/* Greedy choice shared by ordinary and teacher-forced decode: first maximum wins. */
static int32_t greedy(const float *logits,uint32_t n) {
    int32_t id=0;
    for (uint32_t j=0;j<n;j++) {
        if (!isfinite(logits[j])) die("nonfinite logit after successful append");
        if (logits[j]>logits[id]) id=(int32_t)j;
    }
    return id;
}
static Qwen38Bytes bytes(const char *s) { return (Qwen38Bytes){(const uint8_t *)s,strlen(s)}; }
static uint8_t *chat(Qwen38Tokenizer *tok,const char *text,bool thinking,size_t *n) {
    Qwen38Json role={.kind=Q38_JSON_STRING,.as.string=bytes("user")};
    Qwen38Json content={.kind=Q38_JSON_STRING,.as.string=bytes(text)};
    Qwen38JsonMember fields[]={{bytes("role"),&role},{bytes("content"),&content}};
    Qwen38Json message={.kind=Q38_JSON_OBJECT,.as.object={fields,2}};
    Qwen38Json messages={.kind=Q38_JSON_ARRAY,.as.array={&message,1}};
    Qwen38ChatOptions options={.enable_thinking=thinking ? Q38_TRUE : Q38_FALSE,.add_generation_prompt=true};
    Qwen38Error e; q38_check(qwen38_chat_render(tok,&messages,NULL,options,NULL,0,n,&e),&e);
    uint8_t *out=malloc(*n ? *n : 1); if (!out) die("chat allocation");
    q38_check(qwen38_chat_render(tok,&messages,NULL,options,out,*n,n,&e),&e); return out;
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
static void emit(Qwen38Decoder *d,const int32_t *id,const Q38TurnOptions *o) {
    Qwen38Error e; size_t n;
    if (id) q38_check(qwen38_decoder_append(d,id,1,NULL,0,&n,&e),&e);
    else q38_check(qwen38_decoder_finish(d,NULL,0,&n,&e),&e);
    uint8_t *out=malloc(n ? n : 1); if (!out) die("decoder allocation");
    if (id) q38_check(qwen38_decoder_append(d,id,1,out,n,&n,&e),&e);
    else q38_check(qwen38_decoder_finish(d,out,n,&n,&e),&e);
    if (o->session && n) {
        fputs("{\"event\":\"token\",\"hex\":\"",stdout); hex(stdout,out,n); fputs("\"}\n",stdout);
    } else if (n && o->print) o->print(o->print_ctx,out,n);
    else if (!o->session && n && fwrite(out,1,n,stdout)!=n) die("stdout write failed");
    fflush(stdout); free(out);
}
/* Every trunk append goes through here: profile row, stats, and the count
 * a following REPL turn needs to catch the MTP predictor up. */
static void append(Q38Run *r,const int32_t *ids,size_t n,const char *phase,int32_t *id,Qwen38GpuForwardStats *stats) {
    Qwen38Error e;
    q38_profile_begin_append(r->profile,phase,qwen38_gpu_forward_info(r->graph).next_pos);
    if (id) q38_check(qwen38_gpu_forward_argmax(r->graph,ids,n,r->c->trunk_blocks,id,&e),&e);
    else q38_check(qwen38_gpu_forward(r->graph,ids,n,r->c->trunk_blocks,r->y,NULL,NULL,&e),&e);
    r->last_count=n; accumulate(stats,r->graph);
}
void q38_turn(Q38Run *r,const int32_t *tokens,size_t count,Q38TurnOptions o,int32_t *generated,Q38Turn *t) {
    const Qwen38ConfigView *c=r->c; Qwen38GpuForward *graph=r->graph; Qwen38Error e;
    float *logits=r->logits; Qwen38F32Mut *y=&r->y; *t=(Q38Turn){.reason="length"};
    size_t residual_offset=0,got=0; bool queued=false,catchup=false; int32_t catchup_id=0;
    /* T15c: plain greedy reads back only the id (GPU argmax). Logits rows stay for --logits
     * and MTP, which verify against them. */
    const bool use_id=!r->mtp && !o.logfile; int32_t next_id=0;
    t->prefill_start=q38_now();
    for (size_t start=0;start<count && !q38_stopped;start+=o.chunk) {
        size_t n=MIN((size_t)o.chunk,count-start); y->dim[0]=start+n==count ? 1 : 0;
        append(r,tokens+start,n,"prefill",use_id && start+n==count ? &next_id : NULL,&t->prefill);
        residual_offset=n-1;
        if (r->mtp && !q38_stopped) {
            size_t pairs=MIN(n,count-start-1); Qwen38F32Mut empty=*y; empty.dim[0]=0;
            double mtp_start=q38_now();
            if (pairs) q38_check(qwen38_gpu_mtp(graph,0,tokens+start+1,pairs,empty,&e),&e);
            t->mtp_prefill_s+=q38_now()-mtp_start;
            if (pairs) accumulate(&t->mtp,graph);
        }
        if (o.session) {
            fprintf(stdout,"{\"event\":\"prefill\",\"completed\":%zu,\"prompt_tokens\":%zu}\n",start+n,count);
            fflush(stdout);
        }
    }
    t->prefill_end=q38_now(); y->dim[0]=1;
    if (q38_stopped) t->reason="cancelled";
    if (o.forced) {
        /* Teacher-forced decode, the same work as ds4-bench --teacher-forced-decode
         * (ds4 fc80bd6, MIT, ds4_bench.c:main): each forced ID is its own
         * one-token append with one logits row and a greedy argmax, never a
         * batch, and stop tokens do not end the run. --logits keeps the
         * prefill row first, so it lines up with an ordinary run. */
        if (!q38_stopped) {
            if (o.logfile && fwrite(logits,4,c->vocab_size,o.logfile)!=c->vocab_size) die("logits write failed");
            t->first_token=t->decode_start=q38_now();
        }
        for (uint32_t i=0;i<o.forced && !q38_stopped;i++) {
            append(r,tokens+count+i,1,"decode",use_id ? &next_id : NULL,&t->decode); t->target_appends++;
            if (o.logfile && fwrite(logits,4,c->vocab_size,o.logfile)!=c->vocab_size) die("logits write failed");
            generated[got++]=use_id ? next_id : greedy(logits,c->vocab_size);
        }
    } else for (uint32_t step=0;step<MAX(o.max_new,1u) && !q38_stopped;step++) {
        if (got==o.stop_after) { t->reason="cancelled"; break; }
        if (o.logfile && fwrite(logits,4,c->vocab_size,o.logfile)!=c->vocab_size) die("logits write failed");
        if (!o.max_new) break;
        int32_t id=use_id ? next_id : greedy(logits,c->vocab_size);
        generated[got++]=id;
        if (qwen38_token_is_stop(r->tok,id)) t->reason="stop";
        else emit(r->decoder,&id,&o);
        if (t->first_token==0) { t->first_token=q38_now(); t->decode_start=t->first_token; }
        if (!strcmp(t->reason,"stop")) break;
        if (got==o.stop_after) { t->reason="cancelled"; break; }
        if (step+1<o.max_new) {
            if (queued) {
                /* Both rows were already verified causally. The first has
                 * just emitted b; the next iteration emits c from row 1. */
                memcpy(logits,logits+c->vocab_size,(size_t)c->vocab_size*4);
                queued=false; catchup=true; catchup_id=id;
            } else if (r->mtp && o.max_new-got>=2 && o.stop_after-got>=2 &&
                       r->capacity-qwen38_gpu_forward_info(graph).next_pos>=2) {
                int32_t confirmed[2]={catchup ? catchup_id : id,id};
                double part=q38_now();
                q38_check(qwen38_gpu_mtp(graph,catchup ? 0 : residual_offset,confirmed,catchup ? 2 : 1,*y,&e),&e);
                t->draft_s+=q38_now()-part; accumulate(&t->mtp,graph); catchup=false;
                if (q38_stopped) break;
                int32_t draft=0;
                for (uint32_t j=0;j<c->vocab_size;j++) {
                    if (!isfinite(logits[j])) die("nonfinite MTP logit");
                    if (logits[j]>logits[draft]) draft=(int32_t)j;
                }
                t->proposed++;
                /* A proposed stop is checked using only a. It must never
                 * enter the trunk cache, even when it is accepted. */
                bool draft_stop=qwen38_token_is_stop(r->tok,draft);
                Qwen38GpuCheckpoint snapshot; part=q38_now();
                q38_check(qwen38_gpu_checkpoint_save(graph,&snapshot,&e),&e); t->snapshot_s+=q38_now()-part;
                int32_t pair[2]={id,draft}; y->dim[0]=draft_stop ? 1 : 2; part=q38_now(); size_t saved_count=r->last_count;
                append(r,pair,y->dim[0],"decode",NULL,&t->decode);
                t->verify_s+=q38_now()-part; t->verified_tokens+=y->dim[0]; t->target_appends++; y->dim[0]=1;
                if (q38_stopped) { q38_check(qwen38_gpu_checkpoint_restore(graph,snapshot,&e),&e); r->last_count=saved_count; break; }
                int32_t target=0;
                for (uint32_t j=1;j<c->vocab_size;j++) if (logits[j]>logits[target]) target=(int32_t)j;
                if (target==draft) { t->accepted++; queued=!draft_stop; residual_offset=draft_stop ? 0 : 1; }
                else if (!draft_stop) {
                    part=q38_now(); q38_check(qwen38_gpu_checkpoint_restore(graph,snapshot,&e),&e);
                    append(r,&id,1,"decode",NULL,&t->decode);
                    t->replay_s+=q38_now()-part; t->rollback_count++; t->target_appends++; residual_offset=0;
                } else residual_offset=0;
            } else {
                append(r,&id,1,"decode",use_id ? &next_id : NULL,&t->decode);
                t->target_appends++; residual_offset=0;
            }
        }
    }
    t->end=q38_now(); if (q38_stopped) t->reason="cancelled";
    emit(r->decoder,NULL,&o); if (!o.session && !o.print) fputc('\n',stdout);
    t->got=got;
}
void q38_run_graph(Q38Run *r,uint32_t capacity) {
    Qwen38Error e; uint32_t chunk=MIN(r->chunk,capacity);
    if (r->graph) qwen38_gpu_forward_destroy(r->graph);
    free(r->logits); r->graph=NULL; r->capacity=capacity; r->last_count=0;
    q38_check(qwen38_gpu_forward_create(&r->g,r->config,r->weights,r->store,r->device,r->library,
        r->mtp ? MAX(chunk,2u) : chunk,capacity,r->mtp ? 2 : 1,&r->graph,&e),&e);
    if (r->mtp) q38_check(qwen38_gpu_mtp_enable(r->graph,&e),&e);
    if (r->profile) q38_check(q38g_forward_set_profile(r->graph,r->profile,&e),&e);
    const uint32_t vocab=r->c->vocab_size; size_t bytes=(size_t)vocab*4*(r->mtp ? 2u : 1u);
    r->logits=malloc(bytes); if (!r->logits) die("logits allocation");
    r->y=(Qwen38F32Mut){.data=r->logits,.capacity_bytes=bytes,.rank=2,.dim={1,vocab},.stride_bytes={(size_t)vocab*4,4}};
    if (r->decoder) qwen38_decoder_reset(r->decoder);
    else q38_check(qwen38_decoder_create(r->tok,true,&r->decoder,&e),&e);
}
/* ds4-style short form (`./ds4 -p ...`): -m FILE, -p TEXT, -c N, -n N, the rest
 * as in the long form. The model defaults to $PICOFORGE_MODEL. A bare
 * ./picoforge keeps its old meaning (Qwen3-0.6B) unless that is set. */
bool qwen38_short_form(int argc,char **argv) {
    static const char *const starts[]={"-m","-p","-c","-n","--mtp","--thinking",NULL};
    if (argc==1) return getenv("PICOFORGE_MODEL")!=NULL;
    for (size_t i=0;starts[i];i++) if (!strcmp(argv[1],starts[i])) return true;
    return false;
}
int qwen38_short(int argc,char **argv) {
    const char *model=getenv("PICOFORGE_MODEL"); char **out=calloc((size_t)argc+3,sizeof *out); int n=3;
    if (!out) die("argument allocation");
    for (int i=1;i<argc;i++) {
        const char *a=argv[i],*as=!strcmp(a,"-p") ? "--chat" : !strcmp(a,"-c") ? "--context" : !strcmp(a,"-n") ? "--max-new" : NULL;
        if ((as || !strcmp(a,"-m")) && i+1==argc) die("missing value for %s",a);
        if (!strcmp(a,"-m")) model=argv[++i];
        else if (as) { out[n++]=(char *)as; out[n++]=argv[++i]; }
        else out[n++]=argv[i];
    }
    if (!model) die("no model: pass -m FILE.gguf or set PICOFORGE_MODEL");
    out[0]=argv[0]; out[1]="--qwen38"; out[2]=(char *)model;
    int rc=qwen38_cli(n,out); free(out); return rc;
}
int qwen38_cli(int argc,char **argv) {
    double entered=q38_now();
    if (argc==3 && !strcmp(argv[2],"--help")) { usage(); return 0; }
    if (argc<3) { usage(); return 2; }
    const char *input=NULL,*mode=NULL,*logits_path=NULL,*ids_path=NULL,*metallib="picoforge.metallib";
    const char *runs_path=NULL,*profile_path=NULL; char *owned_input=NULL;
    uint32_t capacity=8192,chunk=1024,max_new=32,repeat=1,warmup=0,replay_chunk=0,stop_after=UINT32_MAX,forced=0;
    bool thinking=false,encode_only=false,session=false,mtp=false,max_new_given=false;
    for (int i=3;i<argc;i++) {
        const char *opt=argv[i];
        if (!strcmp(opt,"--session")) { if (mode) die("choose one mode"); session=true; mode="session"; continue; }
        if (!strcmp(opt,"--mtp")) { mtp=true; continue; }
        if (!strcmp(opt,"--thinking")) { thinking=true; continue; }
        if (!strcmp(opt,"--encode-only")) { encode_only=true; continue; }
        if (i+1==argc) die("missing value for %s",opt);
        const char *value=argv[++i];
        if (!strcmp(opt,"--chat") || !strcmp(opt,"--complete") || !strcmp(opt,"--tokens") ||
            !strcmp(opt,"--chat-file") || !strcmp(opt,"--complete-file")) {
            if (mode) die("choose one prompt mode"); mode=opt; input=value;
        } else if (!strcmp(opt,"--max-new")) { max_new=number(value,0,32768,opt); max_new_given=true; }
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
        else if (!strcmp(opt,"--profile")) profile_path=value;
        else if (!strcmp(opt,"--teacher-forced")) forced=number(value,1,32768,opt);
        else die("unknown Qwen3.8 option %s",opt);
    }
    /* No prompt: the interactive chat. It owns its turns, so the measurement
     * and replay options of a one-shot run do not apply. */
    const bool repl=!mode; if (repl) mode="repl";
    if (repl && (runs_path || profile_path || logits_path || encode_only || repeat!=1 || warmup || replay_chunk ||
                 stop_after!=UINT32_MAX || forced)) die("the interactive chat takes only --context, --chunk, --max-new, --mtp, --thinking, --token-ids");
    if (forced && (strcmp(mode,"--tokens") || mtp || session || stop_after!=UINT32_MAX || max_new_given))
        die("--teacher-forced needs --tokens and excludes --mtp, --session, --stop-after and --max-new");
    if (session && (runs_path || profile_path || logits_path || ids_path || encode_only || repeat!=1 || warmup || replay_chunk)) die("session owns its outputs and repetitions");
    if (!strcmp(mode,"--chat-file") || !strcmp(mode,"--complete-file")) {
        owned_input=read_prompt(input); input=owned_input;
        mode=!strcmp(mode,"--chat-file") ? "--chat" : "--complete";
    }
    if (thinking && strcmp(mode,"--chat") && !repl) die("--thinking requires --chat or the interactive chat");
    chunk=MIN(chunk,capacity);
    if (mtp && capacity<2) die("MTP requires context >=2");
    if (replay_chunk>chunk) die("replay chunk exceeds allocated chunk");
    FILE *runs=runs_path ? fopen(runs_path,"wx") : NULL;
    if (runs_path && !runs) die("runs output must be a new file");
    FILE *profile_csv=profile_path ? fopen(profile_path,"wx") : NULL;
    if (profile_path && !profile_csv) die("profile output must be a new file");
    struct sigaction action={0}; action.sa_handler=stop_generation; action.sa_flags=SA_RESTART; sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1,&action,NULL)) die("cannot install stop handler");
    @autoreleasepool {
        Q38Run run={.chunk=chunk,.mtp=mtp}; Q38Run *r=&run; Qwen38Error e;
        gguf_open(argv[2],&r->g);
        q38_check(qwen38_config_load(&r->g,&r->config,&e),&e);
        q38_check(qwen38_weights_describe(&r->g,r->config,&r->weights,&e),&e);
        q38_check(qwen38_tokenizer_load(&r->g,r->config,&r->tok,&e),&e);
        const Qwen38ConfigView *c=r->c=qwen38_config_view(r->config); Qwen38Tokenizer *tok=r->tok;
        if (capacity>c->max_positions) die("context exceeds model maximum");
        int32_t *tokens=malloc((size_t)capacity*4),*generated=malloc((size_t)capacity*4);
        if (!tokens || !generated) die("token allocation");
        size_t count=0;
        if (session || repl) { /* Requests arrive after the model is ready. */ }
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
            q38_check(qwen38_encode(tok,prompt,tokens,capacity,&count,&e),&e); free(rendered);
        }
        /* Teacher forcing: tokens[count..count+forced) stay in the array, outside the prompt. */
        if (forced) {
            if (forced>=count) die("--teacher-forced N must leave at least one prompt token");
            count-=forced; max_new=forced;
        }
        if (!session && !repl && (!count || max_new>capacity-count)) die("prompt and requested generation exceed context, or prompt is empty");
        if (encode_only) {
            fputs("{\"prompt_ids\":",stdout); ids_json(stdout,tokens,count);
            fprintf(stdout,",\"prompt_tokens\":%zu,\"context\":%u}\n",count,capacity);
            qwen38_tokenizer_free(tok); qwen38_weights_free(r->weights); qwen38_config_free(r->config); gguf_close(&r->g);
            free(tokens); free(generated); free(owned_input); if (runs) fclose(runs); if (profile_csv) fclose(profile_csv); return 0;
        }
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        if (!device) die("Metal device unavailable");
        NSError *error=nil; id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(metallib)] error:&error];
        if (!lib) die("cannot load %s: %s",metallib,error.localizedDescription.UTF8String);
        r->device=(__bridge void *)device; r->library=(__bridge void *)lib;
        r->store=gpu_weights_open(r->device,&r->g,0);
        q38_check(qwen38_weights_attach(r->weights,&r->g,gpu_weights_plan(r->store),&e),&e);
        if (profile_csv) q38_check(q38_profile_create(r->device,profile_csv,&r->profile,&e),&e);
        q38_run_graph(r,capacity); Qwen38GpuForward *graph=r->graph;
        fprintf(stderr,"Qwen3.8: %u layers, %zu prompt tokens, context %u, chunk %u; graph buffers %.1f MiB\n",
            c->trunk_blocks,count,capacity,chunk,(double)qwen38_gpu_forward_bytes(graph)/1048576.0);
        double load_s=q38_now()-entered;
        if (repl) q38_repl(r,thinking,max_new_given ? max_new : 0,ids_path);
        if (session) { runs=stdout; fprintf(stdout,"{\"event\":\"ready\",\"load_s\":%.9f,\"context\":%u}\n",load_s,capacity); fflush(stdout); }
        for (uint32_t repetition=0;!repl && (session || repetition<repeat+warmup);repetition++) {
        if (session) {
            int request=session_request(tok,tokens,capacity,&count,&max_new);
            if (!request) break;
            if (request<0) continue;
            if (request==2) {
                q38_check(qwen38_gpu_forward_reset(graph,&e),&e); qwen38_decoder_reset(r->decoder);
                fputs("{\"event\":\"reset\"}\n",stdout); fflush(stdout); continue;
            }
        }
        /* Warm-up repetitions write no profile rows: detached, the graph never splits. */
        if (r->profile) q38_check(q38g_forward_set_profile(graph,repetition<warmup ? NULL : r->profile,&e),&e);
        double request_start=q38_now(),reset_s=0;
        if (session || repetition) {
            q38_check(qwen38_gpu_forward_reset(graph,&e),&e); qwen38_decoder_reset(r->decoder);
            reset_s=q38_now()-request_start;
        }
        q38_stopped=0;
        if (session) { fprintf(stdout,"{\"event\":\"started\",\"prompt_tokens\":%zu}\n",count); fflush(stdout); }
        uint32_t current_chunk=repetition && replay_chunk ? replay_chunk : chunk;
        FILE *logfile=logits_path ? fopen(logits_path,"wb") : NULL;
        if (logits_path && !logfile) die("cannot open logits output");
        Q38Turn t; q38_turn(r,tokens,count,(Q38TurnOptions){.chunk=current_chunk,.max_new=max_new,
            .stop_after=stop_after,.forced=forced,.logfile=logfile,.session=session},generated,&t);
        size_t got=t.got;
        if (logfile && fclose(logfile)) die("logits close failed");
        if (ids_path) {
            FILE *out=fopen(ids_path,"w"); if (!out) die("cannot open token IDs output");
            fputs("{\"prompt_ids\":",out); ids_json(out,tokens,count); fputs(",\"generated_ids\":",out); ids_json(out,generated,got);
            if (forced) { fputs(",\"forced_ids\":",out); ids_json(out,tokens+count,forced); }
            fprintf(out,",\"context\":%u,\"chunk\":%u,\"greedy\":true}\n",capacity,chunk);
            if (fclose(out)) die("token IDs write failed");
        }
        fprintf(stderr,"Qwen3.8: generated %zu tokens%s\n",got,got && qwen38_token_is_stop(tok,generated[got-1]) ? " (stop)" : "");
        fprintf(stderr,"Qwen3.8 memory: {\"metal_allocated_bytes\":%zu,\"metal_recommended_bytes\":%zu,\"graph_buffer_bytes\":%zu}\n",
            (size_t)device.currentAllocatedSize,(size_t)device.recommendedMaxWorkingSetSize,qwen38_gpu_forward_bytes(graph));
        if (runs) {
            fprintf(runs,"{\"event\":\"done\",\"repetition\":%u,\"warmup\":%s,\"load_s\":%.9f,\"reset_s\":%.9f,"
                "\"prefill_s\":%.9f,\"decode_s\":%.9f,\"request_s\":%.9f,\"ttft_request_s\":",
                repetition,repetition<warmup ? "true":"false",load_s,reset_s,t.prefill_end-t.prefill_start,
                t.first_token>0 ? t.end-t.decode_start : 0,t.end-request_start);
            if (t.first_token>0) fprintf(runs,"%.9f",t.first_token-request_start); else fputs("null",runs);
            fprintf(runs,",\"first_token_from_entry_s\":");
            if (t.first_token>0 && !repetition) fprintf(runs,"%.9f",t.first_token-entered); else fputs("null",runs);
            fprintf(runs,",\"prompt_tokens\":%zu,\"generated_tokens\":%zu,\"decode_tokens\":%zu,"
                "\"context\":%u,\"chunk\":%u,\"stop_reason\":\"%s\",\"position\":%llu,"
                "\"metal_allocated_bytes\":%zu,\"metal_recommended_bytes\":%zu,\"graph_buffer_bytes\":%zu,\"generated_ids\":",
                count,got,forced ? got : got ? got-1 : 0,capacity,current_chunk,t.reason,
                (unsigned long long)qwen38_gpu_forward_info(graph).next_pos,
                (size_t)device.currentAllocatedSize,(size_t)device.recommendedMaxWorkingSetSize,qwen38_gpu_forward_bytes(graph));
            ids_json(runs,generated,got);
            if (forced) fprintf(runs,",\"mode\":\"teacher-forced\",\"forced_tokens\":%u",forced);
            fprintf(runs,",\"mtp\":{\"enabled\":%s,\"proposed\":%zu,\"accepted\":%zu,\"rollback_count\":%zu,"
                "\"target_decode_appends\":%zu,\"verified_tokens\":%zu,\"prefill_s\":%.9f,\"draft_s\":%.9f,"
                "\"verify_s\":%.9f,\"snapshot_s\":%.9f,\"replay_s\":%.9f,\"checkpoint_host_bytes\":%zu}",
                mtp ? "true":"false",t.proposed,t.accepted,t.rollback_count,t.target_appends,t.verified_tokens,
                t.mtp_prefill_s,t.draft_s,t.verify_s,t.snapshot_s,t.replay_s,qwen38_gpu_checkpoint_bytes(graph));
            stats_json(runs,"mtp_profile",t.mtp);
            stats_json(runs,"prefill_profile",t.prefill); stats_json(runs,"decode_profile",t.decode);
            fputs("}\n",runs); fflush(runs);
        }
        }
        qwen38_decoder_free(r->decoder); qwen38_gpu_forward_destroy(r->graph); qwen38_tokenizer_free(tok);
        if (!q38_profile_close(r->profile)) die("profile write failed");
        qwen38_weights_free(r->weights); gpu_weights_close(r->store); qwen38_config_free(r->config); gguf_close(&r->g);
        free(tokens); free(generated); free(r->logits);
    }
    free(owned_input); if (runs && runs!=stdout && fclose(runs)) die("runs close failed");
    return 0;
}
