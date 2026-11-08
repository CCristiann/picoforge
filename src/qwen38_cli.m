/* Ordinary greedy Qwen3.8 generation using A's tokenizer/template and G2. */
#import <Metal/Metal.h>
#include "common.h"
#include "qwen38_cli.h"
#include "qwen38_run.h"
#include "qwen38_expert_pool.h"
#include "model_select.h"
#include <fcntl.h>
#include <unistd.h>
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
        "       (MODEL: a path or a short name such as q4, see `picoforge models`; -p is --chat, no -p is the chat)\n"
        "       [--max-new N] [--context N] [--chunk N] [--thinking]\n"
        "       [--logits FILE.f32] [--token-ids FILE.json] [--metallib FILE]\n"
        "       [--chat-file FILE | --complete-file FILE] [--encode-only]\n"
        "       [--repeat N] [--warmup N] [--replay-chunk N] [--runs-jsonl FILE]\n"
        "       [--session] (resident local worker protocol)\n"
        "       [--stop-after N] (SIGUSR1 also stops cooperatively)\n"
        "       [--mtp] [--mtp-draft auto|0..3] (or $PICOFORGE_MTP_DRAFT; drafts per cycle, auto = cost model,\n"
        "        $PICOFORGE_MTP_COST=c2,c3,c4,delta[,mu]; exact greedy verification of up to 1+3 rows)\n"
        "       [--profile FILE.csv] (per-op GPU time of trunk appends; MTP appends are not profiled)\n"
        "       [--teacher-forced N] (--tokens only: prefill all but the last N IDs, append those one by one)\n"
        "       [--prefill-precision fp32|half] (or $PICOFORGE_PREFILL_PRECISION; half: prefill GEMMs on the\n"
        "        Neural Accelerators, half operands, FP32 accumulation; decode stays FP32)\n"
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
/* Defaults measured in R4 (M3, docs/specs/qwen38/R4-mtp.md): a verify of n rows
 * against one decode append (1.27/1.56/1.91 for 2/3/4 rows), one draft (0.095)
 * and the per-token predictor catch-up (0.03), all in decode appends. Priors
 * start optimistic so drafting is tried. */
void q38_policy_init(Q38Policy *p,const char *value,const char *costs) {
    *p=(Q38Policy){.fixed=-1,.cost={0,1,1.27,1.56,1.91},.delta=0.095,.mu=0.03,.alpha={0.9,0.9,0.9}};
    if (value && *value && strcmp(value,"auto")) p->fixed=(int)number(value,0,QWEN38_VERIFY_ROWS-1,"--mtp-draft");
    if (costs && *costs) {
        double v[5]={0,0,0,0,p->mu}; char tail; int n=sscanf(costs,"%lf,%lf,%lf,%lf,%lf%c",v,v+1,v+2,v+3,v+4,&tail);
        if ((n!=4 && n!=5) || !(v[0]>=1 && v[1]>=v[0] && v[2]>=v[1] && v[3]>=0 && v[4]>=0))
            die("PICOFORGE_MTP_COST must be c2,c3,c4,delta[,mu] with 1 <= c2 <= c3 <= c4 and delta, mu >= 0");
        p->cost[2]=v[0]; p->cost[3]=v[1]; p->cost[4]=v[2]; p->delta=v[3]; p->mu=v[4];
    }
}
/* k maximizing (1 + sum_j prod_{i<=j} alpha_i) / (cost[1+k] + mu + k*delta); ties keep the smaller k. */
size_t q38_policy_choose(const Q38Policy *p,size_t kmax) {
    if (p->fixed>=0) return MIN((size_t)p->fixed,kmax);
    size_t best=0; double best_v=1/(p->cost[1]+p->mu),tokens=1,chain=1;
    for (size_t k=1;k<=kmax;k++) {
        chain*=p->alpha[k-1]; tokens+=chain;
        double v=tokens/(p->cost[k+1]+p->mu+(double)k*p->delta);
        if (v>best_v*(1+1e-9)) { best=k; best_v=v; }
    }
    return best;
}
void q38_turn(Q38Run *r,const int32_t *tokens,size_t count,Q38TurnOptions o,int32_t *generated,Q38Turn *t) {
    const Qwen38ConfigView *c=r->c; Qwen38GpuForward *graph=r->graph; Qwen38Error e;
    float *logits=r->logits; Qwen38F32Mut *y=&r->y; *t=(Q38Turn){.reason="length"};
    size_t residual_offset=0,got=0;
    /* T15c: plain greedy reads back only the id (GPU argmax). Logits rows stay for --logits
     * and MTP, which verify against them. */
    const bool use_id=!o.logfile; int32_t next_id=0;
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
    } else {
    /* Decode. With MTP each cycle: catch the predictor up to the emitted id
     * (pend: kept draft tokens whose pairs are still due), draft k tokens (k from
     * the policy; chained after the first), verify [id, drafts] in one append,
     * keep the accepted prefix (no replay) and emit it. k=0 is an ordinary append;
     * the predictor still advances so later cycles can draft. */
    Q38Policy *pol=&r->policy; const uint32_t vocab=c->vocab_size;
    bool have_id=use_id; size_t pend_off=residual_offset,pend_n=0; int32_t pend[QWEN38_VERIFY_ROWS],probe=-1;
    uint64_t mtp_done=r->mtp ? qwen38_gpu_mtp_info(graph).next_pos : 0;
    for (uint32_t step=0;step<MAX(o.max_new,1u) && !q38_stopped;step++) {
        if (got==o.stop_after) { t->reason="cancelled"; break; }
        if (o.logfile && fwrite(logits,4,vocab,o.logfile)!=vocab) die("logits write failed");
        if (!o.max_new) break;
        int32_t id=have_id ? next_id : greedy(logits,vocab);
        if (probe>=0) { pol->alpha[0]+=((probe==id)-pol->alpha[0])/8; probe=-1; }
        generated[got++]=id;
        if (qwen38_token_is_stop(r->tok,id)) t->reason="stop";
        else emit(r->decoder,&id,&o);
        if (t->first_token==0) { t->first_token=q38_now(); t->decode_start=t->first_token; }
        if (!strcmp(t->reason,"stop")) break;
        if (got==o.stop_after) { t->reason="cancelled"; break; }
        if (step+1>=o.max_new) break;
        const uint64_t at=qwen38_gpu_forward_info(graph).next_pos;
        size_t kmax=0;
        if (r->mtp) {   /* drafts that can still be emitted, and verify rows that fit */
            size_t room=MIN((size_t)o.max_new-got,(size_t)o.stop_after-got);
            kmax=MIN((size_t)r->verify_rows-1,room ? room-1 : 0);
            kmax=MIN(kmax,(size_t)(r->capacity-at)-1);
        }
        size_t k=r->mtp ? q38_policy_choose(pol,kmax) : 0;
        bool probing=r->mtp && !k && kmax && pol->fixed<0 && ++pol->idle>=8;
        if (r->mtp && (k || kmax || o.keep_predictor)) {
            int32_t pairs[QWEN38_VERIFY_ROWS+1]; memcpy(pairs,pend,pend_n*sizeof *pend); pairs[pend_n]=id;
            Qwen38F32Mut row=*y; row.dim[0]=k || probing ? 1 : 0; double part=q38_now();
            q38_check(qwen38_gpu_mtp_rewind(graph,mtp_done,&e),&e);
            q38_check(qwen38_gpu_mtp(graph,pend_off,pairs,pend_n+1,row,&e),&e);
            t->draft_s+=q38_now()-part; accumulate(&t->mtp,graph);
            pend_n=0; mtp_done=qwen38_gpu_mtp_info(graph).next_pos;
            if (probing) { probe=greedy(logits,vocab); pol->idle=0; }
            if (q38_stopped) break;
        }
        t->depth[k]++;
        if (!k) {
            append(r,&id,1,"decode",use_id ? &next_id : NULL,&t->decode); have_id=use_id;
            t->target_appends++; pend_off=0; continue;
        }
        int32_t d[QWEN38_VERIFY_ROWS]; size_t nd=0; double part=q38_now();
        d[nd++]=greedy(logits,vocab);
        while (nd<k && !qwen38_token_is_stop(r->tok,d[nd-1])) {
            Qwen38F32Mut row=*y; row.dim[0]=1;
            q38_check(qwen38_gpu_mtp_chain(graph,d[nd-1],row,&e),&e); accumulate(&t->mtp,graph);
            d[nd++]=greedy(logits,vocab);
        }
        t->draft_s+=q38_now()-part;
        /* A proposed stop is checked from the row before it and never enters the trunk. */
        size_t live=nd; bool stop_draft=qwen38_token_is_stop(r->tok,d[nd-1]); if (stop_draft) live=nd-1;
        int32_t rows_ids[QWEN38_VERIFY_ROWS]={id}; memcpy(rows_ids+1,d,live*sizeof *d);
        size_t rows=1+live; y->dim[0]=rows; part=q38_now();
        if (rows>1) {
            q38_profile_begin_append(r->profile,"decode",at);
            q38_check(qwen38_gpu_forward_verify(graph,rows_ids,rows,*y,&e),&e); accumulate(&t->decode,graph);
        } else append(r,rows_ids,1,"decode",NULL,&t->decode);
        y->dim[0]=1; t->verify_s+=q38_now()-part; t->verified_tokens+=rows; t->target_appends++;
        size_t m=0; int32_t target=greedy(logits,vocab);
        while (m<live && target==d[m]) { m++; target=greedy(logits+m*vocab,vocab); }
        bool stop_ok=stop_draft && m==live && target==d[live];
        for (size_t j=0;j<nd && j<=m;j++) {   /* each draft whose predecessors were all accepted */
            bool ok=j<m || (j==live && stop_ok);
            t->proposed_at[j]++; t->accepted_at[j]+=ok; pol->alpha[j]+=(ok-pol->alpha[j])/8;
        }
        t->proposed+=nd; t->accepted+=m+stop_ok; if (m<live) t->rollback_count++;
        size_t keep=q38_stopped ? 1 : m+1;
        if (rows>1) { part=q38_now(); q38_check(qwen38_gpu_forward_keep(graph,keep,&e),&e); t->replay_s+=q38_now()-part; }
        r->last_count=keep; pend_off=0; have_id=false;
        if (q38_stopped) break;
        pend_n=m; memcpy(pend,d,m*sizeof *d);
        bool done=false;
        for (size_t j=0;j<m && !done;j++) {   /* row j's argmax is draft j */
            if (o.logfile && fwrite(logits+j*vocab,4,vocab,o.logfile)!=vocab) die("logits write failed");
            generated[got++]=d[j]; emit(r->decoder,&d[j],&o); step++;
            if (got==o.stop_after) { t->reason="cancelled"; done=true; }
        }
        if (done) break;
        if (m) memmove(logits,logits+m*vocab,(size_t)vocab*4);   /* row m: the next target token */
    }
    /* Leave the predictor at its confirmed position (a chained draft is speculative). */
    if (r->mtp && qwen38_gpu_forward_info(graph).valid && qwen38_gpu_mtp_info(graph).next_pos>mtp_done)
        q38_check(qwen38_gpu_mtp_rewind(graph,mtp_done,&e),&e);
    }
    t->end=q38_now(); if (q38_stopped) t->reason="cancelled";
    emit(r->decoder,NULL,&o); if (!o.session && !o.print) fputc('\n',stdout);
    t->got=got;
}
/* R5.1b: PICOFORGE_STREAM_EXPERTS=N keeps the trunk's routed experts on the SSD and N of them in a
 * pool (LRU over all layers); the MTP layer's experts stay mapped. The pool reads through its own
 * descriptor with F_NOCACHE, so a miss is a real SSD read and not a page-cache copy. */
/* R5.3: PICOFORGE_STREAM_EXPERTS wins (0 = resident). Otherwise a model whose mapped weights fit in
 * 90% of the GPU's recommended working set stays resident (the Q2: ~42 of 52 GiB), and one that does
 * not (the Q4: ~70 GiB) streams with a pool of 85% of that budget minus everything else. The sizes
 * come from the same plan the store will use, so what is counted is exactly what gets mapped. */
static uint32_t stream_slots(const Gguf *g,uint32_t trunk,uint64_t budget) {
    const char *v=getenv("PICOFORGE_STREAM_EXPERTS");
    if (v && *v) {
        char *end; unsigned long n=strtoul(v,&end,10);
        if (*end || n>UINT32_MAX-1) die("PICOFORGE_STREAM_EXPERTS must be a slot count (0 = resident)");
        return (uint32_t)n;
    }
    GgufPlan plan; gguf_plan_ex(g,16384,(size_t)1<<40,trunk,&plan);
    uint64_t rest=0,experts=0,first=0,count=0;
    for (size_t i=0;i<plan.weight_count;i++) rest+=plan.weights[i].tensor.bytes;
    for (size_t i=0;i<plan.streamed_count;i++) experts+=plan.streamed[i].bytes;
    for (int m=0;m<3;m++) { const GgufTensor *t=gguf_plan_streamed(&plan,0,m); if (t) { first+=t->bytes; count=t->shape[2]; } }
    gguf_plan_free(&plan);
    if (!count || rest+experts<=budget/10*9) return 0;
    const uint64_t pool=budget/20*17,per=first/count;
    if (pool<=rest+count*per) die("the model does not fit in GPU memory even with a minimal expert pool");
    const uint32_t slots=(uint32_t)MIN((pool-rest)/per,(uint64_t)trunk*count);
    fprintf(stderr,"expert streaming: automatic, %.1f GiB of weights > 90%% of %.1f GiB GPU budget "
        "(set PICOFORGE_STREAM_EXPERTS to override)\n",(double)(rest+experts)/1073741824.0,(double)budget/1073741824.0);
    return slots;
}
static void open_stream(Q38Run *r,const char *path,uint32_t slots) {
    const GgufPlan *plan=gpu_weights_plan(r->store); const uint32_t layers=r->c->trunk_blocks;
    Q38ExpertLayer *offsets=calloc(layers,sizeof *offsets); size_t bytes[3]={0}; uint32_t experts=0;
    if (!offsets) die("expert pool layer table");
    for (uint32_t l=0;l<layers;l++) for (int m=0;m<3;m++) {
        const GgufTensor *t=gguf_plan_streamed(plan,l,m); if (!t) die("layer %u has no streamed expert tensor %d",l,m);
        if (!l && !m) experts=(uint32_t)t->shape[2];
        if (t->shape[2]!=experts) die("expert count differs across layers");
        if (!l) bytes[m]=t->bytes/experts; else if (bytes[m]!=t->bytes/experts) die("expert size differs across layers");
        offsets[l].offset[m]=t->file_offset;
    }
    /* A prefill chunk may route to every expert of a layer, and all of them stay pinned until its kernels ran. */
    if (slots<experts) die("PICOFORGE_STREAM_EXPERTS=%u: a prefill chunk can pin all %u experts of a layer; use at least %u slots",slots,experts,experts);
    if ((r->pool_fd=open(path,O_RDONLY))<0 || fcntl(r->pool_fd,F_NOCACHE,1)) die("cannot open %s for expert streaming",path);
    if (!(r->pool=q38_pool_create(r->pool_fd,offsets,layers,experts,bytes,slots))) die("expert pool allocation (%u slots)",slots);
    free(offsets);
    const char *rv=getenv("PICOFORGE_STREAM_READERS");   /* R5.2a: parallel miss reads, 1 = serial */
    if (rv && *rv) { char *end; unsigned long n=strtoul(rv,&end,10); if (*end || !n || n>64) die("PICOFORGE_STREAM_READERS must be 1..64"); q38_pool_set_readers(r->pool,(uint32_t)n); }
    void *region[3]; size_t region_bytes[3];
    for (int m=0;m<3;m++) region[m]=q38_pool_region(r->pool,m,&region_bytes[m]);
    gpu_weights_attach_pool(r->store,region,region_bytes,slots);
    fprintf(stderr,"expert streaming: %u slots of %u x %u experts, pool %.1f MiB\n",slots,layers,experts,
        (double)(region_bytes[0]+region_bytes[1]+region_bytes[2])/1048576.0);
}
static void close_stream(Q38Run *r) {
    if (!r->pool) return;
    Q38PoolStats s=q38_pool_stats(r->pool);
    fprintf(stderr,"expert streaming: %llu hits, %llu misses (%.2f%%), %llu evictions, %.2f GiB read, %.3f s waiting in acquire\n",
        (unsigned long long)s.hits,(unsigned long long)s.misses,s.hits+s.misses ? 100.0*(double)s.misses/(double)(s.hits+s.misses) : 0.0,
        (unsigned long long)s.evictions,(double)s.bytes_read/1073741824.0,(double)s.acquire_ns*1e-9);
    q38_pool_destroy(r->pool); close(r->pool_fd); r->pool=NULL;
}
void q38_run_graph(Q38Run *r,uint32_t capacity) {
    Qwen38Error e; uint32_t chunk=MIN(r->chunk,capacity);
    if (r->graph) qwen38_gpu_forward_destroy(r->graph);
    free(r->logits); r->graph=NULL; r->capacity=capacity; r->last_count=0;
    q38_check(qwen38_gpu_forward_create(&r->g,r->config,r->weights,r->store,r->device,r->library,
        r->mtp ? MIN(MAX(chunk,(uint32_t)QWEN38_VERIFY_ROWS),capacity) : chunk,capacity,
        r->mtp ? MIN((uint32_t)QWEN38_VERIFY_ROWS,capacity) : 1,&r->graph,&e),&e);
    r->verify_rows=r->mtp ? MIN((uint32_t)QWEN38_VERIFY_ROWS,capacity) : 1;
    if (r->pool) qwen38_gpu_forward_set_pool(r->graph,r->pool);
    if (r->mtp) q38_check(qwen38_gpu_mtp_enable(r->graph,&e),&e);
    if (r->profile) q38_check(q38g_forward_set_profile(r->graph,r->profile,&e),&e);
    const uint32_t vocab=r->c->vocab_size; size_t bytes=(size_t)vocab*4*r->verify_rows;
    r->logits=malloc(bytes); if (!r->logits) die("logits allocation");
    r->y=(Qwen38F32Mut){.data=r->logits,.capacity_bytes=bytes,.rank=2,.dim={1,vocab},.stride_bytes={(size_t)vocab*4,4}};
    if (r->decoder) qwen38_decoder_reset(r->decoder);
    else q38_check(qwen38_decoder_create(r->tok,true,&r->decoder,&e),&e);
}
/* ds4-style short form (`./ds4 -p ...`): -m MODEL, -p TEXT, -c N, -n N, the rest
 * as in the long form. MODEL is a path or a short name ("q4"); without -m the
 * model comes from $PICOFORGE_MODEL, the only model in models/, or a prompt
 * (model_select.c). A bare ./picoforge is the chat. */
bool qwen38_short_form(int argc,char **argv) {
    static const char *const starts[]={"-m","-p","-c","-n","--mtp","--mtp-draft","--thinking","--prefill-precision",NULL};
    if (argc==1) return true;
    for (size_t i=0;starts[i];i++) if (!strcmp(argv[1],starts[i])) return true;
    return false;
}
int qwen38_short(int argc,char **argv) {
    const char *model=NULL; char **out=calloc((size_t)argc+3,sizeof *out); int n=3;
    if (!out) die("argument allocation");
    for (int i=1;i<argc;i++) {
        const char *a=argv[i],*as=!strcmp(a,"-p") ? "--chat" : !strcmp(a,"-c") ? "--context" : !strcmp(a,"-n") ? "--max-new" : NULL;
        if ((as || !strcmp(a,"-m")) && i+1==argc) die("missing value for %s",a);
        if (!strcmp(a,"-m")) model=argv[++i];
        else if (as) { out[n++]=(char *)as; out[n++]=argv[++i]; }
        else out[n++]=argv[i];
    }
    char *path=model_select(model,argv[0]);
    out[0]=argv[0]; out[1]="--qwen38"; out[2]=path;
    int rc=qwen38_cli(n,out); free(path); free(out); return rc;
}
/* `picoforge bench [short-form options]`: protocol defaults (3 warm-up, 20
 * measured, 128 new tokens), a fixed prompt unless one is given; generated
 * text is discarded, the summary goes to stdout. Later options win. */
static void discard(void *ctx,const uint8_t *s,size_t n) { (void)ctx; (void)s; (void)n; }
int qwen38_bench(int argc,char **argv) {
    static const char *const prompts[]={"-p","--chat","--chat-file","--complete","--complete-file","--tokens",NULL};
    bool has_prompt=false;
    for (int i=2;i<argc;i++) for (size_t j=0;prompts[j];j++) if (!strcmp(argv[i],prompts[j])) has_prompt=true;
    char **out=calloc((size_t)argc+12,sizeof *out); int n=0; if (!out) die("argument allocation");
    bool forced=false; for (int i=2;i<argc;i++) if (!strcmp(argv[i],"--teacher-forced")) forced=true;
    const char *defaults[]={"--bench","--warmup","3","--repeat","20","-n","128"};
    out[n++]=argv[0];   /* teacher forcing fixes its own length: no default -n */
    for (size_t i=0;i<sizeof defaults/sizeof *defaults-(forced ? 2 : 0);i++) out[n++]=(char *)defaults[i];
    if (!has_prompt) { out[n++]="-p"; out[n++]="Write a long story about a lighthouse keeper."; }
    for (int i=2;i<argc;i++) out[n++]=argv[i];
    int rc=qwen38_short(n,out); free(out); return rc;
}
int qwen38_cli(int argc,char **argv) {
    double entered=q38_now();
    if (argc==3 && !strcmp(argv[2],"--help")) { usage(); return 0; }
    if (argc<3) { usage(); return 2; }
    const char *input=NULL,*mode=NULL,*logits_path=NULL,*ids_path=NULL,*metallib="picoforge.metallib";
    const char *runs_path=NULL,*profile_path=NULL,*bench_csv=NULL; char *owned_input=NULL; double weight_mb=0;
    uint32_t capacity=8192,chunk=1024,max_new=32,repeat=1,warmup=0,replay_chunk=0,stop_after=UINT32_MAX,forced=0;
    bool thinking=false,encode_only=false,session=false,mtp=false,max_new_given=false,bench=false;
    const char *mtp_draft=getenv("PICOFORGE_MTP_DRAFT");
    { const char *p=getenv("PICOFORGE_PREFILL_PRECISION");
      if (p && *p && strcmp(p,"fp32") && strcmp(p,"half")) die("PICOFORGE_PREFILL_PRECISION must be fp32 or half"); }
    for (int i=3;i<argc;i++) {
        const char *opt=argv[i];
        if (!strcmp(opt,"--session")) { if (mode) die("choose one mode"); session=true; mode="session"; continue; }
        if (!strcmp(opt,"--mtp")) { mtp=true; continue; }
        if (!strcmp(opt,"--mtp-draft")) { if (i+1==argc) die("missing value for %s",opt); mtp=true; mtp_draft=argv[++i]; continue; }
        if (!strcmp(opt,"--thinking")) { thinking=true; continue; }
        if (!strcmp(opt,"--encode-only")) { encode_only=true; continue; }
        if (!strcmp(opt,"--bench")) { bench=true; continue; }
        if (i+1==argc) die("missing value for %s",opt);
        const char *value=argv[++i];
        if (!strcmp(opt,"--chat") || !strcmp(opt,"--complete") || !strcmp(opt,"--tokens") ||
            !strcmp(opt,"--chat-file") || !strcmp(opt,"--complete-file")) {
            if (mode) die("choose one prompt mode"); mode=opt; input=value;
        } else if (!strcmp(opt,"--max-new")) { max_new=number(value,0,32768,opt); max_new_given=true; }
        else if (!strcmp(opt,"--context")) capacity=number(value,1,32768,opt);
        else if (!strcmp(opt,"--prefill-precision")) {
            /* R3.5: one switch for flag and environment; the packed context
             * reads PICOFORGE_PREFILL_PRECISION when the graph is built. */
            if (strcmp(value,"fp32") && strcmp(value,"half")) die("--prefill-precision must be fp32 or half");
            setenv("PICOFORGE_PREFILL_PRECISION",value,1);
        }
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
        else if (!strcmp(opt,"--bench-csv")) bench_csv=value;
        else if (!strcmp(opt,"--weight-mb")) {
            char *end; errno=0; weight_mb=strtod(value,&end);
            if (errno || end==value || *end || !(weight_mb>0 && weight_mb<1e6)) die("--weight-mb must be a positive number of MB");
        }
        else die("unknown Qwen3.8 option %s",opt);
    }
    /* No prompt: the interactive chat. It owns its turns, so the measurement
     * and replay options of a one-shot run do not apply. */
    const bool repl=!mode; if (repl) mode="repl";
    if (repl && (runs_path || profile_path || logits_path || encode_only || repeat!=1 || warmup || replay_chunk ||
                 stop_after!=UINT32_MAX || forced)) die("the interactive chat takes only --context, --chunk, --max-new, --mtp, --thinking, --token-ids");
    if (forced && (strcmp(mode,"--tokens") || mtp || session || stop_after!=UINT32_MAX || max_new_given))
        die("--teacher-forced needs --tokens and excludes --mtp, --session, --stop-after and --max-new");
    if ((bench_csv || weight_mb>0) && !bench) die("--bench-csv and --weight-mb belong to bench");
    if (weight_mb>0 && mtp) die("--weight-mb assumes one weight read per token; MTP verifies two rows per read");
    if (bench && (session || repl || encode_only)) die("bench needs a prompt and measures a one-shot run");
    if (session && (runs_path || profile_path || logits_path || ids_path || encode_only || repeat!=1 || warmup || replay_chunk)) die("session owns its outputs and repetitions");
    if (!strcmp(mode,"--chat-file") || !strcmp(mode,"--complete-file")) {
        owned_input=read_prompt(input); input=owned_input;
        mode=!strcmp(mode,"--chat-file") ? "--chat" : "--complete";
    }
    if (thinking && strcmp(mode,"--chat") && !repl) die("--thinking requires --chat or the interactive chat");
    const uint32_t chunk_request=chunk; chunk=MIN(chunk,capacity);
    if (mtp && capacity<2) die("MTP requires context >=2");
    if (replay_chunk>chunk) die("replay chunk exceeds allocated chunk");
    FILE *runs=runs_path ? fopen(runs_path,"wx") : NULL;
    if (runs_path && !runs) die("runs output must be a new file");
    FILE *bench_out=bench_csv ? fopen(bench_csv,"wx") : NULL;
    if (bench_csv && !bench_out) die("bench CSV must be a new file");
    Q38BenchRow *bench_rows=bench ? calloc(repeat,sizeof *bench_rows) : NULL; size_t bench_n=0;
    if (bench && !bench_rows) die("bench allocation");
    FILE *profile_csv=profile_path ? fopen(profile_path,"wx") : NULL;
    if (profile_path && !profile_csv) die("profile output must be a new file");
    struct sigaction action={0}; action.sa_handler=stop_generation; action.sa_flags=SA_RESTART; sigemptyset(&action.sa_mask);
    if (sigaction(SIGUSR1,&action,NULL)) die("cannot install stop handler");
    @autoreleasepool {
        Q38Run run={.chunk=chunk_request,.mtp=mtp}; Q38Run *r=&run; Qwen38Error e;  /* chunk clamped per graph */
        q38_policy_init(&r->policy,mtp_draft,getenv("PICOFORGE_MTP_COST"));
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
        const uint32_t slots=stream_slots(&r->g,c->trunk_blocks,device.recommendedMaxWorkingSetSize);
        r->store=gpu_weights_open_ex(r->device,&r->g,0,slots ? c->trunk_blocks : 0);
        if (slots) open_stream(r,argv[2],slots);
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
            .stop_after=stop_after,.forced=forced,.logfile=logfile,.session=session,.print=bench ? discard : NULL},generated,&t);
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
        if (bench && repetition>=warmup) bench_rows[bench_n++]=(Q38BenchRow){count,forced ? got : got ? got-1 : 0,
            t.prefill_end-t.prefill_start,t.first_token>0 ? t.end-t.decode_start : 0,
            t.first_token>0 ? t.first_token-request_start : 0};
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
                "\"verify_s\":%.9f,\"snapshot_s\":%.9f,\"replay_s\":%.9f,\"checkpoint_host_bytes\":%zu,"
                "\"draft\":%d,\"depth\":[%zu,%zu,%zu,%zu],\"proposed_at\":[%zu,%zu,%zu],\"accepted_at\":[%zu,%zu,%zu]}",
                mtp ? "true":"false",t.proposed,t.accepted,t.rollback_count,t.target_appends,t.verified_tokens,
                t.mtp_prefill_s,t.draft_s,t.verify_s,t.snapshot_s,t.replay_s,qwen38_gpu_checkpoint_bytes(graph),
                r->policy.fixed,t.depth[0],t.depth[1],t.depth[2],t.depth[3],t.proposed_at[0],t.proposed_at[1],t.proposed_at[2],
                t.accepted_at[0],t.accepted_at[1],t.accepted_at[2]);
            stats_json(runs,"mtp_profile",t.mtp);
            stats_json(runs,"prefill_profile",t.prefill); stats_json(runs,"decode_profile",t.decode);
            fputs("}\n",runs); fflush(runs);
        }
        }
        if (bench) {
            printf("bench: %zu runs (+%u warm-up), prompt %zu tokens, context %u, chunk %u, mtp %s\n",
                   bench_n,warmup,count,capacity,chunk,mtp ? "on" : "off");
            q38_bench_report(bench_rows,bench_n,weight_mb);
            if (bench_out) { q38_bench_csv(bench_out,bench_rows,bench_n); if (fclose(bench_out)) die("bench CSV write failed"); }
            free(bench_rows);
        }
        qwen38_decoder_free(r->decoder); qwen38_gpu_forward_destroy(r->graph); qwen38_tokenizer_free(tok);
        if (!q38_profile_close(r->profile)) die("profile write failed");
        qwen38_weights_free(r->weights); gpu_weights_close(r->store); close_stream(r); qwen38_config_free(r->config); gguf_close(&r->g);
        free(tokens); free(generated); free(r->logits);
    }
    free(owned_input); if (runs && runs!=stdout && fclose(runs)) die("runs close failed");
    return 0;
}
