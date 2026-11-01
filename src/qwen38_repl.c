/* Interactive multi-turn chat, the shape of ds4's prompt (ds4 fc80bd6, MIT,
 * ds4_cli.c:run_repl/run_chat_turn): the conversation is an array of token
 * IDs, and the graph always holds a prefix transcript[0..next_pos). A new
 * turn renders and appends only its own suffix; a reply is never
 * re-tokenized, so the history is exactly what the model saw. The GDN state
 * cannot rewind to an arbitrary position, so a rollback is reset + replay. */
#include "qwen38_run.h"
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    Q38Run *r; int32_t *ids, *generated; size_t n;  /* transcript ids[0..n) */
    bool thinking; uint32_t max_new; int32_t im_end; const char *ids_path;
} Chat;

/* Ctrl+C only raises the cooperative stop flag (checked between appends).
 * SA_RESTART keeps stdout writes and the line read free of EINTR; at the
 * prompt the terminal already drops the typed line, we just redraw. */
static volatile sig_atomic_t at_prompt;
static void interrupt(int signum) {
    (void)signum; q38_stopped=1;
    if (at_prompt) { ssize_t ignored=write(STDOUT_FILENO,"\npicoforge> ",12); (void)ignored; }
}
static void help(void) {
    puts("Commands:\n"
         "  /help          Show this help.\n"
         "  /read FILE     Send a text file as your message.\n"
         "  /ctx N         New context size (clears the conversation).\n"
         "  /reset         Start a new conversation.\n"
         "  /think, /nothink  Reasoning on or off for the next turns.\n"
         "  /quit, /exit   Leave (Ctrl+D too).\n"
         "  Ctrl+C         Stop the reply and come back here.\n"
         "Anything else is your next chat message.");
}
static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *end=s+strlen(s);
    while (end>s && isspace((unsigned char)end[-1])) *--end=0;
    return s;
}
/* The test oracle: the whole conversation as IDs, and how much the graph holds. */
static void dump_ids(const Chat *ch) {
    if (!ch->ids_path) return;
    FILE *f=fopen(ch->ids_path,"w"); if (!f) die("cannot open token IDs output");
    fputs("{\"ids\":[",f);
    for (size_t i=0;i<ch->n;i++) fprintf(f,"%s%d",i ? "," : "",ch->ids[i]);
    fprintf(f,"],\"kv\":%llu}\n",(unsigned long long)qwen38_gpu_forward_info(ch->r->graph).next_pos);
    if (fclose(f)) die("token IDs write failed");
}
/* Bytes of the next user turn, appended to ids[n..). The first turn is the
 * full template; later ones close the previous reply and render one user
 * message with no system block (effort "medium" emits none, the generation
 * prompt is the same). Returns the number of new IDs, 0 if they do not fit. */
static size_t user_turn(Chat *ch, const char *text) {
    Q38Run *r=ch->r; Qwen38Error e; size_t len=0,count=0;
    Qwen38Json role={.kind=Q38_JSON_STRING,.as.string={(const uint8_t *)"user",4}};
    Qwen38Json content={.kind=Q38_JSON_STRING,.as.string={(const uint8_t *)text,strlen(text)}};
    Qwen38JsonMember fields[]={{{(const uint8_t *)"role",4},&role},{{(const uint8_t *)"content",7},&content}};
    Qwen38Json message={.kind=Q38_JSON_OBJECT,.as.object={fields,2}};
    Qwen38Json messages={.kind=Q38_JSON_ARRAY,.as.array={&message,1}};
    Qwen38ChatOptions o={.enable_thinking=ch->thinking ? Q38_TRUE : Q38_FALSE,.add_generation_prompt=true,
        .reasoning_effort=ch->n && ch->thinking ? Q38_EFFORT_MEDIUM : Q38_EFFORT_OMITTED};
    const char *close=!ch->n ? "" : ch->ids[ch->n-1]==ch->im_end ? "\n" : "<|im_end|>\n";
    size_t skip=strlen(close);
    if (qwen38_chat_render(r->tok,&messages,NULL,o,NULL,0,&len,&e)) { fprintf(stderr,"picoforge: %s\n",e.detail); return 0; }
    uint8_t *bytes=malloc(skip+len+1); if (!bytes) die("render allocation");
    memcpy(bytes,close,skip);
    q38_check(qwen38_chat_render(r->tok,&messages,NULL,o,bytes+skip,len,&len,&e),&e);
    Qwen38Status rc=qwen38_encode(r->tok,(Qwen38Bytes){bytes,skip+len},ch->ids+ch->n,r->capacity-ch->n,&count,&e);
    free(bytes);
    if (rc || count>=r->capacity-ch->n) {
        fprintf(stderr,"picoforge: the context is full (%zu of %u tokens)\n",ch->n,r->capacity);
        return 0;
    }
    return count;
}
/* Reply text: the <think>/</think> tags are dropped and the reasoning is
 * grey on a terminal (idea from ds4 fc80bd6, MIT, ds4_cli.c:token_printer_process).
 * A tag can arrive split across tokens, so a possible prefix waits. */
typedef struct { bool color, in_think, newline; char pending[8]; size_t n; } Printer;
static const char think_open[]="<think>",think_close[]="</think>";
static bool tag_prefix(const char *p, size_t n) {
    return (n<=7 && !memcmp(p,think_open,n)) || (n<=8 && !memcmp(p,think_close,n));
}
static void shade(Printer *p, bool think) {
    if (p->color && think!=p->in_think) fputs(think ? "\x1b[90m" : "\x1b[0m",stdout);
    p->in_think=think;
}
static void print_text(void *ctx, const uint8_t *s, size_t len) {
    Printer *p=ctx;
    for (size_t i=0;i<len;i++) {
        p->pending[p->n++]=(char)s[i];
        if (p->n==7 && !memcmp(p->pending,think_open,7)) { shade(p,true); p->n=0; }
        else if (p->n==8 && !memcmp(p->pending,think_close,8)) {
            shade(p,false); if (!p->newline) fputc('\n',stdout);
            p->newline=true; p->n=0;
        }
        while (p->n && !tag_prefix(p->pending,p->n)) {
            p->newline=p->pending[0]=='\n'; fputc(p->pending[0],stdout); memmove(p->pending,p->pending+1,--p->n);
        }
    }
}
static void print_end(Printer *p) { fwrite(p->pending,1,p->n,stdout); p->n=0; shade(p,false); fflush(stdout); }
/* Prefill what the graph does not hold yet, then decode the reply. */
static void reply(Chat *ch, size_t added) {
    Q38Run *r=ch->r; Qwen38Error e; size_t total=ch->n+added;
    size_t pos=(size_t)qwen38_gpu_forward_info(r->graph).next_pos;
    /* Pair the MTP predictor's pending rows (the last append's) with the IDs
     * that follow them; anything older than that append means replay. */
    size_t mpos=r->mtp ? (size_t)qwen38_gpu_mtp_info(r->graph).next_pos : pos;
    if (pos>ch->n || mpos+r->last_count<pos) {
        q38_check(qwen38_gpu_forward_reset(r->graph,&e),&e); pos=mpos=0; r->last_count=0;
    }
    if (mpos<pos) {
        Qwen38F32Mut empty=r->y; empty.dim[0]=0;
        q38_check(qwen38_gpu_mtp(r->graph,mpos-(pos-r->last_count),ch->ids+mpos+1,pos-mpos,empty,&e),&e);
    }
    size_t room=r->capacity-total; uint32_t max_new=ch->max_new && ch->max_new<room ? ch->max_new : (uint32_t)room;
    Q38Turn t; q38_stopped=0; qwen38_decoder_reset(r->decoder);
    Printer printer={.color=isatty(STDOUT_FILENO)}; shade(&printer,ch->thinking);
    q38_turn(r,ch->ids+pos,total-pos,(Q38TurnOptions){.chunk=r->chunk<r->capacity ? r->chunk : r->capacity,.max_new=max_new,
        .stop_after=UINT32_MAX,.print=print_text,.print_ctx=&printer},ch->generated,&t);
    print_end(&printer); fputc('\n',stdout);
    /* Stopped before the first token: drop the user turn too (as ds4). The
     * graph is then ahead of the transcript, so the next turn replays. */
    bool cancelled=!strcmp(t.reason,"cancelled");
    memcpy(ch->ids+total,ch->generated,t.got*sizeof *ch->ids);
    if (t.got || !cancelled) ch->n=total+t.got;
    double prefill=t.prefill_end-t.prefill_start,decode=t.end-t.decode_start;
    fprintf(stderr,"picoforge: prefill %.2f t/s, generation %.2f t/s, context %zu/%u%s\n",
        prefill>0 ? (double)(total-pos)/prefill : 0.0,decode>0 && t.got>1 ? (double)(t.got-1)/decode : 0.0,
        ch->n,r->capacity,cancelled ? " [interrupted]" : "");
    dump_ids(ch);
}
/* Text file as one message: bounded, NUL-free (UTF-8 is checked by the
 * renderer). Errors are reported, never fatal. */
static char *read_file(const char *path) {
    FILE *f=fopen(path,"rb");
    if (!f) { fprintf(stderr,"picoforge: cannot open %s: %s\n",path,strerror(errno)); return NULL; }
    size_t limit=4u<<20; char *text=malloc(limit+1); if (!text) die("file allocation");
    size_t n=fread(text,1,limit+1,f); bool bad=ferror(f) || n>limit || memchr(text,0,n); fclose(f);
    if (bad) { fprintf(stderr,"picoforge: %s must be NUL-free text <= 4 MiB\n",path); free(text); return NULL; }
    text[n]=0; return text;
}
static void forget(Chat *ch) {
    Qwen38Error e; q38_check(qwen38_gpu_forward_reset(ch->r->graph,&e),&e);
    ch->n=0; ch->r->last_count=0; dump_ids(ch);
}
/* A new graph for N tokens. The old one goes first, so peak memory does not
 * double; the model weights stay mapped. */
static void resize(Chat *ch, const char *arg) {
    char *end; errno=0; unsigned long n=strtoul(arg,&end,10);
    uint32_t top=ch->r->c->max_positions<32768 ? ch->r->c->max_positions : 32768;
    if (errno || end==arg || *end || n<2 || n>top) { fprintf(stderr,"picoforge: /ctx needs N in [2,%u]\n",top); return; }
    q38_run_graph(ch->r,(uint32_t)n);
    int32_t *ids=realloc(ch->ids,n*sizeof *ids); if (!ids) die("transcript allocation"); ch->ids=ids;
    ids=realloc(ch->generated,n*sizeof *ids); if (!ids) die("transcript allocation"); ch->generated=ids;
    ch->n=0; dump_ids(ch);
    printf("Context %lu tokens, graph buffers %.1f MiB. New conversation.\n",n,
        (double)qwen38_gpu_forward_bytes(ch->r->graph)/1048576.0);
}
int q38_repl(Q38Run *r, bool thinking, uint32_t max_new, const char *ids_path) {
    Chat ch={.r=r,.thinking=thinking,.max_new=max_new,.ids_path=ids_path,.im_end=-1};
    ch.ids=malloc((size_t)r->capacity*sizeof *ch.ids); ch.generated=malloc((size_t)r->capacity*sizeof *ch.ids);
    if (!ch.ids || !ch.generated) die("transcript allocation");
    Qwen38Error e; size_t one=0; int32_t probe[2];
    if (!qwen38_encode(r->tok,(Qwen38Bytes){(const uint8_t *)"<|im_end|>",10},probe,2,&one,&e) && one==1) ch.im_end=probe[0];
    struct sigaction action={0},old; action.sa_handler=interrupt; action.sa_flags=SA_RESTART;
    sigemptyset(&action.sa_mask); if (sigaction(SIGINT,&action,&old)) die("cannot install Ctrl+C handler");
    bool tty=isatty(STDIN_FILENO); char *line=NULL; size_t cap=0;
    if (tty) help();
    for (;;) {
        if (tty) { fputs("picoforge> ",stdout); fflush(stdout); }
        at_prompt=tty; errno=0;
        ssize_t got=getline(&line,&cap,stdin);
        at_prompt=0;
        if (got<0 && errno==EINTR) { clearerr(stdin); continue; }
        if (got<0) break;
        if (got>(4<<20)) { fprintf(stderr,"picoforge: line longer than 4 MiB\n"); continue; }
        char *cmd=trim(line);
        if (!*cmd) continue;
        if (!strcmp(cmd,"/help")) help();
        else if (!strcmp(cmd,"/quit") || !strcmp(cmd,"/exit")) break;
        else if (!strcmp(cmd,"/think") || !strcmp(cmd,"/nothink")) {
            ch.thinking=!strcmp(cmd,"/think"); printf("Thinking %s.\n",ch.thinking ? "on" : "off");
        }
        else if (!strcmp(cmd,"/reset")) { forget(&ch); puts("New conversation."); }
        else if (!strncmp(cmd,"/ctx",4) && (!cmd[4] || isspace((unsigned char)cmd[4]))) resize(&ch,trim(cmd+4));
        else if (!strncmp(cmd,"/read",5) && (!cmd[5] || isspace((unsigned char)cmd[5]))) {
            char *text=*trim(cmd+5) ? read_file(trim(cmd+5)) : NULL;
            if (!cmd[5]) fprintf(stderr,"picoforge: /read needs a file path\n");
            size_t added=text ? user_turn(&ch,text) : 0;
            if (added) reply(&ch,added);
            free(text);
        }
        else if (*cmd=='/') fprintf(stderr,"picoforge: unknown command %s (try /help)\n",cmd);
        else { size_t added=user_turn(&ch,cmd); if (added) reply(&ch,added); }
    }
    if (tty) fputc('\n',stdout);
    sigaction(SIGINT,&old,NULL); free(line); free(ch.ids); free(ch.generated);
    return 0;
}
