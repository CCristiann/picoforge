/* Interactive multi-turn chat, the shape of ds4's prompt (ds4 fc80bd6, MIT,
 * ds4_cli.c:run_repl/run_chat_turn): the conversation is an array of token
 * IDs, and the graph always holds a prefix transcript[0..next_pos). A new
 * turn renders and appends only its own suffix; a reply is never
 * re-tokenized, so the history is exactly what the model saw. The GDN state
 * cannot rewind to an arbitrary position, so a rollback is reset + replay. */
#include "qwen38_run.h"
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    Q38Run *r; int32_t *ids, *generated; size_t n;  /* transcript ids[0..n) */
    bool thinking; uint32_t max_new; int32_t im_end; const char *ids_path;
} Chat;

static void help(void) {
    puts("Commands:\n"
         "  /help          Show this help.\n"
         "  /quit, /exit   Leave (Ctrl+D too).\n"
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
    q38_turn(r,ch->ids+pos,total-pos,(Q38TurnOptions){.chunk=r->chunk<r->capacity ? r->chunk : r->capacity,.max_new=max_new,
        .stop_after=UINT32_MAX},ch->generated,&t);
    memcpy(ch->ids+total,ch->generated,t.got*sizeof *ch->ids); ch->n=total+t.got;
    double prefill=t.prefill_end-t.prefill_start,decode=t.end-t.decode_start;
    fprintf(stderr,"picoforge: prefill %.2f t/s, generation %.2f t/s, context %zu/%u%s\n",
        prefill>0 ? (double)(total-pos)/prefill : 0.0,decode>0 && t.got>1 ? (double)(t.got-1)/decode : 0.0,
        ch->n,r->capacity,strcmp(t.reason,"cancelled") ? "" : " [interrupted]");
    dump_ids(ch);
}
int q38_repl(Q38Run *r, bool thinking, uint32_t max_new, const char *ids_path) {
    Chat ch={.r=r,.thinking=thinking,.max_new=max_new,.ids_path=ids_path,.im_end=-1};
    ch.ids=malloc((size_t)r->capacity*sizeof *ch.ids); ch.generated=malloc((size_t)r->capacity*sizeof *ch.ids);
    if (!ch.ids || !ch.generated) die("transcript allocation");
    Qwen38Error e; size_t one=0; int32_t probe[2];
    if (!qwen38_encode(r->tok,(Qwen38Bytes){(const uint8_t *)"<|im_end|>",10},probe,2,&one,&e) && one==1) ch.im_end=probe[0];
    bool tty=isatty(STDIN_FILENO); char *line=NULL; size_t cap=0;
    if (tty) help();
    for (;;) {
        if (tty) { fputs("picoforge> ",stdout); fflush(stdout); }
        ssize_t got=getline(&line,&cap,stdin);
        if (got<0) break;
        if (got>(4<<20)) { fprintf(stderr,"picoforge: line longer than 4 MiB\n"); continue; }
        char *cmd=trim(line);
        if (!*cmd) continue;
        if (!strcmp(cmd,"/help")) help();
        else if (!strcmp(cmd,"/quit") || !strcmp(cmd,"/exit")) break;
        else if (*cmd=='/') fprintf(stderr,"picoforge: unknown command %s (try /help)\n",cmd);
        else { size_t added=user_turn(&ch,cmd); if (added) reply(&ch,added); }
    }
    free(line); free(ch.ids); free(ch.generated);
    return 0;
}
