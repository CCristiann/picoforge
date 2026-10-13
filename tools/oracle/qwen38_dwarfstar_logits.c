/* Observation harness only, linked to the pinned, unmodified DwarfStar build.
 * Never linked into picoforge. Run with the shared GPU lock and upstream cwd.
 * Usage: qwen38_dwarfstar_logits MODEL OUTPUT.f32 TOKEN... */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv) {
    if (argc<4) return 2;
    ds4_engine_options opts={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .n_threads=4,.context_size=8192,.prefill_chunk=1024,.power_percent=100,.quality=true};
    ds4_engine *e=NULL; ds4_session *s=NULL;
    if (ds4_engine_open(&e,&opts) || ds4_session_create(&s,e,8192)) return 1;
    int vocab=ds4_engine_vocab_size(e); float *logits=malloc((size_t)vocab*4);
    FILE *out=fopen(argv[2],"wb"); if (!logits || !out) return 1;
    ds4_tokens prefix={0}; char err[512];
    for (int i=3;i<argc;i++) {
        char *end; long id=strtol(argv[i],&end,10);
        if (*end || id<0 || id>=vocab) return 2;
        ds4_tokens_push(&prefix,(int)id);
        if (ds4_session_sync(s,&prefix,err,sizeof err)) { fprintf(stderr,"DwarfStar: %s\n",err); return 1; }
        if (ds4_session_copy_logits(s,logits,vocab)!=vocab) return 1;
        for (int j=0;j<vocab;j++) if (!isfinite(logits[j])) return 1;
        if (fwrite(logits,4,(size_t)vocab,out)!=(size_t)vocab) return 1;
    }
    if (fclose(out)) return 1;
    printf("{\"tokens\":%d,\"vocab\":%d,\"backend\":\"Metal\",\"quality\":true,\"context\":8192,\"prefill_chunk\":1024,\"mtp\":false}\n",argc-3,vocab);
    free(logits); ds4_tokens_free(&prefix); ds4_session_free(s); ds4_engine_close(e); return 0;
}
