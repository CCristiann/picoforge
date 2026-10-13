/* Bounded adapter to the pinned, unmodified DwarfStar CPU reference.
 * model_open opens its normal private mapping and binds weights, but skips
 * the eager POSIX_MADV_WILLNEED hint over the entire resident weight mapping.
 * The actual forward/MTP functions and arithmetic are upstream unchanged.
 * Disk-only n-grams are still unmapped by upstream model_open.
 * Usage: adapter MODEL; explicit DS4_QWEN4_FT_TOKENS/FT_OUT, optional MTP_OUT.
 */
#define DS4_NO_GPU
#include DS4_REFERENCE_SOURCE

int main(int argc,char **argv) {
    const char *list=getenv("DS4_QWEN4_FT_TOKENS"),*out=getenv("DS4_QWEN4_FT_OUT");
    if (argc!=2 || !list || !*list || !out || !*out) return 2;
    unsigned count=0;
    for (const char *p=list;*p;) {
        char *end; errno=0; long id=strtol(p,&end,10);
        if (errno || end==p || id<0 || id>INT_MAX || (*end && *end!=',') || ++count>256) return 2;
        if (*end==',' && !end[1]) return 2;
        p=*end ? end+1 : end;
    }
    ds4_model model; ds4_weights *weights=xcalloc(1,sizeof(*weights)); ds4_vocab vocab={0};
    model_open(&model,argv[1],false,false);
    config_validate_model(&model);
    if (!ds4_model_is_qwen4()) { model_close(&model); free(weights); return 2; }
    weights_bind(weights,&model,false,0,0,true,false);
    int ids[256]; uint32_t n=qwen4_parse_token_list(list,ids,256);
    for (uint32_t i=0;i<n;i++) if ((uint32_t)ids[i]>=DS4_N_VOCAB) { model_close(&model); free(weights); return 2; }
    vocab_load(&vocab,&model);
    fprintf(stderr,"Bounded CPU oracle: %u tokens; lazy private mapping; unchanged upstream FP32 reference\n",n);
    ds4_tokens prompt={.v=ids,.len=(int)n,.cap=256};
    int result=qwen4_first_token_test(&model,&vocab,weights,&prompt,false);
    vocab_free(&vocab); model_close(&model); free(weights);
    return result;
}
