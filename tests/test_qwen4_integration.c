/* Public ABI seams on A's small fixture, not a full Qwen3.8 forward pass. */
#include "qwen4.h"
#include "qwen4_attention.h"
#include "qwen4_moe.h"
#include "qwen4_ngram.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static Qwen4F32 ro(Qwen4F32Mut v) {
    Qwen4F32 out={.data=v.data,.capacity_bytes=v.capacity_bytes,.rank=v.rank};
    for (size_t i=0;i<v.rank;++i) { out.dim[i]=v.dim[i]; out.stride_bytes[i]=v.stride_bytes[i]; }
    return out;
}
static Qwen4F32Mut matrix(float *p, size_t rows, size_t cols) {
    return (Qwen4F32Mut){.data=p,.capacity_bytes=rows*cols*4,.rank=2,
        .dim={rows,cols},.stride_bytes={cols*4,4}};
}
static Qwen4F32Mut groups(float *p) {
    return (Qwen4F32Mut){.data=p,.capacity_bytes=512*4,.rank=3,
        .dim={1,2,256},.stride_bytes={512*4,256*4,4}};
}
static void packed_router(const Qwen4ConfigView *config, Qwen4Weights *weights) {
    const Qwen4Tensor *tensor;
    assert(qwen4_weights_get(weights,0,QW4_MOE_ROUTER,&tensor,NULL)==QW4_OK);
    assert(config->moe.hidden_size==256 && config->moe.experts==4 && config->moe.top_k==2);
    float data[4*256]={0}, x[256]={1}, logits[4], probabilities[4], selected[2];
    uint32_t ids[2];
    for (size_t e=0;e<4;++e) data[e*256]=(float)e;
    QW4PackedCpuView packed={.span={(const unsigned char *)data,sizeof data}};
    assert(qw4_packed_layout_init(&tensor->storage,tensor->logical_shape[0],&packed.layout,NULL)==QW4_OK);
    assert(qw4_packed_matmul(&packed,0,ro(matrix(x,1,256)),matrix(logits,1,4),NULL)==QW4_OK);
    for (size_t e=0;e<4;++e) assert(logits[e]==(float)e);
    Qwen4MoeRoutes routes={.ids={ids,sizeof ids,1,2,8},.weights=matrix(selected,1,2)};
    assert(qwen4_moe_route(&config->moe,ro(matrix(logits,1,4)),routes,matrix(probabilities,1,4),NULL)==QW4_OK);
    assert(ids[0]==3 && ids[1]==2);
    assert(fabsf(selected[0]-0.7310585786f)<1e-6f && fabsf(selected[1]-0.2689414214f)<1e-6f);
}
static void normalization_gate(void) {
    float key[512]={0}, query[512], normalized[512], gains[256], value[256], gate[2], out[512];
    for (size_t i=0;i<512;++i) query[i]=1;
    for (size_t i=0;i<256;++i) { gains[i]=2; value[i]=(float)i-128; }
    Qwen4F32 gain={.data=gains,.capacity_bytes=sizeof gains,.rank=1,.dim={256},.stride_bytes={4}};
    assert(qwen4_rmsnorm_groups(ro(groups(query)),gain,groups(normalized),1e-6f,NULL)==QW4_OK);
    for (size_t i=0;i<512;++i) assert(fabsf(normalized[i]-1.999999f)<2e-6f);
    assert(qwen4_ple_gate(ro(groups(key)),ro(groups(normalized)),ro(matrix(value,1,256)),
        matrix(gate,1,2),groups(out),NULL)==QW4_OK);
    assert(gate[0]==0.5f && gate[1]==0.5f);
    for (size_t i=0;i<512;++i) assert(out[i]==value[i%256]*0.5f);
}
int main(int argc, char **argv) {
    assert(argc==2); Gguf g; gguf_open(argv[1],&g);
    Qwen4Config *config; Qwen4Weights *weights;
    assert(qwen4_config_load(&g,&config,NULL)==QW4_OK);
    assert(qwen4_weights_describe(&g,config,&weights,NULL)==QW4_OK);
    const Qwen4ConfigView *view=qwen4_config_view(config);
    packed_router(view,weights); normalization_gate();
    const Qwen4NgramConfig *ngram;
    assert(qwen4_config_ngram(config,1,&ngram,NULL)==QW4_OK);
    int32_t prev[]={511,511}, tokens[]={1}, next[2]; uint64_t rows[4];
    assert(qwen4_ngram_ids(ngram,prev,2,tokens,1,rows,4,next,2,NULL)==QW4_OK);
    const uint64_t expected[]={1,8,19,26};
    for (size_t i=0;i<4;++i) assert(rows[i]==expected[i]);
    assert(next[0]==1 && next[1]==511);
    Qwen4AttentionPlan plan; Qwen4AttentionState *state; Qwen4AttentionWorkspace *work;
    assert(qwen4_attention_plan(&view->attention,32,4,&plan,NULL)==QW4_OK);
    Qwen4StateKey key={config,1,1};
    assert(qwen4_attention_create(&view->attention,&plan,key,plan.cache_bytes+plan.workspace_bytes,
        &state,&work,NULL)==QW4_OK);
    Qwen4AttentionTicket *ticket;
    assert(qwen4_attention_begin(state,0,0,1,&ticket,NULL)==QW4_OK);
    assert(qwen4_attention_finish(ticket,true,NULL)==QW4_OK);
    assert(qwen4_attention_reset(state,2,NULL)==QW4_OK);
    assert(qwen4_attention_state_info(state).next_pos==0);
    assert(qwen4_attention_destroy(state,work,NULL)==QW4_OK);
    qwen4_weights_free(weights); qwen4_config_free(config); gguf_close(&g);
    puts("PASS: A catalog -> B projection -> F routing; C normalization -> E gate; A -> D/E config");
    return 0;
}
