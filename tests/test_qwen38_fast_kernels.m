/* R1 fast-kernel harness, native side. `selector` checks Q38_REF_KERNELS
 * parsing (no GPU). Run GPU modes only through qwen38_gpu_lock.py. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "qwen38_b_internal.h"
#include "qwen38_ref.h"
#include <assert.h>
#include <stdlib.h>

/* test_selector_parsing: pure parser cases, then once through the environment. */
static int selector_cases(void) {
    static const struct { const char *value, *family; bool want; } cases[] = {
        {NULL,"linear",false}, {"","linear",false}, {"0","argmax",false},
        {"1","linear",true}, {"1","argmax",true}, {"all","claim",true},
        {"linear,moe","linear",true}, {"linear,moe","moe",true}, {"linear,moe","indexed",false},
        {"linear","linear",true}, {"linear","barrier",false},
        /* R18: any unknown token (typo, number, empty) selects every oracle. */
        {"lineer","moe",true}, {"2","linear",true}, {"linear,,moe","indexed",true},
        {"linear,","fuse",true}, {",moe","argmax",true}, {"Linear","linear",true},
    };
    for (size_t i=0;i<sizeof cases/sizeof *cases;i++)
        if (q38_ref_parse(cases[i].value,cases[i].family)!=cases[i].want) {
            fprintf(stderr,"selector case %zu (\"%s\", %s) expected %d\n",i,
                    cases[i].value ? cases[i].value : "(unset)",cases[i].family,cases[i].want);
            return 1;
        }
    assert(setenv("Q38_REF_KERNELS","indexed",1)==0);
    assert(q38_ref_kernels("indexed") && !q38_ref_kernels("linear"));
    assert(unsetenv("Q38_REF_KERNELS")==0 && !q38_ref_kernels("indexed"));
    return 0;
}

int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"selector")) return selector_cases();
    fprintf(stderr,"usage: %s selector\n",argv[0]);
    return 2;
}
