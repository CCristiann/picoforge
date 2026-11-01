/* `picoforge bench`: the measured repetitions of an ordinary run, reduced to
 * median and p10/p90 (docs/BENCHMARKS.md protocol: >= 3 warm-up, >= 20
 * measured). Raw rows go to an optional CSV so nothing is only a summary. */
#include "qwen38_run.h"
#include <stdlib.h>

static int ascending(const void *a, const void *b) {
    double x=*(const double *)a,y=*(const double *)b;
    return (x>y)-(x<y);
}
/* Linear interpolation between order statistics (numpy's default). */
static double quantile(const double *values, size_t n, double q) {
    double *v=malloc(n*sizeof *v); if (!v) die("quantile allocation");
    memcpy(v,values,n*sizeof *v); qsort(v,n,sizeof *v,ascending);
    double at=q*(double)(n-1); size_t i=(size_t)at; double frac=at-(double)i;
    double out=i+1<n ? v[i]+(v[i+1]-v[i])*frac : v[i];
    free(v); return out;
}
void q38_bench_report(const Q38BenchRow *rows, size_t n, double weight_mb) {
    if (!n) return;
    double *prefill=malloc(n*sizeof *prefill),*decode=malloc(n*sizeof *decode),*ttft=malloc(n*sizeof *ttft);
    if (!prefill || !decode || !ttft) die("bench allocation");
    for (size_t i=0;i<n;i++) {
        prefill[i]=rows[i].prefill_s>0 ? (double)rows[i].prompt_tokens/rows[i].prefill_s : 0;
        decode[i]=rows[i].decode_s>0 ? (double)rows[i].decode_tokens/rows[i].decode_s : 0;
        ttft[i]=rows[i].ttft_s;
    }
    printf("%-16s %10s %10s %10s\n","","median","p10","p90");
    printf("%-16s %10.2f %10.2f %10.2f\n","prefill tok/s",quantile(prefill,n,.5),quantile(prefill,n,.1),quantile(prefill,n,.9));
    printf("%-16s %10.2f %10.2f %10.2f\n","decode tok/s",quantile(decode,n,.5),quantile(decode,n,.1),quantile(decode,n,.9));
    printf("%-16s %10.4f %10.4f %10.4f\n","ttft s",quantile(ttft,n,.5),quantile(ttft,n,.1),quantile(ttft,n,.9));
    /* Decode reads every active weight once per token: GB/s = bytes x tok/s.
     * The bytes come from a measured profile summary, never from here. */
    if (weight_mb>0) {
        double gbps=weight_mb*quantile(decode,n,.5)/1000.0;
        printf("decode weight traffic %.1f GB/s at the median, %.1f%% of 307 GB/s (%.2f MB/token given)\n",
               gbps,100.0*gbps/307.0,weight_mb);
    }
    free(prefill); free(decode); free(ttft);
}
void q38_bench_csv(FILE *f, const Q38BenchRow *rows, size_t n) {
    fputs("repetition,prompt_tokens,decode_tokens,prefill_s,decode_s,ttft_s\n",f);
    for (size_t i=0;i<n;i++)
        fprintf(f,"%zu,%zu,%zu,%.9f,%.9f,%.9f\n",i,rows[i].prompt_tokens,rows[i].decode_tokens,
                rows[i].prefill_s,rows[i].decode_s,rows[i].ttft_s);
}
