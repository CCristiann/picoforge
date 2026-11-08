"""P4 summary: prefill per run, profile top ops, before/after logit bit-identity."""
import json,csv,collections,glob,os,sys,numpy as np
D=sys.argv[1].rstrip('/')+'/'; L=sys.argv[2].rstrip('/')+'/'
for n in sorted(os.listdir(D)):
    if not os.path.exists(D+n+'/runs.jsonl'): continue
    r=[json.loads(l) for l in open(D+n+'/runs.jsonl') if '"done"' in l][-1]
    extra=''
    if n.startswith('profile'):
        agg=collections.defaultdict(float)
        for row in csv.DictReader(open(D+n+'/profile.csv')):
            if row['phase']=='prefill': agg[row['op']]+=float(row['gpu_us'])
        extra=f" | sum {sum(agg.values())/1e6:.2f}s "+" ".join(f"{k} {v/1e6:.2f}" for k,v in sorted(agg.items(),key=lambda x:-x[1])[:6])
    print(f"{n:22s} prefill {r['prefill_s']:6.2f}s ({r['prompt_tokens']/r['prefill_s']:.0f} t/s){extra}")
if os.path.isdir(L):
    for a in sorted(glob.glob(L+'*-before.f32')):
        b=a.replace('-before','-after'); x,y=np.fromfile(a,np.uint32),np.fromfile(b,np.uint32)
        print(os.path.basename(a)[:-11], 'logits bit-identical' if x.size==y.size and (x==y).all() else 'logits DIFFER', x.size)
