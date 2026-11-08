import csv,collections,sys,numpy as np
D='bench/qwen38_p3_20261108/'+sys.argv[1]+'/'
base=None
for name in sys.argv[2:]:
    rows=list(csv.DictReader(open(D+name+'/profile.csv')))
    agg=collections.defaultdict(float); tot=0
    for r in rows:
        if r['phase']=='prefill': agg[r['op']]+=float(r['gpu_us']); tot+=float(r['gpu_us'])
    top=sorted(agg.items(),key=lambda x:-x[1])[:6]
    lg=np.fromfile(D+name+'/logits.f32',dtype=np.float32)
    if base is None: base=lg
    d=np.abs(lg[:len(base)]-base[:len(lg)]).max()
    print(f"{name:14s} prefill {tot/1000:7.1f} ms | maxdiff {d:.2e} | "+"  ".join(f"{k} {v/1000:.0f}" for k,v in top))
