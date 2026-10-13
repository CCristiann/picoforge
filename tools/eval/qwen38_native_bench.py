"""Resident warmup/repetition benchmark; optionally pair original and optimized QSA.

The parent acquires the repository GPU lock once. Every output is exclusive.
OS caches are uncontrolled; each mode loads once, resets per request, and keeps
all warmups in the raw evidence. Never label these process launches cold.
"""
import argparse
import json
from pathlib import Path
import platform
import random
import subprocess
from qwen38_native_suite import ROOT, dump, run_suite, require_lock
from summarize_qwen38_native import summarize


def run(a):
    require_lock(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    order=['cooperative','scalar'] if a.compare_scalar else ['cooperative']
    random.Random(3813).shuffle(order)
    metadata=dict(complete=False,order=order,platform=platform.platform(),
                  hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string','hw.memsize'],text=True).splitlines(),
                  thermal_before=subprocess.check_output(['pmset','-g','therm'],text=True),
                  repetitions=a.repeat,warmups=a.warmup,chunk=a.chunk,cases=a.cases,
                  cache_state='Uncontrolled OS file and shader caches; no cold-cache claim.')
    dump(out/'benchmark.json',metadata)
    for mode in order:
        run_suite(a.suite.resolve(),out/mode,a.cases,a.repeat,a.warmup,0,a.timeout,a.chunk,mode=='scalar')
    summary=summarize(out); dump(out/'summary.json',summary)
    if a.compare_scalar:
        identity={}
        for case in a.cases:
            rows=[[json.loads(l) for l in (out/mode/(case+'.runs.jsonl')).read_text().splitlines()] for mode in order]
            identity[case]=all(r['generated_ids']==rows[0][0]['generated_ids'] for mode in rows for r in mode)
        metadata['paired_greedy_identical']=identity
        if not all(identity.values()):
            dump(out/'benchmark.json',metadata); raise RuntimeError('paired greedy sequence changed')
    metadata.update(complete=True,thermal_after=subprocess.check_output(['pmset','-g','therm'],text=True))
    dump(out/'benchmark.json',metadata)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--cases',nargs='+',default=['ragionamento'])
    p.add_argument('--repeat',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--chunk',type=int,default=128);p.add_argument('--timeout',type=float,default=3600)
    p.add_argument('--compare-scalar',action='store_true')
    a=p.parse_args()
    if a.repeat<1 or a.warmup<1 or a.chunk<1 or a.timeout<=0:p.error('positive repetition, warmup, chunk and timeout required')
    run(a)
