"""Paired scalar/cooperative QSA timings with exact FP32 output comparison."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from qwen38_native_suite import ROOT, dump, sha, require_lock
from summarize_qwen38_native import distribution


def run(out,repeats):
    fd=require_lock();out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build/qwen38-attention-bench';lib=ROOT/'picoforge.metallib'
    meta=dict(complete=False,repetitions=repeats,warmup_gpu_s=.5,clock='Metal GPU timestamps and CLOCK_MONOTONIC wall',
              executable_sha256=sha(binary),metallib_sha256=sha(lib),platform=platform.platform(),
              source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'tools/probe/qwen38_attention_bench.m',ROOT/'src/qwen38_attention.metal')},
              thermal_before=subprocess.check_output(['pmset','-g','therm'],text=True),commands=[],results=[])
    dump(out/'report.json',meta)
    rng=random.Random(3813)
    for n in (64,256,512,1024,2048,8192):
        order=['scalar','cooperative'];rng.shuffle(order)
        result=dict(n=n,selected=min(n,2048),order=order)
        for mode in order:
            prefix=out/f'{n}-{mode}'
            cmd=[str(binary),str(lib),str(prefix),str(n),str(repeats),mode]
            meta['commands'].append(cmd)
            with prefix.with_suffix('.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,pass_fds=(fd,),stdout=log,stderr=log,check=True,timeout=180)
            rows=[json.loads(l) for l in prefix.with_suffix('.jsonl').read_text().splitlines()]
            measured=[r for r in rows if not r['warmup']]
            if len(measured)!=repeats:raise ValueError('wrong sample count')
            result[mode]=dict(gpu_s=distribution([r['gpu_s'] for r in measured]),
                              wall_s=distribution([r['wall_s'] for r in measured]),
                              output_sha256=sha(prefix.with_suffix('.f32')))
        result['bit_identical']=result['scalar']['output_sha256']==result['cooperative']['output_sha256']
        result['speedup']=result['scalar']['gpu_s']['median']/result['cooperative']['gpu_s']['median']
        meta['results'].append(result);dump(out/'report.json',meta)
        print(n,result['bit_identical'],result['speedup'],flush=True)
        if not result['bit_identical']:raise RuntimeError('FP32 output changed')
    meta['complete']=True;meta['thermal_after']=subprocess.check_output(['pmset','-g','therm'],text=True)
    dump(out/'report.json',meta)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--repeat',type=int,default=25)
    a=p.parse_args();run(a.output.resolve(),a.repeat)
