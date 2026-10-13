"""Independent FP32 gate on bounded continuations of the frozen native suite.

The DwarfStar CPU reference consumes the exact prompt + native continuation;
all sampled logit rows, greedy IDs and margins are checked (frozen 1e-3 gate).
Raw logits remain in the exclusive output directory, with SHA256 in the report.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from qwen38_native_suite import ROOT, ENGINE, dump, sha, require_lock
from run_observed import run as observe
sys.path.insert(0,str(ROOT/'tools'))
from qwen38_g3 import logits_report, REVISION, GATE


def run(suite_path, out, ds):
    fd=require_lock()
    if subprocess.check_output(['git','-C',str(ds),'rev-parse','HEAD'],text=True).strip()!=REVISION:
        raise ValueError('reference revision differs')
    suite=json.loads(suite_path.read_text()); out.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,executable_sha256=sha(ENGINE),metallib_sha256=sha(ROOT/'picoforge.metallib'),gate=GATE,dwarfstar_revision=REVISION,source_sha256=sha(Path(__file__)),cases=[])
    dump(out/'report.json',report)
    for name in ('lower-bound','ragionamento','italiano'):
        case=next(c for c in suite['cases'] if c['id']==name)
        tokens=case['prompt_ids'][:24]
        directory=out/name; directory.mkdir()
        native=[str(ENGINE),'--qwen38',suite['checkpoint'],'--tokens',','.join(map(str,tokens)),
                '--context','64','--chunk','7','--max-new','4',
                '--logits',str(directory/'native.f32'),'--token-ids',str(directory/'ids.json')]
        with (directory/'native.stdout').open('w') as stdout,(directory/'native.stderr').open('w') as stderr:
            subprocess.run(native,cwd=ROOT,check=True,pass_fds=(fd,),stdout=stdout,stderr=stderr)
        ids=json.loads((directory/'ids.json').read_text())['generated_ids']
        sequence=tokens+ids[:-1]
        env={k:v for k,v in os.environ.items() if not k.startswith('DS4_')}
        env.update(DS4_QWEN4_FT_TOKENS=','.join(map(str,sequence)),DS4_QWEN4_FT_OUT=str(directory/'reference.f32'))
        oracle=[str(ds/'ds4'),'--cpu','--first-token-test','-m',suite['checkpoint'],'-p','x']
        overrides={k:v for k,v in env.items() if k.startswith('DS4_')}
        if observe(directory/'oracle',ds,oracle,timeout=900,interval=.5,env_overrides=overrides):
            raise RuntimeError('CPU oracle interrupted/failed; observed evidence retained')
        actual=np.fromfile(directory/'native.f32','<f4').reshape(len(ids),-1)
        reference=np.fromfile(directory/'reference.f32','<f4').reshape(len(sequence),-1)[len(tokens)-1:]
        metric=logits_report(actual,reference)
        passed=(metric['scaled']<=GATE and max(metric['per_token_scaled'])<=GATE and
                metric['argmax_matches']==len(ids) and metric['margin_decided']==len(ids) and
                np.array_equal(reference.argmax(1),ids))
        report['cases'].append(dict(case=name,prompt_ids=tokens,generated_ids=ids,metrics=metric,passed=bool(passed),
             commands=[native,oracle],oracle_environment={k:v for k,v in env.items() if k.startswith('DS4_')},
             artifacts={p.name:sha(p) for p in directory.iterdir() if p.is_file()}))
        dump(out/'report.json',report)
        if not passed:
            raise RuntimeError(f'{name}: frozen independent gate failed')
    report['complete']=True; dump(out/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--dwarfstar',type=Path,default=ROOT.parent/'ds4-qwen38-baseline')
    a=p.parse_args(); run(a.suite.resolve(),a.output.resolve(),a.dwarfstar.resolve())
