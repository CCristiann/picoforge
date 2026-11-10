"""Bounded independent FP32 trunk + teacher-forced MTP gate (exclusive evidence).

Prompts are closed through the native tokenizer; continuations are ordinary
native greedy, including several decode rows. No quality claim. All heavy
children inherit the project GPU lock, and CPU follows observed Metal release.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from qwen38_native_suite import ROOT, ENGINE, CHECKPOINT, dump, sha, require_lock
from qwen38_native_bounded import settle_after_metal
from run_observed import run as observe
sys.path.insert(0,str(ROOT/'tools'))
from qwen38_g3 import logits_report, GATE, REVISION
sys.path.insert(0,str(ROOT/'tools/oracle'))
from qwen38_packed import build_reference


def run(output, checkpoint, dwarfstar):
    require_lock(); output.mkdir(parents=True,exist_ok=False)
    oracle=build_reference(dwarfstar,output/'oracle-build',adapter=ROOT/'tools/oracle/qwen38_dwarfstar_cpu.c',
        decoder='ds4.c:qwen4_ref_forward_token + qwen4_ref_mtp (lazy private mapping)')
    probe=ROOT/'build/qwen38-mtp/mtp_real'
    report=dict(complete=False,gate=GATE,dwarfstar_revision=REVISION,
        checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256_recomputed=False,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        diff=subprocess.check_output(['git','diff','--','src','tools','tests'],cwd=ROOT,text=True),
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [ENGINE,probe,ROOT/'picoforge.metallib',Path(__file__)]},cases=[])
    dump(output/'report.json',report)
    for name,mode,prompt in [('coding','--complete','def add(a, b):\n    return'),('italiano','--complete','La capitale della Francia è')]:
        directory=output/name; directory.mkdir()
        command=[str(ENGINE),'--qwen38',str(checkpoint),mode,prompt,'--context','64','--chunk','4','--max-new','6',
                 '--token-ids',str(directory/'ids.json'),'--logits',str(directory/'ordinary.f32')]
        if observe(directory/'ordinary-run',ROOT,command,timeout=240,interval=.5): raise RuntimeError('ordinary run failed')
        ids=json.loads((directory/'ids.json').read_text()); sequence=ids['prompt_ids']+ids['generated_ids'][:-1]
        if len(ids['generated_ids'])<3: raise RuntimeError('bounded continuation did not exercise decode')
        # Wait also between resident native processes; no duplicate mapping.
        settle_after_metal(directory)
        native=[str(probe),str(checkpoint),str(directory),'4','0',*map(str,sequence)]
        if observe(directory/'native-run',ROOT,native,timeout=240,interval=.5): raise RuntimeError('native MTP failed')
        settle_after_metal(directory)
        overrides=dict(DS4_QWEN4_FT_TOKENS=','.join(map(str,sequence)),DS4_QWEN4_FT_OUT=str(directory/'reference.f32'),
                       DS4_QWEN4_MTP_OUT=str(directory/'reference-mtp.f32'))
        if observe(directory/'oracle-run',dwarfstar,[str(oracle),str(checkpoint)],timeout=900,interval=.5,env_overrides=overrides):
            raise RuntimeError('CPU oracle failed/interrupted')
        trunk=np.fromfile(directory/'logits.f32','<f4').reshape(len(sequence),-1)
        ordinary=np.fromfile(directory/'ordinary.f32','<f4').reshape(len(ids['generated_ids']),-1)
        rowmatch=np.array_equal(trunk[len(ids['prompt_ids'])-1:].view('<u4'),ordinary.view('<u4'))
        metrics={}; passed=rowmatch
        for key,a,b,rows in [('trunk','logits.f32','reference.f32',len(sequence)),('mtp','mtp.f32','reference-mtp.f32',len(sequence)-1)]:
            actual=np.fromfile(directory/a,'<f4').reshape(rows,-1); reference=np.fromfile(directory/b,'<f4').reshape(rows,-1)
            metric=logits_report(actual,reference)
            metric['passed']=metric['scaled']<=GATE and max(metric['per_token_scaled'])<=GATE and metric['argmax_matches']==rows and metric['margin_decided']==rows
            passed &= metric['passed']; metrics[key]=metric
        report['cases'].append(dict(name=name,prompt=prompt,mode=mode,sequence=sequence,generated_ids=ids['generated_ids'],
            ordinary_rows_bitexact=rowmatch,metrics=metrics,passed=bool(passed),commands=[command,native],oracle_environment=overrides,
            artifacts={p.name:sha(p) for p in directory.iterdir() if p.is_file()}))
        dump(output/'report.json',report)
        if not passed: raise RuntimeError(f'{name}: frozen gate failed')
    report['complete']=True; dump(output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT); p.add_argument('--dwarfstar',type=Path,default=ROOT.parent/'ds4-qwen38-baseline')
    a=p.parse_args(); run(a.output.resolve(),a.checkpoint.resolve(),a.dwarfstar.resolve())
