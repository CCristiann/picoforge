"""Paired native ordinary/MTP pilot with resident warm-up and three repetitions.

No cache purge or cold-cache claim. Every mode owns the model exclusively;
first mode alternates by case. Decode clocks include proposal, verification,
snapshot and rejection/replay. Sequence identity is required on every run.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
from qwen38_native_suite import ROOT, ENGINE, CHECKPOINT, dump, sha, require_lock, validate_record
from qwen38_native_bounded import settle_after_metal
from run_observed import run as observe


def stats(values):
    return dict(n=len(values),median=statistics.median(values),minimum=min(values),maximum=max(values))


def run(output,checkpoint):
    require_lock();output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=False,checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256_recomputed=False,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        diff=subprocess.check_output(['git','diff','--','src','tools','tests'],cwd=ROOT,text=True),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('qwen38*'))+[Path(__file__),ROOT/'tools/eval/run_observed.py'] if p.is_file()},
        executable_sha256=sha(ENGINE),metallib_sha256=sha(ROOT/'picoforge.metallib'),
        cache_state='Resident model, explicit warm-up excluded; OS/SSD cache not purged or controlled',
        warmup=1,measured_repetitions=3,context=8192,chunk=128,cases=[])
    dump(output/'report.json',report)
    cases=[('coding','--complete','def add(a, b):\n    return',32),
           ('italiano','--chat','Scrivi cinque consigli pratici per verificare un backup e spiega perché una copia riuscita non basta.',64)]
    for index,(name,mode,prompt,max_new) in enumerate(cases):
        directory=output/name;directory.mkdir();case=dict(name=name,mode=mode,prompt=prompt,max_new=max_new,modes={})
        encoded=subprocess.check_output([str(ENGINE),'--qwen38',str(checkpoint),mode,prompt,'--context','8192','--max-new','0','--encode-only'],cwd=ROOT)
        case['prompt_ids']=json.loads(encoded)['prompt_ids'];case['prompt_tokens']=len(case['prompt_ids'])
        if len(case['prompt_ids'])+max_new>8192:raise ValueError('context exceeded')
        report['cases'].append(case);dump(output/'report.json',report)
        expected=None
        for mtp in ((False,True) if index%2==0 else (True,False)):
            label='mtp' if mtp else 'ordinary';runs=directory/(label+'.jsonl')
            command=[str(ENGINE),'--qwen38',str(checkpoint),'--tokens',','.join(map(str,case['prompt_ids'])),
                     '--max-new',str(max_new),'--context','8192','--chunk','128','--warmup','1','--repeat','3',
                     '--runs-jsonl',str(runs),*(['--mtp'] if mtp else [])]
            if observe(directory/label,ROOT,command,timeout=1800,interval=.5):raise RuntimeError(f'{name}/{label} failed')
            records=[validate_record(json.loads(line)) for line in runs.read_text().splitlines()]
            if len(records)!=4 or [r['warmup'] for r in records]!=[True,False,False,False]:raise ValueError('warmup/repetition mismatch')
            if expected is None:expected=records[0]['generated_ids']
            if not expected or any(r['generated_ids']!=expected for r in records):raise ValueError('ordinary/MTP sequence or replay differs')
            measured=records[1:]
            if any(r['stop_reason']=='cancelled' for r in records):raise ValueError('cancelled performance run')
            metrics={k:stats([r[k] for r in measured]) for k in ('load_s','prefill_s','decode_s','request_s','ttft_request_s','metal_allocated_bytes','graph_buffer_bytes')}
            metrics['decode_tokens_per_s']=stats([r['decode_tokens']/r['decode_s'] for r in measured])
            metrics['prefill_tokens_per_s']=stats([r['prompt_tokens']/r['prefill_s'] for r in measured])
            metrics['proposed']=sum(r['mtp']['proposed'] for r in measured);metrics['accepted']=sum(r['mtp']['accepted'] for r in measured)
            metrics['acceptance']=metrics['accepted']/metrics['proposed'] if metrics['proposed'] else None
            metrics['rollback_count']=sum(r['mtp']['rollback_count'] for r in measured)
            metrics['checkpoint_host_bytes']=max(r['mtp']['checkpoint_host_bytes'] for r in measured)
            for k in ('prefill_s','draft_s','verify_s','snapshot_s','replay_s'):
                metrics['mtp_'+k]=stats([r['mtp'][k] for r in measured])
            case['modes'][label]=dict(command=command,metrics=metrics,generated_ids=expected,sequence_identity=True)
            dump(output/'report.json',report)
            release=directory/(label+'-release');release.mkdir();settle_after_metal(release)
        a=case['modes']['ordinary']['metrics'];b=case['modes']['mtp']['metrics']
        case['decode_speed_ratio']=b['decode_tokens_per_s']['median']/a['decode_tokens_per_s']['median']
        case['request_time_ratio']=a['request_s']['median']/b['request_s']['median']
        dump(output/'report.json',report)
    report['complete']=True;dump(output/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    a=p.parse_args();run(a.output.resolve(),a.checkpoint.resolve())
