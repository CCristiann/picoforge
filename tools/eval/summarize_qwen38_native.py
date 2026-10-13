"""Summarize retained native requests; no inference and no evidence overwrite."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

from qwen38_native_suite import dump, validate_record


def distribution(values):
    ordered=sorted(values)
    if not ordered:
        return None
    return dict(median=statistics.median(ordered),p10=ordered[round(.1*(len(ordered)-1))],
                p90=ordered[round(.9*(len(ordered)-1))],n=len(ordered))


def summarize(root):
    groups=[]; incomplete=[]
    for path in sorted(root.rglob('process.json')):
        info=json.loads(path.read_text())
        if not info['complete']:
            incomplete.append(dict(group=str(path.parent.parent.relative_to(root)),case=path.parent.name,
                                   process_s=info.get('process_s'),returncode=info.get('returncode'),
                                   reason=info.get('stop_reason') or ('Processo fallito: consultare i dati grezzi.' if info.get('returncode') is not None else 'Esito non disponibile; prova incompleta o ancora in corso.'),
                                   raw_path=str(path.relative_to(root))))
    for path in sorted(root.rglob('*.runs.jsonl')):
        records=[validate_record(json.loads(line)) for line in path.read_text().splitlines()]
        if not records:
            continue
        name=path.name.removesuffix('.runs.jsonl')
        process=path.parent/name/'process.json'
        if not process.exists():
            continue
        info=json.loads(process.read_text())
        if not info['complete']:
            continue
        rows=[r for r in records if not r['warmup'] and r['stop_reason'] in ('stop','length')]
        if not rows:
            continue
        memory=[json.loads(line) for line in (path.parent/name/'memory.jsonl').read_text().splitlines()]
        before,after=info['system_before'],info['system_after']
        item=dict(group=str(path.parent.relative_to(root)),case=name,warmups=sum(r['warmup'] for r in records),
                  repetitions=len(rows),prompt_tokens=rows[0]['prompt_tokens'],context=rows[0]['context'],
                  chunk=rows[0]['chunk'],generated_tokens=[r['generated_tokens'] for r in rows],
                  stop_reasons=[r['stop_reason'] for r in rows],
                  reset_replay_identical=(all(r['generated_ids']==records[0]['generated_ids'] for r in records) if len(records)>1 else None),
                  process_s=info['process_s'],first_stdout_s=info['first_stream_bytes_s'].get('stdout'),
                  load_s=rows[0]['load_s'],
                  memory=dict(metal_allocated_bytes=max(r['metal_allocated_bytes'] for r in rows),
                       metal_recommended_bytes=rows[0]['metal_recommended_bytes'],
                       graph_buffer_bytes=rows[0]['graph_buffer_bytes'],
                       lifetime_peak_footprint_bytes=info['lifetime_peak_footprint_bytes'],
                       sampled_peak_footprint_bytes=info['sampled_peak_footprint_bytes'],samples=len(memory),
                       peak_system_wired_bytes=max((s['system_wired_pages']*s['page_bytes'] for s in memory),default=None),
                       pressure_levels=sorted({s['system_pressure_level'] for s in memory}),
                       swap_used_delta_bytes=after['system_swap_used_bytes']-before['system_swap_used_bytes'],
                       peak_swap_growth_bytes=max([0]+[s['system_swap_used_bytes']-before['system_swap_used_bytes'] for s in memory]),
                       swapins_delta_pages=after['system_swapins_pages']-before['system_swapins_pages'],
                       swapouts_delta_pages=after['system_swapouts_pages']-before['system_swapouts_pages']),
                  raw_path=str(path.relative_to(root)))
        for metric in ('prefill_s','decode_s','ttft_request_s','request_s','reset_s'):
            item[metric]=distribution([r[metric] for r in rows if r[metric] is not None])
        item['decode_tps']=distribution([r['decode_tokens']/r['decode_s'] for r in rows if r['decode_tokens'] and r['decode_s']>0])
        item['prefill_tps']=distribution([r['prompt_tokens']/r['prefill_s'] for r in rows if r['prefill_s']>0])
        for phase in ('prefill_profile','decode_profile'):
            if all(phase in r for r in rows):
                item[phase]={k:distribution([r[phase][k] for r in rows]) for k in rows[0][phase]}
        if len(records)==1:
            item['answer']=(path.parent/name/'stdout.log').read_text()
        quality=path.parent/(name+'-quality.json')
        if quality.exists():item['quality']=json.loads(quality.read_text())
        elif (path.parent/'manifest.json').exists():
            manifest=json.loads((path.parent/'manifest.json').read_text())
            result=next((r for r in manifest.get('results',[]) if r['case']==name),{})
            if 'quality_exact_json' in result:
                item['quality']=dict(grading='JSON esatto',tests_passed=result['quality_exact_json'])
        groups.append(item)
    bounded=root/'bounded-lazy'/'report.json'
    return dict(schema=1,pilot=True,groups=groups,incomplete=incomplete,
                bounded=json.loads(bounded.read_text()) if bounded.exists() else None,
                notes=['Native FP32 only; DwarfStar 30 token/s belongs to another runtime.',
                       'Warmups retained but excluded. One measured request is not a repeatability benchmark.',
                       'OS file/shader caches are uncontrolled; no cold-cache claim.',
                       'GPU time overlaps wait time. prepare_s includes n-gram I/O, dedup, decode and staging.',
                       'N-gram bytes are logical pread bytes, not physical SSD traffic.',
                       'Footprint, Metal allocations and system wired memory overlap and must not be summed.',
                       'Exact reset/replay within this runtime does not establish independent numerical parity.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists(): p.error('output already exists')
    result=summarize(a.evidence)
    result['evidence_root']=os.path.relpath(a.evidence.resolve(),a.output.resolve().parent)
    dump(a.output,result)
