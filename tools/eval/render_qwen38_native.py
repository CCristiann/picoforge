"""Self-contained view of the recorded G2/G3 gate; no inference or dependencies."""
import argparse
import json
import os
import re
from pathlib import Path


def render(evidence, output, native=None, mtp_gate=None, mtp_bench=None):
    if output.exists():
        raise FileExistsError('Report output must be a new path; historical evidence is immutable')
    def read(name):
        return json.loads((evidence/name).read_text(encoding='utf-8'))
    gate, memory, ids = read('g3.json'), read('memory.json'), read('chat-ids.json')
    samples = [json.loads(line) for line in (evidence/'chat/memory.jsonl').read_text().splitlines()]
    data = dict(layers=gate['hidden_by_layer'], gate=gate['gate'], fp32=gate['fp32_reference'],
                stock=gate['stock_metal'], pass_fp32=gate['pass_fp32'], stock_pass=gate['stock_metal_pass'],
                memory=memory, ids=ids, tie=gate['router_near_tie'],
                answer=(evidence/'chat/stdout.log').read_text().strip(),
                samples=[dict(t=s['elapsed_s'], wired=s['system_wired_pages']*s['page_bytes'],
                              footprint=s['phys_footprint_bytes']) for s in samples])
    if len(data['layers']) != gate['layers'] or len(data['fp32']['per_token_scaled']) != len(gate['tokens']):
        raise ValueError('inconsistent layer/token evidence')
    if native is not None:
        data['native'] = json.loads(native.read_text())
        root=native.parent/data['native'].get('evidence_root','.')
        data['native_root'] = os.path.relpath(root, output.parent)
    if bool(mtp_gate) != bool(mtp_bench):
        raise ValueError('MTP report requires both the independent gate and paired benchmark')
    if mtp_gate:
        mg,mb=(json.loads(p.read_text()) for p in (mtp_gate,mtp_bench))
        if not mg['complete'] or not mb['complete']:
            raise ValueError('MTP report requires completed evidence')
        cases=[]
        for case in mb['cases']:
            item=dict(case)
            item['memory']={}
            for mode in ('ordinary','mtp'):
                folder=mtp_bench.parent/case['name']/mode
                process=json.loads((folder/'process.json').read_text())
                samples=[json.loads(x) for x in (folder/'memory.jsonl').read_text().splitlines()]
                before=process['system_before'];after=process['system_after']
                item['memory'][mode]=dict(footprint=process['lifetime_peak_footprint_bytes'],
                    wired=max(s['system_wired_pages']*s['page_bytes'] for s in samples),
                    pressure=sorted({s['system_pressure_level'] for s in samples}),
                    swap_delta=after['system_swap_used_bytes']-before['system_swap_used_bytes'],
                    peak_swap_growth=max([0]+[s['system_swap_used_bytes']-before['system_swap_used_bytes'] for s in samples]))
            cases.append(item)
        data['mtp']=dict(gate=mg['gate'],gates=mg['cases'],cases=cases,
            raw_gate=os.path.relpath(mtp_gate,output.parent),raw_bench=os.path.relpath(mtp_bench,output.parent))
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('qwen38_native_report.html').read_text(encoding='utf-8')
    relative = os.path.relpath(evidence, output.parent)
    if relative != '.':
        template = re.sub(r'href="(?!https?:|#)([^"]+)"', lambda m: 'href="'+relative+'/'+m[1]+'"', template)
    output.write_text(template.replace('@@DATA@@', payload), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--mtp-gate', type=Path)
    parser.add_argument('--mtp-bench', type=Path)
    args = parser.parse_args()
    render(args.evidence, args.output, args.native, args.mtp_gate, args.mtp_bench)
