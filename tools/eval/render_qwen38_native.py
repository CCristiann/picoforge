"""Self-contained view of the recorded G2/G3 gate; no inference or dependencies."""
import argparse
import json
from pathlib import Path


def render(evidence, output):
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
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('qwen38_native_report.html').read_text(encoding='utf-8')
    output.write_text(template.replace('@@DATA@@', payload), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.evidence, args.output)
