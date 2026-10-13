"""Bounded full-checkpoint G3 gate against pinned DwarfStar FP32 reference.

Run under qwen38_gpu_lock.py from the picoforge worktree. The inherited lock
FD is preserved even when the oracle child changes to the DwarfStar cwd.
No downloads, checkpoint conversion, or full checkpoint hash. Stock Metal
parity is recorded separately because its QSA K/V cache uses FP16.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/eval'))
from run_observed import run as observe
from inspect_qwen38_gguf import inspect
REVISION = 'fc80bd695da76ee14bcb86820bf900a5cdfda806'
GATE = 1e-3
TOKENS = [248045, 846, 198, 7734, 12654, 220, 18, 1970, 25471, 10562, 56761, 2515]


def scaled(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('shape mismatch or nonfinite output')
    return float(np.abs(actual.astype(np.float64)-reference).max()/max(1, float(np.abs(reference).max())))


def logits_report(actual, reference):
    error = np.max(np.abs(actual.astype(np.float64)-reference), axis=1)
    top = np.sort(reference, axis=1)
    margin = top[:, -1]-top[:, -2]
    actual_top5 = np.argsort(-actual, axis=1, kind='stable')[:, :5]
    reference_top5 = np.argsort(-reference, axis=1, kind='stable')[:, :5]
    return dict(scaled=scaled(actual, reference), absolute=float(error.max()),
                argmax_matches=int((actual.argmax(1)==reference.argmax(1)).sum()),
                margin_decided=int((margin>2*error).sum()),
                per_token_scaled=(error/np.maximum(1, np.abs(reference).max(1))).tolist(),
                per_token_margin=margin.tolist(), native_top5=actual_top5.tolist(), reference_top5=reference_top5.tolist(),
                top5_order_matches=int(np.all(actual_top5==reference_top5, axis=1).sum()))


def near_tie(checkpoint, native, out):
    """Replay the frozen layer-7/position-11 boundary with identical scores.

    Read just this F32 router (5 MiB); serial FP32 multiply/add matches B's
    reduction order. No whole-checkpoint payload scan or alternate decoder.
    """
    inventory = inspect(checkpoint)
    tensor = next(t for t in inventory['tensors'] if t['name']=='blk.7.ffn_gate_inp.weight')
    if tensor['format']!='F32' or tensor['shape']!=[2560, 512]:
        raise ValueError('frozen router case has changed')
    with checkpoint.open('rb') as source:
        source.seek(tensor['file_offset']); raw = source.read(tensor['size_bytes'])
    weights = np.frombuffer(raw, '<f4').reshape(512, 2560)
    x = np.fromfile(native/'layer7.ffn_mixed.f32', '<f4').reshape(len(TOKENS), 2560)[11]
    scores = np.zeros(512, np.float32)
    for k in range(2560):
        scores += weights[:, k]*x[k]
    stock = np.fromfile(out/'stock-trace_qwen_router-7_pos11.bin', '<f4')
    if stock.shape!=(512,) or not np.isfinite(stock).all() or not np.isfinite(scores).all():
        raise ValueError('missing or invalid router trace')
    first, second = np.argsort(-scores, kind='stable'), np.argsort(-stock, kind='stable')
    report = dict(layer=7, position=11, logits_error_abs=float(np.abs(scores-stock).max()),
                  native_top12=first[:12].tolist(), reference_top12=second[:12].tolist(),
                  native_margin_10_11=float(scores[first[9]]-scores[first[10]]),
                  reference_margin_10_11=float(stock[second[9]]-stock[second[10]]),
                  header_sha256=inventory['header_sha256'], router_sha256=hashlib.sha256(raw).hexdigest())
    (out/'router-near-tie.json').write_text(json.dumps(report, indent=2)+'\n')
    np.stack([scores, stock]).astype('<f4').tofile(out/'near-tie-scores.f32')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dwarfstar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fd = int(os.environ['PICOFORGE_GPU_LOCK_FD'])
    common = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', '--git-common-dir'], text=True).strip()
    expected = ((ROOT/ common).resolve()/'qwen38-gpu.lock')
    if Path(os.environ['PICOFORGE_GPU_LOCK_PATH']).resolve()!=expected:
        raise RuntimeError('G3 requires the picoforge cross-worktree lock')
    ds = args.dwarfstar.resolve()
    revision = subprocess.check_output(['git', '-C', str(ds), 'rev-parse', 'HEAD'], text=True).strip()
    if revision!=REVISION:
        raise ValueError('DwarfStar revision differs from the frozen oracle')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    native = out/'native'; native.mkdir()
    checkpoint = args.checkpoint.resolve()
    (out/'tokens.json').write_text(json.dumps(TOKENS)+'\n')
    commands = []

    def run(name, command, cwd=ROOT, extra_env=None):
        commands.append(dict(name=name, argv=list(map(str, command)), cwd=str(cwd), env=extra_env or {}))
        child_env = {k: v for k, v in os.environ.items() if not k.startswith('DS4_')}
        child_env.update(extra_env or {})
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(list(map(str, command)), cwd=cwd, env=child_env,
                           pass_fds=(fd,), stdout=log, stderr=log, check=True)

    run('native', [ROOT/'build/qwen38-g2/forward_real', checkpoint, native, 48, 4, *TOKENS])
    objects = ['ds4.o', 'ds4_image.o', 'ds4_distributed.o', 'ds4_tp.o', 'ds4_ssd.o', 'ds4_metal.o', 'ds4_layer_pack.o', 'ds4_engram.o']
    oracle = out/'dwarfstar_logits'
    run('oracle-build', ['clang', '-O2', '-std=c11', '-I'+str(ds), ROOT/'tools/oracle/qwen38_dwarfstar_logits.c',
                        *[ds/p for p in objects], '-framework', 'Metal', '-framework', 'Foundation', '-lm', '-pthread', '-o', oracle])
    run('stock-metal', [oracle, checkpoint, out/'stock-metal.f32', *TOKENS], cwd=ds,
        extra_env={'DS4_METAL_GRAPH_DUMP_PREFIX': str(out/'stock-trace')})
    tie = near_tie(checkpoint, native, out)
    run('near-tie-replay', [ROOT/'build/qwen38-g2/test_parts', ROOT/'build/qwen38-g1/graph.gguf',
                          ROOT/'picoforge.metallib', out/'near-tie-scores.f32'], extra_env={'MTL_DEBUG_LAYER': '1'})
    run('fp32-reference', [ds/'ds4', '--cpu', '--first-token-test', '-m', checkpoint, '-p', 'x'], cwd=ds,
        extra_env={'DS4_QWEN4_FT_TOKENS': ','.join(map(str, TOKENS)),
                   'DS4_QWEN4_FT_OUT': str(out/'fp32-logits.f32'), 'DS4_QWEN4_FT_HIDDEN': str(out/'fp32-hidden.f32')})
    logits = np.fromfile(native/'logits.f32', '<f4').reshape(len(TOKENS), -1)
    fp32 = np.fromfile(out/'fp32-logits.f32', '<f4').reshape(logits.shape)
    stock = np.fromfile(out/'stock-metal.f32', '<f4').reshape(logits.shape)
    hidden = np.fromfile(out/'fp32-hidden.f32', '<f4').reshape(len(TOKENS), 48, -1)
    errors = [scaled(np.fromfile(native/f'layer{i}.output.f32', '<f4').reshape(len(TOKENS), -1), hidden[:, i]) for i in range(48)]
    ref = logits_report(logits, fp32); metal = logits_report(logits, stock)
    passed = max(errors)<=GATE and ref['scaled']<=GATE and ref['argmax_matches']==len(TOKENS) and ref['margin_decided']==len(TOKENS)
    chat_command = list(map(str, [ROOT/'picoforge', '--qwen38', checkpoint, '--chat', 'Rispondi in italiano: quanto fa 2 + 2?',
                                 '--max-new', 16, '--token-ids', out/'chat-ids.json', '--logits', out/'chat-logits.f32']))
    commands.append(dict(name='chat', argv=chat_command, cwd=str(ROOT), observer='tools/eval/run_observed.py'))
    if observe(out/'chat', ROOT, chat_command, timeout=1200, interval=.25):
        raise RuntimeError('chat or memory observation failed; evidence retained')
    memory = json.loads((out/'chat/process.json').read_text())
    metal_memory = [json.loads(line.split(': ', 1)[1]) for line in (out/'chat/stderr.log').read_text().splitlines()
                    if line.startswith('Qwen3.8 memory: ')][-1]
    report = dict(gate=GATE, tokens=TOKENS, layers=48, checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                  dwarfstar_revision=revision, fp32_reference=ref, hidden_by_layer=errors, hidden_max=max(errors),
                  stock_metal=metal, stock_metal_pass=metal['scaled']<=GATE, pass_fp32=passed,
                  router_near_tie=tie,
                  memory={**metal_memory, 'sampled_peak_footprint_bytes': memory['sampled_peak_footprint_bytes'],
                          'lifetime_peak_footprint_bytes': memory['lifetime_peak_footprint_bytes']},
                  note='Stock Metal uses FP16 K/V; its failure is retained, not relabeled as parity.', commands=commands)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    artifacts = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in out.rglob('*') if p.is_file() and p.suffix in ('.f32', '.bin', '.json', '.log', '.jsonl')}
    (out/'artifacts.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('hidden_max', 'fp32_reference', 'stock_metal_pass', 'pass_fp32')}, indent=2))
    if not passed:
        raise SystemExit('G3 FP32 parity gate failed')


if __name__ == '__main__':
    main()
