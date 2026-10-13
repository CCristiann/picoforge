"""Frozen native Qwen3.8 cases and observed runs; standard library only.

Prepare before inference. Long prompts are fitted with the engine's tokenizer,
not an estimated character/token ratio. Run under the common GPU lock.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from run_observed import run as observe

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT/'picoforge'
CHECKPOINT = ROOT/'models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require_lock():
    fd = int(os.environ['PICOFORGE_GPU_LOCK_FD'])
    common = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=ROOT, text=True).strip()
    expected = (ROOT/common).resolve()/'qwen38-gpu.lock'
    if Path(os.environ['PICOFORGE_GPU_LOCK_PATH']).resolve() != expected:
        raise RuntimeError('requires the picoforge common lock')
    st, wanted = os.fstat(fd), expected.stat()
    if (st.st_dev, st.st_ino) != (wanted.st_dev, wanted.st_ino):
        raise RuntimeError('inherited lock descriptor does not match')
    return fd


def encode(checkpoint, path):
    p = subprocess.run([str(ENGINE), '--qwen38', str(checkpoint), '--chat-file', str(path),
                        '--context', '32768', '--max-new', '0', '--encode-only'],
                       cwd=ROOT, check=True, capture_output=True)
    return json.loads(p.stdout)['prompt_ids']


def archive(n):
    lines = ['Archivio sintetico. Usa soltanto i fatti qui forniti.',
             'Il codice del progetto Iris è ROMA47. La chiave di Loto è VERDE82.']
    lines += [f'Riga {i:04d}: lotto L{i:04d}, stato registrato, reparto R{i%17:02d}.' for i in range(n)]
    lines += ['Rettifica: il codice attuale di Iris è BARI29; sostituisce ROMA47.',
              'Rispondi soltanto con JSON con chiavi Iris (codice attuale) e Loto (chiave).']
    return '\n'.join(lines)


def prepare(out, checkpoint):
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT/'tools/eval/qwen38_cases.json').read_text())['cases']
    cases = [dict(c) for c in original[:3]]
    cases += [dict(id='ragionamento', kind='json', mode='nothink', max_new=384,
                   prompt='Trova il più piccolo intero n maggiore di 1000 che lascia resto 2 modulo 7, '
                          '5 modulo 11 e 8 modulo 13. Rispondi soltanto con JSON contenente la chiave n.',
                   expected={'n': 1633}),
              dict(id='italiano', kind='manual', mode='nothink', max_new=512,
                   prompt='Scrivi in italiano una guida pratica di circa 250 parole per verificare un backup. '
                          'Distingui una copia riuscita da un ripristino riuscito, proponi una prova periodica '
                          'e spiega come controllare che i dati recuperati siano integri. Usa esempi concreti.',
                   grading='Manuale: italiano, distinzione copia/ripristino, prova periodica e integrità. Nessun punteggio automatico.')]
    scratch = out/'candidate.txt'
    for capacity in (1024, 2048, 4096, 8192):
        budget = capacity-128
        lo, hi = 0, 450
        while lo < hi:
            mid = (lo+hi+1)//2
            scratch.write_text(archive(mid))
            if len(encode(checkpoint, scratch)) <= budget:
                lo = mid
            else:
                hi = mid-1
        cases.append(dict(id=f'context-{capacity}', kind='json', mode='nothink', max_new=128,
                          context=capacity, prompt=archive(lo), expected={'Iris':'BARI29','Loto':'VERDE82'}))
    scratch.unlink()
    for c in cases:
        text = c.pop('prompt')
        p = out/(c['id']+'.txt'); p.write_text(text)
        ids = encode(checkpoint, p)
        c.update(prompt_file=p.name, prompt_sha256=sha(p), prompt_tokens=len(ids),
                 context=c.get('context',8192), chunk=1024, prompt_ids=ids)
        if len(ids)+c['max_new']>c['context']:
            raise ValueError(f"case exceeds total context: {c['id']}")
    dump(out/'suite.json',dict(version=1, scope='Native runtime and local quality pilot; not a leaderboard',
         checkpoint=str(checkpoint.resolve()), checkpoint_bytes=checkpoint.stat().st_size,
         checkpoint_sha256_recomputed=False, cases=cases))


def validate_record(r):
    for key in ('load_s','reset_s','prefill_s','decode_s','request_s'):
        if not math.isfinite(r[key]) or r[key]<0:
            raise ValueError(f'invalid timing {key}')
    if r['prefill_s']+r['decode_s']>r['request_s']+.001:
        raise ValueError('phase times exceed enclosing timer')
    if r['prompt_tokens']+r['generated_tokens']>r['context']:
        raise ValueError('context exceeded')
    if r['generated_tokens']!=len(r['generated_ids']):
        raise ValueError('token count differs')
    if r['decode_tokens']!=max(0,r['generated_tokens']-1):
        raise ValueError('decode numerator differs')
    if r['ttft_request_s'] is not None and not 0<=r['ttft_request_s']<=r['request_s']:
        raise ValueError('invalid TTFT')
    return r


def run_suite(suite_path, out, select, repeats, warmup, replay_chunk, timeout, chunk=0, scalar=False):
    require_lock()
    suite = json.loads(suite_path.read_text())
    cases = [c for c in suite['cases'] if not select or c['id'] in select]
    if not cases or select and set(select)!={c['id'] for c in cases}:
        raise ValueError('unknown/empty case selection')
    out.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT/'src').glob('qwen38*'))+[Path(__file__), ROOT/'tools/eval/run_observed.py']
    meta = dict(complete=False, checkpoint=suite['checkpoint'], checkpoint_bytes=suite['checkpoint_bytes'],
                checkpoint_sha256_recomputed=False, suite_sha256=sha(suite_path),
                revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                diff=subprocess.check_output(['git','diff','--','src','tools','tests'],cwd=ROOT,text=True),
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources if p.is_file() and p.suffix in ('.c','.m','.h','.metal','.inc','.py')},
                executable_sha256=sha(ENGINE), metallib_sha256=sha(ROOT/'picoforge.metallib'),
                cache_state='Uncontrolled OS file/shader caches; tokenizer preparation read metadata. No cold-cache claim.',
                warmup=warmup, measured_repetitions=repeats, replay_chunk=replay_chunk, chunk_override=chunk,
                environment_overrides={'Q38_QSA_SCALAR':'1'} if scalar else {}, results=[])
    dump(out/'manifest.json',meta)
    for case in cases:
        prompt = suite_path.parent/case['prompt_file']
        if sha(prompt)!=case['prompt_sha256']:
            raise ValueError('prompt changed since freeze')
        record = out/(case['id']+'.runs.jsonl')
        ids_path = out/(case['id']+'.ids.json')
        command = [str(ENGINE),'--qwen38',suite['checkpoint'],'--chat-file',str(prompt.resolve()),
                   '--context',str(case['context']),'--chunk',str(min(chunk or case['chunk'],case['context'])),
                   '--max-new',str(case['max_new']),'--repeat',str(repeats),'--warmup',str(warmup),
                   '--runs-jsonl',str(record),'--token-ids',str(ids_path)]
        if replay_chunk:
            command += ['--replay-chunk',str(replay_chunk)]
        print('Running '+case['id'],flush=True)
        if not scalar and 'Q38_QSA_SCALAR' in os.environ:
            raise ValueError('unset Q38_QSA_SCALAR; select the recorded --scalar option instead')
        code = observe(out/case['id'],ROOT,command,timeout=timeout,interval=.5,
                       env_overrides=meta['environment_overrides'])
        rows = [validate_record(json.loads(x)) for x in record.read_text().splitlines()] if record.exists() else []
        result = dict(case=case['id'],returncode=code,records=len(rows),
                      complete=code==0 and len(rows)==repeats+warmup and
                               all(r['stop_reason'] in ('stop','length') for r in rows))
        if result['complete']:
            result['reset_replay_identical']=(all(r['generated_ids']==rows[0]['generated_ids'] for r in rows) if len(rows)>1 else None)
            if result['reset_replay_identical'] is False:
                result['complete']=False
            if json.loads(ids_path.read_text())['prompt_ids']!=case['prompt_ids']:
                raise ValueError('runtime tokenization differs from frozen prompt')
            # Quality is independent from runtime completion. Never execute generated code here.
            text = (out/case['id']/'stdout.log').read_text()
            if case['kind']=='json' and repeats+warmup==1:
                try:
                    result['quality_exact_json']=json.loads(text.strip())==case['expected']
                except ValueError:
                    result['quality_exact_json']=False
        meta['results'].append(result); dump(out/'manifest.json',meta)
        if not result['complete']:
            raise RuntimeError(f"{case['id']} failed; raw evidence retained")
    meta['complete']=True; dump(out/'manifest.json',meta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('output',type=Path)
    prep.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    run=sub.add_parser('run'); run.add_argument('suite',type=Path); run.add_argument('output',type=Path)
    run.add_argument('--cases',nargs='+'); run.add_argument('--repeat',type=int,default=1)
    run.add_argument('--warmup',type=int,default=0); run.add_argument('--replay-chunk',type=int,default=0)
    run.add_argument('--timeout',type=float,default=3600)
    run.add_argument('--chunk',type=int,default=0,help='Override frozen execution chunk, not prompt/budget')
    run.add_argument('--scalar',action='store_true',help='Recorded original QSA baseline')
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.output.resolve(),a.checkpoint)
    else:
        if a.repeat<1 or a.warmup<0 or a.timeout<=0 or a.chunk<0:
            p.error('invalid repetition or timeout')
        run_suite(a.suite.resolve(),a.output.resolve(),a.cases,a.repeat,a.warmup,a.replay_chunk,a.timeout,a.chunk,a.scalar)


if __name__=='__main__':
    main()
