"""Exact greedy speculation: real graph with controlled transition weights.

The tiny control fixtures make the predictor perfect, wrong, or a false stop;
no production proposal override is used. Additional seeded weights exercise
nontrivial recurrence/PLE/cache state. Tests run under the common GPU lock.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
GGUF=ROOT/'build/qwen38-mtp/cli.gguf'


def control(path, predictor):
    shutil.copyfile(GGUF,path)
    manifest=json.loads(GGUF.with_suffix('.json').read_text())
    def q8(a):
        a=np.asarray(a,dtype=np.int8).reshape(-1,32)
        d=np.ones((len(a),1),dtype='<f2').view(np.uint8)
        return np.concatenate((d,a.view(np.uint8)),axis=1).tobytes()
    # Each embedding is a coordinate vector; the LM head maps it to the next
    # token in 1 -> 2 -> 3 -> 4 -> 5 -> stop(258). All residual blocks are zero.
    emb=np.zeros((512,256),dtype='<f4')
    for i in range(1,6): emb[i,i-1]=1
    emb[0,10]=1
    head=np.zeros((512,256),dtype=np.int8)
    for i in range(1,6): head[i+1 if i<5 else 258,i-1]=1
    # False-stop predictor shifts any embedding into the terminal coordinate.
    eh=np.zeros((256,512),dtype=np.int8)
    if predictor=='perfect': eh[:,:256]=np.eye(256,dtype=np.int8)
    if predictor=='stop': eh[4,:256]=1
    with path.open('r+b') as f:
        for item in manifest['tensors']:
            name=item['name']; blob=bytes(item['size_bytes'])
            if 'norm' in name and item['format']=='F32': blob=np.ones(np.prod(item['shape']),dtype='<f4').tobytes()
            if name=='token_embd.weight': blob=(emb.view('<u4')>>16).astype('<u2').tobytes()
            if name=='output.weight': blob=q8(head)
            if name.endswith('nextn.eh_proj.weight'): blob=q8(eh)
            assert len(blob)==item['size_bytes'],name
            f.seek(item['file_offset']); f.write(blob)


class Speculative(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-mtp'); self.path=Path(self.tmp.name); self.serial=0
    def tearDown(self): self.tmp.cleanup()
    def run_cli(self,gguf,mtp=True,tokens=(1,),n=8,chunk=3,context=32,extra=()):
        self.serial+=1; out=self.path/f'run{self.serial}.jsonl'; logits=self.path/f'run{self.serial}.f32'
        cmd=[str(ROOT/'picoforge'),'--qwen38',str(gguf),'--tokens',','.join(map(str,tokens)),
             '--max-new',str(n),'--chunk',str(chunk),'--context',str(context),'--runs-jsonl',str(out),
             '--logits',str(logits),*(['--mtp'] if mtp else []),*extra]
        p=subprocess.run(cmd,cwd=ROOT,capture_output=True,timeout=45)
        self.assertEqual(p.returncode,0,p.stderr.decode(errors='replace'))
        records=[json.loads(s) for s in out.read_text().splitlines()]
        return records,np.fromfile(logits,'<f4')
    def test_accept_reject_false_stop_and_verified_stop(self):
        for predictor in ('perfect','wrong','stop'):
            with self.subTest(predictor=predictor):
                gguf=self.path/f'{predictor}.gguf'; control(gguf,predictor)
                ordinary,ref=self.run_cli(gguf,False)
                rows,actual=self.run_cli(gguf)
                self.assertEqual(rows[0]['generated_ids'],[2,3,4,5,258])
                self.assertEqual(rows[0]['generated_ids'],ordinary[0]['generated_ids'])
                np.testing.assert_array_equal(actual.view('<u4'),ref.view('<u4'))
                metric=rows[0]['mtp']; self.assertGreater(metric['proposed'],0)
                self.assertEqual(rows[0]['stop_reason'],'stop')
                if predictor=='perfect':
                    self.assertEqual(metric['accepted'],metric['proposed']); self.assertEqual(metric['rollback_count'],0)
                elif predictor=='wrong':
                    self.assertEqual(metric['accepted'],0); self.assertEqual(metric['rollback_count'],metric['proposed'])
                else:
                    self.assertEqual(metric['rollback_count'],0); self.assertEqual(metric['accepted'],1)
                    self.assertGreater(metric['proposed'],1)
    def test_first_stop_budgets_chunk_one_and_cancellation(self):
        gguf=self.path/'perfect.gguf'; control(gguf,'perfect')
        first,_=self.run_cli(gguf,tokens=(5,)); self.assertEqual(first[0]['generated_ids'],[258]); self.assertEqual(first[0]['mtp']['proposed'],0)
        for n in (0,1,2,3,4,5):
            for stop in (0,1,2,3,7):
                with self.subTest(n=n,stop=stop):
                    kw=dict(n=n,chunk=1,context=1+max(n,1),extra=('--stop-after',str(stop)))
                    ref,_=self.run_cli(gguf,False,**kw); rows,_=self.run_cli(gguf,**kw)
                    self.assertEqual(rows[0]['generated_ids'],ref[0]['generated_ids'])
                    self.assertEqual(rows[0]['stop_reason'],ref[0]['stop_reason'])
                    self.assertLessEqual(rows[0]['prompt_tokens']+rows[0]['generated_tokens'],rows[0]['context'])
    def test_stop_at_each_verified_position(self):
        for predictor in ('perfect','wrong','stop'):
            gguf=self.path/f'{predictor}.gguf'; control(gguf,predictor)
            for first in (2,3,4,5):
                with self.subTest(predictor=predictor,first=first):
                    rows,_=self.run_cli(gguf,tokens=(first,))
                    self.assertEqual(rows[0]['generated_ids'],[*range(first+1,6),258])
                    self.assertEqual(rows[0]['stop_reason'],'stop')
                    # The terminal token is never fed back into trunk history.
                    self.assertEqual(rows[0]['position'],rows[0]['generated_tokens'])
    def test_seeded_state_rejection_and_replay(self):
        reference,_=self.run_cli(GGUF,False,tokens=(12,45,19,71,83),n=12,chunk=3)
        rows,_=self.run_cli(GGUF,tokens=(12,45,19,71,83),n=12,chunk=3,extra=('--repeat','2','--replay-chunk','1'))
        for row in rows: self.assertEqual(row['generated_ids'],reference[0]['generated_ids'])
        self.assertGreater(rows[0]['mtp']['rollback_count'],0)
    def test_streaming_signal_stop_and_reset(self):
        p=subprocess.Popen([str(ROOT/'picoforge'),'--qwen38',str(GGUF),'--session','--mtp','--context','512','--chunk','3'],
                           cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        try:
            self.assertEqual(json.loads(p.stdout.readline())['event'],'ready')
            def request(stop_at):
                p.stdin.write('G 64 1\nU 6369616f\n');p.stdin.flush();sent=False
                while True:
                    raw=p.stdout.readline();self.assertTrue(raw);event=json.loads(raw)
                    if not sent and event['event']==stop_at: p.send_signal(signal.SIGUSR1);sent=True
                    if event['event']=='done':return event
            for where in ('started','token'):
                done=request(where); self.assertEqual(done['stop_reason'],'cancelled')
                if where=='started':self.assertEqual(done['generated_tokens'],0)
                else:self.assertGreater(done['generated_tokens'],0)
                p.stdin.write('R\n');p.stdin.flush();self.assertEqual(json.loads(p.stdout.readline())['event'],'reset')
            p.stdin.write('Q\n');p.stdin.flush();self.assertEqual(p.wait(timeout=10),0)
        finally:
            if p.poll() is None:p.kill();p.wait()
            p.stdin.close();p.stdout.close()

if __name__=='__main__':unittest.main()
