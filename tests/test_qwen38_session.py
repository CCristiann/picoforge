"""Resident native protocol: template history, reset, cancellation and capacity."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
ENGINE=Path(os.environ.get('Q38_CLI_BINARY',str(ROOT/'picoforge')))
GGUF=ROOT/'build/qwen38-g2/cli.gguf'

class Session(unittest.TestCase):
    def setUp(self):
        self.p=subprocess.Popen([str(ENGINE),'--qwen38',str(GGUF),'--session','--context','512','--chunk','7'],
                                 cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        self.assertEqual(self.event()['event'],'ready')
    def tearDown(self):
        if self.p.poll() is None:
            self.p.stdin.write('Q\n');self.p.stdin.flush()
        self.p.wait(timeout=10);self.p.stdin.close();self.p.stdout.close()
    def event(self):
        raw=self.p.stdout.readline()
        self.assertTrue(raw,'worker exited unexpectedly')
        return json.loads(raw)
    def send(self,messages,n=5,stop=False):
        text=f'G {n} {len(messages)}\n'+''.join(role+' '+content.encode().hex()+'\n' for role,content in messages)
        self.p.stdin.write(text);self.p.stdin.flush()
        events=[]
        while True:
            event=self.event();events.append(event)
            if event['event']=='started' and stop:self.p.send_signal(signal.SIGUSR1)
            if event['event'] in ('done','error'):return events
    def reference(self,prompt):
        with tempfile.TemporaryDirectory() as d:
            ids=Path(d)/'ids.json'
            subprocess.run([str(ENGINE),'--qwen38',str(GGUF),'--complete',prompt,'--context','512',
                            '--chunk','7','--max-new','5','--token-ids',str(ids)],cwd=ROOT,
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            return json.loads(ids.read_text())
    def test_template_replay_and_reset(self):
        messages=[('U','ciao'),('A','risposta precedente'),('U','continua')]
        # The checkpoint template preserves an empty reasoning block in earlier
        # assistant turns even when thinking is disabled for the new reply.
        rendered='<|im_start|>user\nciao<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nrisposta precedente<|im_end|>\n<|im_start|>user\ncontinua<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        reference=self.reference(rendered)
        for _ in range(2):
            events=self.send(messages);done=events[-1]
            self.assertEqual(done['event'],'done')
            self.assertEqual(done['generated_ids'],reference['generated_ids'])
            self.assertEqual(done['prompt_tokens'],len(reference['prompt_ids']))
            self.assertGreater(done['prefill_profile']['gpu_s'],0)
            self.assertGreater(done['prefill_profile']['ngram_bytes'],0)
        self.p.stdin.write('R\n');self.p.stdin.flush();self.assertEqual(self.event()['event'],'reset')
        self.assertEqual(self.send(messages)[-1]['generated_ids'],reference['generated_ids'])
    def test_stop_capacity_and_recovery(self):
        error=self.send([('U','x'*500)])[-1]
        self.assertEqual(error['event'],'error')
        cancelled=self.send([('U','x'*200)],n=100,stop=True)[-1]
        self.assertEqual(cancelled['stop_reason'],'cancelled')
        self.assertEqual(cancelled['generated_tokens'],0)
        good=self.send([('U','hi')])[-1]
        self.assertEqual(good['event'],'done')
        self.assertGreater(good['generated_tokens'],0)

if __name__=='__main__':unittest.main()
