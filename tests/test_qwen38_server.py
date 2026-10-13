"""Local HTTP contract over the actual small native fixture (worker owns lock)."""
import http.client
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]

class LocalChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process=subprocess.Popen([sys.executable,'-B','tools/ui/qwen38/server.py',
            '--checkpoint','build/qwen38-g2/cli.gguf','--context','512','--chunk','7','--port','0'],
            cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        cls.port=int(cls.process.stdout.readline().strip().rsplit(':',1)[1])
        end=time.monotonic()+30
        while time.monotonic()<end:
            status=cls.request('GET','/api/status')[1]
            if status['state']=='ready':return
            if status['state']=='error':raise RuntimeError(status['error'])
            time.sleep(.05)
        raise RuntimeError('fixture model did not load')

    @classmethod
    def tearDownClass(cls):
        cls.process.send_signal(signal.SIGTERM);cls.process.wait(timeout=20);cls.process.stdout.close()

    @classmethod
    def request(cls,method,path,data=None,headers=None):
        c=http.client.HTTPConnection('127.0.0.1',cls.port,timeout=15)
        c.request(method,path,json.dumps(data) if data is not None else None,
                  headers or {'Content-Type':'application/json'})
        r=c.getresponse();raw=r.read();c.close()
        if r.getheader('Content-Type','').startswith('application/x-ndjson'):
            return r.status,[json.loads(l) for l in raw.splitlines()]
        return r.status,json.loads(raw)

    def test_history_reset_capacity_and_origin(self):
        self.assertEqual(self.request('POST','/api/reset',{})[0],200)
        code,events=self.request('POST','/api/generate',{'prompt':'ciao','max_new':5})
        self.assertEqual(code,200);self.assertEqual(events[-1]['event'],'done')
        answer=''.join(e['text'] for e in events if e['event']=='token')
        status=self.request('GET','/api/status')[1]
        self.assertEqual(status['messages'],[{'role':'user','content':'ciao'},{'role':'assistant','content':answer}])
        code,events2=self.request('POST','/api/generate',{'prompt':'continua','max_new':5})
        self.assertGreater(events2[-1]['prompt_tokens'],events[-1]['prompt_tokens'])
        previous=self.request('GET','/api/status')[1]['messages']
        _,bad=self.request('POST','/api/generate',{'prompt':'x'*500,'max_new':5})
        self.assertEqual(bad[-1]['event'],'error')
        self.assertEqual(self.request('GET','/api/status')[1]['messages'],previous)
        code,_=self.request('POST','/api/reset',{}, {'Content-Type':'application/json','Origin':'https://example.org'})
        self.assertEqual(code,403)
        self.assertEqual(self.request('POST','/api/reset',{})[0],200)
        self.assertEqual(self.request('GET','/api/status')[1]['messages'],[])
        self.assertEqual(self.request('POST','/api/generate',{'prompt':'','max_new':5})[0],400)
        self.assertFalse(self.request('GET','/api/status')[1]['active'])

    def test_busy_stop_and_disconnect(self):
        self.request('POST','/api/reset',{})
        c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=15)
        c.request('POST','/api/generate',json.dumps({'prompt':'x'*100,'max_new':300}),{'Content-Type':'application/json'})
        response=c.getresponse();self.assertEqual(response.status,200)
        self.assertEqual(json.loads(response.readline())['event'],'started')
        self.assertEqual(self.request('POST','/api/reset',{})[0],409)
        self.assertEqual(self.request('POST','/api/stop',{})[0],200)
        events=[json.loads(l) for l in response.read().splitlines()];c.close()
        self.assertEqual(events[-1]['stop_reason'],'cancelled')
        self.assertFalse(self.request('GET','/api/status')[1]['active'])
        self.assertEqual(self.request('POST','/api/reset',{})[0],200)
        c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=15)
        c.request('POST','/api/generate',json.dumps({'prompt':'x'*100,'max_new':300}),{'Content-Type':'application/json'})
        response=c.getresponse();response.readline();response.close();c.close()
        deadline=time.monotonic()+15
        while self.request('GET','/api/status')[1]['active'] and time.monotonic()<deadline:time.sleep(.05)
        self.assertFalse(self.request('GET','/api/status')[1]['active'])
        self.assertEqual(self.request('POST','/api/reset',{})[0],200)

if __name__=='__main__':unittest.main()
