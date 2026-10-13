"""One resident native engine, one local conversation, no additional dependencies."""
import argparse
import codecs
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[3]

class Engine:
    def __init__(self, checkpoint, context=8192, chunk=128, mtp=False):
        self.checkpoint=checkpoint; self.context=context; self.chunk=chunk; self.mtp=mtp
        self.mutex=threading.Lock(); self.control=threading.Lock()
        self.state='loading'; self.error=None; self.process=None; self.history=[]
        self.active=False; self.started=False; self.cancel=False; self.metrics=None; self.logs=[]; self.closing=False
        self.events=queue.Queue()
        threading.Thread(target=self.load,daemon=True).start()

    def load(self):
        command=[sys.executable,str(ROOT/'tools/qwen38_gpu_lock.py'),'--timeout','3600','--',
                 str(ROOT/'picoforge'),'--qwen38',str(self.checkpoint),'--session',
                 '--context',str(self.context),'--chunk',str(self.chunk)]
        if self.mtp: command.append('--mtp')
        try:
            with self.control:
                if self.closing: return
                self.process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE,bufsize=0)
            threading.Thread(target=self.stderr,daemon=True).start()
            for raw in self.process.stdout:
                event=json.loads(raw)
                if event['event']=='ready':
                    with self.control:
                        self.state='ready'; self.metrics=event
                else:
                    self.events.put(event)
            code=self.process.wait()
            with self.control:
                self.state='error'; self.error=f'Motore terminato (codice {code}). Riavvia il servizio. '+''.join(self.logs[-2:])[-500:]
            self.events.put({'event':'error','message':self.error})
        except Exception as error:
            with self.control:
                self.state='error'; self.error=str(error)
            self.events.put({'event':'error','message':str(error)})

    def stderr(self):
        for raw in self.process.stderr:
            line=raw.decode('utf-8',errors='replace')
            self.logs.append(line); self.logs=self.logs[-30:]
            sys.stderr.write(line)

    def status(self):
        with self.control:
            return dict(state=self.state,error=self.error,active=self.active,context=self.context,
                        messages=list(self.history),metrics=self.metrics,
                        session_mode='resident_model_reset_replay',mtp=self.mtp)

    def stop(self):
        with self.control:
            if self.active:
                self.cancel=True
                if self.started and self.process.poll() is None:
                    self.process.send_signal(signal.SIGUSR1)
        return {'stopping':True}

    def generate(self,prompt,max_new):
        # Caller holds mutex until the stream has finished, including disconnect.
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt.encode())>65536:
            raise ValueError('Scrivi un messaggio da 1 a 65536 byte.')
        if not isinstance(max_new,int) or isinstance(max_new,bool) or not 1<=max_new<=min(2048,self.context):
            raise ValueError('Limite di risposta non valido.')
        messages=self.history+[{'role':'user','content':prompt}]
        if len(messages)>127:
            raise ValueError('Troppi turni. Avvia una nuova conversazione.')
        header=f'G {max_new} {len(messages)}\n'
        body=''.join(('U' if m['role']=='user' else 'A')+' '+m['content'].encode().hex()+'\n' for m in messages)
        with self.control:
            self.active=True; self.started=False; self.cancel=False
        # A token limit or cooperative stop can split a UTF-8 character.
        # Publish a replacement for an incomplete final character, just as
        # TextDecoder does, and replay exactly this visible text next turn.
        decoder=codecs.getincrementaldecoder('utf-8')('replace'); answer=''
        try:
            self.process.stdin.write((header+body).encode()); self.process.stdin.flush()
            while True:
                event=self.events.get()
                kind=event['event']
                if kind=='started':
                    with self.control:
                        self.started=True
                        if self.cancel: self.process.send_signal(signal.SIGUSR1)
                elif kind=='token':
                    text=decoder.decode(bytes.fromhex(event.pop('hex')))
                    answer+=text; event['text']=text
                elif kind=='done':
                    tail=decoder.decode(b'',final=True)
                    if tail: answer+=tail; yield {'event':'token','text':tail}
                    # Keep exactly the assistant text seen by the user, including a stopped prefix.
                    # Cancelling before any output leaves the previous conversation intact.
                    with self.control:
                        if answer: self.history=messages+[{'role':'assistant','content':answer}]
                        self.metrics=event
                    yield event; return
                elif kind=='error':
                    yield event; return
                yield event
        except Exception as error:
            self.stop()
            # A broken worker protocol must not admit a second request while
            # the first may still own the device or leave unconsumed events.
            with self.control:
                self.state='error'; self.error='Errore del motore. Riavvia il servizio. '+str(error)
            raise
        finally:
            with self.control:
                self.active=False; self.started=False

    def reset(self):
        self.process.stdin.write(b'R\n'); self.process.stdin.flush()
        event=self.events.get()
        if event['event']!='reset': raise RuntimeError(event.get('message','Reset fallito'))
        with self.control: self.history=[]; self.metrics=None
        return {'reset':True}

    def close(self):
        with self.control: self.closing=True
        self.stop()
        if self.process and self.process.poll() is None:
            try: self.process.stdin.write(b'Q\n'); self.process.stdin.flush()
            except (BrokenPipeError,OSError): pass
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()

class Handler(BaseHTTPRequestHandler):
    def local_request(self):
        port=self.server.server_port
        hosts=(f'127.0.0.1:{port}',f'localhost:{port}')
        origin=self.headers.get('Origin')
        return self.headers.get('Host') in hosts and (not origin or origin in tuple('http://'+h for h in hosts))

    def respond(self,code,value,kind='application/json'):
        data=json.dumps(value,ensure_ascii=False).encode() if kind=='application/json' else value
        self.send_response(code); self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if not self.local_request(): self.respond(403,{'error':'Origine locale richiesta.'}); return
        if self.path=='/': self.respond(200,(Path(__file__).with_name('index.html')).read_bytes(),'text/html; charset=utf-8')
        elif self.path=='/api/status': self.respond(200,self.server.engine.status())
        else: self.respond(404,{'error':'Risorsa inesistente'})

    def do_POST(self):
        if not self.local_request():
            self.respond(403,{'error':'Sono accettate solo richieste locali dalla stessa origine.'}); return
        if self.headers.get('Content-Type')!='application/json':
            self.respond(415,{'error':'È richiesto JSON.'}); return
        engine=self.server.engine
        acquired=False; streaming=False; stream=None; finished=False
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=262144: raise ValueError('Richiesta troppo grande o vuota.')
            data=json.loads(self.rfile.read(size))
            if self.path=='/api/stop': self.respond(200,engine.stop()); return
            if self.path not in ('/api/generate','/api/reset'):
                self.respond(404,{'error':'Risorsa inesistente'}); return
            if engine.state!='ready': self.respond(503,{'error':engine.error or 'Il modello è ancora in caricamento.'}); return
            acquired=engine.mutex.acquire(blocking=False)
            if not acquired: self.respond(409,{'error':'Una generazione è già in corso.'}); return
            if self.path=='/api/reset': self.respond(200,engine.reset()); return
            stream=engine.generate(data.get('prompt'),data.get('max_new',256))
            # Validate/start before sending a 200 response.
            first=next(stream)
            self.send_response(200); self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control','no-store'); self.send_header('Connection','close'); self.end_headers()
            streaming=True
            disconnected=False
            for event in self.events(first,stream):
                if disconnected: continue
                try:
                    self.wfile.write((json.dumps(event,ensure_ascii=False)+'\n').encode()); self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):
                    disconnected=True; engine.stop()
            finished=True
            self.close_connection=True
        except (ValueError,TypeError,AttributeError,StopIteration) as error:
            self.failure(error,400,streaming)
        except Exception as error:
            self.failure(error,500,streaming)
        finally:
            if acquired:
                # Covers a disconnect while sending HTTP headers as well as
                # one in the token loop. Never leave native events for the
                # next conversation request to accidentally consume.
                if stream is not None and not finished:
                    engine.stop()
                    try:
                        for _ in stream: pass
                    except Exception: pass  # Engine has already entered error state.
                engine.mutex.release()

    def failure(self,error,code,streaming):
        try:
            if streaming:
                self.wfile.write((json.dumps({'event':'error','message':str(error)})+'\n').encode())
                self.close_connection=True
            else: self.respond(code,{'error':str(error)})
        except (BrokenPipeError,ConnectionResetError): pass

    @staticmethod
    def events(first,stream):
        yield first
        yield from stream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/Qwen3.8-Flash-Next-Q2/Qwen3.8-Flash-Next-Q2.gguf')
    p.add_argument('--port',type=int,default=8001);p.add_argument('--context',type=int,default=8192)
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--mtp',action='store_true',help='Enable verified greedy MTP proposals in the resident worker')
    a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    server.engine=Engine(a.checkpoint.resolve(),a.context,a.chunk,a.mtp)
    print(f'Chat locale: http://127.0.0.1:{server.server_port}',flush=True)
    def shutdown(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,shutdown)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.engine.close(); server.server_close()

if __name__=='__main__': main()
