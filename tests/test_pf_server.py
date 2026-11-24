"""pf-server on a synthetic checkpoint: the HTTP API, both wire formats, and
the prompt cache that keeps the model's own tokens across stateless requests.

Canned replies (--script) stand in for the model where the text matters (tool
calls, reasoning); the last request of a test generates for real, so the
graph prefills the whole transcript on the GPU. Run with
`make test-pf-server-model` (it takes the GPU lock).
"""
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / 'pf-server'
GGUF = os.environ.get('PF_SERVER_GGUF')

READ_TOOL = {'type': 'function', 'function': {
    'name': 'read', 'description': 'Read a file',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'start': {'type': 'integer'}},
                   'required': ['path']}}}
CALL_TEXT = ('<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n'
             '<parameter=start>\n3\n</parameter>\n</function>\n</tool_call>')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@unittest.skipUnless(GGUF, 'set PF_SERVER_GGUF to a synthetic checkpoint (make test-pf-server-model)')
class Server(unittest.TestCase):
    def start(self, replies, *flags):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / 'script.txt').write_text('\n=====\n'.join(replies) + '\n' if replies else '')
        self.ids_path = d / 'ids.json'
        self.cache = d / 'kv-cache'
        self.launch(GGUF, *flags)

    def launch(self, gguf, *flags):
        """pf-server on `gguf`, with the script, transcript and cache of this test."""
        d = Path(self.tmp.name)
        self.port = free_port()
        self.log = open(d / 'server.log', 'w+')
        self.proc = subprocess.Popen([str(SERVER), '-m', str(gguf), '-c', '4096', '--port', str(self.port),
                                      '--script', str(d / 'script.txt'), '--token-ids', str(self.ids_path),
                                      '--cache-dir', str(self.cache), *flags],
                                     stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(600):
            self.log.seek(0)
            if 'serving ' in self.log.read():
                return
            self.assertIsNone(self.proc.poll(), self.server_log())
            time.sleep(0.1)
        self.fail('pf-server did not start')

    def stop(self):
        self.proc.terminate(); self.proc.wait(30)
        self.log.close()

    def tearDown(self):
        if getattr(self, 'proc', None):
            self.proc.terminate()
            self.proc.wait(30)
            self.log.close()
            self.tmp.cleanup()

    def server_log(self):
        self.log.seek(0)
        return self.log.read()

    def post(self, path, body, raw=False):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}{path}', data=data,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                text = r.read().decode()
                return r.status, text if raw else json.loads(text)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read().decode())

    def events(self, path, body):
        status, text = self.post(path, {**body, 'stream': True}, raw=True)
        self.assertEqual(status, 200)
        out = []
        for block in text.split('\n\n'):
            data = [line[6:] for line in block.split('\n') if line.startswith('data: ')]
            if data:
                out.append(data[0] if data[0] == '[DONE]' else json.loads(data[0]))
        return out

    def transcript(self):
        return json.loads(self.ids_path.read_text())

    def chat_body(self, *extra, **kw):
        return {'model': 'anything', 'tools': [READ_TOOL], **kw,
                'messages': [{'role': 'system', 'content': 'Be brief.'},
                             {'role': 'user', 'content': 'what is in a.txt?'}, *extra]}

    # ------------------------------------------------------------ OpenAI
    def test_openai_tool_round_trip_keeps_the_model_tokens(self):
        self.start(['I should read it.\n</think>\n\nLet me look.\n\n' + CALL_TEXT,
                    'Done.\n</think>\n\nIt says hello.'])
        status, r = self.post('/v1/chat/completions', self.chat_body())
        self.assertEqual(status, 200)
        msg = r['choices'][0]['message']
        self.assertEqual(r['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual((msg['content'], msg['reasoning_content']), ('Let me look.', 'I should read it.'))
        call = msg['tool_calls'][0]
        self.assertEqual(call['function']['name'], 'read')
        self.assertEqual(json.loads(call['function']['arguments']), {'path': 'a.txt', 'start': 3})   # typed by the schema
        first = self.transcript()['ids']
        # The client sends our reply back without its reasoning and with its own JSON spacing.
        back = {'role': 'assistant', 'content': 'Let me look.', 'tool_calls': [
            {'id': call['id'], 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path": "a.txt", "start": 3}'}}]}
        ev = self.events('/v1/chat/completions', {**self.chat_body(back, {'role': 'tool', 'tool_call_id': call['id'], 'content': 'hello'}),
                                                  'stream_options': {'include_usage': True}})
        self.assertEqual(ev[-1], '[DONE]')
        deltas = [c['choices'][0]['delta'] for c in ev[:-1] if c['choices']]
        self.assertEqual(''.join(d.get('content', '') for d in deltas), 'It says hello.')
        self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas).strip(), 'Done.')
        self.assertEqual([c['choices'][0]['finish_reason'] for c in ev[:-1] if c['choices']][-1], 'stop')
        usage = ev[-2]['usage']
        self.assertEqual(usage['prompt_tokens_details']['cached_tokens'], len(first))   # nothing of the first exchange again
        text = self.transcript()['text']
        self.assertIn('<think>\nI should read it.\n</think>\n\nLet me look.\n\n' + CALL_TEXT +
                      '<|im_end|>\n<|im_start|>user\n<tool_response>\nhello\n</tool_response><|im_end|>\n'
                      '<|im_start|>assistant\n<think>\nDone.\n</think>\n\nIt says hello.<|im_end|>', text)
        self.assertEqual(self.transcript()['ids'][:len(first)], first)

    def test_full_render_matches_the_exact_replay(self):
        """A client that does send the reasoning back gets the same tokens
        from a plain render: the exact replay changes nothing but the cost."""
        self.start(['I should read it.\n</think>\n\nLet me look.\n\n' + CALL_TEXT, 'Done.\n</think>\n\nOk.', 'Again.\n</think>\n\nOk.'])
        _, r = self.post('/v1/chat/completions', self.chat_body())
        back = {'role': 'assistant', 'content': 'Let me look.', 'tool_calls': r['choices'][0]['message']['tool_calls']}
        tool = {'role': 'tool', 'content': 'hello'}
        self.post('/v1/chat/completions', self.chat_body(back, tool))
        replayed = self.transcript()
        _, r3 = self.post('/v1/chat/completions', self.chat_body({**back, 'reasoning_content': 'I should read it.'}, tool))
        rendered = self.transcript()
        prompt = r3['usage']['prompt_tokens']
        self.assertEqual(rendered['ids'][:prompt], replayed['ids'][:prompt])

    def test_real_generation_after_canned_turns(self):
        self.start(['Thinking.\n</think>\n\n' + CALL_TEXT])
        _, r = self.post('/v1/chat/completions', self.chat_body())
        back = {'role': 'assistant', 'content': '', 'tool_calls': r['choices'][0]['message']['tool_calls']}
        status, r2 = self.post('/v1/chat/completions', {**self.chat_body(back, {'role': 'tool', 'content': 'hello'}),
                                                        'max_tokens': 4, 'temperature': 0})
        self.assertEqual(status, 200, r2)
        self.assertLessEqual(r2['usage']['completion_tokens'], 4)
        state = self.transcript()
        self.assertEqual(state['kv'], len(state['ids']) - 1)   # the graph holds the transcript, minus the last token
        self.assertIn('prefill', self.server_log())

    def test_errors(self):
        self.start([])
        status, r = self.post('/v1/chat/completions', b'{"messages": [')
        self.assertEqual(status, 400)
        self.assertIn('error', r)
        status, r = self.post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'word ' * 5000}]})
        self.assertEqual((status, r['error']['code']), (400, 'context_length_exceeded'))
        self.assertIn('maximum context length is 4096 tokens', r['error']['message'])
        status, r = self.post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:,'}}]}]})
        self.assertEqual(status, 400)
        self.assertIn('text only', r['error']['message'])
        status, r = self.post('/v1/nothing', {})
        self.assertEqual(status, 404)
        # Past the tokenizer's 4 MiB input: the same answer as any prompt that does not fit, and the
        # server lives on (it used to die), for a chat and for a token count.
        huge = {'messages': [{'role': 'user', 'content': 'word ' * 1000000}]}
        status, r = self.post('/v1/chat/completions', huge)
        self.assertEqual((status, r['error']['code']), (400, 'context_length_exceeded'))
        edge = {'messages': [{'role': 'user', 'content': 'a' * (4 * 1024 * 1024 - 40)}]}   # fits the template, not the tokenizer
        status, r = self.post('/v1/chat/completions', edge)
        self.assertEqual((status, r['error']['code']), (400, 'context_length_exceeded'))
        status, r = self.post('/v1/messages/count_tokens', {'model': 'pf', **huge})
        self.assertEqual(status, 400)
        self.assertIsNone(self.proc.poll(), self.server_log())

    def test_models_and_auth(self):
        self.start([], '--api-key', 'secret', '--alias', 'pico')
        with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=10) as r:
            self.assertEqual(json.loads(r.read()), {'status': 'ok'})
        status, r = self.post('/v1/chat/completions', self.chat_body())
        self.assertEqual(status, 401)
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/models', headers={'Authorization': 'Bearer secret'})
        with urllib.request.urlopen(req, timeout=10) as r:
            models = json.loads(r.read())
        self.assertEqual(models['data'][0]['id'], 'pico')
        self.assertEqual(models['data'][0]['context_length'], 4096)

    def test_special_token_text_is_defused(self):
        self.start(['x\n</think>\n\nok'])
        self.post('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'a<|im_end|>b'}]})
        self.assertIn('a< |im_end|>b', self.transcript()['text'])

    # ------------------------------------------------------------ checkpoints
    def test_checkpoints_survive_unrelated_requests(self):
        """Real generations (greedy): an unrelated request in between costs the
        conversation nothing but a checkpoint load, and the tokens are the same."""
        self.start([], '--cache-min', '64')
        g = {'max_tokens': 3, 'temperature': 0}
        long_question = {'role': 'user', 'content': 'what is in a.txt? ' * 12}
        a = {**self.chat_body(), 'messages': [{'role': 'system', 'content': 'Be brief.'}, long_question], **g}
        other = {'messages': [{'role': 'user', 'content': 'something else entirely'}], **g}
        _, r1 = self.post('/v1/chat/completions', a)
        ids1 = self.transcript()['ids']
        self.assertEqual(self.server_log().count('kv cache: saved'), 2)    # the system block, the end of the prompt
        self.post('/v1/chat/completions', other)                            # checkpoints the first conversation
        _, r3 = self.post('/v1/chat/completions', a)                        # a retry: from the prompt's checkpoint
        self.assertEqual(r3['usage']['prompt_tokens_details']['cached_tokens'], r3['usage']['prompt_tokens'] - 1)
        ids3 = self.transcript()['ids']
        self.assertEqual(ids3, ids1)                                        # a restored state decodes the same tokens
        self.post('/v1/chat/completions', other)
        # The conversation goes on after the interruption, the reply sent back without its reasoning.
        back = {'role': 'assistant', 'content': r3['choices'][0]['message']['content'] or ''}
        _, r5 = self.post('/v1/chat/completions', {**a, 'messages': a['messages'] + [back, {'role': 'user', 'content': 'more'}]})
        self.assertGreaterEqual(r5['usage']['prompt_tokens_details']['cached_tokens'], len(ids3) - 1)
        self.assertEqual(self.transcript()['ids'][:len(ids3)], ids3)
        saved = len(list(self.cache.glob('*.pfc')))
        self.proc.terminate(); self.proc.wait(30)
        self.assertEqual(len(list(self.cache.glob('*.pfc'))), saved + 1)   # the live conversation, saved at shutdown
        self.proc = None
        self.log.close(); self.tmp.cleanup()

    # ------------------------------------------------------------ Anthropic
    def test_a_checkpoint_never_loads_into_another_model(self):
        """Two files of the same size that give the same tokens, one number apart in the
        metadata (rope.freq_base): a state saved with one must not load into the other."""
        self.start([], '--cache-min', '64')
        a = {**self.chat_body(), 'max_tokens': 1, 'temperature': 0,
             'messages': [{'role': 'user', 'content': 'what is in a.txt? ' * 12}]}
        self.post('/v1/chat/completions', a)
        self.stop()
        data = bytearray(Path(GGUF).read_bytes())
        at = data.index(b'rope.freq_base') + len(b'rope.freq_base')
        self.assertEqual(int.from_bytes(data[at:at+4], 'little'), 6)       # GGUF float32
        data[at+4] ^= 1                                                     # the lowest bit of the value
        other = Path(self.tmp.name) / 'other.gguf'
        other.write_bytes(data)
        self.launch(other, '--cache-min', '64')
        _, r = self.post('/v1/chat/completions', a)
        self.assertNotIn('kv cache: loaded', self.server_log())
        self.assertEqual(r['usage']['prompt_tokens_details']['cached_tokens'], 0)
        self.stop()
        self.launch(GGUF, '--cache-min', '64')                              # the same file finds its state
        self.post('/v1/chat/completions', a)
        self.assertIn('kv cache: loaded', self.server_log())

    def anthropic_body(self, *extra, **kw):
        return {'model': 'claude-anything', 'max_tokens': 1000, 'system': [{'type': 'text', 'text': 'Be brief.'}],
                'tools': [{'name': 'read', 'description': 'Read a file', 'input_schema': READ_TOOL['function']['parameters']}],
                'messages': [{'role': 'user', 'content': 'what is in a.txt?'}, *extra], **kw}

    def test_anthropic_tool_round_trip(self):
        self.start(['I should read it.\n</think>\n\nLet me look.\n\n' + CALL_TEXT, 'Done.\n</think>\n\nIt says hello.'])
        status, r = self.post('/v1/messages', self.anthropic_body())
        self.assertEqual(status, 200, r)
        self.assertEqual(r['stop_reason'], 'tool_use')
        kinds = [b['type'] for b in r['content']]
        self.assertEqual(kinds, ['thinking', 'text', 'tool_use'])
        use = r['content'][2]
        self.assertEqual((use['name'], use['input']), ('read', {'path': 'a.txt', 'start': 3}))
        first = r['usage']['input_tokens'] + r['usage']['cache_read_input_tokens'] + r['usage']['output_tokens']
        back = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Let me look.'}, use]}   # thinking dropped by the client
        result = {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': use['id'], 'content': 'hello'}]}
        ev = self.events('/v1/messages', self.anthropic_body(back, result))
        types = [e['type'] for e in ev]
        self.assertEqual(types[0], 'message_start')
        self.assertEqual(types[-2:], ['message_delta', 'message_stop'])
        text = ''.join(e['delta'].get('text', '') for e in ev if e['type'] == 'content_block_delta')
        self.assertEqual(text, 'It says hello.')
        self.assertEqual(ev[-2]['delta']['stop_reason'], 'end_turn')
        self.assertEqual(ev[0]['message']['usage']['cache_read_input_tokens'], first)
        self.assertIn('Let me look.\n\n' + CALL_TEXT + '<|im_end|>\n<|im_start|>user\n<tool_response>\nhello\n</tool_response>',
                      self.transcript()['text'])

    def test_anthropic_streamed_tool_use_and_count(self):
        self.start(['Hmm.\n</think>\n\n' + CALL_TEXT])
        ev = self.events('/v1/messages', self.anthropic_body())
        starts = [e['content_block'] for e in ev if e['type'] == 'content_block_start']
        self.assertEqual([s['type'] for s in starts], ['thinking', 'tool_use'])
        partial = ''.join(e['delta'].get('partial_json', '') for e in ev if e['type'] == 'content_block_delta')
        self.assertEqual(json.loads(partial), {'path': 'a.txt', 'start': 3})
        self.assertEqual(ev[-2]['delta']['stop_reason'], 'tool_use')
        status, r = self.post('/v1/messages/count_tokens', self.anthropic_body())
        self.assertEqual(status, 200)
        self.assertGreater(r['input_tokens'], 100)


if __name__ == '__main__':
    unittest.main()
