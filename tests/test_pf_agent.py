"""pf-agent tools and tool-call parser, offline: no model is loaded.

`pf-agent --run` executes the tool calls written in stdin (the model's XML
format) inside a scratch workspace; `pf-agent --parse` prints what the parser
extracted. Run with `make test-pf-agent`.
"""
import json
import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'pf-agent'


def call(name, **params):
    body = ''.join(f'<parameter={k}>\n{v}\n</parameter>\n' for k, v in params.items())
    return f'<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>\n'


class AgentTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name).resolve() / 'ws'
        self.ws.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_calls(self, *calls, yes=False, env=None):
        args = [str(AGENT), '--run'] + (['--yes'] if yes else [])
        p = subprocess.run(args, input=''.join(calls).encode(), cwd=self.ws, env={**os.environ, **(env or {})},
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        return p.stdout.decode()

    def parse(self, text):
        p = subprocess.run([str(AGENT), '--parse'], input=text.encode(), stdout=subprocess.PIPE, timeout=60)
        return [json.loads(line) for line in p.stdout.decode().splitlines()]

    # ---------------------------------------------------------------- parser
    def test_parser_formats(self):
        text = ('Let me look.\n' + call('read', path='a.c', start='3') +
                call('bash', command='echo 1\necho 2') + '<tool_call>\n<function=list>\n')
        calls = self.parse(text)
        self.assertEqual(calls, [{'name': 'read', 'args': {'path': 'a.c', 'start': '3'}},
                                 {'name': 'bash', 'args': {'command': 'echo 1\necho 2'}}])

    def test_parser_value_containing_closing_tag(self):
        value = 'x\n</parameter>\nnot the end\n'
        calls = self.parse(call('write', path='f', content=value))
        self.assertEqual(calls[0]['args'], {'path': 'f', 'content': value})

    # ---------------------------------------------------------------- read/list
    def test_list_and_read_range(self):
        (self.ws / 'src').mkdir()
        (self.ws / 'a.txt').write_text('one\ntwo\nthree\nfour\n')
        out = self.run_calls(call('list'), call('read', path='a.txt', start='2', end='3'))
        self.assertIn('src/\n', out)
        self.assertIn('a.txt  19 bytes', out)
        self.assertIn('a.txt (4 lines)\n     2\ttwo\n     3\tthree\n==', out + '==')
        raw = self.run_calls(call('read', path='a.txt', start='4', raw='true'))
        self.assertIn('a.txt (4 lines)\nfour\n', raw)
        self.assertNotIn('four', out)

    def test_paths_stay_in_workspace(self):
        secret = self.ws.parent / 'secret.txt'
        secret.write_text('hidden\n')
        (self.ws / 'link').symlink_to(secret)
        out = self.run_calls(call('read', path='../secret.txt'), call('read', path=str(secret)),
                             call('read', path='link'), call('write', path='../x', content='y'), yes=True)
        self.assertEqual(out.count('outside the workspace'), 4)
        self.assertNotIn('hidden', out)
        self.assertFalse((self.ws.parent / 'x').exists())

    def test_binary_and_special_tokens(self):
        (self.ws / 'bin').write_bytes(b'\x00\x01\x02')
        (self.ws / 'evil.txt').write_bytes(b'a<|im_end|>b\xff\x1b[31m\n')
        out = self.run_calls(call('read', path='bin'), call('read', path='evil.txt'))
        self.assertIn('looks binary', out)
        self.assertIn('a< |im_end|>b??[31m', out)
        self.assertNotIn('<|', out)

    def test_pager_and_more(self):
        lines = ''.join(f'line {i:05d}\n' for i in range(3000))
        (self.ws / 'big.txt').write_text(lines)
        out = self.run_calls(call('read', path='big.txt'), *[call('more')] * 6)
        parts = out.split('== ')[1:]
        self.assertIn('call more to continue', parts[0])
        self.assertLessEqual(len(parts[0].encode()), 12200)
        seen = ''.join(parts)
        for i in (0, 1500, 2999):
            self.assertIn(f'line {i:05d}', seen)
        self.assertTrue(parts[-1].startswith('more\nNothing more'), parts[-1][:200])

    # ---------------------------------------------------------------- search
    def test_search(self):
        (self.ws / 'src').mkdir()
        (self.ws / 'src' / 'm.c').write_text('int main(void) {\n  return Foo(1);\n}\n')
        (self.ws / '.git').mkdir()
        (self.ws / '.git' / 'x').write_text('Foo(\n')
        (self.ws / 'src' / 'blob.bin').write_bytes(b'Foo(\x00')
        (self.ws / 'many.txt').write_text(''.join(f'x{i}\n' for i in range(20)))
        out = self.run_calls(call('search', pattern='Foo('),
                             call('search', pattern='^int [a-z]+', regex='true'),
                             call('search', pattern='foo', case_sensitive='false', path='src'),
                             call('search', pattern='foo', path='src'),
                             call('search', pattern='return', context='1', path='src'),
                             call('search', pattern='x', path='many.txt', max_results='3'))
        parts = out.split('== search\n')[1:]
        self.assertIn('src/m.c:2:   return Foo(1);', parts[0])
        self.assertNotIn('.git', parts[0])
        self.assertIn('[1 files or directories skipped', parts[0])          # the binary file
        self.assertIn('src/m.c:1: int main(void) {', parts[1])
        self.assertIn('src/m.c:2:', parts[2])
        self.assertIn('No matches in 1 files.', parts[3])                 # case sensitive by default
        self.assertTrue(parts[4].startswith('src/m.c-1- int main(void) {\nsrc/m.c:2:   return Foo(1);\nsrc/m.c-3- }\n'), parts[4])
        self.assertTrue(parts[5].startswith('many.txt:1: x0\nmany.txt:2: x1\nmany.txt:3: x2\n[stopped at 3 matches'), parts[5])

    # ---------------------------------------------------------------- write/edit
    def test_write_needs_approval(self):
        out = self.run_calls(call('write', path='new.txt', content='hi\n'))
        self.assertIn('declined', out)
        self.assertFalse((self.ws / 'new.txt').exists())

    def test_write_and_edit(self):
        f = self.ws / 'code.py'
        f.write_text('a = 1\nb = 2\nb = 2\n')
        os.chmod(f, 0o755)
        out = self.run_calls(call('edit', path='code.py', old='a = 1', new='a = 10\nc = 3'),
                             call('edit', path='code.py', old='b = 2', new='x'),
                             call('edit', path='code.py', old='zzz', new='x'),
                             call('write', path='sub.txt', content='first\nsecond\n'),
                             call('write', path='nodir/f.txt', content='x'), yes=True)
        self.assertIn('Edited code.py: lines 1-1 replaced by 2 lines.', out)
        self.assertIn('found 2 times', out)
        self.assertIn('not found', out)
        self.assertEqual(f.read_text(), 'a = 10\nc = 3\nb = 2\nb = 2\n')
        self.assertEqual(os.stat(f).st_mode & 0o777, 0o755)
        self.assertEqual((self.ws / 'sub.txt').read_text(), 'first\nsecond\n')
        self.assertIn('does not exist', out)
        self.assertEqual(sorted(p.name for p in self.ws.iterdir()), ['code.py', 'sub.txt'])
        os.link(self.ws / 'sub.txt', self.ws / 'twin.txt')                # hard links: left alone
        out = self.run_calls(call('edit', path='sub.txt', old='first', new='1st'), call('write', path='twin.txt', content='x'), yes=True)
        self.assertEqual(out.count('has 2 hard links'), 2)
        self.assertEqual((self.ws / 'twin.txt').read_text(), 'first\nsecond\n')

    # ---------------------------------------------------------------- bash
    def test_bash_basics(self):
        out = self.run_calls(call('bash', command='echo out; echo err >&2; pwd; cat; exit 3'),
                             call('bash', command='ls /dev/fd'), yes=True)
        self.assertIn(f'out\nerr\n{self.ws}\n[exit code 3 after', out)
        fds = out.split('== bash\n')[2].split('[exit code 0')[0].split()
        alone = subprocess.run(['/bin/bash', '-c', 'ls /dev/fd'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, close_fds=True).stdout.decode().split()
        self.assertEqual(fds, alone)   # what bash alone sees (ls opens two): nothing of ours leaks

    def test_bash_needs_approval(self):
        marker = self.ws / 'ran'
        out = self.run_calls(call('bash', command=f'touch {marker}'))
        self.assertIn('declined', out)
        self.assertFalse(marker.exists())

    def test_bash_jobs(self):
        out = self.run_calls(call('bash', command='echo early; sleep 2; echo late', refresh_sec='1'),
                             call('bash_status'),
                             call('bash', command='sleep 2.5'),
                             call('bash_status', id='1'),
                             call('bash', command='sleep 31.5 & sleep 31.5', refresh_sec='1'),
                             call('bash_stop', id='3'),
                             call('bash', command='sleep 30', timeout_sec='1', refresh_sec='10'),
                             call('bash', command='sleep 32.5', refresh_sec='1'),
                             call('bash_status', id='9'), yes=True)
        parts = out.split('== ')[1:]
        self.assertRegex(parts[0], r'^bash\nearly\n\[job 1 running for 1 s: .*\]\n\[still running as job 1')
        self.assertIn('job 1: [job 1 running', parts[1])
        self.assertRegex(parts[3], r'^bash_status\nearly\nlate\n\[exit code 0 after 2\.\d s\]\n$')
        self.assertIn('[killed by signal 15', parts[5])
        self.assertIn('[stopped by its timeout after 1 s]', parts[6])
        self.assertIn('no job 9', parts[8])
        for _ in range(20):            # the group got SIGTERM; give its last members a moment to go
            left = subprocess.run(['pgrep', '-f', 'sleep 3[012].5|sleep 30$'], stdout=subprocess.PIPE).stdout
            if not left:
                break
            time.sleep(0.05)
        self.assertEqual(left, b'')   # bash_stop and the timeout ended their groups, the exit ended job 5

    def test_bash_long_output(self):
        script = 'for i in $(seq 1 300); do echo line $i; done'
        out = self.run_calls(call('bash', command=script), call('bash', command=f'{script}; sleep 3', refresh_sec='1'),
                             call('bash_status', id='2', refresh_sec='1'), call('bash_status', id='2', refresh_sec='5'), yes=True)
        parts = out.split('== ')[1:]
        self.assertIn('line 100\n[... ', parts[0])               # head: 100 lines
        self.assertNotIn('line 101\n', parts[0].split('not shown')[0])
        self.assertIn('[... 1800 bytes not shown ...]\nline 281\n', parts[0])   # then the last 20
        self.assertIn('line 300\n[exit code 0', parts[0])
        self.assertNotIn('line 280\n', parts[0])
        self.assertRegex(parts[0], r'\[full output: /.*/job1\.log \(2592 bytes\), readable with read\]')
        self.assertIn('line 100\n[... ', parts[1])               # still running: its head first
        self.assertIn('[no new output]', parts[2])                # nothing new while it sleeps
        self.assertIn('line 300\n[exit code 0', parts[3])

    # ---------------------------------------------------------------- web (a fake curl, no network)
    def fake_curl(self, body, status, code=0):
        d = self.ws.parent
        (d / 'body').write_bytes(body if isinstance(body, bytes) else body.encode())
        script = d / 'curl'
        script.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$FAKE_ARGS"\ncat "$FAKE_BODY"\n'
                          'printf \'\\n\\001%s\' "$FAKE_STATUS"\nexit $FAKE_EXIT\n')
        script.chmod(0o755)
        return {'PF_AGENT_CURL': str(script), 'FAKE_ARGS': str(d / 'args'), 'FAKE_BODY': str(d / 'body'),
                'FAKE_STATUS': status, 'FAKE_EXIT': str(code)}

    def test_search(self):
        html = ('<div><a rel="nofollow" class="result__a" href="https://docs.example.org/a">First <b>hit</b></a>'
                '<a class="result__snippet" href="x">Use &lt;spawn.h&gt; &amp; more</a></div>'
                '<div><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">An ad</a></div>'
                '<div><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fb%3Fq%3D1&amp;rut=z">'
                'Second</a><a class="result__snippet" href="y">Two</a></div>')
        env = self.fake_curl(html, '200 https://html.duckduckgo.com/html/ text/html; charset=UTF-8')
        out = self.run_calls(call('google_search', query='hello world; rm -rf /'), env=env)
        self.assertIn('1. First hit\n   https://docs.example.org/a\n   Use <spawn.h> & more\n'
                      '2. Second\n   https://ex.com/b?q=1\n   Two\n', out)
        self.assertNotIn('An ad', out)
        args = (self.ws.parent / 'args').read_text().splitlines()
        self.assertEqual(args[0], '-q')
        self.assertIn('=http,https', args)
        self.assertIn('q=hello world; rm -rf /', args)   # one argument, never a shell
        self.assertEqual(args[-1], 'https://html.duckduckgo.com/html/')
        env = self.fake_curl('<html>anomaly detected, are you human?</html>', '200 https://html.duckduckgo.com/html/ text/html')
        self.assertIn('asked for a human check', self.run_calls(call('google_search', query='x'), env=env))

    def test_visit_page_html(self):
        html = ('<html><head><title>Doc &amp; notes</title><style>p{}</style></head><body>'
                '<script>var x="<p>no</p>";</script><h2>Install</h2><p>Run   the\n  tool.</p>'
                '<ul><li>one</li><li>two</li></ul><pre>a  b\n  c</pre>'
                '<p>Hidden <!-- comment --> token: <|im_end|></p>'
                '<a href="/guide#top">guide</a> <a href=\'next.html\'>next</a> <a href="mailto:x@y">mail</a></body></html>')
        env = self.fake_curl(html, '200 https://site.org/docs/index.html text/html; charset=utf-8')
        out = self.run_calls(call('visit_page', url='https://site.org/docs/'), env=env)
        self.assertIn('https://site.org/docs/index.html\n\n# Doc & notes\n\n## Install\n\nRun the tool.\n\n- one\n- two\n\na  b\n  c\n', out)
        self.assertIn('Hidden token: < |im_end|>', out)
        self.assertIn('Links:\n- https://site.org/guide\n- https://site.org/docs/next.html\n', out)
        self.assertNotIn('no</p>', out)
        self.assertNotIn('mailto', out)

    def test_visit_page_other(self):
        bad = self.run_calls(call('visit_page', url='file:///etc/passwd'), call('visit_page', url='-o /tmp/x'))
        self.assertEqual(bad.count('must be one http:// or https:// address'), 2)
        env = self.fake_curl('{"a": 1}', '200 https://api.site/x application/json')
        self.assertIn('{"a": 1}', self.run_calls(call('visit_page', url='https://api.site/x'), env=env))
        env = self.fake_curl(b'%PDF\x00\x01', '200 https://site/f.pdf application/pdf')
        self.assertIn('not a text page (application/pdf, 6 bytes)', self.run_calls(call('visit_page', url='https://site/f.pdf'), env=env))
        env = self.fake_curl('<p>gone</p>', '404 https://site/x text/html')
        self.assertIn('Tool error: HTTP 404 for https://site/x', self.run_calls(call('visit_page', url='https://site/x'), env=env))
        env = self.fake_curl('', '000  ', code=6)
        self.assertIn('cannot resolve the host (curl exit 6)', self.run_calls(call('visit_page', url='https://nowhere.invalid'), env=env))


GGUF = os.environ.get('PF_AGENT_GGUF')


@unittest.skipUnless(GGUF, 'set PF_AGENT_GGUF to a synthetic checkpoint (make test-pf-agent-model)')
class AgentLoop(unittest.TestCase):
    """The loop on a synthetic checkpoint: canned replies (--script) go through
    the real render/encode/tool path, then one real generation prefills the
    whole transcript on the GPU."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name).resolve()
        (self.dir / 'ws').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, replies, messages, *flags):
        (self.dir / 'script.txt').write_text('\n=====\n'.join(replies) + '\n' if replies else '')
        args = [str(AGENT), '-m', GGUF, '-w', str(self.dir / 'ws'), '--script', str(self.dir / 'script.txt'),
                '--token-ids', str(self.dir / 'ids.json'), '-n', '4', '-c', '16384', *flags]
        env = {**os.environ, 'PICOFORGE_SESSIONS': str(self.dir / 'sessions')}
        p = subprocess.run(args, input=''.join(m + '\n' for m in messages).encode(), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=300, env=env)
        self.assertEqual(p.returncode, 0, p.stderr.decode()[-2000:])
        self.stderr = p.stderr.decode(errors='replace')
        return p.stdout.decode(), json.loads((self.dir / 'ids.json').read_text())

    def prefilled(self):
        return [int(n) for n in re.findall(r'\[prefill (\d+) tokens', self.stderr)]

    def test_tool_round_trip(self):
        out, state = self.agent([call('write', path='a.txt', content='one\ntwo'),
                                 'Reading it.\n' + call('read', path='a.txt'), 'Done.'],
                                ['make a.txt', 'thanks'], '--yes', '--no-think')
        self.assertEqual((self.dir / 'ws' / 'a.txt').read_text(), 'one\ntwo')
        text = state['text']
        self.assertEqual(text.count('<|im_start|>system'), 1)
        self.assertIn('{"type": "function", "function": {"name": "edit"', text)   # tool definitions in the system block
        self.assertIn('</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nWrote 7 bytes to a.txt.\n'
                      '</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nReading it.', text)
        self.assertIn('     2\ttwo\n</tool_response>', text)
        self.assertIn('Done.<|im_end|>\n<|im_start|>user\nthanks<|im_end|>\n<|im_start|>assistant\n', text)
        self.assertEqual(state['kv'], len(state['ids']) - 1)   # the graph holds the transcript, minus the last token

    def test_bash_with_the_model_loaded(self):
        out, state = self.agent([call('bash', command='ls /dev/fd'), 'Ok.'], ['fds?'], '--yes', '--no-think')
        alone = subprocess.run(['/bin/bash', '-c', 'ls /dev/fd'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, close_fds=True).stdout.decode()
        self.assertRegex(state['text'], '<tool_response>\n' + re.escape(alone) + r'\[exit code 0 after \d+\.\d s\]\n</tool_response>')

    def test_saved_session_resumes_exactly(self):
        for mtp in ((), ('--no-mtp',)):
            g = ('--temp', '0', '-n', '6', *mtp)
            _, whole = self.agent([], ['hello', 'again'], *g)
            out, saved = self.agent([], ['hello', '/save s1', '/sessions'], *g)
            self.assertIn('Saved session s1:', out)
            self.assertRegex(out, r's1 +\d{4}-\d\d-\d\d \d\d:\d\d +\d+ tokens .*\n    hello\n')
            self.assertTrue((self.dir / 'sessions' / 's1.txt').read_bytes().decode().endswith(saved['text']))
            out, resumed = self.agent([], ['again'], '--resume', 's1', *g)
            self.assertIn('Resumed session s1', out)
            self.assertEqual(resumed['ids'], whole['ids'])      # the same tokens as never stopping
            self.assertEqual(resumed['kv'], whole['kv'])
            new_turn = len(whole['ids']) - len(saved['ids'])
            self.assertLess(self.prefilled()[0], new_turn)        # loaded, not replayed: only the new turn
            self.assertNotIn('replayed', self.stderr)
            _, fresh = self.agent([], ['again'], *g)              # the test has power: context changes the reply
            self.assertNotEqual(fresh['ids'][-6:], whole['ids'][-6:])
            # /resume inside a session with another context size rebuilds the graph
            out, rebuilt = self.agent([], ['/resume s1', 'again'], '-c', '8192', *g)
            self.assertEqual(rebuilt['ids'], whole['ids'])
            self.assertLess(self.prefilled()[0], new_turn)
            (self.dir / 'sessions' / 's1.pfs').unlink()
        # the file grows with the conversation, not with the context size (zero blocks elided)
        self.agent([], ['hello', '/save small'], '--temp', '0', '-c', '8192')
        self.agent([], ['hello', '/save large'], '--temp', '0', '-c', '16384')
        small, large = ((self.dir / 'sessions' / f'{n}.pfs').stat().st_size for n in ('small', 'large'))
        self.assertLess(large, small * 1.25)
        self.agent([], ['hello', '/save s2'], '--temp', '0')
        _, state = self.agent([], ['/resume s2'], '--temp', '0', '--no-mtp')
        self.assertEqual(state['ids'], [])                         # refused: saved with MTP on

    def test_thinking_and_declined_write(self):
        reasoning = 'Maybe ' + call('list') + 'no, write it.\n</think>\n\n'
        out, state = self.agent([reasoning + call('write', path='b.txt', content='x'), 'Ok.'], ['write b'])
        self.assertFalse((self.dir / 'ws' / 'b.txt').exists())       # stdin is not a terminal: declined
        self.assertEqual(state['text'].count('<tool_response>'), 1)  # the call inside the reasoning did not run
        self.assertIn('the user declined', state['text'])
        self.assertIn('<|im_start|>assistant\n<think>\nMaybe', state['text'])


if __name__ == '__main__':
    unittest.main(verbosity=1)
