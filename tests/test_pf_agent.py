"""pf-agent tools and tool-call parser, offline: no model is loaded.

`pf-agent --run` executes the tool calls written in stdin (the model's XML
format) inside a scratch workspace; `pf-agent --parse` prints what the parser
extracted. Run with `make test-pf-agent`.
"""
import json
import os
import subprocess
import tempfile
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

    def run_calls(self, *calls, yes=False):
        args = [str(AGENT), '--run'] + (['--yes'] if yes else [])
        p = subprocess.run(args, input=''.join(calls).encode(), cwd=self.ws,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
                           stdin=None)
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
        out = self.run_calls(call('search', pattern='Foo('),
                             call('search', pattern='^int [a-z]+', regex='true'),
                             call('search', pattern='foo', ignore_case='true', path='src'),
                             call('search', pattern='nothing here'))
        self.assertIn('src/m.c:2:   return Foo(1);', out)
        self.assertIn('src/m.c:1: int main(void) {', out)
        self.assertNotIn('.git', out)
        self.assertEqual(out.count('src/m.c:2:'), 2)
        self.assertIn('No matches in 1 files.', out)

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


if __name__ == '__main__':
    unittest.main(verbosity=1)
