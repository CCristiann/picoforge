"""Qwen3.8 terminal: the interactive chat (KV reuse against a fresh one-shot run) and bench."""
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / 'build/qwen38-g2/cli.gguf'
ENGINE = str(ROOT / 'picoforge')

class ReplTests(unittest.TestCase):
    def repl(self, lines, *flags):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build/qwen38-g2') as directory:
            ids = Path(directory) / 'ids.json'
            done = subprocess.run([ENGINE, '--qwen38', str(GGUF), '--context', '160', '--chunk', '7',
                                   '--max-new', '6', '--token-ids', str(ids), *flags],
                                  cwd=ROOT, input=''.join(l + '\n' for l in lines).encode(), capture_output=True)
            self.assertEqual(done.returncode, 0, done.stderr.decode(errors='replace'))
            return done, json.loads(ids.read_text()) if ids.exists() else None

    def one_shot(self, prompt, max_new, context=160):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build/qwen38-g2') as directory:
            ids = Path(directory) / 'ids.json'
            subprocess.run([ENGINE, '--qwen38', str(GGUF), '--tokens', ','.join(map(str, prompt)),
                            '--context', str(context), '--chunk', '7', '--max-new', str(max_new), '--token-ids', str(ids)],
                           cwd=ROOT, check=True, capture_output=True)
            return json.loads(ids.read_text())['generated_ids']

    def test_turns_reuse_the_cache_exactly(self):
        done, state = self.repl(['first message', 'second message', 'third'])
        ids = state['ids']
        # The graph holds the whole transcript but the last reply token.
        self.assertEqual(state['kv'], len(ids) - 1)
        self.assertEqual(done.stderr.decode().count('generation'), 3)
        # Each turn starts with <|im_start|>user (257); the last reply is the last 6 IDs
        # (no stop on this model). Prefilling everything before it in one fresh run,
        # chunked differently, must give the same reply: chunk/decode bit-identity.
        starts = [i for i, t in enumerate(ids) if t == 257 and ids[i + 1:i + 5] == [117, 115, 101, 114]]
        self.assertEqual(len(starts), 3)
        self.assertEqual(self.one_shot(ids[:-6], 6), ids[-6:])
        self.assertEqual(self.one_shot(ids[:starts[1] - 2 - 6], 6), ids[starts[1] - 2 - 6:starts[1] - 2])

    def test_mtp_transcript_matches_plain(self):
        _, plain = self.repl(['one', 'two', 'three'])
        _, mtp = self.repl(['one', 'two', 'three'], '--mtp')
        self.assertEqual(plain['ids'], mtp['ids'])
        self.assertEqual(mtp['kv'], len(mtp['ids']) - 1)

    def test_commands_and_full_context(self):
        done, state = self.repl(['/help', '/nope', '/quit', 'never sent'])
        self.assertIn(b'/quit', done.stdout)
        self.assertIn(b'unknown command /nope', done.stderr)
        self.assertIsNone(state)
        done, state = self.repl(['short', 'x' * 400])
        self.assertIn(b'context is full', done.stderr)
        self.assertLessEqual(len(state['ids']), 160)

    def interrupted(self, wait_for_text, *flags):
        """Ctrl+C mid-turn, then one more turn: the process survives, the transcript stays coherent."""
        with tempfile.TemporaryDirectory(dir=ROOT / 'build/qwen38-g2') as directory:
            ids = Path(directory) / 'ids.json'
            p = subprocess.Popen([ENGINE, '--qwen38', str(GGUF), '--context', '1024', '--chunk', '1',
                                  '--token-ids', str(ids), *flags], cwd=ROOT, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.stdin.write(b'a long first turn ' * 20 + b'\n'); p.stdin.flush()
            if wait_for_text: p.stdout.read(1)   # the reply has started streaming
            else: time.sleep(0.3)                 # load + first chunks of the prefill
            p.send_signal(signal.SIGINT)
            out, err = p.communicate(b'second\n', timeout=120)
            self.assertEqual(p.returncode, 0, err.decode(errors='replace'))
            state = json.loads(ids.read_text())
            return err.decode(), state

    def test_ctrl_c_keeps_the_partial_reply(self):
        for flags in ((), ('--mtp',)):
            err, state = self.interrupted(True, *flags)
            self.assertEqual(err.count('[interrupted]'), 1, err)
            self.assertEqual(err.count('generation'), 2)
            ids = state['ids']
            self.assertLessEqual(state['kv'], len(ids))
            # The second turn closed the cut reply: <|im_end|>\n then <|im_start|>user.
            second = max(i for i, t in enumerate(ids) if t == 257 and ids[i + 1:i + 5] == [117, 115, 101, 114])
            self.assertEqual(ids[second - 2:second], [258, 10])
            # The reply after the cut equals a fresh run of everything before it.
            k = max(i for i in range(len(ids) - 2) if ids[i:i + 3] == [281, 10, 10]) + 3
            self.assertEqual(self.one_shot(ids[:k], 6, 1024), ids[k:k + 6])

    def test_ctrl_c_in_prefill_drops_the_turn(self):
        err, state = self.interrupted(False)
        if '[interrupted]' not in err: self.skipTest('the signal landed after the first turn')
        ids = state['ids']
        users = [i for i, t in enumerate(ids) if t == 257 and ids[i + 1:i + 5] == [117, 115, 101, 114]]
        if len(users) == 1:                       # dropped: only "second" remains, replayed from 0
            self.assertEqual(users, [0])

    def test_read_ctx_and_reset(self):
        _, fresh = self.repl(['hello from a file'])
        with tempfile.NamedTemporaryFile('w', dir=ROOT / 'build/qwen38-g2', suffix='.txt') as f:
            f.write('hello from a file\n'); f.flush()
            done, state = self.repl([f'/read {f.name}', '/read /nonexistent/x', '/read'])
        self.assertEqual(state['ids'], fresh['ids'])          # file text == typed text (trimmed)
        self.assertIn(b'cannot open /nonexistent/x', done.stderr)
        self.assertIn(b'/read needs a file path', done.stderr)
        done, state = self.repl(['first', '/ctx 1', '/ctx 300', 'hello from a file'])
        self.assertIn(b'/ctx needs N in [2,', done.stderr)
        self.assertIn(b'Context 300 tokens', done.stdout)
        self.assertEqual(state['ids'], fresh['ids'])          # a new graph, a new conversation
        done, state = self.repl(['first', '/reset', 'hello from a file'], '--mtp')
        self.assertEqual(state['ids'], fresh['ids'])
        self.assertEqual(state['kv'], len(fresh['ids']) - 1)

    def test_thinking_toggle(self):
        _, state = self.repl(['plain', '/think', 'reason', '/nothink', 'plain again'])
        ids = state['ids']
        prompts = [ids[i + 11:i + 15] for i in range(len(ids) - 14)
                   if ids[i:i + 10] == [257, 97, 115, 115, 105, 115, 116, 97, 110, 116]]
        # <think>\n\n</think> when off; an open <think>\n when on. No system block later.
        closed = [280, 10, 10, 281]
        self.assertEqual(len(prompts), 3)
        self.assertEqual((prompts[0], prompts[2]), (closed, closed))
        self.assertEqual(prompts[1][:2], [280, 10]); self.assertNotEqual(prompts[1], closed)
        self.assertEqual(ids[:4], [257, 117, 115, 101])

    def test_short_form(self):
        long = subprocess.run([ENGINE, '--qwen38', str(GGUF), '--chat', 'hi', '--context', '64', '--max-new', '4'],
                              cwd=ROOT, capture_output=True, check=True)
        short = subprocess.run([ENGINE, '-m', str(GGUF), '-p', 'hi', '-c', '64', '-n', '4'],
                               cwd=ROOT, capture_output=True, check=True)
        self.assertEqual(short.stdout, long.stdout)
        env = dict(os.environ, PICOFORGE_MODEL=str(GGUF))
        chat = subprocess.run([ENGINE, '-c', '64', '-n', '4'], cwd=ROOT, input=b'hi\n', env=env,
                              capture_output=True, check=True)
        self.assertEqual(chat.stdout, long.stdout)
        env.pop('PICOFORGE_MODEL')
        none = subprocess.run([ENGINE, '-p', 'hi'], cwd=ROOT, env=env, capture_output=True)
        self.assertNotEqual(none.returncode, 0)
        self.assertIn(b'no model', none.stderr)

class BenchTests(unittest.TestCase):
    def test_summary_matches_the_raw_rows(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=ROOT / 'build/qwen38-g2') as directory:
            csv = Path(directory) / 'bench.csv'
            done = subprocess.run([ENGINE, 'bench', '-m', str(GGUF), '-c', '128', '-n', '8', '--bench-csv', str(csv)],
                                  cwd=ROOT, capture_output=True, check=True)
            rows = np.genfromtxt(csv, delimiter=',', names=True)
        out = done.stdout.decode()
        self.assertIn('bench: 20 runs (+3 warm-up)', out)       # protocol defaults
        self.assertEqual(len(rows), 20)
        self.assertNotIn('lighthouse', out)                      # generated text is discarded
        table = {l[:16].strip(): [float(x) for x in l[16:].split()] for l in out.splitlines()[2:5]}
        for name, values in (('prefill tok/s', rows['prompt_tokens'] / rows['prefill_s']),
                             ('decode tok/s', rows['decode_tokens'] / rows['decode_s']), ('ttft s', rows['ttft_s'])):
            expected = [np.percentile(values, q) for q in (50, 10, 90)]
            np.testing.assert_allclose(table[name], expected, rtol=1e-3, atol=1e-4)

    def test_admission(self):
        for args in (['--qwen38', str(GGUF), '--chat', 'x', '--weight-mb', '1'],
                     ['bench', '-m', str(GGUF), '--weight-mb', '-3'], ['bench', '-m', str(GGUF), '--mtp', '--weight-mb', '6000'], ['bench', '-m', str(GGUF), '--session']):
            done = subprocess.run([ENGINE, *args], cwd=ROOT, capture_output=True)
            self.assertNotEqual(done.returncode, 0, args)

if __name__ == '__main__':
    unittest.main()
