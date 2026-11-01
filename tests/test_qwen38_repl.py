"""Qwen3.8 interactive chat: multi-turn KV reuse against a fresh one-shot run."""
import json
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

    def one_shot(self, prompt, max_new):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build/qwen38-g2') as directory:
            ids = Path(directory) / 'ids.json'
            subprocess.run([ENGINE, '--qwen38', str(GGUF), '--tokens', ','.join(map(str, prompt)),
                            '--context', '160', '--chunk', '7', '--max-new', str(max_new), '--token-ids', str(ids)],
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

if __name__ == '__main__':
    unittest.main()
