"""Qwen3.8 ordinary CLI against the independent CPU graph at each greedy step."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / 'build/qwen38-g2/cli.gguf'
PROMPT = [5, 17, 300]

class CliTests(unittest.TestCase):
    def run_cli(self, command):
        done = subprocess.run(command, cwd=ROOT, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr.decode(errors='replace'))
        return done

    def test_greedy_and_last_logits(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build/qwen38-g2') as directory:
            path = Path(directory)
            cmd = [str(ROOT / 'picoforge'), '--qwen38', str(GGUF), '--tokens', ','.join(map(str, PROMPT)),
                   '--context', '32', '--chunk', '3', '--max-new', '5', '--logits', str(path/'gpu.f32'),
                   '--token-ids', str(path/'ids.json')]
            self.run_cli(cmd)
            info = json.loads((path/'ids.json').read_text())
            self.assertEqual(info['prompt_ids'], PROMPT)
            ids = info['generated_ids']
            self.assertGreater(len(ids), 0)
            cpu = path/'cpu'; cpu.mkdir()
            subprocess.run([str(ROOT/'build/qwen38-g1/test_forward'), str(GGUF), str(cpu), '4', '3',
                            *map(str, PROMPT+ids[:-1])], cwd=ROOT, check=True, capture_output=True)
            reference = np.fromfile(cpu/'logits.f32', '<f4').reshape(-1, 512)[len(PROMPT)-1:]
            actual = np.fromfile(path/'gpu.f32', '<f4').reshape(-1, 512)
            self.assertTrue(np.isfinite(actual).all())
            self.assertLessEqual(np.abs(actual-reference).max()/max(1, np.abs(reference).max()), 1e-3)
            np.testing.assert_array_equal(reference.argmax(1), ids)
            # Zero requested output still supplies the last prefill logit row.
            cmd[cmd.index('--max-new')+1] = '0'
            self.run_cli(cmd)
            info = json.loads((path/'ids.json').read_text())
            self.assertEqual(info['generated_ids'], [])
            last = np.fromfile(path/'gpu.f32', '<f4')
            np.testing.assert_array_equal(last, actual[0])

    def test_text_modes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            ids = Path(directory)/'ids.json'
            base = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--context', '512',
                    '--chunk', '7', '--max-new', '0', '--token-ids', str(ids)]
            self.run_cli(base+['--complete', 'caffè'])
            self.assertEqual(json.loads(ids.read_text())['prompt_ids'], list('caffè'.encode()))
            self.run_cli(base+['--chat', 'hi'])
            expected = [257, *b'user\nhi', 258, 10, 257, *b'assistant\n', 280, 10, 10, 281, 10, 10]
            self.assertEqual(json.loads(ids.read_text())['prompt_ids'], expected)

    def test_admission(self):
        base = [str(ROOT/'picoforge'), '--qwen38', str(GGUF)]
        for args in (['--tokens', '5,', '--max-new', '0'], ['--tokens', '512'],
                     ['--tokens', '5,17,300', '--context', '2'], ['--chunk', '1025', '--tokens', '5'],
                     ['--tokens', '5', '--thinking']):
            with self.subTest(args=args):
                done = subprocess.run(base+args, cwd=ROOT, capture_output=True)
                self.assertNotEqual(done.returncode, 0)
        self.assertEqual(subprocess.run([str(ROOT/'picoforge'), '--qwen38', '--help'],
                         cwd=ROOT, capture_output=True).returncode, 0)

    def test_repeat_cancel_files_and_measurement(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            p = Path(directory)
            (p/'prompt.txt').write_text('caffè')
            base = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--context', '64',
                    '--chunk', '6', '--max-new', '5']
            encoded = self.run_cli(base+['--complete-file', str(p/'prompt.txt'), '--encode-only'])
            self.assertEqual(json.loads(encoded.stdout)['prompt_ids'], list('caffè'.encode()))
            self.run_cli(base+['--tokens', '5,17,300', '--warmup', '1', '--repeat', '2',
                              '--replay-chunk', '1', '--runs-jsonl', str(p/'runs.jsonl')])
            rows = [json.loads(x) for x in (p/'runs.jsonl').read_text().splitlines()]
            self.assertEqual([r['warmup'] for r in rows], [True, False, False])
            self.assertEqual([r['chunk'] for r in rows], [6, 1, 1])
            self.assertTrue(all(r['generated_ids']==rows[0]['generated_ids'] for r in rows))
            for r in rows:
                self.assertEqual(r['decode_tokens'], r['generated_tokens']-1)
                self.assertGreater(r['load_s'], 0)
                self.assertGreater(r['prefill_s'], 0)
                self.assertGreaterEqual(r['request_s'], r['prefill_s']+r['decode_s'])
                self.assertLessEqual(r['ttft_request_s'], r['request_s'])
                self.assertGreater(r['metal_allocated_bytes'], r['graph_buffer_bytes'])
            self.run_cli(base+['--tokens', '5,17,300', '--stop-after', '2', '--repeat', '2',
                              '--runs-jsonl', str(p/'stop.jsonl')])
            stop = [json.loads(x) for x in (p/'stop.jsonl').read_text().splitlines()]
            self.assertEqual(stop[0]['generated_ids'], rows[0]['generated_ids'][:2])
            self.assertEqual(stop[0]['generated_ids'], stop[1]['generated_ids'])
            self.assertEqual(stop[0]['stop_reason'], 'cancelled')
            self.assertEqual(stop[0]['position'], 4)
            done = subprocess.run(base+['--tokens', '5', '--runs-jsonl', str(p/'runs.jsonl')],
                                  cwd=ROOT, capture_output=True)
            self.assertNotEqual(done.returncode, 0)

    def test_teacher_forced_matches_plain_appends(self):
        # R0.4a: forcing the plain greedy continuation must redo exactly the same
        # appends, so its logits are the plain ones bit for bit (ds4-bench parity).
        prompt, k = [5, 17, 300, 9, 44], 6
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            p = Path(directory)
            base = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--context', '32', '--chunk', '3']
            self.run_cli(base+['--tokens', ','.join(map(str, prompt)), '--max-new', str(k),
                               '--logits', str(p/'plain.f32'), '--token-ids', str(p/'plain.json')])
            ids = json.loads((p/'plain.json').read_text())['generated_ids']
            self.assertEqual(len(ids), k, 'prompt must not reach a stop token')
            self.run_cli(base+['--tokens', ','.join(map(str, prompt+ids[:k-1])), '--teacher-forced', str(k-1),
                               '--logits', str(p/'forced.f32'), '--token-ids', str(p/'forced.json'),
                               '--warmup', '1', '--repeat', '2', '--runs-jsonl', str(p/'runs.jsonl')])
            plain = np.fromfile(p/'plain.f32', '<f4').reshape(-1, 512)
            forced = np.fromfile(p/'forced.f32', '<f4').reshape(-1, 512)
            self.assertEqual(plain.shape, (k, 512))
            self.assertEqual(forced.shape, (k, 512))
            self.assertEqual(plain.tobytes(), forced[:k].tobytes())
            info = json.loads((p/'forced.json').read_text())
            self.assertEqual(info['prompt_ids'], prompt)
            self.assertEqual(info['forced_ids'], ids[:k-1])
            self.assertEqual(info['generated_ids'], ids[1:])
            rows = [json.loads(x) for x in (p/'runs.jsonl').read_text().splitlines()]
            self.assertEqual([r['warmup'] for r in rows], [True, False, False])
            for r in rows:
                self.assertEqual((r['mode'], r['forced_tokens'], r['decode_tokens']), ('teacher-forced', k-1, k-1))
                self.assertEqual((r['generated_tokens'], r['generated_ids']), (k-1, ids[1:]))
                self.assertEqual((r['prompt_tokens'], r['position'], r['stop_reason']), (len(prompt), len(prompt)+k-1, 'length'))
                self.assertGreater(r['decode_s'], 0)
                self.assertGreaterEqual(r['request_s'], r['prefill_s']+r['decode_s'])

    def test_id_only_greedy_matches_logits_path(self):
        # T15c: without --logits/--mtp only the GPU argmax id is read back. It must give the
        # ids of the logits path (CPU scan) in default mode and with Q38_REF_KERNELS=argmax,
        # for chunked prefill, decode and teacher-forced runs.
        prompt, k = [5, 17, 300, 9, 44], 6
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            p = Path(directory)
            for chunk in ('1', '2', '5'):
                base = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--context', '32', '--chunk', chunk,
                        '--tokens', ','.join(map(str, prompt)), '--max-new', str(k)]
                self.run_cli(base+['--logits', str(p/'l.f32'), '--token-ids', str(p/'logits.json')])
                want = json.loads((p/'logits.json').read_text())['generated_ids']
                self.assertEqual(len(want), k)
                for mode in ('', 'argmax'):
                    env = {**os.environ, 'Q38_REF_KERNELS': mode}
                    done = subprocess.run(base+['--token-ids', str(p/'ids.json')], cwd=ROOT, capture_output=True, env=env)
                    self.assertEqual(done.returncode, 0, done.stderr.decode(errors='replace'))
                    self.assertEqual(json.loads((p/'ids.json').read_text())['generated_ids'], want, f'{chunk=} {mode=}')
                    forced = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--context', '32', '--chunk', chunk,
                              '--tokens', ','.join(map(str, prompt+want[:k-1])), '--teacher-forced', str(k-1),
                              '--token-ids', str(p/'forced.json')]
                    done = subprocess.run(forced, cwd=ROOT, capture_output=True, env=env)
                    self.assertEqual(done.returncode, 0, done.stderr.decode(errors='replace'))
                    self.assertEqual(json.loads((p/'forced.json').read_text())['generated_ids'], want[1:], f'{chunk=} {mode=}')

    def test_teacher_forced_admission(self):
        base = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--context', '32']
        for args in (['--complete', 'abc', '--teacher-forced', '1'], ['--tokens', '5,17', '--teacher-forced', '0'],
                     ['--tokens', '5,17', '--teacher-forced', '2'], ['--session', '--teacher-forced', '1'],
                     ['--tokens', '5,17', '--teacher-forced', '1', '--mtp'],
                     ['--tokens', '5,17', '--teacher-forced', '1', '--stop-after', '1'],
                     ['--tokens', '5,17', '--teacher-forced', '1', '--max-new', '1']):
            with self.subTest(args=args):
                done = subprocess.run(base+args, cwd=ROOT, capture_output=True)
                self.assertNotEqual(done.returncode, 0)
                self.assertIn(b'teacher-forced', done.stderr)
                self.assertNotIn(b'unknown', done.stderr)

if __name__ == '__main__':
    unittest.main()
