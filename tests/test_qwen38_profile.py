"""R0.2 opt-in per-op GPU profile of the Qwen3.8 trunk on the tiny CLI graph."""
import csv
from pathlib import Path
import re
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / 'build/qwen38-g2/cli.gguf'
HEADER = ['phase', 'token', 'layer', 'op', 'format', 'weight_bytes', 'dispatches', 'barriers', 'gpu_us', 'gbps']
BASE = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--tokens', '5,17,300,9,44',
        '--context', '32', '--chunk', '3', '--max-new', '4']

def read_rows(path):
    with path.open(newline='') as f:
        reader = csv.reader(f)
        assert next(reader) == HEADER
        return [dict(zip(HEADER, r, strict=True)) for r in reader]

class ProfileTests(unittest.TestCase):
    def run_cli(self, extra):
        done = subprocess.run(BASE+extra, cwd=ROOT, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr.decode(errors='replace'))
        return done

    def test_profile_does_not_change_logits(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            p = Path(directory)
            self.run_cli(['--logits', str(p/'plain.f32')])
            self.run_cli(['--logits', str(p/'profiled.f32'), '--profile', str(p/'profile.csv')])
            self.assertEqual((p/'plain.f32').read_bytes(), (p/'profiled.f32').read_bytes())

    def test_profile_csv_covers_every_layer(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            path = Path(directory)/'profile.csv'
            done = self.run_cli(['--profile', str(path)])
            layers = int(re.search(rb'Qwen3\.8: (\d+) layers', done.stderr).group(1))
            rows = read_rows(path)
            # Two prefill chunks (positions 0 and 3), then three decode appends.
            appends = sorted({(r['phase'], int(r['token'])) for r in rows})
            self.assertEqual(appends, [('decode', 5), ('decode', 6), ('decode', 7), ('prefill', 0), ('prefill', 3)])
            for layer in [*map(str, range(layers)), 'global']:
                self.assertTrue(any(r['layer'] == layer for r in rows), layer)
            ops = {r['op'] for r in rows}
            self.assertNotIn('gdn', ops)   # R1.8: the group is split into gdn.* sub-groups
            self.assertNotIn('moe', ops)   # R1.9a: likewise moe.*
            for op in ('embed', 'gdn.hc_pre', 'gdn.proj', 'gdn.conv', 'gdn.prep', 'gdn.scan', 'gdn.norm', 'gdn.out',
                       'gdn.add', 'qsa.proj', 'qsa.core', 'qsa.out', 'hc_attn.down', 'hc_ffn.hc_read',
                       'moe.shared_in', 'moe.route', 'moe.gate_up', 'moe.shared', 'moe.down', 'hc_add', 'final.up', 'lm_head'):
                self.assertIn(op, ops)
            for r in rows:
                self.assertGreater(float(r['gpu_us']), 0, r)
                if r['op'] in ('gdn.hc_pre', 'gdn.proj'):   # R1.8: linears carry their weight bytes
                    self.assertEqual((r['format'], int(r['weight_bytes']) > 0), ('MIXED', True), r)
                if r['op'] in ('gdn.conv', 'gdn.prep', 'gdn.scan', 'gdn.norm', 'gdn.add'):
                    self.assertEqual((r['format'], r['weight_bytes']), ('NONE', '0'), r)
                if r['op'] == 'moe.route':
                    self.assertEqual((r['format'], r['weight_bytes']), ('NONE', '0'), r)
                if r['op'] in ('moe.shared_in', 'moe.gate_up', 'moe.shared', 'moe.down', 'ple', 'qsa.core'):
                    self.assertEqual(r['format'], 'MIXED')
                    # R0.3: composites count the weights they read; qsa.core reads only caches.
                    self.assertEqual(int(r['weight_bytes']) == 0, r['op'] == 'qsa.core', r)
                if r['op'] == 'lm_head':
                    self.assertEqual(r['layer'], 'global')
                    self.assertGreater(int(r['weight_bytes']), 0)
                    expected = int(r['weight_bytes'])/(float(r['gpu_us'])*1e3)
                    self.assertAlmostEqual(float(r['gbps']), expected, delta=1e-3+1e-3*expected)

    def test_profile_counts_dispatches(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            path = Path(directory)/'profile.csv'
            self.run_cli(['--profile', str(path)])
            rows = read_rows(path)
            for r in rows:
                self.assertGreater(int(r['dispatches']), 0, r)
                self.assertGreater(int(r['barriers']), 0, r)
                if r['op'] == 'qsa.proj':  # R1.4: five projections, one multi call: one dispatch, one barrier
                    self.assertEqual((r['format'], r['dispatches'], r['barriers']), ('Q8_0', '1', '1'), r)
            def counts(token):
                return [(r['layer'], r['op'], r['dispatches'], r['barriers'], r['weight_bytes'])
                        for r in rows if (r['phase'], r['token']) == ('decode', token)]
            self.assertTrue(counts('5'))
            self.assertEqual(counts('5'), counts('6'))
            # A 3-token prefill chunk reads more MoE experts than one decode token, the same GDN weights.
            def bytes_of(phase, token, op):
                return [int(r['weight_bytes']) for r in rows if (r['phase'], r['token'], r['op']) == (phase, token, op)]
            for op in ('gdn.hc_pre', 'gdn.proj', 'gdn.out'):
                self.assertEqual(bytes_of('prefill', '0', op), bytes_of('decode', '5', op))
            for op in ('moe.shared_in', 'moe.shared'):   # R1.9a: dense weights once per microbatch
                self.assertEqual(bytes_of('prefill', '0', op), bytes_of('decode', '5', op))
            for op in ('moe.gate_up', 'moe.down'):       # experts scale with the tokens
                for big, small in zip(bytes_of('prefill', '0', op), bytes_of('decode', '5', op), strict=True):
                    self.assertGreater(big, small)

    def test_concurrent_encoder_drops_only_barriers(self):
        """R1.6b: same dispatches and logits as the serial barrier oracle, fewer barriers."""
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            p, rows = Path(directory), {}
            for name, mode in (('fast', ''), ('barrier', 'barrier')):
                done = subprocess.run(BASE+['--logits', str(p/f'{name}.f32'), '--profile', str(p/f'{name}.csv')],
                                      cwd=ROOT, capture_output=True, env={**os.environ, 'Q38_REF_KERNELS': mode})
                self.assertEqual(done.returncode, 0, done.stderr.decode(errors='replace'))
                rows[name] = read_rows(p/f'{name}.csv')
            self.assertEqual((p/'fast.f32').read_bytes(), (p/'barrier.f32').read_bytes())
            total = {m: sum(int(r['barriers']) for r in rows[m] if (r['phase'], r['token']) == ('decode', '5'))
                     for m in rows}
            print(f"\n  decode token barriers: concurrent {total['fast']}, serial oracle {total['barrier']}")
            self.assertLess(total['fast'], total['barrier'])
            for fast, ref in zip(rows['fast'], rows['barrier'], strict=True):
                self.assertEqual([fast[k] for k in HEADER[:7]], [ref[k] for k in HEADER[:7]])
                self.assertLessEqual(int(fast['barriers']), int(ref['barriers']), fast)

    def test_profile_skips_warmup(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            path = Path(directory)/'profile.csv'
            self.run_cli(['--profile', str(path), '--warmup', '1', '--repeat', '1'])
            embeds = [(r['phase'], r['token']) for r in read_rows(path) if r['op'] == 'embed']
            self.assertEqual(embeds, [('prefill', '0'), ('prefill', '3'), ('decode', '5'), ('decode', '6'), ('decode', '7')])

    def test_profile_rejects_existing_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            path = Path(directory)/'profile.csv'
            path.write_text('keep\n')
            done = subprocess.run(BASE+['--profile', str(path)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertEqual(path.read_text(), 'keep\n')

if __name__ == '__main__':
    unittest.main()
