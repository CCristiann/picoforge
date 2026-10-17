"""R0.2 opt-in per-op GPU profile of the Qwen3.8 trunk on the tiny CLI graph."""
import csv
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / 'build/qwen38-g2/cli.gguf'
HEADER = ['phase', 'token', 'layer', 'op', 'format', 'weight_bytes', 'dispatches', 'barriers', 'gpu_us', 'gbps']
BASE = [str(ROOT/'picoforge'), '--qwen38', str(GGUF), '--tokens', '5,17,300,9,44',
        '--context', '32', '--chunk', '3', '--max-new', '4']

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
            with path.open(newline='') as f:
                reader = csv.reader(f)
                self.assertEqual(next(reader), HEADER)
                rows = [dict(zip(HEADER, r, strict=True)) for r in reader]
            # Two prefill chunks (positions 0 and 3), then three decode appends.
            appends = sorted({(r['phase'], int(r['token'])) for r in rows})
            self.assertEqual(appends, [('decode', 5), ('decode', 6), ('decode', 7), ('prefill', 0), ('prefill', 3)])
            for layer in [*map(str, range(layers)), 'global']:
                self.assertTrue(any(r['layer'] == layer for r in rows), layer)
            ops = {r['op'] for r in rows}
            for op in ('embed', 'gdn', 'qsa.qg', 'qsa.core', 'qsa.out', 'hc_attn.down', 'hc_ffn.hc_read',
                       'moe', 'hc_add', 'final.up', 'lm_head'):
                self.assertIn(op, ops)
            for r in rows:
                self.assertGreater(float(r['gpu_us']), 0, r)
                self.assertEqual((r['dispatches'], r['barriers']), ('-1', '-1'))
                if r['op'] in ('gdn', 'moe', 'ple', 'qsa.core'):
                    self.assertEqual((r['format'], r['weight_bytes']), ('MIXED', '0'))
                if r['op'] == 'lm_head':
                    self.assertEqual(r['layer'], 'global')
                    self.assertGreater(int(r['weight_bytes']), 0)
                    expected = int(r['weight_bytes'])/(float(r['gpu_us'])*1e3)
                    self.assertAlmostEqual(float(r['gbps']), expected, delta=1e-3+1e-3*expected)

    def test_profile_rejects_existing_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build/qwen38-g2') as directory:
            path = Path(directory)/'profile.csv'
            path.write_text('keep\n')
            done = subprocess.run(BASE+['--profile', str(path)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertEqual(path.read_text(), 'keep\n')

if __name__ == '__main__':
    unittest.main()
