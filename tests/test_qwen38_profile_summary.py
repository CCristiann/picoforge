"""R0.3 per-op profile summary on fixture CSVs written here (no GPU, no lock)."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/eval/qwen38_profile_summary.py'
HEADER = 'phase,token,layer,op,format,weight_bytes,dispatches,barriers,gpu_us,gbps\n'
# Two decode appends with the same (phase, token), as --repeat writes them; only
# the embed row separates them. The prefill append must not count.
FIRST = HEADER + '''prefill,0,global,embed,BF16,300,2,1,10.0,0.03
prefill,0,0,gdn,MIXED,5000,20,9,100.0,0.05
prefill,0,global,lm_head,Q8_0,9000,1,1,50.0,0.18
decode,3,global,embed,BF16,100,2,1,1.0,0.1
decode,3,0,gdn,MIXED,4000,10,8,40.0,0.1
decode,3,0,hc_add,NONE,0,2,1,2.0,0
decode,3,1,hc_add,NONE,0,2,1,4.0,0
decode,3,global,lm_head,Q8_0,9000,1,1,30.0,0.3
'''
SECOND = HEADER + '''decode,3,global,embed,BF16,100,2,1,3.0,0.03
decode,3,0,gdn,MIXED,4000,10,8,60.0,0.07
decode,3,0,hc_add,NONE,0,2,1,6.0,0
decode,3,1,hc_add,NONE,0,2,1,8.0,0
decode,3,global,lm_head,Q8_0,9000,1,1,50.0,0.18
'''
RUNS = '\n'.join(json.dumps(r) for r in [
    {'event': 'done', 'warmup': True, 'decode_profile': {'gpu_s': 1.0}, 'mtp': {'target_decode_appends': 2}},
    {'event': 'done', 'warmup': False, 'decode_profile': {'gpu_s': 0.0002}, 'mtp': {'target_decode_appends': 2}},
]) + '\n'

class SummaryTests(unittest.TestCase):
    def summarize(self, directory, *extra):
        d = Path(directory)
        (d/'a.csv').write_text(FIRST); (d/'b.csv').write_text(SECOND); (d/'runs.jsonl').write_text(RUNS)
        done = subprocess.run([sys.executable, '-I', str(TOOL), str(d/'a.csv'), str(d/'b.csv'), '--out', str(d/'out'), *extra],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return d/'out'

    def test_summary_aggregates_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.summarize(directory)
            with (out/'summary.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), ['op', 'format', 'calls_per_token', 'us_per_token', 'share',
                                             'weight_bytes_per_token', 'gbps'])
            got = {r['op']: r for r in rows}
            self.assertEqual([r['op'] for r in rows], ['gdn', 'lm_head', 'hc_add', 'embed'])  # by µs, descending
            expect = {'gdn': (1, 50.0, 4000), 'lm_head': (1, 40.0, 9000), 'hc_add': (2, 10.0, 0), 'embed': (1, 2.0, 100)}
            for op, (calls, us, nbytes) in expect.items():
                r = got[op]
                self.assertAlmostEqual(float(r['calls_per_token']), calls)
                self.assertAlmostEqual(float(r['us_per_token']), us)
                self.assertAlmostEqual(float(r['weight_bytes_per_token']), nbytes)
                self.assertAlmostEqual(float(r['share']), us/102.0, places=5)
                self.assertAlmostEqual(float(r['gbps']), nbytes/(us*1e3), places=5)
            self.assertIn('| gdn |', (out/'summary.md').read_text())
            # An existing output directory is refused, never overwritten.
            done = subprocess.run([sys.executable, '-I', str(TOOL), str(Path(directory)/'a.csv'), '--out', str(out)],
                                  capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)

    def test_summary_reports_dispatch_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            out = self.summarize(directory, '--runs-jsonl', str(Path(directory)/'runs.jsonl'))
            totals = json.loads((out/'totals.json').read_text())
            self.assertEqual(totals['decode_appends'], 2)
            self.assertAlmostEqual(totals['us_per_token'], 102.0)
            self.assertAlmostEqual(totals['us_per_token_median'], 102.0)
            self.assertAlmostEqual(totals['weight_bytes_per_token'], 13100)
            self.assertAlmostEqual(totals['dispatches_per_token'], 17)
            self.assertAlmostEqual(totals['barriers_per_token'], 12)
            self.assertAlmostEqual(totals['dispatch_floor_us_per_token'], 17*1.4)
            self.assertAlmostEqual(totals['normal_decode_gpu_us_per_token'], 100.0)  # warm-up line ignored
            self.assertAlmostEqual(totals['groups_over_normal_gpu'], 1.02)

    def test_gdn_subgroups_roll_up(self):
        rows = [('global', 'embed', 'NONE', 0, 30.0), ('0', 'gdn.proj', 'MIXED', 300, 30.0),
                ('0', 'gdn.scan', 'NONE', 0, 20.0), ('0', 'gdn.out', 'Q8_0', 100, 10.0), ('global', 'lm_head', 'Q8_0', 500, 8.0)]
        text = HEADER + ''.join(f'decode,3,{layer},{op},{fmt},{nbytes},1,1,{us},0.1\n' for layer, op, fmt, nbytes, us in rows)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'g.csv').write_text(text)
            done = subprocess.run([sys.executable, '-I', str(TOOL), str(Path(directory)/'g.csv'), '--out', str(Path(directory)/'o')],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            totals = json.loads((Path(directory)/'o'/'totals.json').read_text())
            self.assertAlmostEqual(totals['us_per_token'], 98.0)          # sub-groups sum to the whole
            self.assertAlmostEqual(totals['gdn_us_per_token'], 60.0)
            self.assertAlmostEqual(totals['gdn_share'], 60.0/98.0)
            self.assertAlmostEqual(totals['gdn_weight_bytes_per_token'], 400)
            self.assertIn('gdn rollup', (Path(directory)/'o'/'summary.md').read_text())

    def test_moe_subgroups_roll_up(self):
        rows = [('global', 'embed', 'NONE', 0, 30.0), ('0', 'moe.shared_in', 'MIXED', 300, 10.0),
                ('0', 'moe.route', 'NONE', 0, 5.0), ('0', 'moe.gate_up', 'MIXED', 200, 20.0),
                ('0', 'moe.shared', 'MIXED', 50, 3.0), ('0', 'moe.down', 'MIXED', 100, 12.0), ('global', 'lm_head', 'Q8_0', 500, 8.0)]
        text = HEADER + ''.join(f'decode,3,{layer},{op},{fmt},{nbytes},1,1,{us},0.1\n' for layer, op, fmt, nbytes, us in rows)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'g.csv').write_text(text)
            done = subprocess.run([sys.executable, '-I', str(TOOL), str(Path(directory)/'g.csv'), '--out', str(Path(directory)/'o')],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            totals = json.loads((Path(directory)/'o'/'totals.json').read_text())
            self.assertAlmostEqual(totals['us_per_token'], 88.0)          # sub-groups sum to the whole
            self.assertAlmostEqual(totals['moe_us_per_token'], 50.0)
            self.assertAlmostEqual(totals['moe_share'], 50.0/88.0)
            self.assertAlmostEqual(totals['moe_weight_bytes_per_token'], 650)
            self.assertIn('moe rollup', (Path(directory)/'o'/'summary.md').read_text())

if __name__ == '__main__':
    unittest.main()
