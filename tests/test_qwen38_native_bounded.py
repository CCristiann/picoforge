"""Do not overlap a new CPU mapping with retiring Metal residency."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/eval'))
import qwen38_native_bounded as bounded

def row(gib,pressure=1):
    return dict(system_pressure_level=pressure,system_wired_pages=gib*2**30//16384,page_bytes=16384)

class ReleaseWait(unittest.TestCase):
    def test_release_and_pressure_must_remain_normal(self):
        samples=[row(43),row(10),row(10),row(10,4),row(10),row(10),row(10)]
        with tempfile.TemporaryDirectory() as d, patch.object(bounded,'sample',side_effect=samples) as probe, \
             patch.object(bounded.subprocess,'check_output',return_value=b'68719476736'), \
             patch.object(bounded.time,'sleep'):
            bounded.settle_after_metal(Path(d))
            report=json.loads((Path(d)/'release-wait.json').read_text())
            self.assertTrue(report['complete']);self.assertEqual(probe.call_count,7)
            self.assertEqual(report['samples'],samples)

    def test_timeout_and_missing_probe_prevent_admission(self):
        with tempfile.TemporaryDirectory() as d, patch.object(bounded,'sample',return_value=row(43)), \
             patch.object(bounded.subprocess,'check_output',return_value=b'68719476736'), \
             patch.object(bounded.time,'sleep'), \
             patch.object(bounded.time,'monotonic',side_effect=[0,0,59,61]):
            with self.assertRaisesRegex(RuntimeError,'not released'):
                bounded.settle_after_metal(Path(d))
            self.assertFalse(json.loads((Path(d)/'release-wait.json').read_text())['complete'])
        with tempfile.TemporaryDirectory() as d, patch.object(bounded,'sample',return_value=None), \
             patch.object(bounded.subprocess,'check_output',return_value=b'68719476736'):
            with self.assertRaisesRegex(RuntimeError,'probe failed'):
                bounded.settle_after_metal(Path(d))

if __name__=='__main__':unittest.main()
