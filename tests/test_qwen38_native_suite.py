"""Reject inconsistent benchmark records before they can become results."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/eval'))
from qwen38_native_suite import validate_record

class Records(unittest.TestCase):
    def test_validation(self):
        row=dict(load_s=2.,reset_s=0.,prefill_s=1.,decode_s=2.,request_s=3.1,
                 prompt_tokens=10,generated_tokens=3,context=13,generated_ids=[1,2,3],
                 decode_tokens=2,ttft_request_s=1.05)
        self.assertIs(validate_record(row),row)
        for key,value in [('prefill_s',float('nan')),('decode_s',-1),('request_s',2.9),
                          ('context',12),('generated_tokens',4),('decode_tokens',3),('ttft_request_s',4)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_record({**row,key:value})

if __name__=='__main__': unittest.main()
