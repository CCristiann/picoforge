"""Extra G2 gates: MoE scratch reuse across 32-row microbatches, GPU state/faults.
The five primary oracle tests live in test_qwen38_forward.py and are reused.
"""
import argparse
import os
from pathlib import Path
import unittest
import numpy as np
import test_qwen38_forward as gate

class DeviceGraphTests(unittest.TestCase):
    def test_microbatch_state_and_fault(self):
        tokens = gate.TOKENS * 3  # 72 > 2 * MoE microbatch 32; tail 8
        os.environ['Q38_G2_STATE_CHECKS'] = '1'
        try:
            info, one = gate.run_native(4, len(tokens), tokens, replay=True)
            self.assertEqual(info['replay'], 'identical')
            info, chunks = gate.run_native(4, 17, tokens, replay=True)
            self.assertEqual(info['next_pos'], len(tokens))
            self.assertEqual(info['replay'], 'identical')
            for key in one:
                np.testing.assert_array_equal(one[key], chunks[key], err_msg=key)
        finally:
            os.environ.pop('Q38_G2_STATE_CHECKS', None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    gate.NATIVE, gate.GGUF = args.native.resolve(), args.gguf.resolve()
    unittest.main(argv=[__file__, *remaining])
