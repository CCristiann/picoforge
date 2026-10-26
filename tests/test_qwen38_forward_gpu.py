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

    def test_concurrent_encoder_is_deterministic(self):
        """R1.6b: the concurrent encoder (default) and the serial one with a
        barrier after every op (Q38_REF_KERNELS=barrier) give the same bits,
        logits and every traced stage, on 3 runs each: a missing barrier is a
        race and shows up here as run-to-run or mode-to-mode drift."""
        saved = os.environ.get('Q38_REF_KERNELS')
        try:
            for chunk in (1, 7):  # decode appends; prefill chunks with a tail
                runs = []
                for mode in ('', 'barrier'):
                    os.environ['Q38_REF_KERNELS'] = mode
                    runs += [(mode, gate.run_native(4, chunk)[1]) for _ in range(3)]
                first = runs[0][1]
                for mode, trace in runs[1:]:
                    self.assertEqual(sorted(trace), sorted(first))
                    for key in first:
                        self.assertEqual(trace[key].tobytes(), first[key].tobytes(), f'{chunk=} {mode=} {key}')
        finally:
            if saved is None: os.environ.pop('Q38_REF_KERNELS', None)
            else: os.environ['Q38_REF_KERNELS'] = saved

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    gate.NATIVE, gate.GGUF = args.native.resolve(), args.gguf.resolve()
    unittest.main(argv=[__file__, *remaining])
