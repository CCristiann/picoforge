"""R0.1: the Metal capability probe emits a complete, well-typed JSON object.

The probe opens the GPU, so the Make recipe wraps this whole test in the
shared GPU lock once; the test itself must not take the lock again.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import unittest

NATIVE = Path("build/qwen38-capabilities")
SAMPLING_POINTS = ("stage", "dispatch", "draw", "blit", "tile")
BOOLS = ("timestamp_counter", "command_buffer_gpu_times", "metal4_family",
         "mtl4_command_queue", "residency_set", "unified_memory")
THERMAL = ("nominal", "fair", "serious", "critical")


def is_int(value):  # bool is an int subclass in Python; a byte count never is one
    return isinstance(value, int) and not isinstance(value, bool)


class CapabilityProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        done = subprocess.run([str(NATIVE)], capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            raise AssertionError(f"probe exited {done.returncode}: {done.stderr}")
        cls.caps = json.loads(done.stdout)

    def test_probe_emits_required_keys(self):
        caps = self.caps
        for key in ("device", "os", "thermal_state"):
            self.assertIsInstance(caps[key], str, key)
            self.assertTrue(caps[key], key)
        for key in BOOLS:
            self.assertIsInstance(caps[key], bool, key)
        for key in ("recommended_max_working_set_bytes", "max_buffer_length"):
            self.assertTrue(is_int(caps[key]), key)
        self.assertGreater(caps["recommended_max_working_set_bytes"], 0)
        self.assertGreater(caps["max_buffer_length"], 0)
        self.assertEqual(sorted(caps["counter_sampling"]), sorted(SAMPLING_POINTS))
        for point in SAMPLING_POINTS:
            self.assertIsInstance(caps["counter_sampling"][point], bool, point)
        self.assertIsInstance(caps["counter_sets"], list)
        self.assertTrue(all(isinstance(name, str) for name in caps["counter_sets"]))

    def test_values_are_self_consistent(self):
        caps = self.caps
        self.assertIn(caps["thermal_state"], THERMAL)
        # The timestamp flag must follow from the set list, not be set independently.
        self.assertEqual(caps["timestamp_counter"], "timestamp" in caps["counter_sets"])
        # A Metal 4 queue cannot exist on a device outside the Metal 4 family.
        if caps["mtl4_command_queue"]:
            self.assertTrue(caps["metal4_family"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=NATIVE)
    args, remaining = parser.parse_known_args()
    NATIVE = args.native.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
