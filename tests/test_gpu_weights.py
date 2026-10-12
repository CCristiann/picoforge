"""Failure paths must reject bad ownership/caps before any unsafe GPU access."""
from pathlib import Path
import subprocess
import unittest

BINARY = Path(__file__).resolve().parents[1] / "build/test_gpu_weights"


class WeightStoreTests(unittest.TestCase):
    def test_reject_foreign_owner_and_device_limit(self):
        for option in ("--foreign", "--oversize-cap", "--truncated"):
            result = subprocess.run([str(BINARY), option], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("gguf:", result.stderr)


if __name__ == "__main__":
    unittest.main()
