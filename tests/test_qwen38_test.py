"""A missing native prerequisite must never produce a green aggregate run."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / "tools/qwen38_test.py"


class StrictRunnerTests(unittest.TestCase):
    def test_process_verdicts(self):
        cases = [("pass", 0), ("self.fail('native mismatch')", 1),
                 ("self.skipTest('native harness absent')", 1)]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "suite.py"
            for body, failure in cases:
                with self.subTest(body=body):
                    script.write_text("import unittest\nclass Native(unittest.TestCase):\n"
                                      f"    def test_native(self): {body}\nunittest.main()\n")
                    result = subprocess.run([sys.executable, str(RUNNER), str(script)],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode != 0, bool(failure), result.stderr)


if __name__ == "__main__":
    unittest.main()
