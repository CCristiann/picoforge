"""Exercise capture, nonzero exit, timeout and memory-guard termination."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("observed", ROOT / "tools/eval/run_observed.py")
observed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observed)


class ObservedTests(unittest.TestCase):
    def execute(self, code, timeout=5, *, critical=False):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            def sample(_probe, pid=None):
                return dict(monotonic_s=observed.now(), system_pressure_level=4 if pid and critical else 1,
                            system_swap_used_bytes=0, phys_footprint_bytes=1024,
                            lifetime_peak_phys_footprint_bytes=2048)
            with patch.object(observed, "sample", sample):
                rc = observed.run(output, ROOT, [sys.executable, "-I", "-c", code], timeout, .05)
            metadata = json.loads((output / "process.json").read_text())
            return rc, metadata, (output / "stdout.log").read_bytes(), (output / "stderr.log").read_bytes()

    def test_streams_and_times(self):
        code = "import os,time;os.write(1,b'alpha\\x00');os.write(2,b'error\\n');time.sleep(.2)"
        rc, meta, out, err = self.execute(code)
        self.assertEqual((rc, out, err), (0, b"alpha\0", b"error\n"))
        self.assertTrue(meta["complete"])
        self.assertGreaterEqual(meta["process_s"], .2)
        self.assertLess(meta["first_stream_bytes_s"]["stdout"], meta["process_s"])
        self.assertGreater(meta["samples"], 1)

    def test_failed_child_is_not_complete(self):
        rc, meta, _, _ = self.execute("raise SystemExit(7)")
        self.assertEqual(rc, 7)
        self.assertFalse(meta["complete"])

    def test_preserves_caller_lock_descriptor(self):
        with tempfile.TemporaryFile() as lock:
            with patch.dict(os.environ, {"PICOFORGE_GPU_LOCK_FD": str(lock.fileno())}):
                rc, meta, out, _ = self.execute("import os;os.fstat(int(os.environ['PICOFORGE_GPU_LOCK_FD']));print('inherited')")
        self.assertEqual((rc, out), (0, b"inherited\n"))
        self.assertTrue(meta["complete"])

    def test_timeout_keeps_evidence(self):
        rc, meta, _, _ = self.execute("import time;time.sleep(30)", .15)
        self.assertNotEqual(rc, 0)
        self.assertFalse(meta["complete"])
        self.assertEqual(meta["stop_reason"], "timeout")
        self.assertLess(meta["process_s"], 3)

    def test_critical_pressure_stops_only_the_child(self):
        rc, meta, _, _ = self.execute("import time;time.sleep(30)", critical=True)
        self.assertNotEqual(rc, 0)
        self.assertEqual(meta["stop_reason"], "memory guard")
        self.assertGreaterEqual(meta["samples"], 3)

    def test_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(observed, "sample", return_value={}):
                with self.assertRaises(FileExistsError):
                    observed.run(Path(directory), ROOT, [sys.executable, "-c", "pass"])


if __name__ == "__main__":
    unittest.main()
