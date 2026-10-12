"""CPU-only process tests; never initializes Metal or touches model weights."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WRAPPER = Path(__file__).resolve().parents[1] / "tools/qwen38_gpu_lock.py"


class SharedLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="q38-lock-test-")
        self.root = Path(self.tmp.name)
        self.a, self.b = self.root / "a", self.root / "b"
        self.a.mkdir()
        self.git("init", "-q")
        self.git("-c", "user.name=Lock Test", "-c", "user.email=lock@example.invalid",
                 "commit", "-q", "--allow-empty", "-m", "fixture")
        self.git("worktree", "add", "-q", "--detach", str(self.b))
        self.owner = None
        self.env = os.environ.copy()
        for key in ("PICOFORGE_GPU_LOCK_FD", "PICOFORGE_GPU_LOCK_PATH"):
            self.env.pop(key, None)

    def tearDown(self):
        if self.owner:
            if self.owner.poll() is None:
                self.owner.kill()
            self.owner.communicate(timeout=5)
        self.tmp.cleanup()

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.a), *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run_lock(self, cwd, *args):
        return subprocess.run([sys.executable, str(WRAPPER), *args], cwd=cwd,
                              env=self.env, capture_output=True, text=True, timeout=5)

    def start_owner(self):
        code = ("import os,sys; "
                "fd=int(os.environ['PICOFORGE_GPU_LOCK_FD']); os.fstat(fd); "
                "print(os.getpid(), flush=True); sys.stdin.readline()")
        self.owner = subprocess.Popen(
            [sys.executable, str(WRAPPER), "--", sys.executable, "-c", code],
            cwd=self.a, env=self.env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # select prevents a broken wrapper from hanging the test indefinitely.
        import select
        ready, _, _ = select.select([self.owner.stdout], [], [], 5)
        self.assertTrue(ready, "owner did not start")
        self.assertEqual(int(self.owner.stdout.readline()), self.owner.pid)

    def probe(self, timeout="0"):
        return self.run_lock(self.b, "--timeout", timeout, "--",
                             sys.executable, "-c", "print('entered')")

    def test_common_path_contention_and_release(self):
        a = self.run_lock(self.a, "--print-path")
        b = self.run_lock(self.b, "--print-path")
        self.assertEqual(a.returncode, 0)
        self.assertEqual(a.stdout, b.stdout)
        self.start_owner()
        path = Path(a.stdout.strip())
        inode = path.stat().st_ino
        blocked = self.probe("0.1")
        self.assertEqual(blocked.returncode, 75, blocked.stderr)
        self.assertNotIn("entered", blocked.stdout)
        self.owner.communicate("done\n", timeout=5)
        self.assertEqual(self.owner.returncode, 0)
        self.assertEqual(self.probe().returncode, 0)
        self.assertEqual(path.stat().st_ino, inode, "lock file must never be replaced")

    def test_process_death_releases_lock(self):
        self.start_owner()
        self.assertEqual(self.probe().returncode, 75)
        self.owner.kill()
        self.owner.communicate(timeout=5)
        self.assertEqual(self.probe().returncode, 0)

    def test_failure_status_and_exec_failure_release(self):
        failed = self.run_lock(self.a, "--", sys.executable, "-c", "raise SystemExit(7)")
        self.assertEqual(failed.returncode, 7)
        missing = self.run_lock(self.a, "--", str(self.root / "absent"))
        self.assertEqual(missing.returncode, 126)
        self.assertEqual(self.probe().returncode, 0)

    def test_nested_wrapper_rejected(self):
        nested = self.run_lock(self.a, "--", sys.executable, str(WRAPPER),
                               "--timeout", "0", "--", "true")
        self.assertEqual(nested.returncode, 2)
        self.assertIn("nested", nested.stderr)
        self.assertEqual(self.probe().returncode, 0)


if __name__ == "__main__":
    unittest.main()
