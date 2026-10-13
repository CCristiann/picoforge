"""Run a delivery's unittest entry point, treating every skip as a failure."""
import runpy
import sys
import unittest
from pathlib import Path


def main():
    original = unittest.TextTestRunner.run

    def strict_run(self, test):
        result = original(self, test)
        if result.skipped:
            raise SystemExit("Integration requires zero skipped tests")
        return result

    unittest.TextTestRunner.run = strict_run
    sys.argv = sys.argv[1:]
    if "--report" in sys.argv:
        print("Integration report:", sys.argv[sys.argv.index("--report") + 1], flush=True)
    sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
