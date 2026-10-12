"""Independent Python layout reference versus the native, sanitized reader."""
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest

from test_qwen38_inventory import fixture, inventory, string

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_gguf_inspect"


class NativeGgufTests(unittest.TestCase):
    def run_file(self, data, valid=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            path.write_bytes(data)
            result = subprocess.run([str(BINARY), str(path)], capture_output=True, text=True)
            if not valid:
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("gguf:", result.stderr)
                self.assertNotIn("Sanitizer", result.stderr)
                return
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads(result.stdout)
            expected = inventory.inspect(path)
            self.assertEqual(actual["data_offset"], expected["data_offset"])
            self.assertFalse(actual["payload_verified"])
            for got, want in zip(actual["tensors"], expected["tensors"], strict=True):
                self.assertEqual(got, {key: want[key] for key in got})
            return actual

    def test_all_selected_formats(self):
        for kind, width in [(0, 32), (1, 32), (8, 32), (10, 768),
                            (12, 256), (16, 256), (30, 32), (39, 32)]:
            with self.subTest(kind=kind):
                self.run_file(fixture(kind=kind, dims=(width,)))
        self.run_file(fixture())  # padded down-projection: 768 physical columns

    def test_reject_malformed_layouts(self):
        for kwargs in [dict(dims=(640,)), dict(dims=(0,)), dict(dims=()),
                       dict(dims=(1, 1, 1, 1, 1)), dict(kind=999), dict(offset=1),
                       dict(offset=2**64-32), dict(duplicate=True),
                       dict(dims=(2**63, 2**63)), dict(architecture="qwen3")]:
            with self.subTest(kwargs=kwargs):
                self.run_file(fixture(**kwargs), valid=False)

    def test_every_header_truncation_and_short_payload(self):
        data = fixture()
        end = self.run_file(data)["data_offset"]
        for size in range(end):
            self.run_file(data[:size], valid=False)
        self.run_file(data[:-1], valid=False)

    def test_cursor_failures(self):
        binary = ROOT / "build/test_gguf_reader"
        subprocess.run([str(binary)], check=True, capture_output=True)
        for case in ["truncated", "add", "mul", "array", "type", "string"]:
            result = subprocess.run([str(binary), case], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertNotIn("Sanitizer", result.stderr)

    def test_view_plan(self):
        binary = ROOT / "build/test_gguf_plan"
        subprocess.run([str(binary)], check=True, capture_output=True)
        for case in ["small", "alignment", "overflow", "overlap"]:
            result = subprocess.run([str(binary), case], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertNotIn("Sanitizer", result.stderr)

    def test_distinct_overlapping_tensors_and_magic(self):
        data = fixture(duplicate=True)
        at = data.rindex(b"down.weight")
        self.run_file(data[:at] + b"xxxx.weight" + data[at+11:], valid=False)
        self.run_file(b"FGUG" + fixture()[4:], valid=False)
        self.run_file(fixture()[:4] + struct.pack("<I", 2) + fixture()[8:], valid=False)

    def test_metadata_validation(self):
        architecture = string("general.architecture") + struct.pack("<I", 8) + string("qwen4exp")
        tensor = string("x") + struct.pack("<IQIQ", 1, 32, 0, 0)
        extras = [architecture,
                  string("general.alignment") + struct.pack("<IQ", 10, 32),
                  string("general.alignment") + struct.pack("<II", 4, 3),
                  string("bad") + struct.pack("<IB", 7, 2),
                  string("bad") + struct.pack("<IIQ", 9, 9, 1),
                  string("bad") + struct.pack("<IIQ", 9, 4, 2000001)]
        for extra in extras:
            data = b"GGUF" + struct.pack("<IQQ", 3, 1, 2) + architecture + extra + tensor
            self.run_file(data + bytes(-len(data) % 32) + bytes(128), valid=False)


if __name__ == "__main__":
    unittest.main()
