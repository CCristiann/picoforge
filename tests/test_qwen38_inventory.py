"""Malformed-file checks for metadata-only inspection; no model needed."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inventory", ROOT / "tools/eval/inspect_qwen38_gguf.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def fixture(*, dims=(768, 2), kind=10, offset=0, duplicate=False, architecture="qwen4exp"):
    kv = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    tensor = (string("down.weight") + struct.pack("<I", len(dims)) +
              b"".join(struct.pack("<Q", n) for n in dims) + struct.pack("<IQ", kind, offset))
    data = b"GGUF" + struct.pack("<IQQ", 3, 2 if duplicate else 1, 1) + kv + tensor
    if duplicate:
        data += tensor
    return data + b"\0" * (-len(data) % 32) + bytes(504)


class InventoryTests(unittest.TestCase):
    def inspect(self, data, expected_size=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            path.write_bytes(data)
            return inventory.inspect(path, expected_size)

    def test_q2_padded_physical_stride(self):
        result = self.inspect(fixture())
        self.assertEqual(result["tensors"][0]["size_bytes"], 2 * 3 * 84)
        self.assertEqual(result["tensors"][0]["shape"], [768, 2])
        self.assertFalse(result["payload_verified"])

    def test_quantization_block_sizes(self):
        for kind, width, nbytes in [(16, 256, 66), (12, 256, 144), (8, 32, 34),
                                    (39, 32, 17), (30, 32, 64), (0, 32, 128)]:
            with self.subTest(kind=kind):
                d = self.inspect(fixture(dims=(width,), kind=kind))
                self.assertEqual(d["tensors"][0]["size_bytes"], nbytes)

    def test_header_prefix_does_not_claim_payload(self):
        data = fixture()
        result = self.inspect(data[:-504], len(data))
        self.assertLess(result["actual_file_bytes"], result["expected_file_bytes"])
        self.assertFalse(result["payload_verified"])
        with self.assertRaisesRegex(ValueError, "out-of-file"):
            self.inspect(data[:-504])

    def test_all_header_truncations(self):
        data = fixture()
        header_end = self.inspect(data)["data_offset"]
        for size in range(header_end):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.inspect(data[:size], len(data))

    def test_invalid_layouts(self):
        for kwargs in [dict(dims=(640,)), dict(dims=(0,)), dict(dims=()),
                       dict(dims=(1, 1, 1, 1, 1)), dict(kind=999), dict(offset=1),
                       dict(offset=2**64 - 32), dict(duplicate=True),
                       dict(dims=(2**63, 2**63)), dict(architecture="qwen3")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.inspect(fixture(**kwargs))

    def test_oversized_string_and_counts(self):
        for data in [b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 2**63),
                     b"GGUF" + struct.pack("<IQQ", 3, 2**63, 1), b"FGUG" + bytes(20)]:
            with self.assertRaises(ValueError):
                self.inspect(data)


if __name__ == "__main__":
    unittest.main()
