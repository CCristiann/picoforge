"""Independent CPU references for A. Does not test a native Qwen3.8 graph.

Run with the existing venv. QWEN38_A_NATIVE_INVENTORY optionally supplies a
fresh dump of tests/gguf_inspect.c for comparison with the Python inventory.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/qwen38/a"
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_config as oracle


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hf = json.loads((FIXTURES / "config.json").read_text())
        cls.inventory = json.loads((ROOT / "bench/qwen38_q2_20261011_phase0/inventory.json").read_text())
        cls.metadata = cls.inventory["metadata"]

    def test_pinned_config(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(hashlib.sha256((FIXTURES / "config.json").read_bytes()).hexdigest(),
                         manifest["config"]["sha256"])

    def test_original_config_against_checkpoint(self):
        result = oracle.check_metadata(self.hf, self.metadata)
        self.assertEqual((result["ordinary_layers"], result["mtp_layers"]), (48, 1))
        self.assertEqual(result["ple_hash_rows"], 320001446)
        self.assertEqual(result["ple_padding_rows"], 90)

    def test_text_scope_and_dimension_derivation(self):
        hf = copy.deepcopy(self.hf)
        hf["vision_config"]["hidden_size"] = 7
        self.assertEqual(oracle.reference_metadata(hf), oracle.reference_metadata(self.hf))
        hf["text_config"]["hidden_size"] = 512
        hf["text_config"]["shared_expert_intermediate_size"] = 128
        self.assertEqual(oracle.reference_metadata(hf)["qwen4exp.embedding_length"], 512)
        ts = oracle.reference_tensors(hf, 128)
        self.assertEqual(ts["blk.0.ffn_down_shexp.weight"][0], [128, 512])
        self.assertEqual(ts["blk.0.ffn_down_exps.weight"][0], [768, 512, 512])

    def test_missing_and_inconsistent_metadata(self):
        for key, value in [("qwen4exp.block_count", 48), ("qwen4exp.embedding_length", True),
                           ("qwen4exp.ple.layers", [2]), ("qwen4exp.ple.row_count", 320001446),
                           ("ds4.qwen4.down.physical_input", 640),
                           ("qwen4exp.attention.layer_norm_rms_epsilon", float("nan"))]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.metadata)
                bad[key] = value
                with self.assertRaises(ValueError): oracle.check_metadata(self.hf, bad)
                del bad[key]
                with self.assertRaises(KeyError): oracle.check_metadata(self.hf, bad)

    def test_array_types_and_ple_ranges(self):
        for key, values in [("attention.compress_ratios", [False]+self.metadata["qwen4exp.attention.compress_ratios"][1:]),
                            ("ple.head_offsets", [1]*16), ("ple.head_vocab_sizes", [0]*16),
                            ("ple.layer_multipliers", [1, 2]), ("ple.layer_multipliers", [1, 2, 2**64])]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.metadata)
                bad["qwen4exp."+key] = values
                with self.assertRaises(ValueError): oracle.check_metadata(self.hf, bad)

    def test_complete_tensor_catalog(self):
        counts = oracle.check_tensors(self.hf, self.metadata, self.inventory["tensors"])
        self.assertEqual(counts, {"ordinary": 1223, "mtp": 32, "disk_ple": 1})

    def test_wrong_tensor_shape_format_size_and_duplicates(self):
        tensors = self.inventory["tensors"]
        index = next(i for i, t in enumerate(tensors) if t["name"] == "blk.0.ffn_down_exps.weight")
        for key, value in [("shape", [640, 2560, 512]), ("format", "Q4_K"), ("size_bytes", 1), ("type_id", 12)]:
            with self.subTest(key=key):
                bad = copy.deepcopy(tensors)
                bad[index][key] = value
                with self.assertRaises(ValueError): oracle.check_tensors(self.hf, self.metadata, bad)
        for bad in (tensors[:-1], tensors+[tensors[0]]):
            with self.assertRaises(ValueError): oracle.check_tensors(self.hf, self.metadata, bad)

    def test_mtp_is_not_an_ordinary_layer(self):
        hf = copy.deepcopy(self.hf)
        hf["text_config"]["mtp_num_hidden_layers"] = 0
        hf["text_config"]["mtp"] = {"layer_types": [], "num_hidden_layers": 0}
        self.assertEqual(oracle.reference_metadata(hf)["qwen4exp.block_count"], 48)
        ts = oracle.reference_tensors(hf, 128)
        self.assertFalse(any(n.startswith("blk.48.") for n in ts))
        self.assertFalse(any(v[1] in ("Q4_K", "MXFP4") for v in ts.values()))

    @unittest.skipUnless(os.environ.get("QWEN38_A_NATIVE_INVENTORY"), "native dump not supplied")
    def test_native_header_inventory(self):
        native = json.loads(Path(os.environ["QWEN38_A_NATIVE_INVENTORY"]).read_text())
        self.assertFalse(native["payload_verified"])
        fields = ("name", "shape", "type_id", "format", "size_bytes", "file_offset")
        normalized = lambda ts: sorted(({k: t[k] for k in fields} for t in ts), key=lambda t: t["name"])
        self.assertEqual(normalized(native["tensors"]), normalized(self.inventory["tensors"]))


if __name__ == "__main__":
    unittest.main()
