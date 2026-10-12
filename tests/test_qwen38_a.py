"""Independent CPU references for A. Does not test a native Qwen3.8 graph.

Run with the existing venv. QWEN38_A_NATIVE_INVENTORY optionally supplies a
fresh dump of tests/gguf_inspect.c for comparison with the Python inventory.
"""
import copy
import hashlib
import json
import os
import subprocess
import struct
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/qwen38/a"
sys.path.insert(0, str(ROOT / "tools/oracle"))
import qwen38_config as oracle
import qwen38_text as text_oracle


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

    @unittest.skipUnless(os.environ.get("QWEN38_A_NATIVE_CONFIG"), "native config not supplied")
    def test_native_config_accessors(self):
        got = json.loads(Path(os.environ["QWEN38_A_NATIVE_CONFIG"]).read_text())
        c = self.hf["text_config"]
        pick = lambda *keys: [c[k] for k in keys]
        eps = c["rms_norm_eps"]
        expected = dict(profile="qwen38-q2-text-f32-v1",
            dimensions=pick("hidden_size","vocab_size")+[c["num_hidden_layers"]+c["mtp_num_hidden_layers"]]+
                pick("num_hidden_layers","mtp_num_hidden_layers","max_position_embeddings","linear_conv_kernel_dim"),
            gdn=pick("linear_num_key_heads","linear_num_value_heads","linear_key_head_dim","linear_value_head_dim")+[eps,1e-6],
            hc=pick("hc_count","hidden_size","hc_lowrank")+[eps],
            attention=pick("num_attention_heads","num_key_value_heads","head_dim","indexer_n_heads","indexer_kv_heads",
                           "indexer_head_dim","indexer_compress_ratio","indexer_budget")+
                [int(c["head_dim"]*c["partial_rotary_factor"]),eps,c["rope_parameters"]["rope_theta"]],
            moe=pick("hidden_size","num_experts","num_experts_per_tok","moe_intermediate_size","shared_expert_intermediate_size")+
                [(c["moe_intermediate_size"]+255)//256*256,True],
            layers=[[int(k=="full_attention"),int(i>=c["num_hidden_layers"]),
                     c["indexer_compress_ratio"] if k=="full_attention" else 0,i+1 in c["ple_layer_ids"]]
                    for i,k in enumerate(c["layer_types"]+c["mtp"]["layer_types"])],
            ngram=pick("vocab_size","eos_token_id")+[c["ple_layer_ids"][0]-1]+pick("ngram_size","heads_per_ngram")+
                [c["ple_embed_dim"]//((c["ngram_size"]-1)*c["heads_per_ngram"])]+
                pick("hidden_size","hc_count","ple_conv_kernel_size")+[eps,self.metadata[oracle.PREFIX+"ple.row_count"]],
            multipliers=self.metadata[oracle.PREFIX+"ple.layer_multipliers"],
            offsets=self.metadata[oracle.PREFIX+"ple.head_offsets"],vocabs=self.metadata[oracle.PREFIX+"ple.head_vocab_sizes"])
        # C exports decimal FP32 round-trips; compare storage bits for float fields.
        for key, indexes in {"gdn":[4,5],"hc":[3],"attention":[9,10],"ngram":[9]}.items():
            for i in indexes:
                self.assertEqual(struct.pack("<f",got[key][i]),struct.pack("<f",expected[key][i]))
                got[key][i] = expected[key][i]
        self.assertEqual(got,expected)


class TextFixtureTests(unittest.TestCase):
    def test_frozen_fixture_hash_and_case_counts(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())["text_fixture"]
        raw = (FIXTURES / "text.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["sha256"])
        data = json.loads(raw)
        for group in ("encodes", "decodes", "templates"):
            self.assertEqual(len(data[group]), manifest[group])
        self.assertFalse(manifest["native_parity_tested"])


@unittest.skipUnless(os.environ.get("QWEN38_A_REFERENCE_ROOT"), "HF reference root not supplied")
class TextReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_root = Path(os.environ["QWEN38_A_REFERENCE_ROOT"])
        cls.actual = text_oracle.generate(cls.reference_root)
        cls.templates = {c["name"]: c["expected"] for c in cls.actual["templates"]}

    def test_all_cases_reproduce_from_pinned_hf(self):
        self.assertEqual(self.actual, json.loads((FIXTURES / "text.json").read_text()))

    def test_extended_cases_reproduce(self):
        self.assertEqual(text_oracle.extended_cases(self.reference_root),
                         json.loads((FIXTURES / "text_extended.json").read_text()))
        self.assertEqual(text_oracle.extended_chat_cases(self.reference_root),
                         json.loads((FIXTURES / "chat_extended.json").read_text()))

    @unittest.skipUnless(os.environ.get("QWEN38_A_TEXT_EXE"), "native text harness not supplied")
    def test_native_nfc_all_unicode_scalars(self):
        tokenizer, _ = text_oracle.load_reference(self.reference_root)
        directory = ROOT / "build/qwen38-a"
        directory.mkdir(parents=True, exist_ok=True)
        for part in range(5):
            text = "".join(chr(cp) for cp in range(part*0x40000, min((part+1)*0x40000, 0x110000))
                           if not 0xD800 <= cp < 0xE000)
            source, result = directory / f"nfc-{part}.in", directory / f"nfc-{part}.out"
            source.write_bytes(text.encode())
            subprocess.run([os.environ["QWEN38_A_TEXT_EXE"], "--nfc", str(source), str(result)], check=True)
            self.assertEqual(result.read_bytes(), tokenizer.normalizer.normalize_str(text).encode())

    def test_nul_unicode_and_special_semantics(self):
        encodes = {case["text"]: case for case in self.actual["encodes"]}
        self.assertEqual(encodes["A\0B"]["ids"], [32, 188, 33])
        self.assertEqual(encodes["e\u0301 vs é"]["decoded"], "é vs é")
        decodes = {tuple(case["ids"]): case for case in self.actual["decodes"]}
        self.assertEqual(decodes[(248060, 248068, 248044)]["skip"], "<|fim_prefix|><think>")
        self.assertEqual(decodes[(248077, 248319)]["keep"], "")
        self.assertNotEqual(encodes["[PAD248077]"]["ids"], [248077])

    def test_thinking_and_history_boundaries(self):
        cases = self.templates
        self.assertTrue(cases["thinking-default"]["rendered"].startswith("<|im_start|>system\n"))
        self.assertTrue(cases["thinking-default"]["rendered"].endswith("<think>\n"))
        self.assertTrue(cases["thinking-disabled"]["rendered"].endswith("<think>\n\n</think>\n\n"))
        self.assertFalse(cases["effort-medium"]["rendered"].startswith("<|im_start|>system"))
        self.assertIn("Ragiono", cases["preserve-default"]["rendered"])
        self.assertNotIn("Ragiono", cases["preserve-false"]["rendered"])
        self.assertIn("<think>\nR\n</think>", cases["assistant-after-query"]["rendered"])

    def test_tools_use_hf_json_and_group_responses(self):
        rendered = self.templates["tools-and-responses"]["rendered"]
        self.assertIn("leggi <&> caffè", rendered)
        self.assertIn("[true, null, 1, 1.0]", rendered)
        self.assertIn('{"z": "è", "a": "<&>"}', rendered)
        self.assertIn("<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response>", rendered)

    def test_error_cases_and_disabled_effort(self):
        expected = {"invalid-effort", "late-system", "invalid-role", "no-user-query", "only-tool-response",
                    "invalid-part", "system-image", "empty-messages"}
        self.assertEqual({name for name, result in self.templates.items() if "error" in result}, expected)
        self.assertEqual(self.templates["empty-messages"]["error_type"], "ValueError")
        self.assertEqual(self.templates["ignored-effort-when-disabled"], self.templates["thinking-disabled"])

    def test_changed_source_hash_is_rejected(self):
        output = ROOT / "build/qwen38/a"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as directory:
            root = Path(directory)
            (root / "build").mkdir()
            (root / "build/qwen38-tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "source hash changed"):
                text_oracle.load_reference(root)

    @unittest.skipUnless(os.environ.get("QWEN38_A_GGUF"), "GGUF path not supplied")
    def test_full_gguf_tokenizer_tables(self):
        result = text_oracle.check_gguf_tables(os.environ["QWEN38_A_GGUF"], self.reference_root)
        self.assertEqual((result["base_tokens"], result["merges"], result["added_tokens"]), (248044, 247587, 33))
        self.assertEqual(result["unused_tokens"], 243)
        self.assertLess(result["metadata_bytes_read"], 12*1024**2)
        self.assertFalse(result["payload_verified"])


if __name__ == "__main__":
    unittest.main()
