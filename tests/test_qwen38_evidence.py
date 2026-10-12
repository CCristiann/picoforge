"""Reject incomplete/invalid checkpoint and measurement evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/eval"))
import summarize_qwen38 as summary
import verify_qwen38_download as verify
import download_qwen38_q2 as downloader
import subprocess


class EvidenceTests(unittest.TestCase):
    def test_retry_resumes_the_new_partial_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = root / (downloader.FILENAME + ".part")
            part.write_bytes(b"abc")
            offsets = []
            def transfer(command, **_kwargs):
                offsets.append(part.stat().st_size)
                self.assertIn("--continue-at", command)
                with part.open("ab") as stream: stream.write(b"def")
                return subprocess.CompletedProcess(command, 18 if len(offsets) == 1 else 0)
            with patch.object(downloader, "EXPECTED_SIZE", 9), patch.object(downloader.subprocess, "run", transfer), patch.object(downloader.time, "sleep"):
                self.assertEqual(downloader.download(root, root / "evidence"), 0)
            self.assertEqual(offsets, [3, 6])
            self.assertEqual(part.read_bytes(), b"abcdefdef")
            self.assertFalse((root / downloader.FILENAME).exists())

    def test_publish_verified_file_without_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            part = Path(directory) / "test.gguf.part"
            part.write_bytes(b"test checkpoint")
            inode = part.stat().st_ino
            with patch.object(verify, "EXPECTED_SIZE", 15), patch.object(verify, "EXPECTED_HASH", hashlib.sha256(part.read_bytes()).hexdigest()):
                result = verify.verify(part, Path(directory) / "verified.json")
            self.assertTrue(result["matched"])
            self.assertFalse(part.exists())
            self.assertEqual(Path(result["path"]).stat().st_ino, inode)

    def test_mismatch_preserves_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            part, report = Path(directory) / "test.gguf.part", Path(directory) / "verified.json"
            part.write_bytes(b"bad")
            with patch.object(verify, "EXPECTED_SIZE", 3), self.assertRaises(ValueError):
                verify.verify(part, report)
            self.assertFalse(json.loads(report.read_text())["matched"])
            self.assertTrue(part.exists())

    def test_incomplete_refuses_hash_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            part = Path(directory) / "test.gguf.part"
            part.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                verify.verify(part, Path(directory) / "verified.json")
            part.with_suffix("").write_bytes(b"another session")
            with patch.object(verify, "EXPECTED_SIZE", 3), self.assertRaises(FileExistsError):
                verify.verify(part, Path(directory) / "verified.json")
            self.assertEqual(part.with_suffix("").read_bytes(), b"another session")

    def test_summary_rejects_bad_timing_missing_rows_and_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(repeat=i, prompt_tokens=10, generated=2, decode_evals=1,
                         load_s=.2, session_s=.005, warmup_s=.005, prefill_s=.1,
                         decode_s=.2, request_s=.32, ttft_session_s=.12,
                         decode_start_monotonic_s=100.36+i*.32, engine_start_monotonic_s=100.05,
                         first_token_from_start_s=.32+i*.32) for i in range(2)]
            system = dict(system_swap_used_bytes=0, system_swapouts_pages=0)
            (root / "process.json").write_text(json.dumps(dict(complete=True, returncode=0, process_s=1, start_monotonic_s=100,
                                                              system_before=system, system_after=system)))
            for i in range(2): (root / f"answer.{i}.ids").write_text("4\n5\n")
            (root / "memory.jsonl").write_text(json.dumps(dict(phys_footprint_bytes=100,
                lifetime_peak_phys_footprint_bytes=120, system_pressure_level=1)) + "\n")
            log = []
            for i in range(6):
                log.append("ngram_io: " + json.dumps(dict(phase=["before_prefill", "after_prefill", "after_decode"][i%3],
                    calls=i, bytes=320*i, summed_service_ns=100*i, errors=0)))
            (root / "stderr.log").write_text("\n".join(log) + "\nmetal_allocated_bytes: 90\nmetal_allocated_bytes: 91\n")
            def write_rows(values):
                (root / "stdout.log").write_text("".join(json.dumps(r) + "\n" for r in values))
            write_rows(rows)
            self.assertEqual(summary.case_summary(root, 1, 1)["decode_tok_s"], 5.0)
            write_rows(rows[:1])
            with self.assertRaisesRegex(ValueError, "repetitions"): summary.case_summary(root, 1, 1)
            write_rows([rows[0], {**rows[1], "decode_s": -1}])
            with self.assertRaisesRegex(ValueError, "timer"): summary.case_summary(root, 1, 1)
            write_rows(rows)
            (root / "answer.1.ids").write_text("4\n6\n")
            with self.assertRaisesRegex(ValueError, "diverged"): summary.case_summary(root, 1, 1)


if __name__ == "__main__":
    unittest.main()
