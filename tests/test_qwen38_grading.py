"""Prove the pilot grader accepts solutions and rejects the supplied bugs."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("grading", ROOT / "tools/eval/grade_qwen38.py")
grading = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grading)
CASES = {c["id"]: c for c in json.loads((ROOT / "tools/eval/qwen38_cases.json").read_text())["cases"]}

MERGE = '''def merge_ranges(ranges):
    result = []
    for a,b in sorted((a,b) for a,b in ranges if a != b):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else: result.append((a,b))
    return result
'''
LOWER = '''size_t lower_bound_int(const int *a, size_t n, int x) {
    size_t lo=0, hi=n;
    while(lo<hi) {
        size_t mid=lo+(hi-lo)/2;
        if(a[mid]<x) lo=mid+1; else hi=mid;
    }
    return lo;
}
'''


class GradingTests(unittest.TestCase):
    def test_merge_oracle(self):
        case = CASES["merge-ranges"]
        self.assertTrue(grading.grade(case, MERGE)["tests_passed"])
        self.assertFalse(grading.grade(case, MERGE.replace("a <=", "a <"))["tests_passed"])

    def test_c_oracle(self):
        case = CASES["lower-bound"]
        self.assertTrue(grading.grade(case, LOWER)["tests_passed"])
        self.assertFalse(grading.grade(case, LOWER.replace("a[mid]<x", "a[mid]<=x"))["tests_passed"])

    def test_two_files(self):
        case = CASES["two-file-fix"]
        good = {"slices.py": "def windows(xs,k):\n    if k<=0: raise ValueError('width')\n    return [xs[i:i+k] for i in range(len(xs)-k+1)]\n",
                "summaries.py": "from slices import windows\ndef rolling_means(xs,k):\n    return [sum(w)/k for w in windows(xs,k)]\n"}
        self.assertTrue(grading.grade(case, json.dumps(good))["tests_passed"])
        good["slices.py"] = good["slices.py"].replace("len(xs)-k+1", "len(xs)-k")
        self.assertFalse(grading.grade(case, json.dumps(good))["tests_passed"])
        self.assertFalse(grading.grade(case, "not json")["tests_passed"])

    def test_reasoning_and_truncation(self):
        case = CASES["crt-italian"]
        self.assertTrue(grading.grade(case, 'reasoning...</think>\n{"n":1633}')["tests_passed"])
        self.assertFalse(grading.grade(case, '{"n":1632}')["tests_passed"])
        self.assertFalse(grading.grade(case, 'I still need to think')["tests_passed"])


if __name__ == "__main__":
    unittest.main()
