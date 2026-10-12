"""Executable grading for the local pilot. Review generated source before use.

This executes model-produced code in a temporary directory with timeouts; it
is a correctness scaffold, not a security sandbox or a standard benchmark.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

PYTHON_TESTS = '''
import copy, random
from answer import merge_ranges
assert merge_ranges([]) == []
assert merge_ranges(iter([(3, 5), (1, 3), (9, 9)])) == [(1, 5)]
rng = random.Random(7319)
for _ in range(120):
    xs = [tuple(sorted([rng.randrange(-30, 31), rng.randrange(-30, 31)]))
          for _ in range(rng.randrange(20))]
    original = copy.deepcopy(xs)
    covered = set().union(*(set(range(a, b)) for a, b in xs))
    expected = []
    for x in sorted(covered):
        if expected and expected[-1][1] == x:
            expected[-1] = (expected[-1][0], x + 1)
        else:
            expected.append((x, x + 1))
    assert merge_ranges(xs) == expected, (xs, expected)
    assert xs == original
print("PASS: interval oracle, generator and no mutation")
'''
FILES_TESTS = '''
from slices import windows
from summaries import rolling_means
for function in (windows, rolling_means):
    for k in (0, -1, -20):
        try: function([1, 2], k)
        except ValueError: pass
        else: raise AssertionError((function.__name__, k))
assert windows([], 1) == []
assert windows([3], 2) == []
assert windows([3], 1) == [[3]]
assert windows([1, 2, 3], 2) == [[1, 2], [2, 3]]
assert windows([1, 2, 3], 3) == [[1, 2, 3]]
assert rolling_means([-2, 1, 4], 2) == [-0.5, 2.5]
assert rolling_means([0.5, 1.0, 2.0], 2) == [0.75, 1.5]
xs = [1, 2, 3]; windows(xs, 2); rolling_means(xs, 2)
assert xs == [1, 2, 3]
assert rolling_means([], 1) == []
assert rolling_means([2], 2) == []
assert rolling_means([2, 4], 2) == [3.0]
print("PASS: both files, boundaries, validation, true division")
'''
C_TESTS = '''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
static int cmp(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}
int main(void) {
    assert(lower_bound_int(NULL, 0, INT_MIN) == 0);
    uint32_t state = 9181;
    for (size_t n = 1; n <= 64; n++) {
        int a[64];
        for (size_t i = 0; i < n; i++) {
            state = state * UINT32_C(1664525) + UINT32_C(1013904223);
            a[i] = (int)(state % 31) - 15;
        }
        a[0] = INT_MIN; if (n > 1) a[1] = INT_MAX;
        qsort(a, n, sizeof(a[0]), cmp);
        int extreme[] = {INT_MIN, INT_MAX};
        for (int j = -20; j <= 21; j++) {
            int x = j >= 20 ? extreme[j - 20] : j;
            size_t expected = 0;
            while (expected < n && a[expected] < x) expected++;
            assert(lower_bound_int(a, n, x) == expected);
        }
    }
    puts("PASS: lower_bound linear oracle and sanitizers");
}
'''


def _grade(case, response):
    final = response.rsplit("</think>", 1)[-1].strip()
    fence = re.fullmatch(r"```(?:python|c|C|json)?\s*\n(.*?)\n```", final, re.S)
    source = fence.group(1) if fence else final
    result = dict(case=case["id"], format_followed=fence is None, tests_passed=False)
    if case["kind"] == "json":
        try:
            result["tests_passed"] = json.loads(final) == case["expected"]
        except ValueError:
            pass
        return result
    with tempfile.TemporaryDirectory(prefix="picoforge-qwen38-grade-") as directory:
        path = Path(directory)
        if case["kind"] == "python-files":
            files = json.loads(source)
            if set(files) != {"slices.py", "summaries.py"} or not all(isinstance(v, str) for v in files.values()):
                raise ValueError("expected exactly the two source files")
            for name, text in files.items():
                (path / name).write_text(text)
            tests = FILES_TESTS
        elif case["kind"] == "python":
            (path / "answer.py").write_text(source)
            tests = PYTHON_TESTS
        else:
            (path / "answer.c").write_text("#include <stddef.h>\n#include <stdlib.h>\n" + source + C_TESTS)
            compile_result = subprocess.run(["clang", "-std=c11", "-Wall", "-Wextra", "-Wconversion", "-Werror",
                "-fsanitize=address,undefined", "answer.c", "-o", "answer"], cwd=path, capture_output=True, text=True, timeout=30)
            if compile_result.returncode:
                return {**result, "compiler_output": compile_result.stderr}
            tests = None
        if tests is not None:
            script = "import sys\nsys.path.insert(0, " + repr(str(path)) + ")\n" + tests
            (path / "check.py").write_text(script)
        command = [sys.executable, "-I", str(path / "check.py")] if tests else [str(path / "answer")]
        executed = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=5)
        result.update(tests_passed=executed.returncode == 0, returncode=executed.returncode,
                      stdout=executed.stdout, stderr=executed.stderr)
    return result


def grade(case, response):
    try:
        return _grade(case, response)
    except (ValueError, TypeError, subprocess.TimeoutExpired) as error:
        return dict(case=case["id"], tests_passed=False, format_followed=False,
                    error=str(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    case = next(c for c in json.loads(args.suite.read_text())["cases"] if c["id"] == args.case_id)
    print(json.dumps(grade(case, args.response.read_text()), indent=2))
