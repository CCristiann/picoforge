"""G3 gate launches: every step that maps the checkpoint goes through Launcher (fakes, no GPU, no lock)."""
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'tools/eval')]
import qwen38_compare as qc  # noqa: E402
import qwen38_g3 as g3  # noqa: E402

AWAKE = lambda: dict(graphics=True, lid_open=True, ac_power=True)  # noqa: E731  no test reads the real Mac
STEPS = ['native', 'stock-metal', 'fp32-reference', 'chat']


class G3LaunchTests(unittest.TestCase):
    def test_checkpoint_steps_never_bypass_the_launcher(self):
        source = (ROOT/'tools/qwen38_g3.py').read_text()
        self.assertEqual(re.findall(r'\bobserve\(', source), [])  # the old bare observer is gone
        # The four steps that map the 46 GiB checkpoint are fenced, in this order.
        self.assertEqual(re.findall(r"\bfenced\(launch, out, '([\w-]+)'", source), STEPS)
        # What stays plain never maps it: clang, and the tiny synthetic graph.gguf replay.
        self.assertEqual(re.findall(r"\brun\('([\w-]+)'", source), ['oracle-build', 'near-tie-replay'])
        self.assertEqual(source.count('subprocess.run('), 1)  # inside that plain run()
        for line in source.splitlines():
            if 'check_output(' in line:
                self.assertIn("'git'", line)

    def test_fenced_gives_each_step_its_own_launch_directory(self):
        calls = []
        def launch(directory, cwd, command, timeout, env=None):
            calls.append((directory.name, cwd, command, timeout, env))
            return 0
        out = Path('/nowhere')
        record = g3.fenced(launch, out, 'stock-metal', [out/'oracle', 7], Path('/ds'), env={'DS4_X': '1'})
        g3.fenced(launch, out, 'native', ['forward_real'], ROOT)
        # 'native' is forward_real's own output directory: the launch directory must not be it.
        self.assertEqual(calls, [('launch-stock-metal', Path('/ds'), ['/nowhere/oracle', '7'], 900, {'DS4_X': '1'}),
                                 ('launch-native', ROOT, ['forward_real'], 900, None)])
        self.assertEqual(record, dict(name='stock-metal', argv=['/nowhere/oracle', '7'], cwd='/ds',
                                      env={'DS4_X': '1'}, launcher='qwen38_compare.Launcher'))
        with self.assertRaisesRegex(RuntimeError, r'fp32-reference.*code 9.*evidence'):
            g3.fenced(lambda *a, **k: 9, out, 'fp32-reference', ['ds4'], ROOT)

    def test_launcher_forwards_env_and_fenced_step_waits_for_release(self):
        seen = []
        def observe(output, cwd, command, timeout, interval, guard, env_overrides='unset'):
            seen.append(('observe', output.parent.name, env_overrides, guard))
            output.mkdir()
            return 0
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            launcher = qc.Launcher(lambda path: seen.append(('settle', path.name)), observe, dict, AWAKE)
            launcher(d/'direct', ROOT, ['x'], 1, env={'A': '1'})
            self.assertEqual(seen[1], ('observe', 'direct', {'A': '1'}, qc.GUARD))
            seen.clear()
            launcher(d/'plain', ROOT, ['x'], 1)  # env stays optional
            self.assertIsNone(seen[1][2])
            seen.clear()
            g3.fenced(launcher, d, 'native', ['x'], ROOT, env={'B': '2'})
            self.assertEqual(seen, [('settle', 'release-before'), ('observe', 'launch-native', {'B': '2'}, qc.GUARD),
                                    ('settle', 'release-after')])


if __name__ == '__main__':
    unittest.main()
