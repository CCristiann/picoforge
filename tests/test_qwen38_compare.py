"""R0.4 paired picoforge/DwarfStar harness: pure logic with fake tokenizers (no GPU, no lock)."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT/'tools/eval/qwen38_compare.py'
sys.path.insert(0, str(TOOL.parent))
import qwen38_compare as qc  # noqa: E402
import run_observed  # noqa: E402

SPECIAL = {'<|endoftext|>': 1000, '<|im_end|>': 1002}


def fake_ids(text):
    """Character tokenizer that knows the two stop spellings, like both real ones."""
    ids, i = [], 0
    while i < len(text):
        hit = next((s for s in SPECIAL if text.startswith(s, i)), None)
        ids.append(SPECIAL[hit] if hit else ord(text[i]))
        i += len(hit) if hit else 1
    return ids


class Fake:
    """Stands in for the two CLIs and for picoforge's greedy continuation run."""
    def __init__(self, defect=None, greedy=([99]*128, 'c'*128)):
        self.defect, self.continuation = defect, greedy

    def tokenize(self, engine, path):
        ids = fake_ids(path.read_text())
        if engine == 'dwarfstar' and self.defect == 'long' and len(ids) > 5000:
            ids[5000] += 1  # one differing ID deep inside the 8K frontier
        return ids

    def greedy(self, prompt_ids, count, directory):
        return self.continuation


CASE = dict(prompt_ids=[97, 98], prompt_text='ab', recorded_ids=[99]*32)


class FakeObserve:
    """In place of run_observed.run: logs each launch and prints what the real CLI would."""
    def __init__(self, calls, code=0):
        self.calls, self.code = calls, code

    def __call__(self, output, cwd, command, timeout, interval, guard):
        def arg(flag):
            return command[command.index(flag) + 1]
        kind = 'tokenize' if '--encode-only' in command or '--dump-tokens' in command else 'greedy'
        self.calls.append(('launch', kind, Path(command[0]).name, timeout, interval, guard))
        output.mkdir()
        if self.code:
            return self.code  # a failed process prints nothing
        if '--encode-only' in command:
            text = json.dumps({'prompt_ids': fake_ids(Path(arg('--complete-file')).read_text())})
        elif '--dump-tokens' in command:
            text = json.dumps(fake_ids(Path(arg('--prompt-file')).read_text())) + '\n    99  c\n'
        else:
            prompt = [int(t) for t in arg('--tokens').split(',')]
            Path(arg('--token-ids')).write_text(json.dumps({'prompt_ids': prompt, 'generated_ids': [99]*128}))
            text = 'c'*128 + '\n'
        (output/'stdout.log').write_text(text)
        return self.code


def memory_sample(pressure=1, swap=0):
    """Fake build/process_memory rows: calm for the system sample, (pressure, swap) for the child."""
    def sample(_probe, pid=None):
        return dict(monotonic_s=run_observed.now(), system_pressure_level=pressure if pid else 1,
                    system_swap_used_bytes=swap if pid else 0, phys_footprint_bytes=1,
                    lifetime_peak_phys_footprint_bytes=1)
    return sample


def bench_launch(calls):
    """In place of Launcher for `run`: writes what each engine's sample leaves behind."""
    def launch(directory, cwd, command, timeout):
        def arg(flag):
            return command[command.index(flag) + 1]
        name = Path(command[0]).name
        calls.append((name, command))
        (directory/'process').mkdir(parents=True)
        if name == 'picoforge':
            ids, n = [int(t) for t in arg('--tokens').split(',')], int(arg('--teacher-forced'))
            Path(arg('--runs-jsonl')).write_text(json.dumps(dict(mode='teacher-forced', decode_tokens=n,
                                                 stop_reason='length', decode_s=n/1.25, prefill_s=2.0)) + '\n')
            Path(arg('--token-ids')).write_text(json.dumps(dict(prompt_ids=ids[:-n], forced_ids=ids[-n:])))
        else:
            Path(arg('--csv')).write_text('ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,'
                                          f"gen_steady_tokens,gen_steady_tps,kvcache_bytes\n{arg('--ctx-start')},2,"
                                          f"9.0,{arg('--gen-tokens')},31.00,40.0,127,31.10,0\n")
        memory = dict(system_swap_used_bytes=0)
        (directory/'process/process.json').write_text(json.dumps(dict(
            system_before=memory, system_after=memory, lifetime_peak_footprint_bytes=1, process_s=1.0)))
        (directory/'conditions.json').write_text(json.dumps(dict(before=dict(ac_power=True), after=dict(ac_power=True))))
        return 0
    return launch


class CompareTests(unittest.TestCase):
    def test_schedule_alternates_blocks(self):
        plan = qc.schedule({'picoforge': (1, 3), 'dwarfstar': (1, 3)}, 2)
        self.assertEqual(plan, [('picoforge', [True, False]), ('dwarfstar', [True, False]),
                                ('picoforge', [False, False]), ('dwarfstar', [False, False])])
        # The slow 8K baseline: alternate while both engines have samples left, then finish the other.
        counts = {'picoforge': (0, 2), 'dwarfstar': (3, 20)}
        plan = qc.schedule(counts, 1)
        self.assertEqual([e for e, _ in plan[:4]], ['picoforge', 'dwarfstar']*2)
        self.assertEqual({e for e, _ in plan[4:]}, {'dwarfstar'})
        for engine, (warmup, measure) in counts.items():
            flags = [f for e, block in plan if e == engine for f in block]
            self.assertEqual(flags, [True]*warmup + [False]*measure)  # warm-ups first, per engine
        self.assertTrue(all(len(block) == 1 for _, block in plan))
        for bad in ({'picoforge': (0, 0), 'dwarfstar': (0, 1)}, {'picoforge': (-1, 1), 'dwarfstar': (0, 1)}):
            with self.assertRaises(ValueError):
                qc.schedule(bad, 1)
        with self.assertRaises(ValueError):
            qc.schedule({'picoforge': (0, 1), 'dwarfstar': (0, 1)}, 0)

    def test_stats_nearest_sample_percentiles(self):
        values = [float(v) for v in (7, 1, 20, 3, 14, 9, 2, 18, 5, 11, 16, 4, 13, 8, 19, 6, 12, 10, 17, 15)]
        s = qc.stats(values)
        # Nearest sorted sample, as native-next: an interpolated p10 would be 2.9, p90 18.1.
        self.assertEqual((s['n'], s['median'], s['p10'], s['p90']), (20, 10.5, 3.0, 18.0))
        samples = [dict(engine='picoforge', warmup=True, tok_s=100.0)]
        samples += [dict(engine='picoforge', warmup=False, tok_s=v) for v in (1.0, 1.2, 1.1)]
        samples += [dict(engine='dwarfstar', warmup=True, tok_s=0.1)]
        samples += [dict(engine='dwarfstar', warmup=False, tok_s=v) for v in (30.0, 32.0, 31.0)]
        summary = qc.summary(samples)
        self.assertEqual(summary['picoforge']['median'], 1.1)  # warm-ups excluded
        self.assertEqual(summary['dwarfstar']['median'], 31.0)
        self.assertEqual(summary['warmups_excluded'], 2)
        self.assertAlmostEqual(summary['ratio_picoforge_over_dwarfstar'], 1.1/31.0)

    def test_prepare_rejects_token_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            corpus = d/'corpus.txt'
            corpus.write_text(''.join(f'riga {i:05d} del romanzo\n' for i in range(4000)))
            # Control: agreeing tokenizers give both frontiers and the stop IDs.
            qc.prepare(d/'good', Fake(), corpus, CASE)
            suite = json.loads((d/'good/suite.json').read_text())
            self.assertEqual(suite['stop_ids'], [1000, 1002])
            short, long = suite['frontiers']['short'], suite['frontiers']['8k']
            self.assertEqual((short['prompt_ids'], short['forced_ids']), ([97, 98], [99]*128))
            self.assertTrue(short['frozen_prefix_match'])
            self.assertEqual((len(long['prompt_ids']), len(long['forced_ids'])), (8063, 128))
            self.assertEqual(long['prompt_ids'][:4], fake_ids('riga'))
            cases = {'long': (Fake('long'), 'disagree'),
                     'short': (Fake(greedy=([99]*128, 'd'*128)), 'reproduce'),
                     'stop': (Fake(greedy=([99]*127 + [1002], 'c'*127 + '<|im_end|>')), 'stop'),
                     'early': (Fake(greedy=([99]*100, 'c'*100)), 'early')}
            for name, (fake, message) in cases.items():
                with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                    qc.prepare(d/name, fake, corpus, CASE)


    def test_every_launch_waits_for_release(self):
        with tempfile.TemporaryDirectory() as directory:
            d, calls = Path(directory), []
            corpus = d/'corpus.txt'
            corpus.write_text(''.join(f'riga {i:05d} del romanzo\n' for i in range(4000)))
            launch = qc.Launcher(settle=lambda path: calls.append(('settle', path.name)),
                                 observe=FakeObserve(calls), conditions=dict)
            qc.prepare(d/'suite', qc.Engines(d/'suite', launch), corpus, CASE)
            # Every process is fenced by a release wait on both sides, so no mapping of the
            # checkpoint can start while a previous Metal process's pages are still wired.
            self.assertEqual([c[:2] if c[0] == 'settle' else c[0] for c in calls],
                             [('settle', 'release-before'), 'launch', ('settle', 'release-after')]*7)
            launches = [c for c in calls if c[0] == 'launch']
            # Tokenizers needing no model run go first; the short text only after greedy.
            self.assertEqual([(c[1], c[2]) for c in launches],
                             [('tokenize', 'picoforge'), ('tokenize', 'ds4')]*2 + [('greedy', 'picoforge')] +
                             [('tokenize', 'picoforge'), ('tokenize', 'ds4')])
            self.assertTrue(all(c[3] == qc.TOKENIZE_TIMEOUT == 60 for c in launches if c[1] == 'tokenize'))
            self.assertTrue(all(c[4] == .25 and c[5] == qc.GUARD for c in launches))
            self.assertEqual(json.loads((d/'suite/suite.json').read_text())['frontiers']['short']['forced_ids'], [99]*128)
            # A failed or crashed process is still followed by the release wait.
            calls.clear()
            self.assertEqual(qc.Launcher(lambda path: calls.append(('settle', path.name)), FakeObserve(calls, 9), dict)(
                d/'failed', d, [sys.executable, '--encode-only'], 1), 9)
            def crash(*args, **kwargs):
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                qc.Launcher(lambda path: calls.append(('settle', path.name)), crash, dict)(d/'crashed', d, ['x'], 1)
            self.assertEqual([c[1] if c[0] == 'settle' else c[0] for c in calls],
                             ['release-before', 'launch', 'release-after', 'release-before', 'release-after'])
        # Statically: the Launcher holds the only observe() call, and no other subprocess
        # call in the tool can start an engine (only pmset, git and the capability probe).
        source = Path(qc.__file__).read_text()
        self.assertEqual(source.count('self.observe('), 1)
        self.assertEqual(re.findall(r'(?<!self\.)\bobserve\(', source), [])
        for line in source.splitlines():
            if 'subprocess.' in line:
                self.assertTrue(any(k in line for k in ("['pmset'", "['git'", '[str(probe)]')), line)

    def test_watchdog_kills_guard_breach(self):
        self.assertEqual(qc.GUARD, {'pressure': 2, 'swap_bytes': 256 << 20, 'samples': 2})
        sleeper = [sys.executable, '-c', 'import time; time.sleep(30)']
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            launch = qc.Launcher(settle=lambda path: None, conditions=dict)  # the real run_observed
            for name, sample in (('pressure', memory_sample(pressure=2)), ('swap', memory_sample(swap=257 << 20))):
                with self.subTest(name), patch.object(run_observed, 'sample', sample):
                    self.assertNotEqual(launch(d/name, d, sleeper, 20), 0)
                    meta = json.loads((d/name/'process/process.json').read_text())
                    self.assertEqual(meta['stop_reason'], 'memory guard')
                    self.assertLess(meta['process_s'], 10)
                    self.assertGreaterEqual(meta['samples'], 2)
                    self.assertEqual(meta['guard_limits'], qc.GUARD)
            # The other tools keep the old default: pressure 2 alone does not stop them.
            with patch.object(run_observed, 'sample', memory_sample(pressure=2)):
                rc = run_observed.run(d/'default', d, [sys.executable, '-c', 'import time; time.sleep(.4)'], 20, .05)
            meta = json.loads((d/'default/process.json').read_text())
            self.assertEqual((rc, meta['stop_reason'], meta['guard_limits']),
                             (0, None, {'pressure': 4, 'swap_bytes': 2 << 30, 'samples': 3}))
            self.assertEqual(meta['guard'], 'stop after 3 samples: critical pressure or swap growth > 2 GiB')


    def test_run_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'out'
            out.mkdir()
            (out/'keep.txt').write_text('evidence')
            done = subprocess.run([sys.executable, '-B', str(TOOL), 'run', str(Path(directory)/'suite'), str(out),
                                   '--frontier', 'short', '--warmup', '0', '--measure', '1', '--block', '1'],
                                  capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn('exists', done.stderr)
            self.assertEqual([p.name for p in out.iterdir()], ['keep.txt'])
            self.assertEqual((out/'keep.txt').read_text(), 'evidence')


    def test_run_alternates_and_summarizes(self):
        with tempfile.TemporaryDirectory() as directory:
            d, calls = Path(directory), []
            corpus = d/'corpus.txt'
            corpus.write_text(''.join(f'riga {i:05d} del romanzo\n' for i in range(4000)))
            qc.prepare(d/'suite', Fake(), corpus, CASE)
            with patch.object(qc, 'bounded', lambda: (None, 'pinned')), patch.object(qc, 'provenance', lambda r: {}), \
                    patch.object(qc, 'conditions', dict), patch.object(qc, 'thermal', lambda: None):
                qc.run(d/'suite', d/'out', 'short', {'picoforge': (1, 2), 'dwarfstar': (1, 2)}, 1, 10, bench_launch(calls))
            self.assertEqual([c[0] for c in calls], ['picoforge', 'ds4-bench']*3)
            pico, bench = calls[0][1], calls[1][1]
            self.assertNotIn('--logits', pico)  # its fwrite would be timed inside decode_s
            self.assertEqual(pico[pico.index('--tokens') + 1], ','.join(map(str, [97, 98] + [99]*128)))
            for flag, value in (('--ctx-start', '2'), ('--ctx-max', '2'), ('--ctx-alloc', '8192'), ('--gen-tokens', '128')):
                self.assertEqual(bench[bench.index(flag) + 1], value)
            self.assertIn('--teacher-forced-decode', bench)
            self.assertEqual(Path(bench[bench.index('--prompt-file') + 1]).name, 'short.txt')
            samples = [json.loads(line) for line in (d/'out/samples.jsonl').read_text().splitlines()]
            self.assertEqual([(x['engine'], x['warmup']) for x in samples],
                             [('picoforge', True), ('dwarfstar', True)] + [('picoforge', False), ('dwarfstar', False)]*2)
            summary = json.loads((d/'out/summary.json').read_text())
            self.assertEqual((summary['picoforge']['median'], summary['dwarfstar']['median']), (1.25, 31.0))
            self.assertAlmostEqual(summary['ratio_picoforge_over_dwarfstar'], 1.25/31.0)
            self.assertTrue(json.loads((d/'out/manifest.json').read_text())['complete'])
            self.assertEqual((d/'out/samples.csv').read_text().splitlines()[0], ','.join(qc.COLUMNS))


if __name__ == '__main__':
    unittest.main()
