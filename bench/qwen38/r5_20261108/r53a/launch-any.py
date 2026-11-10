"""Controller helper (not committed): one guarded run of any checkpoint-mapping command.

usage: launch-any.py OUT_DIR CWD [VAR=VALUE ...] -- command args...   ({d} = OUT_DIR)
Same Launcher as launch-cmd.py (release wait, strict memory guard, awake + AC check). Run under the GPU lock.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/eval'))
from qwen38_compare import Launcher

out = Path(sys.argv[1]).resolve(); cwd = Path(sys.argv[2]).resolve(); rest = sys.argv[3:]
assert 'PICOFORGE_GPU_LOCK_FD' in os.environ and not out.exists()
cut = rest.index('--'); env = dict(a.split('=', 1) for a in rest[:cut])
env = {k: v.replace('{d}', str(out)) for k, v in env.items()}
cmd = [a.replace('{d}', str(out)) for a in rest[cut + 1:]]
code = Launcher()(out, cwd, cmd, 3600, env=env or None)
print(f'{out.name}: exit {code}', flush=True)
sys.exit(code)
