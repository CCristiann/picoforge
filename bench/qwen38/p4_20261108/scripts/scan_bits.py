"""Old vs new q38c_gdn_scan_fast: bitwise equality of error word, out and state on seeded inputs."""
import subprocess, sys, os, numpy as np
sys.path.insert(0, "tests")
native, old, new = sys.argv[1:4]
def run(lib, qkv, g, beta, state, dims):
    t = len(qkv); hk, hv, dk, dv = dims
    env = {k: v for k, v in os.environ.items() if k != "Q38_REF_KERNELS"}; env["Q38_PACKED_LIBRARY"] = lib
    r = subprocess.run([native, "scan", "fast", str(t), *map(str, dims), "0"], env=env, capture_output=True, pass_fds=(int(os.environ["PICOFORGE_GPU_LOCK_FD"]),),
                       input=b"".join(np.ascontiguousarray(a, "<f4").tobytes() for a in (qkv, g, beta, state)))
    assert r.returncode == 0, r.stderr
    return r.stdout
rng = np.random.default_rng(97)
n = 0
for dims in ((16, 48, 128, 128), (2, 6, 32, 8), (4, 4, 64, 12), (2, 4, 96, 8), (1, 2, 256, 4)):
    for t in (1, 7, 64, 333):
        hk, hv, dk, dv = dims
        qk = rng.normal(0, 1, (t, 2 * hk, dk)); qk /= np.linalg.norm(qk, axis=-1, keepdims=True); qk[:, :hk] /= np.sqrt(dk)
        qkv = np.concatenate([qk.reshape(t, -1), rng.normal(0, 1, (t, hv * dv))], 1)
        ins = [a.astype(np.float32) for a in (qkv, -rng.uniform(0, 3, (t, hv)), rng.uniform(0, 1, (t, hv)), rng.normal(0, .5, (hv, dv, dk)))]
        a, b = run(old, *ins, dims), run(new, *ins, dims)
        assert a == b, (dims, t); n += 1
print(f"bit-identical: {n} cases")
