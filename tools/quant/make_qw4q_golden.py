"""make_qw4q_golden.py — expected bytes for the qw4q compat encoders, from DwarfStar.

Run once, by hand, with a DwarfStar checkout; only the outputs are committed.
DwarfStar is never linked into picoforge: its quantizer is compiled into a
temporary library here and called on the synthetic rows of golden_inputs.py.

  tools/venv/bin/python tools/quant/make_qw4q_golden.py ../ds4 tests/fixtures/qw4q
"""

import ctypes
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_inputs import golden_inputs  # noqa: E402

# (file, GGML type, seed, rows, columns, zero columns at the end): the Q2_K case
# mimics the expert down projection padded from 640 to 768 inputs.
CASES = [("iq2_xxs.bin", 16, 1234, 64, 2560, 0), ("q2_k.bin", 10, 99, 64, 768, 128)]


def main(ds4: Path, out: Path) -> None:
    commit = subprocess.run(["git", "-C", str(ds4), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        lib_path = Path(tmp) / "libds4quants.dylib"
        subprocess.run(["cc", "-O3", "-std=c11", "-dynamiclib", "-fPIC", "-o", str(lib_path),
                        str(ds4 / "gguf-tools/quants.c"), "-lm", "-pthread"], check=True)
        lib = ctypes.CDLL(str(lib_path))
        f32 = ctypes.POINTER(ctypes.c_float)
        lib.ds4q_quantize_chunk.argtypes = [ctypes.c_int, f32, ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int64, f32]
        lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ds4q_row_size.restype = ctypes.c_size_t
        record = {"dwarfstar_commit": commit, "cases": []}
        for name, qtype, seed, rows, cols, pad in CASES:
            x, imp = golden_inputs(seed, rows, cols)
            if pad:
                x[:, cols - pad:] = 0
                imp[cols - pad:] = 0
            dst = np.empty(rows * lib.ds4q_row_size(qtype, cols), np.uint8)
            n = lib.ds4q_quantize_chunk(qtype, x.ctypes.data_as(f32), dst.ctypes.data, 0, rows, cols,
                                        imp.ctypes.data_as(f32))
            assert n == dst.nbytes, (name, n, dst.nbytes)
            (out / name).write_bytes(dst.tobytes())
            record["cases"].append({"file": name, "type": qtype, "seed": seed, "rows": rows,
                                    "cols": cols, "zero_tail_cols": pad,
                                    "sha256": hashlib.sha256(dst.tobytes()).hexdigest()})
    (out / "golden.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
