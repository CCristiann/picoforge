"""golden_inputs.py — deterministic synthetic rows for the encoder golden tests.

Weights of real expert matrices are heavy-tailed, so rows are Student-t with 3
degrees of freedom and a per-row scale spread over three decades. Two blocks
probe the degenerate paths: block 0 of row 0 is all zero, block 1 of row 1 is
below the encoders' 1e-15 group threshold. The importance (an imatrix row) is
log-normal with 10% zero columns and block 2 all zero.
"""

import numpy as np


def golden_inputs(seed: int, nrows: int, ncols: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-3, 0, size=(nrows, 1))
    x = (rng.standard_t(3, size=(nrows, ncols)) * scale).astype("<f4")
    if nrows > 0 and ncols >= 256:
        x[0, :256] = 0
    if nrows > 1 and ncols >= 512:
        x[1, 256:512] = 1e-18
    importance = rng.lognormal(0.0, 1.5, size=ncols).astype("<f4")
    importance[rng.random(ncols) < 0.1] = 0
    if ncols >= 768:
        importance[512:768] = 0
    return np.ascontiguousarray(x), np.ascontiguousarray(importance)
