"""imatrix.py — importance per input column of each routed expert, from an imatrix GGUF.

For tensor T the file holds T.in_sum2 [experts x width] (sum of squared inputs
over the calibration tokens routed to that expert) and T.counts [experts]
(how many tokens). Importance = in_sum2 / count. An expert the router never
picked has count 0 and no data: like DwarfStar, fall back to the energy of its
own weights per column, so large weights are protected.
"""

import hashlib
from pathlib import Path

import numpy as np

from gguf_header import read_header


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Imatrix:
    def __init__(self, path: Path, sha256: str):
        actual = file_sha256(path)
        if actual != sha256:
            raise ValueError(f"{path}: sha256 {actual} does not match the pinned {sha256}")
        self.path = Path(path)
        meta, self.tensors = read_header(path)
        if meta.get("general.type") != "imatrix":
            raise ValueError(f"{path}: general.type is not imatrix")
        self.datasets = list(meta["imatrix.datasets"])
        self.chunk_count = int(meta["imatrix.chunk_count"])
        self.chunk_size = int(meta["imatrix.chunk_size"])
        self._cache = {}

    def _entry(self, tensor: str) -> tuple[np.ndarray, np.ndarray]:
        if tensor not in self._cache:
            sums_name, counts_name = tensor + ".in_sum2", tensor + ".counts"
            if sums_name not in self.tensors or counts_name not in self.tensors:
                raise KeyError(f"imatrix has no entry for {tensor}")
            (st, sdims, soff), (ct, cdims, coff) = self.tensors[sums_name], self.tensors[counts_name]
            width, experts = sdims
            if st != 0 or ct != 0 or cdims != [1, experts]:
                raise ValueError(f"{tensor}: unexpected imatrix layout {sdims} / {cdims}")
            sums = np.memmap(self.path, "<f4", "r", soff, (experts, width))
            counts = np.memmap(self.path, "<f4", "r", coff, (experts,))
            if not (np.isfinite(sums).all() and (np.asarray(sums) >= 0).all() and np.isfinite(counts).all()):
                raise ValueError(f"{tensor}: non-finite or negative imatrix statistics")
            self._cache[tensor] = (sums, counts)
        return self._cache[tensor]

    def expert_importance(self, tensor: str, expert: int, source_f32: np.ndarray) -> np.ndarray:
        """source_f32: the expert's weights [rows, width], used only for the zero-count fallback."""
        sums, counts = self._entry(tensor)
        if counts[expert] > 0:
            return np.ascontiguousarray(sums[expert] / counts[expert], dtype="<f4")
        return np.ascontiguousarray(np.square(source_f32, dtype=np.float32).sum(axis=0, dtype=np.float32), dtype="<f4")
