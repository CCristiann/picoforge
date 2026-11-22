"""qw4q.py — ctypes binding of build/libqw4q.dylib (tools/quant/qw4q.c).

GGML type ids: 16 = IQ2_XXS, 10 = Q2_K. Modes: 0 = compat (byte-identical to
DwarfStar b3f5018), 1 = refine (O1, IQ2_XXS only).
"""

import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = ROOT / "build/libqw4q.dylib"
IQ2_XXS, Q2_K = 16, 10
COMPAT, REFINE = 0, 1

_F32 = ctypes.POINTER(ctypes.c_float)
_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(LIB_PATH))
        _lib.qw4q_row_bytes.argtypes = [ctypes.c_int, ctypes.c_int64]
        _lib.qw4q_row_bytes.restype = ctypes.c_size_t
        _lib.qw4q_quantize_iq2_xxs.argtypes = [_F32, ctypes.c_int64, ctypes.c_int64, _F32, ctypes.c_int,
                                               ctypes.c_void_p]
        _lib.qw4q_quantize_iq2_xxs.restype = ctypes.c_int
        if hasattr(_lib, "qw4q_quantize_q2_k"):
            _lib.qw4q_quantize_q2_k.argtypes = [_F32, ctypes.c_int64, ctypes.c_int64, _F32, ctypes.c_void_p]
            _lib.qw4q_quantize_q2_k.restype = ctypes.c_int
        _lib.qw4q_dequantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, _F32]
        _lib.qw4q_dequantize.restype = ctypes.c_int
    return _lib


def quantize(qtype: int, x: np.ndarray, importance: np.ndarray, mode: int = COMPAT) -> bytes:
    """Quantize rows x [nrows, ncols]; importance has ncols values shared by every row."""
    x = np.ascontiguousarray(x, dtype="<f4")
    importance = np.ascontiguousarray(importance, dtype="<f4")
    nrows, ncols = x.shape
    if importance.shape != (ncols,):
        raise ValueError(f"importance shape {importance.shape} does not match {ncols} columns")
    size = lib().qw4q_row_bytes(qtype, ncols)
    if size == 0:
        raise ValueError(f"type {qtype} cannot hold rows of {ncols} columns")
    out = np.empty(nrows * size, np.uint8)
    if qtype == IQ2_XXS:
        rc = lib().qw4q_quantize_iq2_xxs(x.ctypes.data_as(_F32), nrows, ncols, importance.ctypes.data_as(_F32),
                                        mode, out.ctypes.data)
    elif qtype == Q2_K and mode == COMPAT:
        rc = lib().qw4q_quantize_q2_k(x.ctypes.data_as(_F32), nrows, ncols, importance.ctypes.data_as(_F32),
                                     out.ctypes.data)
    else:
        raise ValueError(f"unsupported type {qtype} / mode {mode}")
    if rc != 0:
        raise ValueError(f"qw4q rejected type {qtype}, {nrows}x{ncols}")
    return out.tobytes()


def dequantize(qtype: int, data: bytes, nrows: int, ncols: int) -> np.ndarray:
    out = np.empty((nrows, ncols), "<f4")
    buf = np.frombuffer(data, np.uint8)
    if lib().qw4q_dequantize(qtype, buf.ctypes.data, nrows, ncols, out.ctypes.data_as(_F32)) != 0:
        raise ValueError(f"qw4q cannot decode type {qtype}, {nrows}x{ncols} from {len(data)} bytes")
    return out
