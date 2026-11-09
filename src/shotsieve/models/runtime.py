"""ONNX Runtime session factory: CoreML EP with CPU fallback."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from shotsieve._ort import import_onnxruntime
from shotsieve.log import get_logger

ort = import_onnxruntime()

log = get_logger("runtime")

COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"

_NP_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
}


def coreml_available() -> bool:
    return COREML in ort.get_available_providers()


def np_dtype(ort_type: str) -> type[np.generic]:
    """Map an ORT type string such as 'tensor(float16)' to a numpy dtype."""
    try:
        return _NP_DTYPES[ort_type]
    except KeyError as exc:
        raise ValueError(f"unsupported ONNX input type {ort_type!r}") from exc


def _options(threads: int, arena: bool = True) -> ort.SessionOptions:
    so = ort.SessionOptions()
    if not arena:  # big transformer towers: the CPU arena keeps every peak activation resident
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return so


def create_session(
    path: str | Path,
    warmup: Callable[[ort.InferenceSession], object] | None = None,
    use_coreml: bool = True,
    threads: int = 4,
    arena: bool = True,
) -> tuple[ort.InferenceSession, str]:
    """Create an ORT session; returns (session, active provider name).

    CoreML (MLComputeUnits=ALL) is tried first when available. If session creation or the
    `warmup(session)` callback raises, the CPU provider is used instead.
    """
    path = str(path)
    if use_coreml and coreml_available():
        try:
            sess = ort.InferenceSession(
                path,
                sess_options=_options(threads, arena),
                providers=[(COREML, {"MLComputeUnits": "ALL"}), CPU],
            )
            if sess.get_providers()[0] == COREML:
                if warmup is not None:
                    warmup(sess)
                return sess, COREML
        except Exception as exc:
            log.warning("CoreML unavailable for %s (%s); falling back to CPU", path, exc)
    sess = ort.InferenceSession(path, sess_options=_options(threads, arena), providers=[CPU])
    if warmup is not None:
        warmup(sess)
    return sess, CPU
