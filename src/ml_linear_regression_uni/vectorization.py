from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def as_vector(x: np.ndarray) -> np.ndarray:
    """Convert input to a 1D NumPy array."""
    return np.asarray(x).reshape(-1)


def my_dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product via Python loop (educational).
    Returns a scalar float.
    """
    a = as_vector(a)
    b = as_vector(b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a{a.shape} vs b{b.shape}")

    s = 0.0
    for i in range(a.shape[0]):
        s += float(a[i]) * float(b[i])
    return float(s)


def dot_np(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product via NumPy (vectorized)."""
    a = as_vector(a)
    b = as_vector(b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a{a.shape} vs b{b.shape}")

    return float(np.dot(a, b))


@dataclass(frozen=True)
class DotBenchResult:
    n: int
    repeats: int
    loop_ms: float
    numpy_ms: float

    @property
    def speedup(self) -> float:
        if self.numpy_ms == 0:
            return float("inf")
        return self.loop_ms / self.numpy_ms


def benchmark_dot(n: int = 1_000_000, repeats: int = 3, seed: int = 1) -> DotBenchResult:
    """
    Benchmark my_dot vs np.dot.
    Not for CI assertions; use it from a CLI script.
    """
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    b = rng.random(n)

    # warm-up
    _ = dot_np(a, b)

    loop_times = []
    np_times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = my_dot(a, b)
        t1 = time.perf_counter()
        loop_times.append((t1 - t0) * 1000)

        t0 = time.perf_counter()
        _ = dot_np(a, b)
        t1 = time.perf_counter()
        np_times.append((t1 - t0) * 1000)

    return DotBenchResult(
        n=n,
        repeats=repeats,
        loop_ms=float(np.mean(loop_times)),
        numpy_ms=float(np.mean(np_times)),
    )
