# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lloyd-Max optimal scalar quantizer for TurboQuant.

After rotating a d-dimensional unit vector by a random orthogonal matrix,
each coordinate approximately follows N(0, 1/d) for d >= 64.
We solve the Lloyd-Max conditions to find optimal centroids.

Based on: turboquant-pytorch/lloyd_max.py (Zandieh et al.)
"""

import json
import math
import os
import tempfile
from functools import lru_cache

import torch


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _trapz(f, a: float, b: float, n: int = 200) -> float:
    """Trapezoidal numerical integration (replaces scipy.integrate.quad)."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution.

    Args:
        d: Vector dimension (determines variance = 1/d).
        bits: Number of quantization bits.
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        centroids: Sorted tensor of 2^bits optimal centroids.
        boundaries: Sorted tensor of 2^bits - 1 decision boundaries.
    """
    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


_CENTROIDS_DISK_CACHE: dict[tuple[int, int], tuple[float, ...]] | None = None


def _centroids_cache_path() -> str:
    return os.environ.get(
        "VLLM_TURBOQUANT_CENTROIDS_CACHE",
        os.path.expanduser("~/.cache/vllm/turboquant_centroids.json"),
    )


def _encode_centroids_key(key: tuple[int, int]) -> str:
    return f"{key[0]}:{key[1]}"


def _decode_centroids_key(key: str) -> tuple[int, int] | None:
    try:
        d_str, bits_str = key.split(":", 1)
        return int(d_str), int(bits_str)
    except Exception:
        return None


def _load_centroids_disk_cache() -> dict[tuple[int, int], tuple[float, ...]]:
    global _CENTROIDS_DISK_CACHE
    if _CENTROIDS_DISK_CACHE is not None:
        return _CENTROIDS_DISK_CACHE
    path = _centroids_cache_path()
    try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                raw_cache = json.load(f)
            cache: dict[tuple[int, int], tuple[float, ...]] = {}
            if isinstance(raw_cache, dict):
                for raw_key, raw_values in raw_cache.items():
                    key = _decode_centroids_key(str(raw_key))
                    if key is not None and isinstance(raw_values, list):
                        cache[key] = tuple(float(x) for x in raw_values)
            _CENTROIDS_DISK_CACHE = cache
        else:
            _CENTROIDS_DISK_CACHE = {}
    except Exception:
        _CENTROIDS_DISK_CACHE = {}
    return _CENTROIDS_DISK_CACHE


def _save_centroids_disk_cache() -> None:
    if _CENTROIDS_DISK_CACHE is None:
        return
    path = _centroids_cache_path()
    try:
        cache_dir = os.path.dirname(path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        serializable_cache = {
            _encode_centroids_key(key): list(values)
            for key, values in _CENTROIDS_DISK_CACHE.items()
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=cache_dir or None, delete=False
        ) as tmp_file:
            json.dump(serializable_cache, tmp_file)
            tmp_name = tmp_file.name
        os.replace(tmp_name, path)
    except Exception:
        pass


@lru_cache(maxsize=32)
def get_centroids(d: int, bits: int) -> torch.Tensor:
    """Get precomputed Lloyd-Max centroids (memory and disk cached)."""
    key = (int(d), int(bits))
    disk_cache = _load_centroids_disk_cache()
    if key in disk_cache:
        try:
            return torch.tensor(disk_cache[key], dtype=torch.float32)
        except Exception:
            pass

    centroids, _ = solve_lloyd_max(d, bits)
    try:
        disk_cache[key] = tuple(float(x) for x in centroids.cpu().tolist())
        _save_centroids_disk_cache()
    except Exception:
        pass
    return centroids
