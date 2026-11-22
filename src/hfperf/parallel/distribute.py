"""Static distribution of the unique pair rows ij among P workers.

Row ij of the unique-quartet triangle holds ij + 1 quartets (all kl <= ij), so the
rows have very different costs: row 0 has 1 quartet, the last row npair of them.

    block     contiguous, equal NUMBER of rows   -> the last worker gets ~2x the mean work
    cyclic    round robin  r = rank, rank+P, ... -> balanced to within one row
    weighted  contiguous, equal estimated COST (sum of ij+1)   -> balanced, and each
              worker's rows are contiguous (better locality of the pair data)
"""
from __future__ import annotations

import numpy as np

SCHEMES = ("block", "cyclic", "weighted")


def row_costs(npair: int) -> np.ndarray:
    """Estimated cost of each row = number of quartets in it (ignores screening and l)."""
    return np.arange(1, npair + 1, dtype=np.int64)


def distribute(npair: int, nworkers: int, rank: int, scheme: str = "cyclic") -> np.ndarray:
    """Pair-row indices (int64, ascending) owned by `rank` out of `nworkers`."""
    if not 0 <= rank < nworkers:
        raise ValueError("need 0 <= rank < nworkers")
    if scheme == "cyclic":
        return np.arange(rank, npair, nworkers, dtype=np.int64)
    if scheme == "block":
        edges = np.linspace(0, npair, nworkers + 1).round().astype(np.int64)
    elif scheme == "weighted":
        cum = np.concatenate([[0], np.cumsum(row_costs(npair))])
        targets = np.linspace(0, cum[-1], nworkers + 1)
        edges = np.searchsorted(cum, targets).astype(np.int64)
        edges[0], edges[-1] = 0, npair
    else:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    return np.arange(edges[rank], edges[rank + 1], dtype=np.int64)


def imbalance(npair: int, nworkers: int, scheme: str) -> float:
    """max/mean of the estimated cost per worker (1.0 = perfect balance)."""
    c = row_costs(npair)
    loads = np.array([c[distribute(npair, nworkers, r, scheme)].sum() for r in range(nworkers)], float)
    return float(loads.max() / loads.mean())
