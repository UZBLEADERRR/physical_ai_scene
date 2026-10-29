"""Minimum-cost assignment (Hungarian / Kuhn–Munkres) without SciPy.

Same result as ``scipy.optimize.linear_sum_assignment`` for a rectangular cost matrix; the team
environment does not ship SciPy, and the matrices here are small (tracks × detections < 50).
"""

from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of the minimum-cost matching; every row or every column is used once."""
    C = np.asarray(cost, dtype=float)
    if C.size == 0:
        return np.empty(0, int), np.empty(0, int)
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape                         # n <= m
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)         # p[j] = row (1-based) matched to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows, cols = np.array(rows, int), np.array(cols, int)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]
