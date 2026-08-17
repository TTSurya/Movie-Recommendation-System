"""Singular Value Thresholding for matrix completion."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


@dataclass
class LowRank:
    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray

    @property
    def rank(self):
        return len(self.s)

    def values_at(self, rows, cols):
        if not self.rank:
            return np.zeros(len(rows))
        return np.sum((self.u[rows] * self.s) * self.vt[:, cols].T, axis=1)

    def dense(self):
        return (self.u * self.s) @ self.vt if self.rank else np.zeros((len(self.u), self.vt.shape[1]))


@dataclass
class Result:
    x: LowRank
    iterations: int
    residuals: list
    ranks: list
    delta: float


def empty(shape):
    return LowRank(np.empty((shape[0], 0)), np.empty(0), np.empty((0, shape[1])))


def shrink(y, tau, old_rank=0, step=5):
    """Soft-threshold singular values, computing only the useful ones."""
    n = min(y.shape)
    if n < 2 or (sparse.issparse(y) and y.nnz == 0) or (not sparse.issparse(y) and not np.any(y)):
        return empty(y.shape)

    k = min(max(1, old_rank + 1), n - 1)
    while True:
        u, s, vt = svds(y, k=k, which="LM")
        order = np.argsort(s)[::-1]
        u, s, vt = u[:, order], s[order], vt[order]
        if s[-1] <= tau:
            break
        if k == n - 1:
            a = y.toarray() if sparse.issparse(y) else y
            u, s, vt = np.linalg.svd(a, full_matrices=False)
            break
        k = min(n - 1, k + step)

    keep = s > tau
    return LowRank(u[:, keep], s[keep] - tau, vt[keep]) if np.any(keep) else empty(y.shape)


def complete(m, mask, tau, delta=None, eps=1e-4, iters=1000, svd_step=5):
    """Run the equality-constrained SVT iteration from the paper."""
    m, mask = np.asarray(m, float), np.asarray(mask, bool)
    if m.shape != mask.shape or not mask.any():
        raise ValueError("m and mask must have the same non-empty shape")
    if svd_step < 1:
        raise ValueError("svd_step must be positive")

    r, c = np.nonzero(mask)
    b = m[r, c]
    p = len(b) / m.size
    delta = 1.2 / p if delta is None else delta
    if delta <= 0:
        raise ValueError("delta must be positive")

    obs = sparse.csr_matrix((b, (r, c)), shape=m.shape)
    sigma = float(svds(obs, k=1, which="LM", return_singular_vectors=False)[0])
    skip = max(0, int(np.ceil(tau / (delta * sigma)) - 1))
    y = obs * (skip * delta)
    scale = np.linalg.norm(b)
    x, residuals, ranks = empty(m.shape), [], []

    for it in range(1, iters + 1):
        x = shrink(y, tau, x.rank, svd_step)
        err = b - x.values_at(r, c)
        residuals.append(float(np.linalg.norm(err) / scale))
        ranks.append(x.rank)
        if residuals[-1] <= eps:
            break
        y += sparse.csr_matrix((delta * err, (r, c)), shape=m.shape)
    return Result(x, it, residuals, ranks, delta)
