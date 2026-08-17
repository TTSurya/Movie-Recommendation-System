import argparse
import json
import time
from pathlib import Path

import numpy as np

from svt import complete


def run(n=100, rank=5, oversampling=6, seed=42, max_iter=1000, delta=None):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, n))
    dof = rank * (2 * n - rank)
    count = min(round(oversampling * dof), n * n)
    mask = np.zeros((n, n), bool)
    mask.flat[rng.choice(n * n, count, replace=False)] = True

    start = time.perf_counter()
    out = complete(m * mask, mask, tau=5 * n, delta=delta, iters=max_iter)
    elapsed = time.perf_counter() - start
    error = np.linalg.norm(out.x.dense() - m) / np.linalg.norm(m)
    return {
        "matrix_size": n, "true_rank": rank, "degrees_of_freedom": dof,
        "oversampling_ratio": oversampling, "observed_entries": int(mask.sum()),
        "observed_fraction": float(mask.mean()), "iterations": out.iterations,
        "final_observed_residual": out.residuals[-1],
        "full_matrix_relative_error": float(error), "final_estimated_rank": out.x.rank,
        "step_size": out.delta, "threshold": 5 * n, "elapsed_seconds": elapsed,
        "rank_history": out.ranks, "residual_history": out.residuals,
    }


def main():
    p = argparse.ArgumentParser(description="SVT matrix completion experiment")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--oversampling-ratio", type=float, default=6)
    p.add_argument("--delta", type=float)
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("artifacts/experiment.json"))
    a = p.parse_args()
    if a.size < 2 or a.rank < 1 or a.oversampling_ratio <= 0:
        p.error("size and rank must be positive; size must be at least 2")
    result = run(a.size, a.rank, a.oversampling_ratio, a.seed, a.max_iterations, a.delta)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("history")}, indent=2))


if __name__ == "__main__":
    main()
