import numpy as np

from experiment import run
from svt import shrink


def test_shrink():
    x = shrink(np.diag([5.0, 2.0]), 3)
    assert x.rank == 1
    assert np.allclose(x.dense(), np.diag([2.0, 0.0]))


def test_completion():
    out = run(n=30, rank=2, oversampling=8, seed=7, max_iter=400)
    assert out["final_observed_residual"] < 1e-4
    assert out["full_matrix_relative_error"] < 0.1


if __name__ == "__main__":
    test_shrink()
    test_completion()
    print("All checks passed.")
