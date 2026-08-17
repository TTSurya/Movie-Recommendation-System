# Movie Recommendation with Singular Value Thresholding

This project implements Singular Value Thresholding (SVT) for matrix completion. The motivating problem is a movie recommender: estimate how a user would rate movies they have not rated yet.

## Problem

Let `M` be a user-movie ratings matrix. Row `i` represents a user, column `j` represents a movie, and `M[i, j]` is a rating when it is available. Most entries are missing because each user rates only a small fraction of the catalogue.

Given only the observed entries `Ω`, recover a complete matrix `X` that agrees with the known ratings:

```math
X_{ij} = M_{ij}, \quad (i,j) \in \Omega.
```

The recommendation for a user is then the unseen movie with the largest predicted value in that user's row of `X`.

## Why low rank?

Ratings are not independent. A small set of hidden preferences—such as genre, pacing, era, or style—often explains a large part of what users rate highly. In a latent-factor view,

```math
M \approx U V^T,
```

where `U` represents user preferences and `V` represents movie attributes. If only a few latent factors are needed, `M` is approximately low rank.

The direct formulation would be:

```math
\min_X \operatorname{rank}(X) \quad \text{subject to} \quad P_\Omega(X)=P_\Omega(M).
```

Rank minimization is computationally hard. The nuclear norm, `||X||_*` (the sum of singular values), is its standard convex relaxation. SVT solves the regularized version:

```math
\min_X \; \tau ||X||_* + \frac{1}{2}||X||_F^2
\quad \text{subject to} \quad P_\Omega(X)=P_\Omega(M).
```

At every iteration it soft-thresholds the singular values:

```math
X^k = D_\tau(Y^{k-1}), \qquad
Y^k = Y^{k-1} + \delta P_\Omega(M-X^k),
```

where `D_τ` replaces each singular value `σ` with `max(σ - τ, 0)`. This promotes a low-rank completion while repeatedly correcting errors on known ratings.

## Implementation

The solver is intentionally compact but retains the important large-scale ideas from the paper:

- keeps `Y` sparse, because only observed ratings are updated;
- stores `X` as a reduced SVD rather than a dense matrix;
- computes only singular values above the threshold using an adaptive partial SVD;
- skips initial iterations that would produce an all-zero estimate;
- uses the paper's synthetic setup: Gaussian low-rank factors, `τ = 5n`, and an oversampling ratio of 6.

`experiment.py` uses a synthetic matrix so the true answer is known. That makes it possible to measure reconstruction error directly. To apply the solver to MovieLens or another ratings dataset, create a user-by-movie matrix, mark known ratings in `mask`, and call `complete` from `svt.py`. For a real recommender evaluation, hold out some known ratings and report RMSE or MAE on that held-out set.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python test_svt.py
python experiment.py --size 1000 --rank 10 --oversampling-ratio 6
```

The experiment writes its metrics and convergence history to `artifacts/experiment.json`.

## Testing

Run:

```powershell
python test_svt.py
```

The test script checks two things:

1. **Operator check:** `D₃(diag(5, 2))` must equal `diag(2, 0)`. This verifies the core singular-value shrinkage step.
2. **Recovery check:** it creates a small rank-2 matrix, hides entries, runs SVT, and verifies both the observed-entry residual and reconstruction error are small. Because the test matrix is generated in code, its ground truth is available.

## Example result

For a `1000 × 1000` rank-10 synthetic matrix with oversampling ratio 6, this implementation converged in 121 iterations, recovered rank 10, and achieved a relative reconstruction error of approximately `1.70e-04`.

## Bibliography

J.-F. Cai, E. J. Candès, and Z. Shen. *A Singular Value Thresholding Algorithm for Matrix Completion*. SIAM Journal on Optimization, 20(4):1956-1982, 2010. https://doi.org/10.1137/080738970
