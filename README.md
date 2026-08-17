# Movie Recommendation System

This project implements **Singular Value Thresholding (SVT)** for matrix completion, using movie recommendation as the motivating application.

## Problem

Let $M$ be a user-movie ratings matrix:

- Each row represents a user.
- Each column represents a movie.
- $M[i, j]$ is a rating when it is available.

Most entries are missing because each user rates only a small fraction of the movie catalogue.

Given only the observed entries $\Omega$, the goal is to recover a complete matrix $X$ that agrees with the known ratings: $X_{ij} = M_{ij}, \qquad (i,j) \in \Omega.$

The recommendation for a user is then the unseen movie with the largest predicted rating in that user's row of $X$.

## Why Low Rank?

Ratings are not independent. A small set of hidden preferences, such as genre, pacing, era, or style, can explain a large part of what users rate highly.

In a latent-factor view, $M \approx UV^T.$ Here, $U$ represents user preferences and $V$ represents movie attributes. If only a few latent factors are needed, $M$ is approximately low rank.

The direct formulation is: $\min_X \mathrm{rank}(X) \quad \text{subject to} \quad P_\Omega(X) = P_\Omega(M).$

Rank minimization is computationally hard. The nuclear norm, denoted by $\Vert{}X\Vert{}_*$ and equal to the sum of the singular values of $X$, provides a standard convex relaxation.

SVT solves the regularized problem: 
$$
\min_X \tau \Vert{}X\Vert{}_* + \frac{1}{2} \Vert{}X\Vert{}_F^2 \quad \text{subject to} \quad P_\Omega(X) = P_\Omega(M)
$$

## Singular Value Thresholding

At each iteration, SVT soft-thresholds the singular values.

Given $Y^{k-1} = U\Sigma V^T,$ the singular values are transformed according to $\sigma_i' = \max(\sigma_i - \tau, 0).$

The resulting matrix is $X^k = D_\tau(Y^{k-1}),$ followed by an update on the observed entries: $Y^k = Y^{k-1} + \delta P_\Omega(M - X^k).$

Thus, small singular values are removed while the observed ratings are repeatedly used to correct the estimate.

## Implementation

The solver is intentionally compact while retaining the important large-scale ideas from the SVT approach:

- `Y` is kept sparse because only observed ratings are updated.
- `X` is stored as a reduced SVD rather than a dense matrix.
- Only singular values above the threshold are computed using an adaptive partial SVD.
- Initial iterations that would produce an all-zero estimate are skipped.
- The experiment uses the paper's synthetic setup: Gaussian low-rank factors, $\tau = 5n$, and an oversampling ratio of 6.

`experiment.py` uses a synthetic matrix so the true answer is known. This makes it possible to measure reconstruction error directly.

To apply the solver to MovieLens or another ratings dataset, create a user-by-movie matrix, mark known ratings in `mask`, and call `complete` from `svt.py`. For a real recommender evaluation, hold out some known ratings and report RMSE or MAE on the held-out entries.

## Run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

Install dependencies:

```powershell
pip install -r requirements.txt

```

Run the tests:

```powershell
python test_svt.py

```

Run the experiment:

```powershell
python experiment.py --size 1000 --rank 10 --oversampling-ratio 6

```

The experiment writes its metrics and convergence history to:

```text
artifacts/experiment.json

```

## Testing

Run:

```powershell
python test_svt.py

```

The test script checks:

1. **Operator check:** $D_3(\mathrm{diag}(5, 2))$ must equal $\mathrm{diag}(2, 0)$, verifying the core singular-value shrinkage step.
2. **Recovery check:** A small rank-2 matrix is generated, entries are hidden, and SVT is run. The test verifies that both the observed-entry residual and reconstruction error are small.

Because the test matrix is generated in code, its ground truth is available.

## Example Result

For a $1000 \times 1000$ rank-10 synthetic matrix with an oversampling ratio of 6, the implementation converged in 121 iterations, recovered rank 10, and achieved a relative reconstruction error of approximately $1.70 \times 10^{-4}$.

## Bibliography

J.-F. Cai, E. J. Candès, and Z. Shen. *A Singular Value Thresholding Algorithm for Matrix Completion*. SIAM Journal on Optimization, 20(4):1956–1982, 2010.

DOI: https://doi.org/10.1137/080738970
