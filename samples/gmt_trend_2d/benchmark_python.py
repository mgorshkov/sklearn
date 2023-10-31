#!/usr/bin/env python3
"""
Python benchmark helper for GMT_trend2d cross-language comparison.

Called by benchmark.cpp via subprocess. Receives parameters via command-line
arguments and prints timing results to stdout for the C++ program to parse.
"""

import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.special as sc


def GMT_trend2d(data, rank):
    """Pure Python reimplementation of GMT_trend2d (identical to test_time.py)."""
    MAD_NORMALIZE = 1.4826
    sig_threshold = 0.51

    if rank not in [1, 2, 3]:
        raise Exception('Number of model parameters "rank" should be 1, 2, or 3')

    def gmtstat_f_q(chisq1, nu1, chisq2, nu2):
        if chisq1 == 0.0:
            return 1
        if chisq2 == 0.0:
            return 0
        return sc.betainc(0.5 * nu2, 0.5 * nu1, chisq2 / (chisq2 + chisq1))

    if rank in [2, 3]:
        x = data[:, 0]
        x = np.interp(x, (x.min(), x.max()), (-1, +1))
    if rank == 3:
        y = data[:, 1]
        y = np.interp(y, (y.min(), y.max()), (-1, +1))
    z = data[:, 2]
    w = np.ones(z.shape)

    if rank == 1:
        xy = np.expand_dims(np.zeros(z.shape), 1)
    elif rank == 2:
        xy = np.expand_dims(x, 1)
    elif rank == 3:
        xy = np.stack([x, y]).transpose()

    mlr = LinearRegression()
    chisqs = []
    coeffs = []
    while True:
        mlr.fit(xy, z, sample_weight=w)
        r = np.abs(z - mlr.predict(xy))
        chisq = np.sum((r ** 2 * w)) / (z.size - 3)
        chisqs.append(chisq)
        k = 1.5 * MAD_NORMALIZE * np.median(r)
        w = np.where(r <= k, 1, (2 * k / r) - (k * k / (r ** 2)))
        sig = 1 if len(chisqs) == 1 else gmtstat_f_q(chisqs[-1], z.size - 3, chisqs[-2], z.size - 3)
        if len(chisqs) == 1 or chisqs[-2] > chisqs[-1]:
            coeffs = [mlr.intercept_, *mlr.coef_]
        if sig < sig_threshold:
            break
    return coeffs[:rank]


def generate_data(rank, num_points, noise_level):
    """Generate synthetic test data (identical to test_time.py)."""
    np.random.seed(42)
    x = np.linspace(-10, 10, num_points)
    y = np.linspace(-10, 10, num_points)
    if rank == 1:
        z = 3 * x + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    elif rank == 2:
        z = 2 * x + 3 * y + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    elif rank == 3:
        z = 2 * x ** 2 + 3 * y ** 2 + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    return data


def main():
    if len(sys.argv) != 5:
        print("Usage: benchmark_python.py <rank> <num_points> <noise_level> <n_runs>", file=sys.stderr)
        sys.exit(1)

    rank = int(sys.argv[1])
    num_points = int(sys.argv[2])
    noise_level = int(sys.argv[3])
    n_runs = int(sys.argv[4])

    data = generate_data(rank, num_points, noise_level)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Warm-up run
        _ = GMT_trend2d(data, rank)

        # Timed runs
        start = time.perf_counter()
        for _ in range(n_runs):
            GMT_trend2d(data, rank)
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000.0
    # Print result for C++ to parse: "PYTHON_TIME <avg_ms>"
    print(f"PYTHON_TIME {avg_ms:.6f}")


if __name__ == "__main__":
    main()
