# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Error bars for correlated samples.

A Markov chain does not give independent samples, so the naive standard error
`sigma / sqrt(n)` understates the true uncertainty -- often by a large factor
and always in the optimistic direction. Two standard remedies, both here:

  * **blocking**: average consecutive samples into blocks, and grow the block
    until the block means stop being correlated. The error estimate rises with
    block size and then plateaus; the plateau is the honest answer.
  * **integrated autocorrelation time** `tau`: the number of steps between
    effectively independent samples, so the effective sample size is `n / 2 tau`.

Both are estimates from one finite chain, and neither is reliable if the run
is short compared with `tau` -- which is exactly when a naive error bar looks
best.
"""

import math

import numpy as np
from numpy.typing import NDArray


def blocked_error(values: NDArray[np.float64], min_blocks: int = 8) -> float:
    """
    Standard error of the mean, from the blocking plateau.

    Parameters
    ----------
    values : array of float
        The time series, in order. Order matters -- that is the point.
    min_blocks : int
        Stop when fewer than this many blocks remain; block means from a
        handful of blocks have a huge variance of their own.

    Returns
    -------
    float
        The largest error estimate over all block sizes, which is the plateau
        for a chain long enough to have one.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.size < 2:
        return math.nan
    worst = 0.0
    current = series
    while current.size >= min_blocks:
        error = float(np.std(current, ddof=1) / math.sqrt(current.size))
        worst = max(worst, error)
        pairs = current.size // 2
        current = 0.5 * (current[: 2 * pairs : 2] + current[1 : 2 * pairs : 2])
    return worst


def autocorrelation_time(values: NDArray[np.float64], cutoff: float = 0.05) -> float:
    """
    Integrated autocorrelation time, summed until the correlation dies.

    `tau = 1/2 + sum_k rho(k)`, truncated where `rho` first falls below
    `cutoff` -- the sum has to be truncated because the tail of `rho` is pure
    noise for a finite chain, and including it makes `tau` a random walk.

    Returns
    -------
    float
        Steps between effectively independent samples. `1/2` for white noise.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.size < 2:
        return math.nan
    centred = series - series.mean()
    variance = float(np.dot(centred, centred)) / series.size
    if variance <= 0.0:
        return 0.5
    total = 0.5
    for lag in range(1, series.size // 2):
        rho = float(np.dot(centred[:-lag], centred[lag:])) / (
            (series.size - lag) * variance
        )
        if rho < cutoff:
            break
        total += rho
    return total


__all__ = ["autocorrelation_time", "blocked_error"]
