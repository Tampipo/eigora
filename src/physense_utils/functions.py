# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Common mathematical functions used across simulation modules.
"""

import numpy as np
from numpy.typing import NDArray


def gaussian(
    x: NDArray[np.float64],
    x0: float = 0.0,
    sigma: float = 1.0,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """
    Normalised Gaussian function.

    Parameters
    ----------
    x : array
        Evaluation points.
    x0 : float
        Centre of the Gaussian.
    sigma : float
        Standard deviation (width).
    amplitude : float
        Peak amplitude. Default 1.0 gives a unit-amplitude Gaussian.
        Use amplitude=1/(sigma*sqrt(2*pi)) for L2-normalised Gaussian.

    Returns
    -------
    array of same shape as x.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return amplitude * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def sinc(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Unnormalised sinc function: sin(x) / x, with sinc(0) = 1.

    Note: numpy.sinc uses the normalised convention sin(πx)/(πx).
    This function uses the physics convention sin(x)/x.

    Parameters
    ----------
    x : array
        Evaluation points.

    Returns
    -------
    array of same shape as x.
    """
    return np.sinc(x / np.pi)


def heaviside(
    x: NDArray[np.float64],
    x0: float = 0.0,
    value_at_zero: float = 0.5,
) -> NDArray[np.float64]:
    """
    Heaviside step function centred at x0.

    Parameters
    ----------
    x : array
        Evaluation points.
    x0 : float
        Location of the step.
    value_at_zero : float
        Value at x == x0. Default 0.5 (symmetric convention).

    Returns
    -------
    array of same shape as x.
    """
    return np.heaviside(x - x0, value_at_zero)

__all__ = ["gaussian", "sinc", "heaviside"]