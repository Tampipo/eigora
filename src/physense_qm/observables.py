# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Quantum observables computed from a wavefunction.
Atomic units: hbar = 1.
"""

import numpy as np
from numpy.typing import NDArray

from physense_utils.grids import GridND
from physense_utils.fft import fft1d, fft_frequencies


def expectation_x(psi: NDArray[np.complex128], grid: GridND) -> float:
    """<x> = integral x |psi|^2 dx"""
    return float(np.trapezoid(grid.x * np.abs(psi) ** 2, grid.x))


def expectation_x2(psi: NDArray[np.complex128], grid: GridND) -> float:
    """<x^2> = integral x^2 |psi|^2 dx"""
    return float(np.trapezoid(grid.x**2 * np.abs(psi) ** 2, grid.x))


def uncertainty_x(psi: NDArray[np.complex128], grid: GridND) -> float:
    """Delta x = sqrt(<x^2> - <x>^2)"""
    x2 = expectation_x2(psi, grid)
    x = expectation_x(psi, grid)
    return float(np.sqrt(max(x2 - x**2, 0.0)))


def expectation_p(psi: NDArray[np.complex128], grid: GridND) -> float:
    """
    <p> = integral k |psi_k|^2 dk / (2*pi)
    Computed in momentum space.
    """
    k = fft_frequencies(grid.n_points, grid.dx)
    psi_k = fft1d(psi, grid.dx)
    dk = k[1] - k[0]
    return float(np.trapezoid(k * np.abs(psi_k) ** 2, k) / (2 * np.pi))


def expectation_p2(psi: NDArray[np.complex128], grid: GridND) -> float:
    """<p^2> computed in momentum space."""
    k = fft_frequencies(grid.n_points, grid.dx)
    psi_k = fft1d(psi, grid.dx)
    return float(np.trapezoid(k**2 * np.abs(psi_k) ** 2, k) / (2 * np.pi))


def uncertainty_p(psi: NDArray[np.complex128], grid: GridND) -> float:
    """Delta p = sqrt(<p^2> - <p>^2)"""
    p2 = expectation_p2(psi, grid)
    p = expectation_p(psi, grid)
    return float(np.sqrt(max(p2 - p**2, 0.0)))


def heisenberg_product(psi: NDArray[np.complex128], grid: GridND) -> float:
    """
    Delta x * Delta p — should be >= 0.5 (hbar=1).
    Useful to verify wavepacket preparation.
    """
    return uncertainty_x(psi, grid) * uncertainty_p(psi, grid)


def norm(psi: NDArray[np.complex128], grid: GridND) -> float:
    """integral |psi|^2 dx — should remain 1 during evolution."""
    return float(np.trapezoid(np.abs(psi) ** 2, grid.x))


__all__ = [
    "expectation_x",
    "expectation_x2",
    "uncertainty_x",
    "expectation_p",
    "expectation_p2",
    "uncertainty_p",
    "heisenberg_product",
    "norm",
]
