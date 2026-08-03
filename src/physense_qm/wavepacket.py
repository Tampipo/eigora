# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Initial quantum states for time evolution simulations.
All states are normalised in L2.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physense_utils.grids import GridND


class InitialState(ABC):
    """Abstract base class for initial quantum states."""

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Return the complex wavefunction psi(x) on the grid."""

    def normalised(self, grid: GridND) -> NDArray[np.complex128]:
        """Return psi(x) normalised so that integral |psi|^2 dx = 1."""
        psi = self(grid.x)
        norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, grid.x))
        return psi / norm


@dataclass
class GaussianWavepacket(InitialState):
    """
    psi(x) = A * exp(-(x - x0)^2 / (4 * sigma^2)) * exp(i * k0 * x)

    A Gaussian envelope modulated by a plane wave of momentum k0.

    Parameters
    ----------
    x0 : float
        Centre of the wavepacket.
    k0 : float
        Central momentum (group velocity = k0 in atomic units).
    sigma : float
        Spatial width of the wavepacket.
    """

    x0: float = 0.0
    k0: float = 1.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        envelope = np.exp(-((x - self.x0) ** 2) / (4 * self.sigma**2))
        plane_wave = np.exp(1j * self.k0 * x)
        return envelope * plane_wave


__all__ = ["InitialState", "GaussianWavepacket"]
