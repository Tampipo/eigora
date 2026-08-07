# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
A `Spectrum` backed by the finite-difference eigensolver.

This is what a block gets when its potential has no analytic solution. It
exposes the same interface as the exact spectra, so a separable system can mix
the two freely -- `is_exact` then reports False for the whole system.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from physense_utils.grids import GridND

from physense_qm.potentials.base import PotentialND
from physense_qm.solvers.eigensolver import EigenSolution, solve_eigenstates
from physense_qm.spectra.base import Label, Spectrum


class NumericalSpectrum(Spectrum):
    """
    Eigenstates computed on a grid, labelled (n,) with n = 0, 1, ... .

    Wavefunctions are linearly interpolated off the grid points and vanish
    outside the grid, so they can be multiplied with other blocks' states at
    arbitrary coordinates.
    """

    def __init__(self, solution: EigenSolution) -> None:
        self.solution = solution

    @classmethod
    def solve(
        cls,
        potential: "PotentialND | Callable[..., NDArray[np.float64]]",
        grid: GridND,
        n_states: int = 20,
    ) -> "NumericalSpectrum":
        """Solve a 1D potential on `grid` and wrap the result."""
        if grid.ndim != 1:
            raise ValueError(f"the numerical solver is 1D, got a {grid.ndim}D grid")
        return cls(solve_eigenstates(grid, potential, n_states=n_states))

    @property
    def ndim(self) -> int:
        return 1

    @property
    def quantum_numbers(self) -> tuple[str, ...]:
        return ("n",)

    @property
    def is_exact(self) -> bool:
        return False

    @property
    def n_available(self) -> int:
        return self.solution.n_states

    def energy(self, label: Label) -> float:
        return float(self.solution.energies[self._index(label)])

    def wavefunction(self, label: Label) -> Callable[..., NDArray[np.float64]]:
        psi_grid = self.solution.wavefunctions[self._index(label)]
        x_grid = self.solution.grid.x

        def psi(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.interp(np.asarray(x, dtype=np.float64), x_grid, psi_grid,
                             left=0.0, right=0.0)

        return psi

    def states(self, n: int) -> list[Label]:
        if n < 1:
            raise ValueError(f"number of states must be at least 1, got {n}")
        return [(i,) for i in range(min(n, self.n_available))]

    def _index(self, label: Label) -> int:
        self._check_arity(label)
        n = label[0]
        if n < 0 or n >= self.n_available:
            raise IndexError(
                f"state {n} out of range; {self.n_available} states were computed"
            )
        return n


__all__ = ["NumericalSpectrum"]
