# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Eigenstate solver for 1D quantum systems.
Uses finite difference discretisation of the Hamiltonian (atomic units).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from physense_utils.grids import GridND

from physense_qm.potentials import Potential


@dataclass(frozen=True)
class EigenSolution:
    """
    Result of an eigenstate calculation.

    Attributes
    ----------
    energies : NDArray of shape (n_states,)
        Eigenvalues in ascending order (atomic units).
    wavefunctions : NDArray of shape (n_states, n_points)
        Normalised eigenfunctions. wavefunctions[i] corresponds to energies[i].
    grid : GridND
        The spatial grid used for the calculation.
    potential : NDArray of shape (n_points,)
        The potential evaluated on the grid.
    """

    energies: NDArray[np.float64]
    wavefunctions: NDArray[np.float64]
    grid: GridND
    potential: NDArray[np.float64]

    @property
    def n_states(self) -> int:
        return len(self.energies)

    @property
    def ground_state(self) -> NDArray[np.float64]:
        return self.wavefunctions[0]

    @property
    def ground_energy(self) -> float:
        return float(self.energies[0])

    def wavefunction(self, n: int) -> NDArray[np.float64]:
        """Return the n-th eigenfunction (0-indexed)."""
        if n < 0 or n >= self.n_states:
            raise IndexError(f"State index {n} out of range [0, {self.n_states - 1}]")
        return self.wavefunctions[n]

    def probability_density(self, n: int) -> NDArray[np.float64]:
        """Return |psi_n(x)|^2."""
        return self.wavefunction(n) ** 2


def solve_eigenstates(
    grid: GridND,
    potential: Potential,
    n_states: int = 10,
) -> EigenSolution:
    """
    Solve the time-independent Schrödinger equation on a 1D grid.

    Uses finite differences to discretise H = -0.5 * d²/dx² + V(x)
    (atomic units: hbar = m = 1), then finds the n_states lowest eigenpairs
    via scipy sparse eigensolver.

    Parameters
    ----------
    grid : GridND
        Spatial grid.
    potential : Potential
        Potential function V(x).
    n_states : int
        Number of eigenstates to compute.

    Returns
    -------
    EigenSolution
    """
    if n_states < 1:
        raise ValueError(f"n_states must be at least 1, got {n_states}")
    if n_states >= grid.n_points - 1:
        raise ValueError(
            f"n_states ({n_states}) must be less than n_points - 1 ({grid.n_points - 1})"
        )

    x = grid.x
    dx = grid.dx
    n = grid.n_points
    V = potential(x)

    # Kinetic energy: -0.5 * d²/dx² via finite differences
    diag = 1.0 / (dx**2) + V
    off = -0.5 / (dx**2) * np.ones(n - 1)

    H = sp.diags([off, diag, off], offsets=[-1, 0, 1], format="csr")

    energies, vecs = spla.eigsh(H, k=n_states, which="LM", sigma=V.min(), maxiter=10000)

    # Sort by energy
    idx = np.argsort(energies)
    energies = energies[idx]
    vecs = vecs[:, idx]

    # Normalise: integral |psi|^2 dx = 1
    wavefunctions = []
    for i in range(n_states):
        psi = vecs[:, i]
        norm = np.sqrt(np.trapezoid(psi**2, x))
        psi = psi / norm
        # Fix phase: largest component positive
        if psi[np.argmax(np.abs(psi))] < 0:
            psi = -psi
        wavefunctions.append(psi)

    return EigenSolution(
        energies=energies,
        wavefunctions=np.array(wavefunctions),
        grid=grid,
        potential=V,
    )


__all__ = ["EigenSolution", "solve_eigenstates"]
