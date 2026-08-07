# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Time evolution of states on a discrete Hilbert space.
Atomic units: hbar = 1.

Unlike the continuous side, which Trotter-splits exp(-iHt) and has to march
in steps of dt, a finite-dimensional H can simply be diagonalised. Writing
psi(0) in the energy eigenbasis,

    |psi(t)> = sum_n <E_n|psi(0)> exp(-i E_n t) |E_n>

every requested time is computed directly from psi(0). There is no time step,
no stability criterion, and no error accumulating from frame to frame -- the
times may be unevenly spaced, out of order, or negative.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eigora.qm.discrete.operations import Matrix
from eigora.qm.discrete.operators import Hamiltonian, Observable


@dataclass(frozen=True)
class Evolution:
    """
    Result of a time evolution.

    Attributes
    ----------
    psi : NDArray of shape (n_times, dim), complex
        State vector at each requested time, in the computational basis.
    times : NDArray of shape (n_times,)
        The times the state was evaluated at.
    hamiltonian : Hamiltonian
        The Hamiltonian that generated the evolution.
    """

    psi: Matrix
    times: NDArray[np.float64]
    hamiltonian: Hamiltonian

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def dim(self) -> int:
        """Dimension of the Hilbert space."""
        return self.psi.shape[1]

    def state(self, n: int) -> Matrix:
        """Return the state at the n-th requested time (0-indexed)."""
        if n < 0 or n >= self.n_times:
            raise IndexError(f"Time index {n} out of range [0, {self.n_times - 1}]")
        return self.psi[n]

    def probabilities(self, n: int) -> NDArray[np.float64]:
        """
        |psi_i(t_n)|^2 -- the probability of each basis state.

        The discrete counterpart of `Evolution.probability_density` on the
        continuous side: a distribution over basis states, not a density.
        """
        return np.abs(self.state(n)) ** 2

    def norm(self, n: int) -> float:
        """<psi|psi> at the n-th time -- should stay 1."""
        return float(np.sum(self.probabilities(n)))

    def expectation(self, observable: Observable) -> NDArray[np.float64]:
        """
        <A>(t) at every requested time.

        Parameters
        ----------
        observable : Observable
            The observable to track, acting on the same space.

        Returns
        -------
        NDArray of shape (n_times,)
            Real, because `observable` is Hermitian.
        """
        if observable.dim != self.dim:
            raise ValueError(
                f"observable is {observable.dim}D but the states are {self.dim}D"
            )
        # (A psi_t)_i = sum_j A_ij psi_t[j], for states stored one per row.
        applied = self.psi @ observable.matrix.T
        return np.real(np.sum(self.psi.conj() * applied, axis=1))


def evolve(
    hamiltonian: Hamiltonian,
    psi0: Matrix,
    times: ArrayLike,
) -> Evolution:
    """
    Evolve a state under a time-independent Hamiltonian.

    Expands `psi0` in the energy eigenbasis once and applies the phase
    exp(-i E_n t) at each requested time, so the cost is one diagonalisation
    (cached on `hamiltonian`) plus a single matrix product for all times.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Generator of the evolution.
    psi0 : Matrix of shape (dim,)
        Initial state. Normalised on the way in.
    times : array-like of shape (n_times,)
        Times to evaluate at (atomic units). Need not be evenly spaced,
        ascending, or positive.

    Returns
    -------
    Evolution
    """
    psi0 = np.asarray(psi0, dtype=complex)
    if psi0.shape != (hamiltonian.dim,):
        raise ValueError(
            f"psi0 must have shape ({hamiltonian.dim},), got {psi0.shape}"
        )
    norm = np.linalg.norm(psi0)
    if norm == 0:
        raise ValueError("psi0 must be a non-zero vector")
    psi0 = psi0 / norm

    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"times must be one-dimensional, got shape {times.shape}")
    if times.size == 0:
        raise ValueError("times must contain at least one time")

    eigensystem = hamiltonian.eigensystem
    vectors = eigensystem.eigenvectors  # one eigenvector per row

    coefficients = vectors.conj() @ psi0                             # <E_n|psi0>
    phases = np.exp(-1j * np.outer(times, eigensystem.eigenvalues))  # (n_times, dim)
    psi = (coefficients * phases) @ vectors                          # (n_times, dim)

    return Evolution(psi=psi, times=times, hamiltonian=hamiltonian)


__all__ = ["Evolution", "evolve"]
