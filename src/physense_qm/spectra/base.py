# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The `Spectrum` interface: the eigenstates of a system, however they were found.

A state is labelled by a tuple of quantum numbers, and a spectrum answers three
questions about it -- its energy, its wavefunction, and where it sits in the
ascending list of states. Everything else (energies, energy levels,
degeneracies, sampling on a grid) is derived from those here, so an analytic
solution and a numerical one expose exactly the same surface and can be
composed with each other.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physense_utils.grids import GridND

# Quantum numbers identifying one state, e.g. (2,) or (n, l, m) or (n_1, n_2).
Label = tuple[int, ...]

# Enumeration never grows past this many states while searching for levels.
_MAX_SEARCH = 4096


@dataclass(frozen=True)
class EnergyLevel:
    """A group of states sharing an energy."""

    energy: float
    states: tuple[Label, ...]

    @property
    def degeneracy(self) -> int:
        """Number of states at this energy."""
        return len(self.states)


class Spectrum(ABC):
    """
    Abstract base class for the eigenstates of a quantum system.

    Subclasses provide `ndim`, `quantum_numbers`, `is_exact`, `n_available`,
    and implement `energy`, `wavefunction` and `states`.
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of spatial coordinates the wavefunctions take."""

    @property
    @abstractmethod
    def quantum_numbers(self) -> tuple[str, ...]:
        """Names of the quantum numbers making up a label."""

    @property
    @abstractmethod
    def is_exact(self) -> bool:
        """True if every energy and wavefunction is analytic."""

    @property
    @abstractmethod
    def n_available(self) -> int | None:
        """How many states this spectrum can produce; None if unbounded."""

    @abstractmethod
    def energy(self, label: Label) -> float:
        """Energy of the state with these quantum numbers (atomic units)."""

    @abstractmethod
    def wavefunction(self, label: Label) -> Callable[..., NDArray[np.float64]]:
        """
        The normalised wavefunction of a state, as a callable of `ndim`
        coordinate arrays.
        """

    @abstractmethod
    def states(self, n: int) -> list[Label]:
        """
        The `n` lowest states, ascending in energy.

        Returns fewer than `n` labels if the spectrum is exhausted.
        """

    # -- derived ----------------------------------------------------------

    @property
    def ground_state(self) -> Label:
        """Label of the lowest-energy state."""
        return self.states(1)[0]

    @property
    def ground_energy(self) -> float:
        """Energy of the lowest state."""
        return self.energy(self.ground_state)

    def energies(self, n: int) -> NDArray[np.float64]:
        """Energies of the `n` lowest states, ascending."""
        return np.array([self.energy(label) for label in self.states(n)])

    def levels(self, n_levels: int = 5, tol: float = 1e-9) -> list[EnergyLevel]:
        """
        The `n_levels` lowest distinct energies, each with its degenerate states.

        Parameters
        ----------
        n_levels : int
            Number of distinct energies to return.
        tol : float
            Relative/absolute tolerance for calling two energies equal. States
            closer than this are treated as degenerate.

        Returns
        -------
        list of EnergyLevel
        """
        if n_levels < 1:
            raise ValueError(f"n_levels must be at least 1, got {n_levels}")

        n = max(4 * n_levels, 8)
        while True:
            labels = self.states(n)
            groups = self._group_by_energy(labels, tol)
            exhausted = len(labels) < n or n >= _MAX_SEARCH
            # The last group may still be missing partners beyond `n` states.
            complete = groups if exhausted else groups[:-1]
            if len(complete) >= n_levels or exhausted:
                return complete[:n_levels]
            n *= 2

    def degeneracy(self, label: Label, tol: float = 1e-9) -> int:
        """
        Number of states sharing the energy of `label`.

        Parameters
        ----------
        label : tuple of int
            Quantum numbers of the state.
        tol : float
            Tolerance for calling two energies equal.
        """
        energy = self.energy(label)
        n = 16
        while True:
            labels = self.states(n)
            matches = [
                other for other in labels if _close(self.energy(other), energy, tol)
            ]
            exhausted = len(labels) < n or n >= _MAX_SEARCH
            # Only trust the count once a strictly higher energy has been seen,
            # otherwise degenerate partners may lie just past the horizon.
            if exhausted or self.energy(labels[-1]) > energy + max(tol, tol * abs(energy)):
                return len(matches)
            n *= 2

    def label_text(self, label: Label) -> str:
        """Human-readable form of a label, e.g. 'n_x=1, n_y=0'."""
        self._check_arity(label)
        return ", ".join(
            f"{name}={value}" for name, value in zip(self.quantum_numbers, label)
        )

    def wavefunction_on_grid(
        self,
        label: Label,
        grid: GridND,
        sparse: bool = False,
    ) -> NDArray[np.float64]:
        """Sample a state's wavefunction on a grid, returning `grid.shape`."""
        if grid.ndim != self.ndim:
            raise ValueError(
                f"{type(self).__name__} is {self.ndim}D but the grid is {grid.ndim}D"
            )
        psi = self.wavefunction(label)(*grid.coordinates(sparse=sparse))
        psi = np.asarray(psi)
        if psi.shape != grid.shape:
            psi = np.broadcast_to(psi, grid.shape).copy()
        return psi

    def density_on_grid(self, label: Label, grid: GridND) -> NDArray[np.float64]:
        """Sample |psi|^2 on a grid."""
        return np.abs(self.wavefunction_on_grid(label, grid)) ** 2

    # -- helpers ----------------------------------------------------------

    def _group_by_energy(self, labels: list[Label], tol: float) -> list[EnergyLevel]:
        """Group energy-ascending labels into levels."""
        levels: list[EnergyLevel] = []
        current: list[Label] = []
        reference = math.nan
        for label in labels:
            energy = self.energy(label)
            if current and not _close(energy, reference, tol):
                levels.append(EnergyLevel(reference, tuple(current)))
                current = []
            if not current:
                reference = energy
            current.append(label)
        if current:
            levels.append(EnergyLevel(reference, tuple(current)))
        return levels

    def _check_arity(self, label: Label) -> None:
        if len(label) != len(self.quantum_numbers):
            raise ValueError(
                f"expected {len(self.quantum_numbers)} quantum number(s) "
                f"{self.quantum_numbers}, got {label}"
            )


def _close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


__all__ = ["Label", "EnergyLevel", "Spectrum"]
