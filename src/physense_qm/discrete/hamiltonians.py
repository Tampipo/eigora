# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Known discrete Hamiltonians, and ways of combining them.

Each system is built from its physical parameters rather than a matrix, and
those that have a closed-form spectrum override `eigensystem` to return it
directly instead of diagonalising. The tests hold the two against each other:
an analytic solution and `np.linalg.eigh` must agree.

Arithmetic on these degrades to a plain `Hamiltonian` -- see
`KnownHamiltonian._wrap`. Scaling, adding or embedding one of them generally
produces a system that is no longer of that family, and must not keep the
closed form.

Atomic units throughout (hbar = 1). Spin operators follow the Pauli
convention of `operators`, so a splitting of `w` means eigenvalues +/- w/2.
"""

from collections.abc import Sequence
# `field` is a parameter name here (magnetic field), so the dataclass
# helper of the same name is aliased out of the way.
from dataclasses import dataclass, field as _field
from functools import cached_property

import numpy as np

from physense_qm.discrete.operations import Matrix
from physense_qm.discrete.operators import (
    SX,
    SY,
    SZ,
    Eigensystem,
    Hamiltonian,
    embed,
)


@dataclass(frozen=True)
class KnownHamiltonian(Hamiltonian):
    """
    Base for systems built from physical parameters.

    Subclasses declare their parameters as fields and derive `matrix` in
    `__post_init__`, so `TwoLevel(bias=1.0)` reads as the physics rather than
    as a 2x2 array.
    """

    @classmethod
    def _wrap(cls, matrix: Matrix) -> Hamiltonian:
        """
        Degrade to a plain `Hamiltonian` after any operation.

        `2 * TwoLevel(1.0, 1.0)` is a different two-level system and
        `embed(TwoLevel(...), 0, (2, 2))` is not two-level at all, so neither
        may inherit the closed-form spectrum -- nor the constructor, which
        takes parameters and not a matrix.
        """
        return Hamiltonian(matrix)


@dataclass(frozen=True)
class TwoLevel(KnownHamiltonian):
    """
    The generic two-level system.

        H = (bias/2) SZ + (coupling/2) SX

    Exact spectrum
    --------------
        E± = ± sqrt(bias^2 + coupling^2) / 2

    The gap is the quadrature sum of the two scales, so it never closes
    unless both vanish -- the avoided crossing.

    Parameters
    ----------
    bias : float
        Energy difference between the two basis states (the SZ splitting).
    coupling : float
        Off-diagonal tunnelling amplitude between them.
    """

    bias: float = 0.0
    coupling: float = 1.0
    matrix: Matrix = _field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "matrix", (self.bias / 2 * SZ + self.coupling / 2 * SX).matrix
        )
        super().__post_init__()

    @cached_property
    def eigensystem(self) -> Eigensystem:
        """The closed form, in place of diagonalising."""
        return _spin_half_eigensystem(self.coupling, 0.0, self.bias)


@dataclass(frozen=True)
class Rabi(KnownHamiltonian):
    """
    A driven two-level system in the rotating frame (rotating-wave approx).

        H = (detuning/2) SZ + (rabi_frequency/2) SX

    The same matrix as `TwoLevel`, named for the problem it describes.

    Exact spectrum
    --------------
        E± = ± W / 2,   W = sqrt(rabi_frequency^2 + detuning^2)

    with W the generalised Rabi frequency.

    Exact dynamics
    --------------
    Starting from the lower basis state, the population transferred to the
    upper one is

        P(t) = (rabi_frequency / W)^2 * sin^2(W t / 2)

    On resonance (detuning = 0) the transfer is complete; off resonance it
    never reaches 1.

    Parameters
    ----------
    detuning : float
        Drive frequency minus the transition frequency.
    rabi_frequency : float
        Strength of the drive.
    """

    detuning: float = 0.0
    rabi_frequency: float = 1.0
    matrix: Matrix = _field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix",
            (self.detuning / 2 * SZ + self.rabi_frequency / 2 * SX).matrix,
        )
        super().__post_init__()

    @property
    def generalised_frequency(self) -> float:
        """W = sqrt(rabi_frequency^2 + detuning^2), the oscillation rate."""
        return float(np.hypot(self.rabi_frequency, self.detuning))

    @property
    def max_transfer(self) -> float:
        """Peak excited-state population, 1 only on resonance."""
        if self.generalised_frequency == 0.0:
            return 0.0
        return float((self.rabi_frequency / self.generalised_frequency) ** 2)

    @cached_property
    def eigensystem(self) -> Eigensystem:
        """The closed form, in place of diagonalising."""
        return _spin_half_eigensystem(self.rabi_frequency, 0.0, self.detuning)


@dataclass(frozen=True)
class SpinInField(KnownHamiltonian):
    """
    A spin-1/2 in a uniform magnetic field.

        H = (Bx SX + By SY + Bz SZ) / 2

    Exact spectrum
    --------------
        E± = ± |B| / 2

    The gap |B| is the Larmor frequency: the spin precesses about the field
    at that rate, whatever direction the field points. The eigenvectors are
    the spin-up and spin-down states along B.

    Parameters
    ----------
    field : sequence of float
        The three components (Bx, By, Bz).
    """

    field: tuple[float, float, float] = (0.0, 0.0, 1.0)
    matrix: Matrix = _field(init=False, repr=False)

    def __post_init__(self) -> None:
        components = np.asarray(self.field, dtype=float)
        if components.shape != (3,):
            raise ValueError(
                f"field must have three components, got shape {components.shape}"
            )
        object.__setattr__(self, "field", tuple(float(b) for b in components))

        bx, by, bz = components
        object.__setattr__(self, "matrix", ((bx * SX + by * SY + bz * SZ) * 0.5).matrix)
        super().__post_init__()

    @property
    def strength(self) -> float:
        """|B|, the Larmor frequency and the level splitting."""
        return float(np.linalg.norm(self.field))

    @cached_property
    def eigensystem(self) -> Eigensystem:
        """The closed form, in place of diagonalising."""
        return _spin_half_eigensystem(*self.field)


@dataclass(frozen=True)
class HarmonicLadder(KnownHamiltonian):
    """
    A harmonic oscillator truncated to its lowest `n_levels` states.

    Exact spectrum
    --------------
        E_n = omega (n + 1/2),  n = 0, 1, ..., n_levels - 1

    Already diagonal, so the closed form is exact by construction. Useful as
    a bridge to the continuous side, where `HarmonicWell` has the same
    spectrum but reached by solving a differential equation.

    Parameters
    ----------
    n_levels : int
        How many rungs to keep.
    omega : float
        Level spacing.
    """

    n_levels: int = 2
    omega: float = 1.0
    matrix: Matrix = _field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_levels < 1:
            raise ValueError(f"n_levels must be at least 1, got {self.n_levels}")
        if self.omega <= 0:
            raise ValueError(f"omega must be positive, got {self.omega}")
        object.__setattr__(self, "matrix", np.diag(self._energies))
        super().__post_init__()

    @property
    def _energies(self) -> np.ndarray:
        return self.omega * (np.arange(self.n_levels) + 0.5)

    @cached_property
    def eigensystem(self) -> Eigensystem:
        """The closed form: already diagonal, so the basis is the eigenbasis."""
        return Eigensystem(
            eigenvalues=self._energies,
            eigenvectors=np.eye(self.n_levels, dtype=complex),
        )


@dataclass(frozen=True)
class HeisenbergChain(KnownHamiltonian):
    """
    A chain of spin-1/2 sites with isotropic nearest-neighbour exchange.

        H = coupling * sum_<ij> (SX_i SX_j + SY_i SY_j + SZ_i SZ_j)
            + field * sum_i SZ_i

    Partially exact
    ---------------
    For two sites and no field the four states split into a singlet and a
    triplet,

        E_singlet = -3 * coupling      (once)
        E_triplet =  +1 * coupling     (three times)

    since the Pauli dot product has eigenvalues -3 and +1. Longer chains have
    no elementary closed form, so this class does not override `eigensystem`
    and falls back to diagonalisation. H always commutes with the total SZ,
    whatever the length, so magnetisation stays a good quantum number.

    Parameters
    ----------
    n_sites : int
        Number of spins; the Hilbert space has dimension 2**n_sites.
    coupling : float
        Exchange strength. Positive is antiferromagnetic here.
    field : float
        Uniform longitudinal field.
    periodic : bool
        Close the chain into a ring. Ignored for fewer than three sites,
        where it would double the single bond.
    """

    n_sites: int = 2
    coupling: float = 1.0
    field: float = 0.0
    periodic: bool = False
    matrix: Matrix = _field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_sites < 1:
            raise ValueError(f"n_sites must be at least 1, got {self.n_sites}")

        dims = (2,) * self.n_sites
        dimension = 2**self.n_sites
        matrix = np.zeros((dimension, dimension), dtype=complex)

        for i, j in self.bonds:
            for pauli in (SX, SY, SZ):
                matrix += (
                    self.coupling * (embed(pauli, i, dims) @ embed(pauli, j, dims)).matrix
                )

        if self.field:
            for site in range(self.n_sites):
                matrix += self.field * embed(SZ, site, dims).matrix

        object.__setattr__(self, "matrix", matrix)
        super().__post_init__()

    @property
    def bonds(self) -> list[tuple[int, int]]:
        """The coupled site pairs, including the closing bond if periodic."""
        bonds = [(i, i + 1) for i in range(self.n_sites - 1)]
        if self.periodic and self.n_sites > 2:
            bonds.append((self.n_sites - 1, 0))
        return bonds


def noninteracting(hamiltonians: Sequence[Hamiltonian]) -> Hamiltonian:
    """
    Combine independent subsystems into one Hamiltonian.

    H = sum_k I ⊗ ... ⊗ H_k ⊗ ... ⊗ I, acting on the tensor product of the
    subsystem spaces. Energies of the composite system are sums of subsystem
    energies -- not products, which is what a bare H_1 ⊗ H_2 would give.

    Parameters
    ----------
    hamiltonians : sequence of Hamiltonian
        The subsystem Hamiltonians, in tensor-factor order.

    Returns
    -------
    Hamiltonian
        Hamiltonian on the product space, of dimension prod(h.dim).
    """
    if not hamiltonians:
        raise ValueError("at least one Hamiltonian is required")

    dims = [h.dim for h in hamiltonians]
    total = embed(hamiltonians[0], 0, dims)
    for site, hamiltonian in enumerate(hamiltonians[1:], start=1):
        total = total + embed(hamiltonian, site, dims)
    return total


def _spin_half_eigensystem(bx: float, by: float, bz: float) -> Eigensystem:
    """
    Closed-form eigensystem of H = (B . sigma) / 2.

    Eigenvalues are ±|B|/2 and the eigenvectors are the spin states along B,
    parametrised by its polar angles. A zero field leaves every state
    degenerate, so any orthonormal basis will do.
    """
    strength = float(np.sqrt(bx**2 + by**2 + bz**2))
    if strength == 0.0:
        return Eigensystem(np.zeros(2), np.eye(2, dtype=complex))

    theta = np.arccos(np.clip(bz / strength, -1.0, 1.0))
    phi = np.arctan2(by, bx)
    cosine, sine = np.cos(theta / 2), np.sin(theta / 2)

    lower = np.array([-np.exp(-1j * phi) * sine, cosine])
    upper = np.array([cosine, np.exp(1j * phi) * sine])
    return Eigensystem(
        eigenvalues=np.array([-strength / 2, strength / 2]),
        eigenvectors=np.array([lower, upper]),
    )


__all__ = [
    "KnownHamiltonian",
    "TwoLevel",
    "Rabi",
    "SpinInField",
    "HarmonicLadder",
    "HeisenbergChain",
    "noninteracting",
]
