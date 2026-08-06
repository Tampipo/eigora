# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract definition of operators.

Three layers, each adding a guarantee the one below cannot make:
`Operator` is any square matrix (ladder operators and propagators live here),
`Observable` is Hermitian and therefore has a real eigenvalue spectrum and an
orthonormal eigenbasis, and `Hamiltonian` is the observable that generates
time translation.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from physense_qm.discrete.measurement import (
    Outcome,
    outcomes as _outcomes,
    sample as _sample,
)
from physense_qm.discrete.operations import (
    Matrix,
    anticommutator as _anticommutator,
    commutator as _commutator,
    dagger as _dagger,
    is_hermitian as _is_hermitian,
    is_unitary as _is_unitary,
    tensor_product as _tensor_product,
)


@dataclass(frozen=True)
class Operator:
    """
    Any linear operator on the space. Not necessarily Hermitian.

    Attributes
    ----------
    matrix : Matrix
        Square matrix representation, in the computational basis.
    """

    matrix: Matrix

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"the operator must be a square matrix, got shape {matrix.shape}"
            )
        object.__setattr__(self, "matrix", matrix)

    @property
    def dim(self) -> int:
        """Dimension of the Hilbert space."""
        return self.matrix.shape[0]

    def dagger(self) -> "Operator":
        """The adjoint A†. Hermiticity is preserved, so the class is kept."""
        return type(self)(_dagger(self.matrix))

    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """True if A† = A, so that `Observable(self.matrix)` would be accepted."""
        return _is_hermitian(self.matrix, tol)

    def is_unitary(self, tol: float = 1e-10) -> bool:
        """True if A†A = AA† = I."""
        return _is_unitary(self.matrix, tol)

    # The product of two Hermitian operators is Hermitian only when they
    # commute, so this one always drops to `Operator`.
    def __matmul__(self, other: "Operator") -> "Operator":
        return Operator(self.matrix @ other.matrix)

    def __add__(self, other: "Operator") -> "Operator":
        return _common(self, other)(self.matrix + other.matrix)

    def __sub__(self, other: "Operator") -> "Operator":
        return _common(self, other)(self.matrix - other.matrix)

    def __mul__(self, scalar: complex) -> "Operator":
        # A real multiple of a Hermitian operator stays Hermitian; 1j * A
        # is anti-Hermitian.
        cls = type(self) if np.isreal(scalar) else Operator
        return cls(self.matrix * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> "Operator":
        return type(self)(-self.matrix)


@dataclass(frozen=True)
class Eigensystem:
    """
    Result of diagonalising an observable.

    Attributes
    ----------
    eigenvalues : NDArray of shape (dim,)
        Real eigenvalues, in ascending order.
    eigenvectors : NDArray of shape (dim, dim), complex
        Normalised eigenvectors, one per row. eigenvectors[i] corresponds to
        eigenvalues[i] -- note this is the transpose of what `np.linalg.eigh`
        returns, and matches `EigenSolution.wavefunctions` on the continuous
        side.
    """

    eigenvalues: NDArray[np.float64]
    eigenvectors: Matrix

    @property
    def n_states(self) -> int:
        return len(self.eigenvalues)

    def eigenvector(self, n: int) -> Matrix:
        """Return the n-th eigenvector (0-indexed, ascending in eigenvalue)."""
        if n < 0 or n >= self.n_states:
            raise IndexError(f"State index {n} out of range [0, {self.n_states - 1}]")
        return self.eigenvectors[n]


@dataclass(frozen=True)
class Observable(Operator):
    """
    A Hermitian operator: real eigenvalues and an orthonormal eigenbasis.

    Being Hermitian is what makes `np.linalg.eigh` valid here, and with it
    everything the measurement postulate needs.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _is_hermitian(self.matrix):
            raise ValueError("an observable must be Hermitian")

    @cached_property
    def eigensystem(self) -> Eigensystem:
        """
        Eigenvalues and eigenvectors, ascending in eigenvalue.

        Diagonalised once and cached. Can be overridden for exact known systems.
        """
        values, vectors = np.linalg.eigh(self.matrix)
        # eigh returns eigenvectors as columns -- transpose so eigenvectors[i]
        # is the eigenvector for eigenvalues[i].
        return Eigensystem(eigenvalues=values, eigenvectors=vectors.T)

    def eigenvalues(self) -> NDArray[np.float64]:
        """The real eigenvalues, ascending -- the possible measurement outcomes."""
        return self.eigensystem.eigenvalues

    def eigenvectors(self) -> Matrix:
        """The eigenvectors, one per row, ascending in eigenvalue."""
        return self.eigensystem.eigenvectors

    def expectation(self, psi: Matrix) -> float:
        """
        <psi|A|psi>, real because A is Hermitian.

        Assumes `psi` is normalised.
        """
        return float(np.real(np.vdot(psi, self.matrix @ psi)))

    def outcomes(self, psi: Matrix, tol: float = 1e-9) -> list[Outcome]:
        """
        The possible results of measuring this observable on `psi`.

        One entry per *distinct* eigenvalue, so a degenerate eigenvalue
        appears once, with its probability summed over the whole subspace.

        Parameters
        ----------
        psi : Matrix of shape (dim,)
            State to measure. Normalised on the way in.
        tol : float
            Tolerance for calling two eigenvalues degenerate.

        Returns
        -------
        list of Outcome
        """
        return _outcomes(
            self.eigensystem.eigenvalues, self.eigensystem.eigenvectors, psi, tol
        )

    def measure(
        self,
        psi: Matrix,
        rng: np.random.Generator | None = None,
        tol: float = 1e-9,
    ) -> tuple[Outcome, Matrix]:
        """
        Perform a projective measurement: draw an outcome, then collapse.

        Parameters
        ----------
        psi : Matrix of shape (dim,)
            State to measure.
        rng : np.random.Generator, optional
            Source of randomness; a fresh default generator if omitted.
        tol : float
            Tolerance for calling two eigenvalues degenerate.

        Returns
        -------
        (Outcome, Matrix)
            The result observed and the post-measurement state.
        """
        return _sample(
            self.eigensystem.eigenvalues, self.eigensystem.eigenvectors, psi, rng, tol
        )


@dataclass(frozen=True)
class Hamiltonian(Observable):
    """
    The generator of time translation.

    Adds the energy vocabulary and the propagator; the evolution of a state
    itself lives in `physense_qm.discrete.evolution`.
    """

    def eigenenergies(self) -> NDArray[np.float64]:
        """Eigenenergies, ascending (atomic units)."""
        return self.eigensystem.eigenvalues

    @property
    def ground_state(self) -> Matrix:
        """Eigenvector of the lowest energy."""
        return self.eigensystem.eigenvectors[0]

    @property
    def ground_energy(self) -> float:
        """The lowest energy (atomic units)."""
        return float(self.eigensystem.eigenvalues[0])

    def propagator(self, t: float) -> Operator:
        """
        U(t) = exp(-iHt), built in the energy eigenbasis.

        Unitary, and exact at any `t` -- no time stepping involved. To push a
        state through time, prefer `evolution.evolve`, which reuses one
        decomposition across all requested times.

        Parameters
        ----------
        t : float
            Elapsed time (atomic units). May be negative.

        Returns
        -------
        Operator
            Unitary, hence an `Operator` rather than an `Observable`.
        """
        phases = np.exp(-1j * self.eigensystem.eigenvalues * t)
        # Columns again, so that column i is scaled by its own phase.
        vectors = self.eigensystem.eigenvectors.T
        return Operator((vectors * phases) @ _dagger(vectors))


def commutator(A: Operator, B: Operator) -> Operator:
    """
    [A, B] = AB - BA.

    Returns a plain `Operator`: the commutator of two observables is
    anti-Hermitian ([A,B]† = -[A,B]), so it is not itself an observable --
    i[A,B] is the one you can measure.
    """
    return Operator(_commutator(A.matrix, B.matrix))


def anticommutator(A: Operator, B: Operator) -> Operator:
    """
    {A, B} = AB + BA.

    Unlike the commutator this is Hermitian whenever A and B are
    ({A,B}† = B†A† + A†B† = {A,B}), so the class is kept.
    """
    return _common(A, B)(_anticommutator(A.matrix, B.matrix))


def tensor_product(A: Operator, B: Operator) -> Operator:
    """
    A ⊗ B, acting on the product space.

    Returns a plain `Operator` even when both arguments are observables --
    re-wrap deliberately with `Observable(...)` if you want the guarantee back.
    """
    return Operator(_tensor_product(A.matrix, B.matrix))


def _common(a: Operator, b: Operator) -> type[Operator]:
    """
    The more specific of two classes, when one derives from the other.

    Used by the operations that provably preserve Hermiticity, so that
    `H_free + H_interaction` comes back a `Hamiltonian` rather than a bare
    `Operator`. Symmetric: the result does not depend on operand order.

    Note this reconstructs via `cls(matrix)`, so any subclass must keep the
    single-matrix constructor signature -- which is why the known systems in
    `hamiltonians` are factory functions rather than subclasses.
    """
    if isinstance(b, type(a)):
        return type(a)
    if isinstance(a, type(b)):
        return type(b)
    return Operator


__all__ = [
    "Operator",
    "Eigensystem",
    "Observable",
    "Hamiltonian",
    "commutator",
    "anticommutator",
    "tensor_product",
]
