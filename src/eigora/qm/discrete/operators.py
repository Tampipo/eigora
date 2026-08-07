# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Abstract definition of operators.

Three layers, each adding a guarantee the one below cannot make:
`Operator` is any square matrix (ladder operators and propagators live here),
`Observable` is Hermitian and therefore has a real eigenvalue spectrum and an
orthonormal eigenbasis, and `Hamiltonian` is the observable that generates
time translation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, reduce

import numpy as np
from numpy.typing import NDArray

from eigora.qm.discrete.measurement import (
    Outcome,
    outcomes as _outcomes,
    sample as _sample,
)
from eigora.qm.discrete.operations import (
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

    @classmethod
    def _wrap(cls, matrix: Matrix) -> "Operator":
        """
        Rebuild this class around a new matrix, after an operation.

        Subclasses built from physical parameters rather than a matrix
        override this to degrade to their generic ancestor: a scaled or
        embedded `TwoLevel` is no longer that two-level system, and must not
        keep its closed-form spectrum.
        """
        return cls(matrix)

    def dagger(self) -> "Operator":
        """The adjoint A†. Hermiticity is preserved, so the class is kept."""
        return type(self)._wrap(_dagger(self.matrix))

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
        return _common(self, other)._wrap(self.matrix + other.matrix)

    def __sub__(self, other: "Operator") -> "Operator":
        return _common(self, other)._wrap(self.matrix - other.matrix)

    def __mul__(self, scalar: complex) -> "Operator":
        # A real multiple of a Hermitian operator stays Hermitian; 1j * A
        # is anti-Hermitian.
        cls = type(self) if np.isreal(scalar) else Operator
        return cls._wrap(self.matrix * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> "Operator":
        return type(self)._wrap(-self.matrix)


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
    itself lives in `eigora.qm.discrete.evolution`.
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
    return _common(A, B)._wrap(_anticommutator(A.matrix, B.matrix))


def tensor_product(A: Operator, B: Operator) -> Operator:
    """
    A ⊗ B, acting on the product space.

    Returns a plain `Operator` even when both arguments are observables --
    re-wrap deliberately with `Observable(...)` if you want the guarantee back.
    """
    return Operator(_tensor_product(A.matrix, B.matrix))


def identity(dim: int) -> Observable:
    """
    The identity operator on a `dim`-dimensional space.

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space.

    Returns
    -------
    Observable
        Hermitian (and unitary), so usable as a tensor factor anywhere.
    """
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}")
    return _constant(Observable, np.eye(dim))


def embed(operator: Operator, site: int, dims: Sequence[int]) -> Operator:
    """
    Lift a single-subsystem operator to the whole product space.

    Returns I ⊗ ... ⊗ A ⊗ ... ⊗ I, with `operator` in the `site`-th tensor
    slot. This is the way to build a composite observable: total S_z on two
    spins is `embed(SZ, 0, dims) + embed(SZ, 1, dims)`, whose eigenvalues are
    the sums +2, 0, 0, -2 -- unlike `tensor_product(SZ, SZ)`, which multiplies
    them.

    Unlike `tensor_product`, this preserves the class: padding with identities
    changes neither Hermiticity nor what the operator means, so an `Observable`
    stays measurable and a `Hamiltonian` stays a Hamiltonian (of a composite
    system in which only that subsystem carries energy).

    Parameters
    ----------
    operator : Operator
        Acts on subsystem `site` alone.
    site : int
        Which tensor factor it occupies.
    dims : sequence of int
        Dimension of every subsystem, in tensor-factor order.

    Returns
    -------
    Operator
        Of dimension prod(dims), the same class as `operator`.
    """
    dims = list(dims)
    if not dims:
        raise ValueError("dims must name at least one subsystem")
    if not 0 <= site < len(dims):
        raise ValueError(f"site {site} out of range for {len(dims)} subsystem(s)")
    if operator.dim != dims[site]:
        raise ValueError(
            f"operator is {operator.dim}D but subsystem {site} is {dims[site]}D"
        )

    factors: list[Matrix] = [np.eye(d, dtype=complex) for d in dims]
    factors[site] = operator.matrix
    return type(operator)._wrap(reduce(_tensor_product, factors))


def _constant(cls: type[Operator], matrix: Matrix) -> Operator:
    """
    Build a shared operator whose matrix cannot be mutated in place.

    These are module-level singletons, so an in-place write by one caller
    would be visible to every other one.
    """
    array = np.asarray(matrix, dtype=complex)
    array.setflags(write=False)
    return cls(array)


#: Pauli matrices, eigenvalues ±1. Note [SX, SY] = 2i·SZ, and SX² = I.
SX = _constant(Observable, [[0, 1], [1, 0]])
SY = _constant(Observable, [[0, -1j], [1j, 0]])
SZ = _constant(Observable, [[1, 0], [0, -1]])

#: Ladder operators, S± = (SX ± i·SY)/2. Not Hermitian, hence not observables
#: -- they raise and lower rather than being measured.
SPLUS = _constant(Operator, [[0, 1], [0, 0]])
SMINUS = _constant(Operator, [[0, 0], [1, 0]])


def _common(a: Operator, b: Operator) -> type[Operator]:
    """
    The nearest class both operands belong to.

    Used by the operations that provably preserve Hermiticity, so that
    `H_free + H_interaction` comes back a `Hamiltonian` rather than a bare
    `Operator`. Symmetric: the result does not depend on operand order.

    Walking the ancestry rather than comparing the two types directly matters
    for siblings: `TwoLevel + SpinInField` are unrelated to each other but
    both Hamiltonians, and the sum should say so.

    The result is rebuilt through `_wrap`, so classes whose constructor takes
    physical parameters instead of a matrix can degrade gracefully.
    """
    # Most-derived first; the first class `b` also belongs to is their nearest
    # common ancestor. `Operator` always matches, so this cannot come up empty.
    return next(
        cls
        for cls in type(a).__mro__
        if issubclass(cls, Operator) and isinstance(b, cls)
    )


__all__ = [
    "Operator",
    "Eigensystem",
    "Observable",
    "Hamiltonian",
    "commutator",
    "anticommutator",
    "tensor_product",
    "identity",
    "embed",
    "SX",
    "SY",
    "SZ",
    "SPLUS",
    "SMINUS",
]
