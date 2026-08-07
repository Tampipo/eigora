# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The measurement postulate on a discrete Hilbert space.

Plain functions over arrays -- they take an eigenvalue/eigenvector pair and a
state, never an `Observable`, so that `Observable` can call into here without
this module having to import it back.

Everything is grouped by *distinct eigenvalue*, not by eigenvector. A
degenerate eigenvalue is one outcome whose probability sums over its whole
subspace, and measuring it projects onto that subspace rather than onto any
single eigenvector -- picking one basis vector out of a degenerate pair is the
classic way to get plausible-looking but wrong answers.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physense_qm.discrete.operations import Matrix


@dataclass(frozen=True)
class Outcome:
    """
    One possible result of measuring an observable.

    Attributes
    ----------
    value : float
        The eigenvalue observed.
    probability : float
        Born probability of this outcome, summed over the degenerate subspace.
    indices : tuple of int
        Positions in the eigenbasis spanning the subspace for this eigenvalue.
    """

    value: float
    probability: float
    indices: tuple[int, ...]

    @property
    def degeneracy(self) -> int:
        """Dimension of the eigenspace for this value."""
        return len(self.indices)


def degenerate_groups(
    eigenvalues: NDArray[np.float64],
    tol: float = 1e-9,
) -> list[tuple[float, tuple[int, ...]]]:
    """
    Group ascending eigenvalues into distinct values.

    Parameters
    ----------
    eigenvalues : NDArray of shape (dim,)
        Eigenvalues in ascending order, as `Observable.eigenvalues` returns.
    tol : float
        Relative/absolute tolerance for calling two eigenvalues equal.

    Returns
    -------
    list of (value, indices)
        One entry per distinct eigenvalue, ascending.
    """
    groups: list[tuple[float, tuple[int, ...]]] = []
    current: list[int] = []
    reference = math.nan
    for index, value in enumerate(eigenvalues):
        if current and not _close(float(value), reference, tol):
            groups.append((reference, tuple(current)))
            current = []
        if not current:
            reference = float(value)
        current.append(index)
    if current:
        groups.append((reference, tuple(current)))
    return groups


def outcomes(
    eigenvalues: NDArray[np.float64],
    eigenvectors: Matrix,
    psi: Matrix,
    tol: float = 1e-9,
) -> list[Outcome]:
    """
    The possible measurement results and their Born probabilities.

    P(lambda) = sum over the eigenspace of |<v_i|psi>|^2.

    Parameters
    ----------
    eigenvalues : NDArray of shape (dim,)
        Ascending eigenvalues.
    eigenvectors : Matrix of shape (dim, dim)
        Orthonormal eigenvectors, one per row.
    psi : Matrix of shape (dim,)
        State to measure. Normalised on the way in.
    tol : float
        Tolerance for calling two eigenvalues degenerate.

    Returns
    -------
    list of Outcome
        Ascending in value; probabilities sum to 1.
    """
    psi = _normalised(psi)
    weights = np.abs(eigenvectors.conj() @ psi) ** 2
    return [
        Outcome(value=value, probability=float(weights[list(indices)].sum()), indices=indices)
        for value, indices in degenerate_groups(eigenvalues, tol)
    ]


def collapse(eigenvectors: Matrix, indices: tuple[int, ...], psi: Matrix) -> Matrix:
    """
    Project a state onto an eigenspace and renormalise.

    |psi'> = P|psi> / || P|psi> ||, with P the projector onto the subspace
    spanned by `indices`.

    Parameters
    ----------
    eigenvectors : Matrix of shape (dim, dim)
        Orthonormal eigenvectors, one per row.
    indices : tuple of int
        Rows spanning the eigenspace.
    psi : Matrix of shape (dim,)
        State to project.

    Returns
    -------
    Matrix of shape (dim,)
        Normalised post-measurement state.

    Raises
    ------
    ValueError
        If `psi` has no component in the subspace, so the outcome was
        impossible to begin with.
    """
    psi = _normalised(psi)
    basis = eigenvectors[list(indices)]
    projected = (basis.conj() @ psi) @ basis
    norm = np.linalg.norm(projected)
    if norm == 0:
        raise ValueError("the state has no component in this eigenspace")
    return projected / norm


def sample(
    eigenvalues: NDArray[np.float64],
    eigenvectors: Matrix,
    psi: Matrix,
    rng: np.random.Generator | None = None,
    tol: float = 1e-9,
) -> tuple[Outcome, Matrix]:
    """
    Perform a projective measurement: draw an outcome, then collapse.

    Parameters
    ----------
    eigenvalues : NDArray of shape (dim,)
        Ascending eigenvalues.
    eigenvectors : Matrix of shape (dim, dim)
        Orthonormal eigenvectors, one per row.
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
    if rng is None:
        rng = np.random.default_rng()
    results = outcomes(eigenvalues, eigenvectors, psi, tol)
    probabilities = np.array([result.probability for result in results])
    # Guard against the probabilities summing to 1 only up to rounding.
    probabilities = probabilities / probabilities.sum()
    chosen = results[int(rng.choice(len(results), p=probabilities))]
    return chosen, collapse(eigenvectors, chosen.indices, psi)


def _normalised(psi: Matrix) -> Matrix:
    psi = np.asarray(psi, dtype=complex)
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("psi must be a non-zero vector")
    return psi / norm


def _close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


__all__ = ["Outcome", "degenerate_groups", "outcomes", "collapse", "sample"]
