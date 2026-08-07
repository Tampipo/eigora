# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Several operations on matrices
"""

import numpy as np
from numpy.typing import NDArray

# A linear operator on a discrete Hilbert space, as a square matrix.
Matrix = NDArray[np.complex128]


def dagger(matrix: Matrix) -> Matrix:
    """
    Compute the conjugate transpose (dagger) of a matrix.

    Parameters
    ----------
    matrix : Matrix
        The input matrix.

    Returns
    -------
    Matrix
        The conjugate transpose of the input matrix.
    """
    return matrix.conj().T


def is_hermitian(matrix: Matrix, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is Hermitian.

    Parameters
    ----------
    matrix : Matrix
        The input matrix.
    tol : float, optional
        Absolute tolerance for numerical comparison, by default 1e-10.

    Returns
    -------
    bool
        True if the matrix is Hermitian, False otherwise.
    """
    if not _is_square(matrix):
        return False
    # rtol=0: np.allclose otherwise keeps its 1e-5 relative term, which swamps
    # `tol` entirely for large matrix elements.
    return bool(np.allclose(matrix, dagger(matrix), rtol=0.0, atol=tol))


def is_unitary(matrix: Matrix, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is unitary.

    Parameters
    ----------
    matrix : Matrix
        The input matrix.
    tol : float, optional
        Absolute tolerance for numerical comparison, by default 1e-10.

    Returns
    -------
    bool
        True if the matrix is unitary, False otherwise.
    """
    if not _is_square(matrix):
        return False
    identity = np.eye(matrix.shape[0])
    return bool(
        np.allclose(matrix @ dagger(matrix), identity, rtol=0.0, atol=tol)
        and np.allclose(dagger(matrix) @ matrix, identity, rtol=0.0, atol=tol)
    )


def commutator(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the commutator of two matrices A and B.

    Parameters
    ----------
    A : Matrix
        The first matrix.
    B : Matrix
        The second matrix.

    Returns
    -------
    Matrix
        The commutator [A, B] = AB - BA.
    """
    return A @ B - B @ A


def anticommutator(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the anticommutator of two matrices A and B.

    Parameters
    ----------
    A : Matrix
        The first matrix.
    B : Matrix
        The second matrix.

    Returns
    -------
    Matrix
        The anticommutator {A, B} = AB + BA.
    """
    return A @ B + B @ A


def tensor_product(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the tensor product of two matrices A and B.

    This is the raw Kronecker product. Note that the Hamiltonian of two
    independent subsystems is *not* H_A ⊗ H_B but H_A ⊗ I + I ⊗ H_B —
    see `hamiltonians.noninteracting`.

    Parameters
    ----------
    A : Matrix
        The first matrix.
    B : Matrix
        The second matrix.

    Returns
    -------
    Matrix
        The tensor product A ⊗ B.
    """
    return np.kron(A, B)


def _is_square(matrix: Matrix) -> bool:
    """True for a 2D array with matching dimensions. Never raises."""
    shape = np.shape(matrix)
    return len(shape) == 2 and shape[0] == shape[1]


__all__ = [
    "Matrix",
    "dagger",
    "is_hermitian",
    "is_unitary",
    "commutator",
    "anticommutator",
    "tensor_product",
]
