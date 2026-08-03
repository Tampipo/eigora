# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Base class for potentials in an arbitrary number of dimensions.

A potential of dimension n is called with n coordinate arrays, which may be
full meshgrid arrays or the sparse (open) meshgrid of
`GridND.coordinates(sparse=True)`; the result is broadcast to their common
shape. A 1D `PotentialND` is therefore a drop-in replacement for a
`potentials.known.Potential` in `solve_eigenstates` and `evolve`.

Atomic units (hbar = m = 1), as everywhere else in the package.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physense_utils.grids import GridND


class PotentialND(ABC):
    """
    Abstract base class for a potential in `ndim` spatial dimensions.

    Subclasses implement `_evaluate`; `__call__` checks the arguments and
    broadcasts the result.
    """

    def __init__(self, ndim: int) -> None:
        if ndim < 1:
            raise ValueError(f"ndim must be at least 1, got {ndim}")
        self._ndim = int(ndim)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self._ndim

    @abstractmethod
    def _evaluate(self, *coords: NDArray[np.float64]) -> ArrayLike:
        """Evaluate V on `ndim` float coordinate arrays."""

    def __call__(self, *coords: ArrayLike) -> NDArray[np.float64]:
        """
        Evaluate V on `ndim` coordinate arrays.

        Parameters
        ----------
        *coords : array_like
            One array per dimension, mutually broadcastable.

        Returns
        -------
        NDArray
            V on the common broadcast shape of the coordinates.
        """
        if len(coords) != self.ndim:
            raise ValueError(
                f"{type(self).__name__} is {self.ndim}D and expects "
                f"{self.ndim} coordinate array(s), got {len(coords)}"
            )
        arrays = tuple(np.asarray(c, dtype=np.float64) for c in coords)
        shape = np.broadcast_shapes(*(a.shape for a in arrays))
        V = np.asarray(self._evaluate(*arrays), dtype=np.float64)
        if V.shape != shape:
            V = np.broadcast_to(V, shape).copy()
        return V

    def on_grid(self, grid: GridND, sparse: bool = False) -> NDArray[np.float64]:
        """
        Evaluate the potential on a grid, returning an array of `grid.shape`.

        Parameters
        ----------
        grid : GridND
            Grid whose dimension must match `ndim`.
        sparse : bool
            Use the open meshgrid -- cheaper in memory, safe for any potential
            built from these classes.
        """
        if grid.ndim != self.ndim:
            raise ValueError(
                f"{type(self).__name__} is {self.ndim}D but the grid is {grid.ndim}D"
            )
        return self(*grid.coordinates(sparse=sparse))

    def __add__(self, other: "PotentialND") -> "SumPotential":
        return SumPotential(self, other)


class SumPotential(PotentialND):
    """
    Sum of potentials sharing the same coordinates.

    Unlike `SeparablePotential`, the terms all see every coordinate, so the
    result carries no separable structure and has no analytic spectrum.
    """

    def __init__(self, *terms: PotentialND) -> None:
        flat = _flatten(terms)
        super().__init__(flat[0].ndim)
        self.terms = flat

    def _evaluate(self, *coords: NDArray[np.float64]) -> ArrayLike:
        total = self.terms[0](*coords)
        for term in self.terms[1:]:
            total = total + term(*coords)
        return total


def _flatten(terms: Sequence[PotentialND]) -> tuple[PotentialND, ...]:
    """Check types and dimensions, and inline nested sums."""
    if not terms:
        raise ValueError("at least one term is required")
    ndim: int | None = None
    flat: list[PotentialND] = []
    for term in terms:
        if not isinstance(term, PotentialND):
            raise TypeError(f"expected a PotentialND, got {type(term).__name__}")
        if ndim is None:
            ndim = term.ndim
        elif term.ndim != ndim:
            raise ValueError(f"cannot add a {term.ndim}D potential to a {ndim}D one")
        flat.extend(term.terms if isinstance(term, SumPotential) else [term])
    return tuple(flat)


__all__ = ["PotentialND", "SumPotential"]
