# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Potentials defined by an arbitrary callable, in any dimension.

This is the escape hatch: it accepts any V(x_1, ..., x_n) but carries no
structure, so it has no analytic spectrum. Build a `SeparablePotential` from
known pieces instead whenever the system actually separates.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physense_qm.potentials.base import PotentialND


class GenericPotential(PotentialND):
    """
    V(x_1, ..., x_n) given by a Python callable.

    Example
    -------
    >>> V = GenericPotential(lambda x, y: 0.5 * (x**2 + y**2) + 0.1 * x * y, ndim=2)
    >>> V(1.0, 1.0)
    array(1.1)

    Parameters
    ----------
    func : callable
        Receives one array per dimension, returns something broadcastable to
        their common shape.
    ndim : int
        Number of spatial dimensions.
    """

    def __init__(self, func: Callable[..., ArrayLike], ndim: int) -> None:
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")
        super().__init__(ndim)
        self.func = func

    def _evaluate(self, *coords: NDArray[np.float64]) -> ArrayLike:
        return self.func(*coords)


__all__ = ["GenericPotential"]
