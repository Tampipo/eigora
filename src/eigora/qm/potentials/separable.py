# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Separable potentials: sums of sub-system potentials over disjoint coordinate
blocks.

    V(x_1, ..., x_n) = V_A(block A) + V_B(block B) + ...

Each block has its own dimension -- a 1D well, a 3D central potential, one of
several non-interacting particles. The blocks are kept as objects rather than
collapsed into a single callable, because that structure is exactly what makes
the solution composable: energies add, wavefunctions multiply, quantum numbers
concatenate (see `physense_qm.spectra`).
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physense_qm.potentials.base import PotentialND


class Block1D(PotentialND):
    """
    Adapts a 1D potential to a one-dimensional `PotentialND` block.

    The wrapped object is kept as `.potential`, so the spectrum registry can
    still recognise a `HarmonicWell` inside a separable system.
    """

    def __init__(self, potential: Callable[..., ArrayLike]) -> None:
        if not callable(potential):
            raise TypeError(f"potential must be callable, got {type(potential).__name__}")
        super().__init__(1)
        self.potential = potential

    def _evaluate(self, x: NDArray[np.float64]) -> ArrayLike:
        return self.potential(x)


def as_block(potential: "PotentialND | Callable[..., ArrayLike]") -> PotentialND:
    """
    Coerce a potential into a `PotentialND` block.

    `PotentialND` instances pass through; any other callable is treated as a
    1D potential and wrapped in `Block1D`.
    """
    if isinstance(potential, PotentialND):
        return potential
    if callable(potential):
        return Block1D(potential)
    raise TypeError(f"cannot use {type(potential).__name__} as a potential block")


class SeparablePotential(PotentialND):
    """
    Sum of independent sub-system potentials over disjoint coordinate blocks.

    Coordinates are split contiguously in block order: with blocks of dimension
    (1, 3), a call takes 4 coordinates, the first going to block 0 and the last
    three to block 1.

    Example
    -------
    >>> from physense_qm.potentials import HarmonicWell
    >>> V = SeparablePotential([HarmonicWell(omega=1.0)] * 3)   # 3D isotropic trap
    >>> V.ndim
    3
    >>> float(V(1.0, 0.0, 0.0))
    0.5

    Parameters
    ----------
    blocks : sequence
        One potential per block; 1D potentials are wrapped by `as_block`.
        Nested `SeparablePotential`s are inlined.
    names : sequence of str, optional
        Block names, used to label quantum numbers. Defaults to "1", "2", ...
    """

    def __init__(
        self,
        blocks: Sequence["PotentialND | Callable[..., ArrayLike]"],
        names: Sequence[str] | None = None,
    ) -> None:
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("a separable potential needs at least one block")

        given_names = tuple(names) if names is not None else None
        if given_names is not None and len(given_names) != len(blocks):
            raise ValueError(
                f"expected {len(blocks)} block name(s), got {len(given_names)}"
            )

        flat_blocks: list[PotentialND] = []
        flat_names: list[str] = []
        for index, block in enumerate(blocks):
            block = as_block(block)
            name = given_names[index] if given_names is not None else None
            if isinstance(block, SeparablePotential):
                flat_blocks.extend(block.blocks)
                flat_names.extend(
                    f"{name}.{inner}" if name is not None else inner
                    for inner in block.names
                )
            else:
                flat_blocks.append(block)
                flat_names.append(name if name is not None else str(len(flat_names) + 1))

        super().__init__(sum(block.ndim for block in flat_blocks))
        self.blocks = tuple(flat_blocks)
        self.names = tuple(flat_names)

    @property
    def block_slices(self) -> tuple[slice, ...]:
        """The slice of coordinates owned by each block."""
        slices = []
        start = 0
        for block in self.blocks:
            slices.append(slice(start, start + block.ndim))
            start += block.ndim
        return tuple(slices)

    def _evaluate(self, *coords: NDArray[np.float64]) -> ArrayLike:
        total: ArrayLike = np.float64(0.0)
        for block, block_slice in zip(self.blocks, self.block_slices):
            total = total + block(*coords[block_slice])
        return total


__all__ = ["Block1D", "SeparablePotential", "as_block"]
