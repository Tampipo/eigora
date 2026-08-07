# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The spectrum of a separable system: a product of its blocks' spectra.

Because the blocks act on disjoint coordinates, energies add and wavefunctions
multiply, and a state of the whole system is labelled by the concatenation of
its blocks' quantum numbers. Nothing here cares whether a block was solved
analytically or numerically.
"""

import heapq
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from eigora.qm.spectra.base import Label, Spectrum


class SeparableSpectrum(Spectrum):
    """
    Eigenstates of a sum of independent sub-systems.

        E(n_A, n_B) = E_A(n_A) + E_B(n_B)
        psi(x_A, x_B) = psi_A(x_A) * psi_B(x_B)

    Example
    -------
    >>> from eigora.qm.potentials import HarmonicWell, SeparablePotential
    >>> from eigora.qm.spectra import spectrum_for
    >>> sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 3))
    >>> sol.energy((0, 0, 0))
    1.5
    >>> [level.degeneracy for level in sol.levels(4)]
    [1, 3, 6, 10]

    Parameters
    ----------
    blocks : sequence of Spectrum
        One spectrum per sub-system, in coordinate order.
    names : sequence of str, optional
        Block names used to disambiguate quantum numbers ("n_1", "n_2", ...).
    """

    def __init__(self, blocks: Sequence[Spectrum], names: Sequence[str] | None = None) -> None:
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("a separable spectrum needs at least one block")
        for block in blocks:
            if not isinstance(block, Spectrum):
                raise TypeError(f"expected a Spectrum, got {type(block).__name__}")

        self.blocks = blocks
        self.names = tuple(names) if names is not None else tuple(
            str(i + 1) for i in range(len(blocks))
        )
        if len(self.names) != len(blocks):
            raise ValueError(f"expected {len(blocks)} block name(s), got {len(self.names)}")

        # Labels of each block, ascending in energy, grown on demand.
        self._known: list[list[Label]] = [[] for _ in blocks]

    @property
    def ndim(self) -> int:
        return sum(block.ndim for block in self.blocks)

    @property
    def quantum_numbers(self) -> tuple[str, ...]:
        return tuple(
            f"{name}_{block_name}"
            for block, block_name in zip(self.blocks, self.names)
            for name in block.quantum_numbers
        )

    @property
    def is_exact(self) -> bool:
        return all(block.is_exact for block in self.blocks)

    @property
    def n_available(self) -> int | None:
        counts = [block.n_available for block in self.blocks]
        if any(count is None for count in counts):
            return None
        total = 1
        for count in counts:
            total *= count
        return total

    @property
    def block_slices(self) -> tuple[slice, ...]:
        """The slice of quantum numbers belonging to each block."""
        slices = []
        start = 0
        for block in self.blocks:
            width = len(block.quantum_numbers)
            slices.append(slice(start, start + width))
            start += width
        return tuple(slices)

    @property
    def coordinate_slices(self) -> tuple[slice, ...]:
        """The slice of coordinates belonging to each block."""
        slices = []
        start = 0
        for block in self.blocks:
            slices.append(slice(start, start + block.ndim))
            start += block.ndim
        return tuple(slices)

    def split(self, label: Label) -> tuple[Label, ...]:
        """Split a flat label into one label per block."""
        self._check_arity(label)
        return tuple(tuple(label[block_slice]) for block_slice in self.block_slices)

    def energy(self, label: Label) -> float:
        return float(
            sum(
                block.energy(part)
                for block, part in zip(self.blocks, self.split(label))
            )
        )

    def wavefunction(self, label: Label) -> Callable[..., NDArray[np.float64]]:
        parts = [
            block.wavefunction(part)
            for block, part in zip(self.blocks, self.split(label))
        ]
        coordinate_slices = self.coordinate_slices
        ndim = self.ndim

        def psi(*coords: NDArray[np.float64]) -> NDArray[np.float64]:
            if len(coords) != ndim:
                raise ValueError(f"expected {ndim} coordinate array(s), got {len(coords)}")
            product = parts[0](*coords[coordinate_slices[0]])
            for part, block_slice in zip(parts[1:], coordinate_slices[1:]):
                product = product * part(*coords[block_slice])
            return product

        return psi

    def states(self, n: int) -> list[Label]:
        """
        The `n` lowest states, ascending in energy.

        Each block's states are already ascending, so the composite ones are
        enumerated best-first over the lattice of per-block ranks: pop the
        cheapest combination, then offer the combinations one step up along
        each block. A block that runs out simply stops producing successors.
        """
        if n < 1:
            raise ValueError(f"number of states must be at least 1, got {n}")

        start = (0,) * len(self.blocks)
        if self._labels_at(start) is None:
            return []

        heap: list[tuple[float, tuple[int, ...]]] = [(self._energy_at(start), start)]
        seen = {start}
        found: list[Label] = []

        while heap and len(found) < n:
            _, ranks = heapq.heappop(heap)
            found.append(self._flat_label(ranks))
            for block_index in range(len(self.blocks)):
                nxt = list(ranks)
                nxt[block_index] += 1
                candidate = tuple(nxt)
                if candidate in seen or self._labels_at(candidate) is None:
                    continue
                seen.add(candidate)
                heapq.heappush(heap, (self._energy_at(candidate), candidate))

        return found

    # -- rank bookkeeping --------------------------------------------------

    def _block_label(self, block_index: int, rank: int) -> Label | None:
        """The block's rank-th state, or None once its tower is exhausted."""
        known = self._known[block_index]
        if rank >= len(known):
            known = self.blocks[block_index].states(rank + 1)
            self._known[block_index] = known
        return known[rank] if rank < len(known) else None

    def _labels_at(self, ranks: tuple[int, ...]) -> tuple[Label, ...] | None:
        """The per-block labels of a rank tuple, or None if out of range."""
        labels = []
        for block_index, rank in enumerate(ranks):
            label = self._block_label(block_index, rank)
            if label is None:
                return None
            labels.append(label)
        return tuple(labels)

    def _energy_at(self, ranks: tuple[int, ...]) -> float:
        labels = self._labels_at(ranks)
        assert labels is not None  # callers check availability first
        return float(
            sum(block.energy(label) for block, label in zip(self.blocks, labels))
        )

    def _flat_label(self, ranks: tuple[int, ...]) -> Label:
        labels = self._labels_at(ranks)
        assert labels is not None
        return tuple(value for label in labels for value in label)


__all__ = ["SeparableSpectrum"]
