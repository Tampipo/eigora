# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Systems built by putting independent subsystems side by side.

    log Z(A x B) = log Z(A) + log Z(B)

because the microstates of the whole are pairs of microstates of the parts and
their energies add. Everything canonical follows from that one line: mean
energies add, and so do variances, because the parts are independent under the
product Gibbs measure -- which makes heat capacities additive too.

Nothing is enumerated. A composite deliberately does *not* implement
`SpectralSystem`: the level stream of a product is combinatorial in the number
of blocks, while its partition function costs one call per block. Making it a
plain `System` is how the type layering says so, and it is why
`HarmonicMode(omega) ** 3000` -- an Einstein solid of a thousand atoms -- is
free rather than impossible.

The subsystems are **distinguishable**. `A * A` is two labelled copies, not two
identical particles; there is no symmetrisation here and none is implied. For
indistinguishable particles use `IdenticalParticles`, which asks for the
statistics by name.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from eigora.statphys.systems.base import (
    Couplings,
    Level,
    Moments,
    SpectralSystem,
    System,
)

# A composite of incommensurate blocks has as many distinct energies as it has
# microstates, so the convolution has to be bounded. Commensurate spectra --
# the ones worth convolving -- collapse to far fewer.
_MAX_DISTINCT = 200_000

# Energies are grouped by rounding onto this grid. Summing `w` three times need
# not give exactly `3 * w`, so degeneracies would fragment without it.
_ENERGY_GRID = 1e-9


@dataclass(frozen=True)
class CompositeSystem(System):
    """
    Independent subsystems treated as one, distinguishable from each other.

    Example
    -------
    >>> from eigora.statphys.systems.known import HarmonicMode, TwoLevel
    >>> einstein = HarmonicMode(omega=1.0) ** 300     # 100 atoms, 3 modes each
    >>> paramagnet = TwoLevel(splitting=1.0) ** 50
    >>> both = einstein * paramagnet

    Parameters
    ----------
    blocks : sequence of System
        The subsystems. Nested composites are flattened, so `(A * B) * C` and
        `A * (B * C)` build the same object.
    """

    blocks: tuple[System, ...]

    def __init__(self, blocks: Sequence[System]) -> None:
        flat: list[System] = []
        for block in blocks:
            if not isinstance(block, System):
                raise TypeError(f"expected a System, got {type(block).__name__}")
            if isinstance(block, CompositeSystem):
                flat.extend(block.blocks)
            else:
                flat.append(block)
        if not flat:
            raise ValueError("a composite system needs at least one block")
        object.__setattr__(self, "blocks", tuple(flat))

    @property
    def is_exact(self) -> bool:
        return all(block.is_exact for block in self.blocks)

    @property
    def extensive_variables(self) -> frozenset[str]:
        """The union over blocks: the composite can report whatever any of them can."""
        return frozenset().union(
            *(block.extensive_variables for block in self.blocks)
        )

    def log_z(self, beta: float, couplings: Couplings = ()) -> float:
        return sum(
            block.log_z(beta, _carried_by(block, couplings)) for block in self.blocks
        )

    def moments(self, beta: float, couplings: Couplings = ()) -> Moments:
        """
        Every moment adds over the blocks, freed variables included.

        The weight factorises across independent subsystems, so `log Z` adds;
        and because the blocks are then independent under the product measure,
        so do all the first and second cumulants -- which is also why the heat
        capacity is extensive. A magnet of N spins therefore costs N partition
        functions, not one sum over 2^N states.

        Each block is handed only the couplings its own microstates can report.
        A block carrying no magnetisation is not an error in a magnetic
        ensemble -- it contributes zero to `<M>` and to `Var(M)`, which is
        exactly what a spin glued to a two-level means. Handing every coupling
        to every block would refuse `Spin(1.5) * TwoLevel(1.0)` in a field, a
        system with nothing wrong with it.
        """
        couplings = tuple(couplings)
        self._check_couplings(couplings)
        names = tuple(name for name, _ in couplings)
        parts = [
            block.moments(beta, _carried_by(block, couplings))
            for block in self.blocks
        ]
        return Moments(
            log_z=sum(part.log_z for part in parts),
            energy=sum(part.energy for part in parts),
            energy_variance=sum(part.energy_variance for part in parts),
            means={
                name: sum(part.means.get(name, 0.0) for part in parts)
                for name in names
            },
            variances={
                name: sum(part.variances.get(name, 0.0) for part in parts)
                for name in names
            },
        )


def _carried_by(block: System, couplings: Couplings) -> Couplings:
    """The subset of `couplings` whose variables this block's microstates report."""
    if not couplings:
        return ()
    carried = block.extensive_variables
    return tuple((name, value) for name, value in couplings if name in carried)


def convolve_levels(
    system: System,
    tol: float = _ENERGY_GRID,
    max_distinct: int = _MAX_DISTINCT,
) -> tuple[Level, ...]:
    """
    The joint level structure of a composite, by convolving degeneracies.

    Energies add and degeneracies multiply, so the joint spectrum is the
    discrete convolution `g = g_A * g_B` -- accumulate `(E_A + E_B, g_A g_B)`
    into a table keyed by energy. That is the *only* way a composite ever gets
    an enumerable spectrum, and it is needed exactly where the partition
    function is not enough: counting microstates at fixed energy.

    Convolution rather than a merge over individual states, for two reasons.
    It yields degeneracies directly, with no tolerance-based regrouping pass
    afterwards. And it keeps them as exact Python integers, so `TwoLevel(w) **
    200` gives the binomial coefficients themselves -- `C(200, 100)` is a
    59-digit integer that `math.log` handles and a float cannot hold.

    Energies are rounded onto a grid of width `tol` before grouping: summing
    `w` three times need not give bit-identical `3 * w`, and without the grid a
    single level would fragment into several with degeneracy 1 each.

    Parameters
    ----------
    system : System
        A `SpectralSystem` with finitely many states, or a `CompositeSystem`
        of them.
    tol : float
        Energy grid for grouping.
    max_distinct : int
        Cap on the number of distinct energies. Note it is *different*
        incommensurate blocks that explode, not repeated ones: `A ** n` gives
        multisets of one block's levels, `C(n + d - 1, d - 1)` of them, which
        is polynomial. Distinct blocks that share no common divisor give one
        energy per microstate.

    Returns
    -------
    tuple of Level
        Ascending in energy, with exact integer degeneracies.

    Raises
    ------
    ValueError
        If a block is unbounded, carries extensive variables, or the joint
        spectrum exceeds `max_distinct` distinct energies.
    """
    blocks = system.blocks if isinstance(system, CompositeSystem) else (system,)
    table: dict[int, int] = {0: 1}

    for block in blocks:
        table = _convolve_one(table, _block_levels(block), tol, max_distinct)

    return tuple(
        Level(key * tol, degeneracy)
        for key, degeneracy in sorted(table.items())
    )


def _block_levels(block: System) -> tuple[Level, ...]:
    """
    One block's levels, checked for the guarantees convolution needs.

    No nested-composite case: `CompositeSystem` flattens on construction, so
    `blocks` never contains one.
    """
    if not isinstance(block, SpectralSystem):
        raise ValueError(
            f"convolution needs enumerable levels, and "
            f"{type(block).__name__} has none"
        )
    if block.n_states is None:
        raise ValueError(
            f"{type(block).__name__} has unboundedly many states, so its "
            f"levels cannot be convolved; truncate it first"
        )
    levels = tuple(block.levels())
    if any(level.extensive for level in levels):
        raise ValueError(
            f"{type(block).__name__} carries extensive variables, which "
            f"convolution does not track -- it groups by energy alone"
        )
    return levels


def _convolve_one(
    table: dict[int, int],
    levels: tuple[Level, ...],
    tol: float,
    max_distinct: int,
) -> dict[int, int]:
    """Fold one block's levels into the running energy -> degeneracy table."""
    folded: dict[int, int] = {}
    for key, degeneracy in table.items():
        for level in levels:
            joint = key + round(level.energy / tol)
            folded[joint] = folded.get(joint, 0) + degeneracy * level.degeneracy
    if len(folded) > max_distinct:
        raise ValueError(
            f"the convolved spectrum has more than {max_distinct} distinct "
            f"energies; blocks whose levels share no common divisor put every "
            f"microstate on its own energy, and convolution only pays when "
            f"they collapse together"
        )
    return folded


def total_states(levels: Sequence[Level]) -> int:
    """The exact number of microstates in a convolved spectrum."""
    return sum(level.degeneracy for level in levels)


def log_omega(levels: Sequence[Level], energy: float, tol: float = 1e-9) -> float:
    """
    `log` of the number of microstates at `energy`, or `-inf` if there are none.

    Kept in logs and taken of an exact integer, so a degeneracy far past the
    float range is still reported correctly.
    """
    for level in levels:
        if abs(level.energy - energy) <= tol:
            return math.log(level.degeneracy)
    return -math.inf


__all__ = [
    "CompositeSystem",
    "convolve_levels",
    "log_omega",
    "total_states",
]
