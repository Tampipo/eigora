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

from collections.abc import Sequence
from dataclasses import dataclass

from eigora.statphys.systems.base import Couplings, Moments, System


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

    def log_z(self, beta: float, couplings: Couplings = ()) -> float:
        return sum(block.log_z(beta, couplings) for block in self.blocks)

    def moments(self, beta: float, couplings: Couplings = ()) -> Moments:
        """
        Every moment adds over the blocks, freed variables included.

        The weight factorises across independent subsystems, so `log Z` adds;
        and because the blocks are then independent under the product measure,
        so do all the first and second cumulants -- which is also why the heat
        capacity is extensive. A magnet of N spins therefore costs N partition
        functions, not one sum over 2^N states.
        """
        couplings = tuple(couplings)
        parts = [block.moments(beta, couplings) for block in self.blocks]
        names = tuple(name for name, _ in couplings)
        return Moments(
            log_z=sum(part.log_z for part in parts),
            energy=sum(part.energy for part in parts),
            energy_variance=sum(part.energy_variance for part in parts),
            means={name: sum(part.means[name] for part in parts) for name in names},
            variances={
                name: sum(part.variances[name] for part in parts) for name in names
            },
        )


__all__ = ["CompositeSystem"]
