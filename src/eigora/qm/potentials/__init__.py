# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Quantum potentials.

`known` holds the catalogue of analytic 1D potentials (harmonic well, barrier,
double well, ...). `base`, `generic` and `separable` generalise to any
dimension: wrap an arbitrary callable with `GenericPotential`, or -- when the
system splits into independent sub-systems -- combine known pieces with
`SeparablePotential`, which keeps the structure that `eigora.qm.spectra`
needs to solve it.
"""

from eigora.qm.potentials.known import (
    Potential,
    CompositePotential,
    FreeParticle,
    HarmonicWell,
    InfiniteSquareWell,
    FiniteSquareWell,
    RectangularBarrier,
    PotentialStep,
    DoubleWell,
)
from eigora.qm.potentials.base import PotentialND, SumPotential
from eigora.qm.potentials.generic import GenericPotential
from eigora.qm.potentials.separable import Block1D, SeparablePotential, as_block

__all__ = [
    "Potential",
    "CompositePotential",
    "FreeParticle",
    "HarmonicWell",
    "InfiniteSquareWell",
    "FiniteSquareWell",
    "RectangularBarrier",
    "PotentialStep",
    "DoubleWell",
    "PotentialND",
    "SumPotential",
    "GenericPotential",
    "Block1D",
    "SeparablePotential",
    "as_block",
]
