# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Quantum potentials.

`known` holds the catalogue of analytic 1D potentials (harmonic well, barrier,
double well, ...). Everything is re-exported here, so
`from physense_qm.potentials import HarmonicWell` keeps working.
"""

from physense_qm.potentials.known import (
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
]
