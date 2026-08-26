# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Systems: whatever supplies energies to statistical mechanics.

`base` holds the three-layer interface -- a partition function, then an
enumerable spectrum, then variable extensive parameters. `known` is the
catalogue of named systems, and `composite` puts them side by side with `*`
and `**`.
"""

from eigora.statphys.systems.base import (
    Level,
    Moments,
    ParametrisedSystem,
    SpectralSystem,
    System,
)
from eigora.statphys.systems.composite import CompositeSystem
from eigora.statphys.systems.identical import (
    BOLTZMANN,
    BOSE,
    FERMI,
    IdenticalParticles,
    Statistics,
    WithDegeneracy,
    with_degeneracy,
)
from eigora.statphys.systems.known import (
    Box1D,
    Degenerate,
    HarmonicMode,
    IdealGas,
    NLevel,
    Rotor,
    Spin,
    TwoLevel,
    particle_in_box,
)

__all__ = [
    "Level",
    "Moments",
    "System",
    "SpectralSystem",
    "ParametrisedSystem",
    "NLevel",
    "TwoLevel",
    "Degenerate",
    "HarmonicMode",
    "IdealGas",
    "IdenticalParticles",
    "Statistics",
    "WithDegeneracy",
    "with_degeneracy",
    "FERMI",
    "BOSE",
    "BOLTZMANN",
    "Spin",
    "Rotor",
    "Box1D",
    "particle_in_box",
    "CompositeSystem",
]
