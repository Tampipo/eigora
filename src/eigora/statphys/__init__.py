# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Equilibrium statistical mechanics.

Two halves that meet at `equilibrium`. A **system** says what the microstates
are and what they cost: a two-level system, a harmonic mode, a rigid rotor, or
any of them multiplied together with `*` and `**`. An **ensemble** says how
those microstates are weighted -- which extensive variables are held fixed, and
which are set free in exchange for a conjugate field. Binding the two gives a
`ThermalState`, which is where free energies, entropies, heat capacities and
susceptibilities live.

Keeping systems and ensembles apart is the point: the same system can be
studied in several ensembles, and comparing them is most of what the subject
is about.

    from eigora.statphys import Canonical, HarmonicMode, equilibrium

    state = equilibrium(HarmonicMode(omega=1.0), Canonical(temperature=0.5))
    state.energy            # 0.582, the closed form (omega/2) coth(beta omega/2)
    state.heat_capacity     # 0.171

> **Units:** k_B = 1, so temperature *is* an energy and beta = 1/T. Entropy is
> dimensionless. This matches the atomic units (hbar = m = 1) used everywhere
> else in the package, so nothing here imports `constants`.
"""

from eigora.statphys.systems import (
    BOLTZMANN,
    BOSE,
    FERMI,
    Box1D,
    CompositeSystem,
    Degenerate,
    HarmonicMode,
    IdealGas,
    IdenticalParticles,
    Level,
    Moments,
    NLevel,
    ParametrisedSystem,
    Rotor,
    SpectralSystem,
    Statistics,
    WithDegeneracy,
    Spin,
    System,
    TwoLevel,
    particle_in_box,
    with_degeneracy,
)
from eigora.statphys.state import ThermalState, equilibrium
from eigora.statphys.ensembles import (
    Canonical,
    Ensemble,
    Field,
    Generalised,
    GrandCanonical,
    IsothermalIsobaric,
    Magnetic,
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
    "Field",
    "Ensemble",
    "Canonical",
    "Magnetic",
    "GrandCanonical",
    "IsothermalIsobaric",
    "Generalised",
    "ThermalState",
    "equilibrium",
]
