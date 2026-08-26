# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Monte Carlo: the ensemble's second consumer.

An `Ensemble` is a log-weight on a microstate. Summed over a spectrum it gives
the exact thermodynamics in `statphys.state`; compared across a proposed move
it gives the Metropolis acceptance rule here. Same object, two readings -- so
canonical, magnetic and grand canonical sampling differ only in which `Field`s
the ensemble carries, and there is one sampler rather than one per ensemble.

`Sampling` deliberately shares its vocabulary with `ThermalState`: `energy`,
`heat_capacity`, `magnetisation`, `susceptibility`, `mean`, `variance`,
`response`. A sampled estimate and an exact result are then compared directly,
which is how every model here is validated.
"""

from eigora.statphys.monte_carlo.base import Configuration, Proposal
from eigora.statphys.monte_carlo.measure import autocorrelation_time, blocked_error
from eigora.statphys.monte_carlo.metropolis import Sampling, metropolis
from eigora.statphys.monte_carlo.models import (
    FermionGas,
    HeisenbergLattice,
    IsingLattice,
)

__all__ = [
    "Configuration",
    "Proposal",
    "Sampling",
    "metropolis",
    "FermionGas",
    "HeisenbergLattice",
    "IsingLattice",
    "autocorrelation_time",
    "blocked_error",
]
