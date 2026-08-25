# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ensembles: how the microstates of a system are weighted.

`base` holds the general form -- an ensemble is a log-weight, `-beta[E - sum
f_i X_i]`, and nothing more. `known` names the standard choices of which
variables to free.

Binding an ensemble to a system lives one level up, in `statphys.state`: a
`ThermalState` belongs to neither half, and an ensemble is read by more than
one consumer -- the exact thermodynamics here, and the Metropolis acceptance
rule in `statphys.monte_carlo` later.
"""

from eigora.statphys.ensembles.base import Ensemble, Field
from eigora.statphys.ensembles.known import (
    Canonical,
    Generalised,
    IsothermalIsobaric,
    Magnetic,
)

__all__ = [
    "Field",
    "Ensemble",
    "Canonical",
    "Magnetic",
    "IsothermalIsobaric",
    "Generalised",
]
