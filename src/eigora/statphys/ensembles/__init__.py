# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ensembles: how the microstates of a system are weighted, and what follows.

`base` holds the general form -- an ensemble is a log-weight, `-beta[E - sum
f_i X_i]`, and nothing more. `known` names the standard choices of which
variables to free, and `state` binds a system to an ensemble and reads the
thermodynamics off it.
"""

from eigora.statphys.ensembles.base import Ensemble, Field
from eigora.statphys.ensembles.known import Canonical, Generalised, Magnetic
from eigora.statphys.ensembles.state import ThermalState, equilibrium

__all__ = [
    "Field",
    "Ensemble",
    "Canonical",
    "Magnetic",
    "Generalised",
    "ThermalState",
    "equilibrium",
]
