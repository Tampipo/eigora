# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Time evolution of quantum states.

`split_step` implements the split-step Fourier propagator. Its entry points are
re-exported here, so `from eigora.qm.evolution import evolve` keeps working.

`trajectory` reduces an evolution to expectation values, and integrates the
matching classical path for comparison.
"""

from eigora.qm.evolution.split_step import Evolution, evolve
from eigora.qm.evolution.trajectory import (
    QuantumTrajectory,
    classical_trajectory,
    quantum_trajectory,
)

__all__ = [
    "Evolution",
    "evolve",
    "QuantumTrajectory",
    "quantum_trajectory",
    "classical_trajectory",
]
