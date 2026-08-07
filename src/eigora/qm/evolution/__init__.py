# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Time evolution of quantum states.

`split_step` implements the split-step Fourier propagator. Its entry points are
re-exported here, so `from eigora.qm.evolution import evolve` keeps working.
"""

from eigora.qm.evolution.split_step import Evolution, evolve

__all__ = ["Evolution", "evolve"]
