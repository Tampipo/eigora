# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Numerical solvers for the Schrödinger equation.

`eigensolver` discretises the 1D Hamiltonian by finite differences and finds
its lowest eigenpairs.
"""

from eigora.qm.solvers.eigensolver import EigenSolution, solve_eigenstates

__all__ = ["EigenSolution", "solve_eigenstates"]
