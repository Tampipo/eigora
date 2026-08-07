# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fundamental physical constants in SI units.
Values from CODATA 2022.
"""

# Planck constant (J·s)
H = 6.62607015e-34

# Reduced Planck constant (J·s)
HBAR = 1.054571817e-34

# Speed of light in vacuum (m/s)
C = 299792458.0

# Elementary charge (C)
E = 1.602176634e-19

# Electron mass (kg)
M_E = 9.1093837015e-31

# Proton mass (kg)
M_P = 1.67262192369e-27

# Boltzmann constant (J/K)
K_B = 1.380649e-23

# Avogadro constant (mol⁻¹)
N_A = 6.02214076e23

# Vacuum permittivity (F/m)
EPSILON_0 = 8.8541878128e-12

# Vacuum permeability (H/m)
MU_0 = 1.25663706212e-6

# Gravitational constant (m³/kg/s²)
G = 6.67430e-11

# Fine structure constant (dimensionless)
ALPHA = 7.2973525693e-3

# Bohr radius (m)
A_0 = 5.29177210903e-11

# Electron volt in joules (J)
EV = 1.602176634e-19

__all__ = [
    "H",
    "HBAR",
    "C",
    "E",
    "M_E",
    "M_P",
    "K_B",
    "N_A",
    "EPSILON_0",
    "MU_0",
    "G",
    "ALPHA",
    "A_0",
    "EV",
]