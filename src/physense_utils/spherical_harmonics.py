# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Spherical harmonics implementation.
"""

from scipy.special import sph_harm_y

def spherical_harmonic(l: int, m: int, theta: float, phi: float) -> complex:
    """
    Compute the value of the spherical harmonic Y_l^m(theta, phi).

    Parameters
    ----------
    l : int
        Degree of the spherical harmonic (l >= 0).
    m : int
        Order of the spherical harmonic (-l <= m <= l).
    theta : float
        Polar angle in radians (0 <= theta <= pi).
    phi : float
        Azimuthal angle in radians (0 <= phi < 2*pi).

    Returns
    -------
    complex
        Value of the spherical harmonic Y_l^m(theta, phi).
    """

    if l < 0:
        raise ValueError(f"Degree l must be non-negative, got {l}")
    if abs(m) > l:
        raise ValueError(f"Order m must satisfy -l <= m <= l, got m={m}, l={l}")
    
    return sph_harm_y(l, m, theta, phi) # Physicists convention


__all__ = ["spherical_harmonic"]