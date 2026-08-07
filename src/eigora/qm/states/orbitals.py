# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from dataclasses import dataclass
import numpy as np
from scipy.special import eval_genlaguerre
from eigora.spherical_harmonics import spherical_harmonic
from eigora.grids import GridND, to_spherical, map_spherical
"""
Atomic orbitals for hydrogen-like atoms, expressed in spherical coordinates (r, theta, phi).
"""



class SingleAtomState:
    """
    A single atom system defined by its atomic number Z.

    Example
    -------
    >>> from eigora.qm import SingleAtomState
    >>> atom = SingleAtomState(Z=1, n=1, l=0, m=0)  # Hydrogen atom
    >>> r = 1.0
    >>> theta = 0.5
    >>> phi = 1.0
    >>> l = 0
    >>> m = 0
    >>> orbital_value = atom.orbital(r, theta, phi, l, m)
    """

    def __init__(self, Z: int, n: int = 1, l: int = 0, m: int = 0):
        self.Z = Z #Atomic number
        self.n = n #Principal quantum number
        self.l = l #Orbital angular momentum quantum number
        self.m = m #Magnetic quantum number

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"Principal quantum number n must be >= 1, got {value}")
        self._n = value

    @property
    def l(self) -> int:
        return self._l

    @l.setter
    def l(self, value: int) -> None:
        if value < 0 or value >= self.n:
            raise ValueError(f"Orbital angular momentum quantum number l must satisfy 0 <= l < n, got l={value}, n={self.n}")
        self._l = value

    @property
    def m(self) -> int:
        return self._m

    @m.setter
    def m(self, value: int) -> None:
        if abs(value) > self.l:
            raise ValueError(f"Magnetic quantum number m must satisfy -l <= m <= l, got m={value}, l={self.l}")
        self._m = value

    def _orbital(self, r: float, theta: float, phi: float) -> complex:
        """
        Compute the value of the hydrogen-like atomic orbital at given spherical coordinates (r, theta, phi).

        Parameters
        ----------
        r : float
            Radial distance from the nucleus.
        theta : float
            Polar angle in radians (0 <= theta <= pi).
        phi : float
            Azimuthal angle in radians (0 <= phi < 2*pi).

        Returns
        -------
        complex
            Value of the atomic orbital at (r, theta, phi).
        """
        # Radial part R_{nl}(r)
        rho = 2 * self.Z * r / self.n

        radial_part = (
            np.sqrt((2 * self.Z / self.n)**3 * math.factorial(self.n - self.l - 1)
                    / (2 * self.n * math.factorial(self.n + self.l)))
            * np.exp(-rho / 2) * rho**self.l
            * eval_genlaguerre(self.n - self.l - 1, 2 * self.l + 1, rho)
        )

        # Angular part Y_{lm}(theta, phi)
        angular_part = spherical_harmonic(self.l, self.m, theta, phi)

        return radial_part * angular_part

    def __call__(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Evaluate the atomic orbital on a grid of spherical coordinates.

        Parameters
        ----------
        r : np.ndarray
            Radial distances from the nucleus.
        theta : np.ndarray
            Polar angles in radians.
        phi : np.ndarray
            Azimuthal angles in radians.

        Returns
        -------
        np.ndarray
            Values of the atomic orbital at the given coordinates.
        """
        return self._orbital(r, theta, phi)

    def density(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute the probability density of the atomic orbital on a grid of spherical coordinates.

        Parameters
        ----------
        r : np.ndarray
            Radial distances from the nucleus.
        theta : np.ndarray
            Polar angles in radians.
        phi : np.ndarray
            Azimuthal angles in radians.

        Returns
        -------
        np.ndarray
            Probability density of the atomic orbital at the given coordinates.
        """
        orbital_values = self._orbital(r, theta, phi)
        return np.abs(orbital_values)**2

    def density_on_grid(self, grid: GridND) -> np.ndarray:
        """
        Compute the probability density of the atomic orbital on a 3D grid.

        Parameters
        ----------
        grid : GridND
            A 3D Cartesian grid.

        Returns
        -------
        np.ndarray
            Probability density of the atomic orbital on the 3D grid.
        """
        r, theta, phi = to_spherical(grid)
        return map_spherical(grid, r.flatten(), theta.flatten(), phi.flatten(), self.density)

    def wavefunction(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute the real part of the atomic orbital wavefunction on a grid of
        spherical coordinates — unlike `density`, this keeps the sign/phase,
        which is what gives each lobe of a p/d/f orbital its alternating
        color in a standard orbital diagram.

        Parameters
        ----------
        r : np.ndarray
            Radial distances from the nucleus.
        theta : np.ndarray
            Polar angles in radians.
        phi : np.ndarray
            Azimuthal angles in radians.

        Returns
        -------
        np.ndarray
            Real part of the atomic orbital wavefunction at the given coordinates.
        """
        return self._orbital(r, theta, phi).real

    def wavefunction_on_grid(self, grid: GridND) -> np.ndarray:
        """
        Compute the real part of the atomic orbital wavefunction on a 3D grid.

        Parameters
        ----------
        grid : GridND
            A 3D Cartesian grid.

        Returns
        -------
        np.ndarray
            Real part of the atomic orbital wavefunction on the 3D grid.
        """
        r, theta, phi = to_spherical(grid)
        return map_spherical(grid, r.flatten(), theta.flatten(), phi.flatten(), self.wavefunction)


__all__ = ["SingleAtomState"]