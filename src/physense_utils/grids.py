# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Uniform grids for 1D and 2D simulation domains.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D grid on [x_min, x_max] with n_points points."""

    x_min: float
    x_max: float
    n_points: int

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.n_points < 2:
            raise ValueError(f"n_points ({self.n_points}) must be at least 2")

    @property
    def x(self) -> NDArray[np.float64]:
        return np.linspace(self.x_min, self.x_max, self.n_points)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.n_points - 1)

    @property
    def length(self) -> float:
        return self.x_max - self.x_min


@dataclass(frozen=True)
class Grid2D:
    """Uniform 2D grid on [x_min, x_max] x [y_min, y_max]."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")
        if self.nx < 2:
            raise ValueError(f"nx ({self.nx}) must be at least 2")
        if self.ny < 2:
            raise ValueError(f"ny ({self.ny}) must be at least 2")

    @property
    def x(self) -> NDArray[np.float64]:
        return np.linspace(self.x_min, self.x_max, self.nx)

    @property
    def y(self) -> NDArray[np.float64]:
        return np.linspace(self.y_min, self.y_max, self.ny)

    @property
    def meshgrid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.ny - 1)

@dataclass(frozen=True)
class Grid3D:
    """Uniform 3D grid on [x_min, x_max] x [y_min, y_max] x [z_min, z_max]."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be less than z_max ({self.z_max})")
        if self.nx < 2:
            raise ValueError(f"nx ({self.nx}) must be at least 2")
        if self.ny < 2:
            raise ValueError(f"ny ({self.ny}) must be at least 2")
        if self.nz < 2:
            raise ValueError(f"nz ({self.nz}) must be at least 2")

    @property
    def x(self) -> NDArray[np.float64]:
        return np.linspace(self.x_min, self.x_max, self.nx)

    @property
    def y(self) -> NDArray[np.float64]:
        return np.linspace(self.y_min, self.y_max, self.ny)

    @property
    def z(self) -> NDArray[np.float64]:
        return np.linspace(self.z_min, self.z_max, self.nz)

    @property
    def meshgrid(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.ny - 1)

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / (self.nz - 1)

    def to_spherical(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Convert the 3D Cartesian grid to spherical coordinates (r, theta, phi).

        Returns
        -------
        r : array
            Radial distances.
        theta : array
            Polar angles (0 <= theta <= pi).
        phi : array
            Azimuthal angles (0 <= phi < 2*pi).
        """
        X, Y, Z = self.meshgrid
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.zeros_like(r)
        nonzero = r > 0
        theta[nonzero] = np.arccos(np.clip(Z[nonzero] / r[nonzero], -1.0, 1.0))  # theta undefined at r=0, default to 0
        phi = np.arctan2(Y, X) % (2 * np.pi)
        return r, theta, phi

    @staticmethod
    def to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

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
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Cartesian coordinates (x, y, z).
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    
    def to_grid(self,r: np.ndarray, theta: np.ndarray, phi: np.ndarray, f: callable) -> np.ndarray:
        """
        Convert a function defined in spherical coordinates to a 3D grid.

        Parameters
        ----------
        r : np.ndarray
            Radial distances from the nucleus.
        theta : np.ndarray
            Polar angles in radians.
        phi : np.ndarray
            Azimuthal angles in radians.
        f : callable
            Function defined in spherical coordinates (r, theta, phi).
        grid : Grid3D
            3D grid to which the function will be mapped.
        """
        x, y, z = self.to_cartesian(r, theta, phi)
        values = f(r, theta, phi)

        # Create an empty grid
        grid_values = np.zeros((self.nx, self.ny, self.nz), dtype=values.dtype)

        # Map the spherical coordinates to the grid indices
        for i in range(len(r)):
            xi = int((x[i] - self.x_min) / (self.x_max - self.x_min) * (self.nx - 1))
            yi = int((y[i] - self.y_min) / (self.y_max - self.y_min) * (self.ny - 1))
            zi = int((z[i] - self.z_min) / (self.z_max - self.z_min) * (self.nz - 1))

            if 0 <= xi < self.nx and 0 <= yi < self.ny and 0 <= zi < self.nz:
                grid_values[xi, yi, zi] = values[i]

        return grid_values


__all__ = ["Grid1D", "Grid2D", "Grid3D"]



