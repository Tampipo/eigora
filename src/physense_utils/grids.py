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

__all__ = ["Grid1D", "Grid2D", "Grid3D"]



