# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Uniform grids for simulation domains of any dimension.

A grid is a tuple of independent uniform `Axis` objects, so 1D, 2D, 3D and
beyond are the same type. Coordinate arrays are always produced with
`indexing="ij"`, i.e. `coordinates()[k]` varies along axis `k`.
"""

from dataclasses import dataclass, field
from math import prod
from typing import Callable

import numpy as np
from numpy.typing import NDArray

_DEFAULT_NAMES = ("x", "y", "z")


def default_axis_names(ndim: int) -> tuple[str, ...]:
    """Axis names used when none are given: (x,), (x, y), (x, y, z), then (x1, ..., xn)."""
    if ndim < 1:
        raise ValueError(f"ndim must be at least 1, got {ndim}")
    if ndim <= len(_DEFAULT_NAMES):
        return _DEFAULT_NAMES[:ndim]
    return tuple(f"x{i + 1}" for i in range(ndim))


@dataclass(frozen=True)
class Axis:
    """A single uniform axis: `n_points` samples covering [min, max] inclusive."""

    min: float
    max: float
    n_points: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.min >= self.max:
            raise ValueError(
                f"axis {self.name or '?'}: min ({self.min}) must be less than max ({self.max})"
            )
        if self.n_points < 2:
            raise ValueError(
                f"axis {self.name or '?'}: n_points ({self.n_points}) must be at least 2"
            )

    @property
    def values(self) -> NDArray[np.float64]:
        """The sample points."""
        return np.linspace(self.min, self.max, self.n_points)

    @property
    def spacing(self) -> float:
        """Distance between two consecutive samples."""
        return (self.max - self.min) / (self.n_points - 1)

    @property
    def length(self) -> float:
        """Extent of the axis."""
        return self.max - self.min


@dataclass(frozen=True)
class GridND:
    """
    Uniform grid of arbitrary dimension, defined by its axes.

    Example
    -------
    >>> grid = GridND.line(-10.0, 10.0, 512)          # 1D
    >>> grid = GridND.uniform([(-5, 5), (-5, 5)], (128, 128))
    >>> X, Y = grid.coordinates()
    """

    axes: tuple[Axis, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        if not self.axes:
            raise ValueError("a grid needs at least one axis")
        named = [axis.name for axis in self.axes if axis.name]
        if len(set(named)) != len(named):
            raise ValueError(f"axis names must be unique, got {named}")

    # -- construction ------------------------------------------------------

    @classmethod
    def uniform(
        cls,
        bounds: "list[tuple[float, float]] | tuple[tuple[float, float], ...]",
        shape: "int | list[int] | tuple[int, ...]",
        names: "list[str] | tuple[str, ...] | None" = None,
    ) -> "GridND":
        """
        Build a grid from per-axis (min, max) bounds and a number of points.

        Parameters
        ----------
        bounds : sequence of (float, float)
            One (min, max) pair per axis.
        shape : int or sequence of int
            Points per axis; a single int applies to every axis.
        names : sequence of str, optional
            Axis names; defaults to (x, y, z, ...).
        """
        bounds = tuple(bounds)
        ndim = len(bounds)
        if isinstance(shape, int):
            shape = (shape,) * ndim
        shape = tuple(shape)
        if len(shape) != ndim:
            raise ValueError(f"shape has {len(shape)} entries but bounds have {ndim}")
        names = tuple(names) if names is not None else default_axis_names(ndim)
        if len(names) != ndim:
            raise ValueError(f"names has {len(names)} entries but bounds have {ndim}")
        return cls(
            tuple(
                Axis(min=float(lo), max=float(hi), n_points=int(n), name=name)
                for (lo, hi), n, name in zip(bounds, shape, names)
            )
        )

    @classmethod
    def line(
        cls,
        x_min: float,
        x_max: float,
        n_points: int,
        name: str = "x",
    ) -> "GridND":
        """Build a 1D grid on [x_min, x_max]."""
        return cls((Axis(min=float(x_min), max=float(x_max), n_points=int(n_points), name=name),))

    @classmethod
    def cube(
        cls,
        half_width: float,
        n_points: int,
        ndim: int = 3,
        names: "list[str] | tuple[str, ...] | None" = None,
    ) -> "GridND":
        """Build a grid on [-half_width, half_width] in every one of `ndim` axes."""
        return cls.uniform([(-half_width, half_width)] * ndim, n_points, names)

    # -- geometry ----------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Points per axis."""
        return tuple(axis.n_points for axis in self.axes)

    @property
    def size(self) -> int:
        """Total number of grid points."""
        return prod(self.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        """Sample spacing per axis."""
        return tuple(axis.spacing for axis in self.axes)

    @property
    def volume_element(self) -> float:
        """Product of the spacings -- the dV of an integral over the grid."""
        return prod(self.spacing)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """(min, max) per axis."""
        return tuple((axis.min, axis.max) for axis in self.axes)

    @property
    def names(self) -> tuple[str, ...]:
        """Axis names."""
        return tuple(axis.name for axis in self.axes)

    def axis(self, key: "int | str") -> Axis:
        """Look up an axis by index or by name."""
        if isinstance(key, str):
            for axis in self.axes:
                if axis.name == key:
                    return axis
            raise KeyError(f"no axis named {key!r}; grid has {list(self.names)}")
        return self.axes[key]

    def values(self, key: "int | str" = 0) -> NDArray[np.float64]:
        """Sample points of one axis."""
        return self.axis(key).values

    def coordinates(self, sparse: bool = False) -> tuple[NDArray[np.float64], ...]:
        """
        Coordinate arrays of the grid, one per axis, in 'ij' indexing.

        Parameters
        ----------
        sparse : bool
            If True return the open meshgrid (each array of shape
            (..., 1, n_k, 1, ...)), which broadcasts identically but costs far
            less memory in 3D and beyond.
        """
        return tuple(
            np.meshgrid(*(axis.values for axis in self.axes), indexing="ij", sparse=sparse)
        )

    def sub(self, start: int, stop: "int | None" = None) -> "GridND":
        """
        Sub-grid spanning axes [start, stop) -- the grid of a coordinate block.

        Used to give each block of a separable system its own grid.
        """
        stop = self.ndim if stop is None else stop
        axes = self.axes[start:stop]
        if not axes:
            raise ValueError(f"empty sub-grid for axes [{start}, {stop}) of {self.ndim}")
        return GridND(axes)

    def __len__(self) -> int:
        return self.ndim

    def __getitem__(self, key: "int | str") -> Axis:
        return self.axis(key)

    # -- 1D conveniences ---------------------------------------------------

    def _require_1d(self, attribute: str) -> Axis:
        if self.ndim != 1:
            raise ValueError(
                f"'{attribute}' is only defined for a 1D grid, this one is {self.ndim}D; "
                f"use coordinates(), spacing or shape instead"
            )
        return self.axes[0]

    @property
    def x(self) -> NDArray[np.float64]:
        """Sample points (1D grids only)."""
        return self._require_1d("x").values

    @property
    def dx(self) -> float:
        """Sample spacing (1D grids only)."""
        return self._require_1d("dx").spacing

    @property
    def n_points(self) -> int:
        """Number of points (1D grids only); see `size` and `shape` otherwise."""
        return self._require_1d("n_points").n_points

    @property
    def length(self) -> float:
        """Extent of the domain (1D grids only)."""
        return self._require_1d("length").length


def to_spherical(
    grid: GridND,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert a 3D Cartesian grid to spherical coordinates (r, theta, phi).

    Returns
    -------
    r : array
        Radial distances.
    theta : array
        Polar angles (0 <= theta <= pi); undefined at r = 0, where it is 0.
    phi : array
        Azimuthal angles (0 <= phi < 2*pi).
    """
    _require_3d(grid)
    X, Y, Z = grid.coordinates()
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.zeros_like(r)
    nonzero = r > 0
    theta[nonzero] = np.arccos(np.clip(Z[nonzero] / r[nonzero], -1.0, 1.0))
    phi = np.arctan2(Y, X) % (2 * np.pi)
    return r, theta, phi


def to_cartesian(
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def map_spherical(
    grid: GridND,
    r: NDArray[np.float64],
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    f: Callable[..., NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Sample f(r, theta, phi) and bin the values onto a 3D Cartesian grid.

    Each spherical point is assigned to the grid cell containing it; cells no
    point falls into keep the value 0.

    Parameters
    ----------
    grid : GridND
        Target 3D grid.
    r, theta, phi : array
        Flat spherical coordinates to sample.
    f : callable
        Function of (r, theta, phi).
    """
    _require_3d(grid)
    x, y, z = to_cartesian(r, theta, phi)
    values = f(r, theta, phi)

    grid_values = np.zeros(grid.shape, dtype=values.dtype)
    indices = []
    for coord, axis in zip((x, y, z), grid.axes):
        scaled = (coord - axis.min) / axis.length * (axis.n_points - 1)
        indices.append(scaled.astype(int))

    xi, yi, zi = indices
    inside = (
        (xi >= 0) & (xi < grid.shape[0])
        & (yi >= 0) & (yi < grid.shape[1])
        & (zi >= 0) & (zi < grid.shape[2])
    )
    grid_values[xi[inside], yi[inside], zi[inside]] = values[inside]
    return grid_values


def _require_3d(grid: GridND) -> None:
    if grid.ndim != 3:
        raise ValueError(f"spherical coordinates need a 3D grid, got {grid.ndim}D")


__all__ = [
    "Axis",
    "GridND",
    "default_axis_names",
    "to_spherical",
    "to_cartesian",
    "map_spherical",
]
