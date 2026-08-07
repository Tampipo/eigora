# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from physense_utils.grids import (
    Axis,
    GridND,
    default_axis_names,
    to_spherical,
    to_cartesian,
    map_spherical,
)


class TestAxis:
    def test_values(self):
        axis = Axis(min=0.0, max=1.0, n_points=101)
        assert len(axis.values) == 101
        assert axis.values[0] == pytest.approx(0.0)
        assert axis.values[-1] == pytest.approx(1.0)

    def test_spacing(self):
        assert Axis(min=0.0, max=1.0, n_points=11).spacing == pytest.approx(0.1)

    def test_length(self):
        assert Axis(min=-5.0, max=5.0, n_points=100).length == pytest.approx(10.0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Axis(min=1.0, max=0.0, n_points=10)

    def test_equal_bounds(self):
        with pytest.raises(ValueError):
            Axis(min=1.0, max=1.0, n_points=10)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            Axis(min=0.0, max=1.0, n_points=1)

    def test_immutable(self):
        axis = Axis(min=0.0, max=1.0, n_points=10)
        with pytest.raises(Exception):
            axis.min = 2.0


class TestDefaultAxisNames:
    def test_named_axes_up_to_3d(self):
        assert default_axis_names(1) == ("x",)
        assert default_axis_names(2) == ("x", "y")
        assert default_axis_names(3) == ("x", "y", "z")

    def test_numbered_axes_beyond_3d(self):
        assert default_axis_names(5) == ("x1", "x2", "x3", "x4", "x5")

    def test_invalid_ndim(self):
        with pytest.raises(ValueError):
            default_axis_names(0)


class TestConstruction:
    def test_line(self):
        grid = GridND.line(-5.0, 5.0, 128)
        assert grid.ndim == 1
        assert grid.shape == (128,)
        assert grid.names == ("x",)

    def test_uniform_with_per_axis_points(self):
        grid = GridND.uniform([(0.0, 1.0), (0.0, 2.0)], (5, 7))
        assert grid.shape == (5, 7)
        assert grid.bounds == ((0.0, 1.0), (0.0, 2.0))
        assert grid.names == ("x", "y")

    def test_uniform_with_shared_point_count(self):
        grid = GridND.uniform([(0.0, 1.0)] * 3, 11)
        assert grid.shape == (11, 11, 11)

    def test_uniform_beyond_3d(self):
        grid = GridND.uniform([(-1.0, 1.0)] * 5, 4)
        assert grid.ndim == 5
        assert grid.names == ("x1", "x2", "x3", "x4", "x5")

    def test_cube(self):
        grid = GridND.cube(half_width=2.0, n_points=9, ndim=3)
        assert grid.shape == (9, 9, 9)
        assert grid.bounds == ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0))

    def test_custom_names(self):
        grid = GridND.uniform([(0.0, 1.0), (0.0, 1.0)], 4, names=("u", "v"))
        assert grid.names == ("u", "v")

    def test_shape_length_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            GridND.uniform([(0.0, 1.0), (0.0, 1.0)], (5, 7, 9))

    def test_names_length_mismatch(self):
        with pytest.raises(ValueError, match="names"):
            GridND.uniform([(0.0, 1.0), (0.0, 1.0)], 5, names=("u",))

    def test_no_axes(self):
        with pytest.raises(ValueError, match="at least one axis"):
            GridND(())

    def test_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            GridND.uniform([(0.0, 1.0), (0.0, 1.0)], 5, names=("u", "u"))

    def test_axes_normalised_to_tuple(self):
        grid = GridND([Axis(0.0, 1.0, 5, "x")])
        assert isinstance(grid.axes, tuple)

    def test_invalid_bounds_propagate(self):
        with pytest.raises(ValueError):
            GridND.uniform([(1.0, 0.0)], 10)

    def test_too_few_points_propagate(self):
        with pytest.raises(ValueError):
            GridND.uniform([(0.0, 1.0), (0.0, 1.0)], (10, 1))


class TestGeometry:
    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    def test_coordinate_shapes(self, ndim):
        grid = GridND.uniform([(0.0, 1.0)] * ndim, 4)
        coords = grid.coordinates()
        assert len(coords) == ndim
        assert all(c.shape == grid.shape for c in coords)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_sparse_coordinates_broadcast_to_dense(self, ndim):
        grid = GridND.uniform([(0.0, 1.0)] * ndim, 4)
        sparse = grid.coordinates(sparse=True)
        dense = grid.coordinates()
        for s, d in zip(sparse, dense):
            assert np.allclose(np.broadcast_to(s, grid.shape), d)

    def test_ij_indexing(self):
        grid = GridND.uniform([(0.0, 1.0), (0.0, 2.0)], (5, 7))
        X, Y = grid.coordinates()
        assert X.shape == (5, 7)
        assert np.allclose(X[:, 0], grid.values(0))
        assert np.allclose(Y[0, :], grid.values(1))

    def test_spacing_and_volume_element(self):
        grid = GridND.uniform([(0.0, 1.0), (0.0, 2.0)], (11, 21))
        assert grid.spacing == pytest.approx((0.1, 0.1))
        assert grid.volume_element == pytest.approx(0.01)

    def test_size(self):
        grid = GridND.uniform([(0.0, 1.0)] * 3, (4, 5, 6))
        assert grid.size == 120
        assert len(grid) == 3

    def test_axis_lookup(self):
        grid = GridND.uniform([(0.0, 1.0), (0.0, 2.0)], 5)
        assert grid.axis("y").max == pytest.approx(2.0)
        assert grid["y"] is grid.axes[1]
        assert grid.axis(0) is grid.axes[0]

    def test_unknown_axis_name(self):
        with pytest.raises(KeyError):
            GridND.line(0.0, 1.0, 5).axis("y")

    def test_sub_grid(self):
        grid = GridND.uniform([(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)], (4, 5, 6))
        block = grid.sub(1, 3)
        assert block.ndim == 2
        assert block.shape == (5, 6)
        assert block.bounds == ((0.0, 2.0), (0.0, 3.0))
        assert grid.sub(2).shape == (6,)

    def test_empty_sub_grid(self):
        with pytest.raises(ValueError, match="empty sub-grid"):
            GridND.line(0.0, 1.0, 5).sub(1)

    def test_immutable(self):
        grid = GridND.line(0.0, 1.0, 10)
        with pytest.raises(Exception):
            grid.axes = ()


class TestOneDimensionalConveniences:
    def test_accessors(self):
        grid = GridND.line(0.0, 1.0, 11)
        assert len(grid.x) == 11
        assert grid.dx == pytest.approx(0.1)
        assert grid.n_points == 11
        assert grid.length == pytest.approx(1.0)

    @pytest.mark.parametrize("attribute", ["x", "dx", "n_points", "length"])
    def test_rejected_beyond_1d(self, attribute):
        grid = GridND.uniform([(0.0, 1.0), (0.0, 1.0)], 5)
        with pytest.raises(ValueError, match="only defined for a 1D grid"):
            getattr(grid, attribute)


class TestSpherical:
    @pytest.fixture
    def grid(self):
        return GridND.cube(half_width=1.0, n_points=3, ndim=3)

    def test_to_spherical(self, grid):
        r, theta, phi = to_spherical(grid)

        assert r.shape == (3, 3, 3)
        assert theta.shape == (3, 3, 3)
        assert phi.shape == (3, 3, 3)

        assert r[1, 1, 1] == pytest.approx(0.0)  # centre
        assert theta[2, 1, 1] == pytest.approx(np.pi / 2)  # on the xy-plane
        assert phi[1, 2, 1] == pytest.approx(np.pi / 2)  # on the y-axis

    def test_to_cartesian_round_trip(self, grid):
        r, theta, phi = to_spherical(grid)
        x, y, z = to_cartesian(r, theta, phi)
        X, Y, Z = grid.coordinates()

        assert np.allclose(x, X)
        assert np.allclose(y, Y)
        assert np.allclose(z, Z)

    def test_map_spherical(self):
        grid = GridND.cube(half_width=1.0, n_points=5, ndim=3)
        r, theta, phi = to_spherical(grid)

        def f(r, theta, phi):
            return r * np.sin(theta) * np.cos(phi)

        values = map_spherical(grid, r.flatten(), theta.flatten(), phi.flatten(), f)

        assert values.shape == (5, 5, 5)
        assert values[2, 2, 2] == pytest.approx(0.0)  # centre

    @pytest.mark.parametrize("ndim", [1, 2, 4])
    def test_requires_3d(self, ndim):
        grid = GridND.uniform([(-1.0, 1.0)] * ndim, 3)
        with pytest.raises(ValueError, match="3D grid"):
            to_spherical(grid)
