# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from physense_utils.grids import Grid1D, Grid2D, Grid3D


class TestGrid1D:
    def test_basic(self):
        g = Grid1D(x_min=0.0, x_max=1.0, n_points=101)
        assert len(g.x) == 101
        assert g.x[0] == pytest.approx(0.0)
        assert g.x[-1] == pytest.approx(1.0)

    def test_dx(self):
        g = Grid1D(x_min=0.0, x_max=1.0, n_points=11)
        assert g.dx == pytest.approx(0.1)

    def test_length(self):
        g = Grid1D(x_min=-5.0, x_max=5.0, n_points=100)
        assert g.length == pytest.approx(10.0)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Grid1D(x_min=1.0, x_max=0.0, n_points=10)

    def test_equal_bounds(self):
        with pytest.raises(ValueError):
            Grid1D(x_min=1.0, x_max=1.0, n_points=10)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            Grid1D(x_min=0.0, x_max=1.0, n_points=1)

    def test_immutable(self):
        g = Grid1D(x_min=0.0, x_max=1.0, n_points=10)
        with pytest.raises(Exception):
            g.x_min = 2.0


class TestGrid2D:
    def test_basic(self):
        g = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=11, ny=21)
        assert len(g.x) == 11
        assert len(g.y) == 21

    def test_meshgrid_shape(self):
        g = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=2.0, nx=5, ny=7)
        X, Y = g.meshgrid
        assert X.shape == (5, 7)
        assert Y.shape == (5, 7)

    def test_dx_dy(self):
        g = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=2.0, nx=11, ny=21)
        assert g.dx == pytest.approx(0.1)
        assert g.dy == pytest.approx(0.1)

    def test_invalid_x_bounds(self):
        with pytest.raises(ValueError):
            Grid2D(x_min=1.0, x_max=0.0, y_min=0.0, y_max=1.0, nx=10, ny=10)

    def test_invalid_y_bounds(self):
        with pytest.raises(ValueError):
            Grid2D(x_min=0.0, x_max=1.0, y_min=1.0, y_max=0.0, nx=10, ny=10)

    def test_too_few_nx(self):
        with pytest.raises(ValueError):
            Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=1, ny=10)


class TestGrid3D:
    def test_basic(self):
        g = Grid3D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0, nx=11, ny=21, nz=31)
        assert len(g.x) == 11
        assert len(g.y) == 21
        assert len(g.z) == 31

    def test_meshgrid_shape(self):
        g = Grid3D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=2.0, z_min=0.0, z_max=3.0, nx=5, ny=7, nz=9)
        X, Y, Z = g.meshgrid
        assert X.shape == (5, 7, 9)
        assert Y.shape == (5, 7, 9)
        assert Z.shape == (5, 7, 9)

    def test_dx_dy_dz(self):
        g = Grid3D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=2.0, z_min=0.0, z_max=3.0, nx=11, ny=21, nz=31)
        assert g.dx == pytest.approx(0.1)
        assert g.dy == pytest.approx(0.1)
        assert g.dz == pytest.approx(0.1)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Grid3D(x_min=1.0, x_max=0.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0, nx=10, ny=10, nz=10)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            Grid3D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0,
                   nx=10, ny=10, nz=1)

    def test_to_spherical(self):
        g = Grid3D(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=1.0,
                   nx=3, ny=3, nz=3)
        r, theta, phi = g.to_spherical()

        # Check shapes
        assert r.shape == (3, 3, 3)
        assert theta.shape == (3, 3, 3)
        assert phi.shape == (3, 3, 3)

        # Check known values
        assert r[1, 1, 1] == pytest.approx(0.0)  # Center point
        assert theta[2, 1, 1] == pytest.approx(np.pi / 2)  # Point on xy-plane
        assert phi[1, 2, 1] == pytest.approx(np.pi / 2)  # Point on y-axis