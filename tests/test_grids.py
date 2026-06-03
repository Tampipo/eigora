# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from physense_utils.grids import Grid1D, Grid2D


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
