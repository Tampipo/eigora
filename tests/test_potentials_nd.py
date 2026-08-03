# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np

from physense_utils.grids import GridND

from physense_qm.potentials import (
    Block1D,
    DoubleWell,
    GenericPotential,
    HarmonicWell,
    RectangularBarrier,
    SeparablePotential,
    SumPotential,
    as_block,
)
from physense_qm.solvers import solve_eigenstates


class TestCallConvention:
    def test_wrong_number_of_coordinates(self):
        V = GenericPotential(lambda x, y: x + y, ndim=2)
        with pytest.raises(ValueError, match="2 coordinate"):
            V(np.array([0.0]))

    def test_result_broadcast_to_common_shape(self):
        V = GenericPotential(lambda x, y: np.float64(3.0), ndim=2)
        x = np.linspace(-1, 1, 4)[:, None]
        y = np.linspace(-1, 1, 7)[None, :]
        assert V(x, y).shape == (4, 7)
        assert np.all(V(x, y) == 3.0)

    def test_sparse_and_dense_coordinates_agree(self):
        V = SeparablePotential([HarmonicWell(omega=1.0), HarmonicWell(omega=2.0)])
        grid = GridND.uniform([(-2.0, 2.0), (-3.0, 3.0)], (8, 5))
        assert np.allclose(V.on_grid(grid, sparse=True), V.on_grid(grid))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="ndim"):
            GenericPotential(lambda x: x, ndim=0)

    def test_non_callable(self):
        with pytest.raises(TypeError):
            GenericPotential(3.0, ndim=1)


class TestOnGrid:
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_shape_matches_grid(self, ndim):
        V = SeparablePotential([HarmonicWell(omega=1.0)] * ndim)
        grid = GridND.cube(half_width=2.0, n_points=6, ndim=ndim)
        assert V.on_grid(grid).shape == grid.shape

    def test_dimension_mismatch(self):
        V = SeparablePotential([HarmonicWell(omega=1.0)] * 2)
        with pytest.raises(ValueError, match="2D but the grid is 1D"):
            V.on_grid(GridND.line(-1.0, 1.0, 8))

    def test_values_match_direct_evaluation(self):
        V = SeparablePotential([HarmonicWell(omega=1.0), HarmonicWell(omega=2.0)])
        grid = GridND.uniform([(-2.0, 2.0), (-2.0, 2.0)], (16, 16))
        X, Y = grid.coordinates()
        assert np.allclose(V.on_grid(grid), 0.5 * X**2 + 0.5 * 4 * Y**2)


class TestBlock1D:
    def test_wraps_known_potential(self):
        block = Block1D(HarmonicWell(omega=2.0))
        assert block.ndim == 1
        assert block(1.0) == pytest.approx(2.0)

    def test_keeps_the_original(self):
        well = HarmonicWell(omega=1.0)
        assert Block1D(well).potential is well

    def test_as_block_passes_through_potential_nd(self):
        V = GenericPotential(lambda x, y: x + y, ndim=2)
        assert as_block(V) is V

    def test_as_block_wraps_callables(self):
        assert isinstance(as_block(HarmonicWell()), Block1D)
        assert isinstance(as_block(lambda x: x**2), Block1D)

    def test_as_block_rejects_non_callable(self):
        with pytest.raises(TypeError):
            as_block(3.0)


class TestSeparablePotential:
    def test_sum_of_blocks(self):
        V = SeparablePotential([HarmonicWell(omega=1.0), HarmonicWell(omega=2.0)])
        # 0.5*1*x^2 + 0.5*4*y^2
        assert V(2.0, 1.0) == pytest.approx(2.0 + 2.0)

    def test_ndim_is_the_sum_of_block_dimensions(self):
        V = SeparablePotential(
            [HarmonicWell(omega=1.0), GenericPotential(lambda x, y, z: x + y + z, ndim=3)]
        )
        assert V.ndim == 4
        assert V.block_slices == (slice(0, 1), slice(1, 4))

    def test_coordinates_are_split_contiguously(self):
        three_d = GenericPotential(lambda x, y, z: 100 * x + 10 * y + z, ndim=3)
        V = SeparablePotential([HarmonicWell(omega=1.0), three_d])
        # first coordinate to the well, last three to the 3D block
        assert V(2.0, 1.0, 2.0, 3.0) == pytest.approx(2.0 + 123.0)

    def test_nested_separables_are_flattened(self):
        inner = SeparablePotential([HarmonicWell(omega=1.0)] * 2)
        V = SeparablePotential([inner, HarmonicWell(omega=1.0)])
        assert len(V.blocks) == 3
        assert V.ndim == 3

    def test_nested_names_are_qualified(self):
        inner = SeparablePotential([HarmonicWell(omega=1.0)] * 2, names=["x", "y"])
        V = SeparablePotential([inner, HarmonicWell(omega=1.0)], names=["inner", "z"])
        assert V.names == ("inner.x", "inner.y", "z")

    def test_default_names(self):
        V = SeparablePotential([HarmonicWell(omega=1.0)] * 3)
        assert V.names == ("1", "2", "3")

    def test_custom_names(self):
        V = SeparablePotential([HarmonicWell(omega=1.0)] * 2, names=["e1", "e2"])
        assert V.names == ("e1", "e2")

    def test_name_count_must_match(self):
        with pytest.raises(ValueError, match="block name"):
            SeparablePotential([HarmonicWell()] * 2, names=["only-one"])

    def test_no_blocks(self):
        with pytest.raises(ValueError, match="at least one block"):
            SeparablePotential([])

    def test_two_identical_particles_is_just_two_blocks(self):
        particle = GenericPotential(lambda x, y, z: -1.0 / np.sqrt(x**2 + y**2 + z**2 + 0.1), ndim=3)
        pair = SeparablePotential([particle, particle], names=["a", "b"])
        assert pair.ndim == 6
        assert pair(1.0, 0.0, 0.0, 2.0, 0.0, 0.0) == pytest.approx(
            particle(1.0, 0.0, 0.0) + particle(2.0, 0.0, 0.0)
        )


class TestSumPotential:
    def test_add_shares_coordinates(self):
        V = SeparablePotential([HarmonicWell(omega=1.0)] * 2) + GenericPotential(
            lambda x, y: 0.1 * x * y, ndim=2
        )
        assert isinstance(V, SumPotential)
        assert V(1.0, 2.0) == pytest.approx(0.5 + 2.0 + 0.2)

    def test_flattening(self):
        a = GenericPotential(lambda x: x, ndim=1)
        V = a + a + a
        assert len(V.terms) == 3

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="cannot add"):
            GenericPotential(lambda x: x, ndim=1) + GenericPotential(lambda x, y: x, ndim=2)

    def test_type_error(self):
        with pytest.raises(TypeError):
            GenericPotential(lambda x: x, ndim=1) + 1.0


class TestOneDimensionalInterop:
    def test_separable_1d_solves_like_the_known_potential(self):
        grid = GridND.line(-8.0, 8.0, 512)
        V = SeparablePotential([HarmonicWell(omega=1.0)])
        assert solve_eigenstates(grid, V, n_states=3).energies == pytest.approx(
            [0.5, 1.5, 2.5], abs=1e-3
        )

    def test_generic_1d_matches_known_potential_pointwise(self):
        x = np.linspace(-5, 5, 100)
        built = GenericPotential(lambda x: 0.5 * 4 * (x - 1) ** 2, ndim=1)
        assert np.allclose(built(x), HarmonicWell(omega=2.0, x0=1.0)(x))

    def test_barrier_block_keeps_its_shape(self):
        V = SeparablePotential([RectangularBarrier(height=4.0, width=1.0), DoubleWell()])
        assert V(0.0, 0.0) == pytest.approx(4.0)
        assert V(5.0, 0.0) == pytest.approx(0.0)
