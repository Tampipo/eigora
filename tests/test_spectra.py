# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np

from eigora.grids import GridND

from eigora.qm import QuantumSystem, QuantumSystem1D
from eigora.qm.potentials import (
    Block1D,
    DoubleWell,
    GenericPotential,
    HarmonicWell,
    InfiniteSquareWell,
    SeparablePotential,
)
from eigora.qm.solvers import solve_eigenstates
from eigora.qm.spectra import (
    BoxSpectrum,
    HarmonicSpectrum,
    NumericalSpectrum,
    SeparableSpectrum,
    exact_spectrum,
    has_exact_spectrum,
    spectrum_for,
)

GRID = GridND.line(-8.0, 8.0, 512)


class TestHarmonicSpectrum:
    @pytest.fixture
    def spectrum(self):
        return HarmonicSpectrum(HarmonicWell(omega=1.0))

    def test_energies(self, spectrum):
        assert spectrum.energies(4) == pytest.approx([0.5, 1.5, 2.5, 3.5])

    def test_energies_scale_with_omega(self):
        spectrum = HarmonicSpectrum(HarmonicWell(omega=2.5))
        assert spectrum.energy((3,)) == pytest.approx(2.5 * 3.5)

    def test_metadata(self, spectrum):
        assert spectrum.ndim == 1
        assert spectrum.quantum_numbers == ("n",)
        assert spectrum.is_exact
        assert spectrum.n_available is None

    def test_ground_state(self, spectrum):
        assert spectrum.ground_state == (0,)
        assert spectrum.ground_energy == pytest.approx(0.5)

    def test_states_are_zero_indexed(self, spectrum):
        assert spectrum.states(3) == [(0,), (1,), (2,)]

    @pytest.mark.parametrize("n", [0, 1, 2, 5])
    def test_wavefunctions_are_normalised(self, spectrum, n):
        x = GRID.x
        psi = spectrum.wavefunction((n,))(x)
        assert np.trapezoid(psi**2, x) == pytest.approx(1.0, abs=1e-6)

    def test_wavefunctions_are_orthogonal(self, spectrum):
        x = GRID.x
        psi0 = spectrum.wavefunction((0,))(x)
        psi2 = spectrum.wavefunction((2,))(x)
        assert np.trapezoid(psi0 * psi2, x) == pytest.approx(0.0, abs=1e-6)

    def test_ground_state_is_the_expected_gaussian(self):
        spectrum = HarmonicSpectrum(HarmonicWell(omega=2.0, x0=1.0))
        x = np.linspace(-3, 5, 200)
        expected = (2.0 / np.pi) ** 0.25 * np.exp(-2.0 * (x - 1.0) ** 2 / 2)
        assert np.allclose(spectrum.wavefunction((0,))(x), expected)

    @pytest.mark.parametrize("n", [40, 100, 160])
    def test_high_states_stay_finite_and_normalised(self, spectrum, n):
        # 2^n n! overflows long before the Hermite polynomial does, hence the
        # log-space prefactor. The window must cover the classical turning
        # point at sqrt(2E), which grows with n.
        turning_point = np.sqrt(2 * (n + 0.5))
        x = np.linspace(-1.6 * turning_point, 1.6 * turning_point, 40001)
        psi = spectrum.wavefunction((n,))(x)
        assert np.all(np.isfinite(psi))
        assert np.trapezoid(psi**2, x) == pytest.approx(1.0, abs=1e-4)

    def test_negative_quantum_number(self, spectrum):
        with pytest.raises(ValueError, match="n must be >= 0"):
            spectrum.energy((-1,))

    def test_wrong_arity(self, spectrum):
        with pytest.raises(ValueError, match="quantum number"):
            spectrum.energy((0, 0))


class TestBoxSpectrum:
    @pytest.fixture
    def spectrum(self):
        return BoxSpectrum(InfiniteSquareWell(width=2.0))

    def test_energies(self, spectrum):
        expected = [n**2 * np.pi**2 / (2 * 2.0**2) for n in (1, 2, 3)]
        assert spectrum.energies(3) == pytest.approx(expected)

    def test_states_are_one_indexed(self, spectrum):
        assert spectrum.states(3) == [(1,), (2,), (3,)]
        assert spectrum.ground_state == (1,)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_wavefunctions_are_normalised(self, spectrum, n):
        x = np.linspace(-3.0, 3.0, 4001)
        psi = spectrum.wavefunction((n,))(x)
        assert np.trapezoid(psi**2, x) == pytest.approx(1.0, abs=1e-4)

    def test_vanishes_outside_the_well(self, spectrum):
        psi = spectrum.wavefunction((1,))(np.array([-2.0, -1.0, 1.0, 2.0]))
        assert psi == pytest.approx([0.0, 0.0, 0.0, 0.0])

    def test_offset_well(self):
        spectrum = BoxSpectrum(InfiniteSquareWell(width=2.0, x0=5.0))
        assert spectrum.wavefunction((1,))(np.array([5.0]))[0] == pytest.approx(1.0)

    def test_zero_quantum_number(self, spectrum):
        with pytest.raises(ValueError, match="n must be >= 1"):
            spectrum.energy((0,))

    def test_metadata(self, spectrum):
        assert spectrum.ndim == 1
        assert spectrum.is_exact
        assert spectrum.n_available is None

    def test_states_needs_a_positive_count(self, spectrum):
        with pytest.raises(ValueError, match="at least 1"):
            spectrum.states(0)


class TestRegistry:
    def test_known_potentials(self):
        assert has_exact_spectrum(HarmonicWell())
        assert has_exact_spectrum(InfiniteSquareWell())

    def test_looks_through_a_block(self):
        assert has_exact_spectrum(Block1D(HarmonicWell()))
        assert isinstance(exact_spectrum(Block1D(HarmonicWell())), HarmonicSpectrum)

    def test_unknown_potential(self):
        assert not has_exact_spectrum(DoubleWell())
        with pytest.raises(ValueError, match="no exact spectrum for DoubleWell"):
            exact_spectrum(DoubleWell())

    def test_spectrum_carries_its_potential(self):
        well = HarmonicWell(omega=3.0)
        assert exact_spectrum(well).potential is well


class TestNumericalSpectrum:
    @pytest.fixture
    def spectrum(self):
        return NumericalSpectrum.solve(HarmonicWell(omega=1.0), GRID, n_states=6)

    def test_matches_the_exact_harmonic(self, spectrum):
        assert spectrum.energies(4) == pytest.approx([0.5, 1.5, 2.5, 3.5], abs=1e-3)

    def test_metadata(self, spectrum):
        assert not spectrum.is_exact
        assert spectrum.n_available == 6
        assert spectrum.quantum_numbers == ("n",)

    def test_states_stop_at_what_was_computed(self, spectrum):
        assert len(spectrum.states(100)) == 6

    def test_wavefunction_interpolates_off_grid(self, spectrum):
        exact = HarmonicSpectrum(HarmonicWell(omega=1.0))
        x = np.array([-1.234, 0.0, 0.777])  # deliberately between grid points
        psi = spectrum.wavefunction((0,))(x)
        assert psi == pytest.approx(exact.wavefunction((0,))(x), abs=1e-4)

    def test_wavefunction_vanishes_outside_the_grid(self, spectrum):
        assert spectrum.wavefunction((0,))(np.array([-50.0, 50.0])) == pytest.approx([0.0, 0.0])

    def test_state_out_of_range(self, spectrum):
        with pytest.raises(IndexError, match="out of range"):
            spectrum.energy((6,))

    def test_needs_a_1d_grid(self):
        with pytest.raises(ValueError, match="1D"):
            NumericalSpectrum.solve(HarmonicWell(), GridND.cube(4.0, 8, ndim=2))


class TestSeparableDegeneracies:
    """The headline property: degeneracies of a separable system."""

    @pytest.mark.parametrize("n, expected", [(0, 1), (1, 2), (2, 3), (3, 4)])
    def test_isotropic_2d_harmonic(self, n, expected):
        sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 2))
        level = sol.levels(n + 1)[n]
        assert level.energy == pytest.approx(n + 1.0)
        assert level.degeneracy == expected

    @pytest.mark.parametrize("n, expected", [(0, 1), (1, 3), (2, 6), (3, 10), (4, 15)])
    def test_isotropic_3d_harmonic(self, n, expected):
        sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 3))
        level = sol.levels(n + 1)[n]
        assert level.energy == pytest.approx(n + 1.5)
        assert level.degeneracy == expected

    def test_anisotropy_breaks_the_degeneracies(self):
        aniso = SeparablePotential([HarmonicWell(1.0), HarmonicWell(1.0), HarmonicWell(2.0)])
        levels = spectrum_for(aniso).levels(4)
        assert [level.degeneracy for level in levels] == [1, 2, 4, 6]

    def test_square_box_degeneracies(self):
        sol = spectrum_for(SeparablePotential([InfiniteSquareWell(width=1.0)] * 2))
        assert sol.degeneracy((1, 1)) == 1
        assert sol.degeneracy((1, 2)) == 2  # (1,2) and (2,1)
        assert sol.degeneracy((2, 2)) == 1

    def test_degeneracy_is_found_beyond_the_initial_search_window(self):
        # Level n=5 of the 3D trap has 21 states, all of which lie past the
        # first batch the search looks at -- the window has to grow.
        sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 3))
        assert sol.degeneracy((5, 0, 0)) == 21
        assert sol.levels(6)[5].degeneracy == 21

    def test_degenerate_states_are_listed(self):
        sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 2))
        assert set(sol.levels(2)[1].states) == {(1, 0), (0, 1)}

    def test_levels_energies_are_distinct_and_ascending(self):
        sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 3))
        energies = [level.energy for level in sol.levels(6)]
        assert energies == sorted(energies)
        assert len(set(np.round(energies, 9))) == 6


class TestSeparableSpectrum:
    @pytest.fixture
    def trap(self):
        return spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 3))

    def test_metadata(self, trap):
        assert trap.ndim == 3
        assert trap.is_exact
        assert trap.n_available is None
        assert trap.quantum_numbers == ("n_1", "n_2", "n_3")

    def test_energies_add(self, trap):
        assert trap.energy((1, 2, 3)) == pytest.approx(0.5 * 3 + 6)

    def test_states_ascend_in_energy(self, trap):
        energies = [trap.energy(label) for label in trap.states(30)]
        assert np.all(np.diff(energies) >= -1e-12)

    def test_states_are_unique(self, trap):
        labels = trap.states(30)
        assert len(set(labels)) == 30

    def test_ground_state(self, trap):
        assert trap.ground_state == (0, 0, 0)
        assert trap.ground_energy == pytest.approx(1.5)

    def test_labels_split_per_block(self, trap):
        assert trap.split((1, 2, 3)) == ((1,), (2,), (3,))

    def test_label_text(self, trap):
        assert trap.label_text((1, 0, 2)) == "n_1=1, n_2=0, n_3=2"

    def test_block_names_appear_in_quantum_numbers(self):
        pair = SeparablePotential([HarmonicWell(1.0), HarmonicWell(1.0)], names=["a", "b"])
        assert spectrum_for(pair).quantum_numbers == ("n_a", "n_b")

    def test_wavefunction_is_the_product_of_block_states(self, trap):
        exact = HarmonicSpectrum(HarmonicWell(omega=1.0))
        psi = trap.wavefunction((1, 0, 2))
        x, y, z = 0.3, -0.7, 1.1
        expected = (
            exact.wavefunction((1,))(np.array([x]))
            * exact.wavefunction((0,))(np.array([y]))
            * exact.wavefunction((2,))(np.array([z]))
        )
        assert psi(np.array([x]), np.array([y]), np.array([z])) == pytest.approx(expected)

    def test_wavefunction_rejects_wrong_coordinate_count(self, trap):
        with pytest.raises(ValueError, match="3 coordinate"):
            trap.wavefunction((0, 0, 0))(np.array([0.0]))

    @pytest.mark.parametrize("label", [(0, 0), (1, 0), (1, 2)])
    def test_product_states_are_normalised_on_a_grid(self, label):
        grid = GridND.cube(half_width=8.0, n_points=256, ndim=2)
        sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 2))
        psi = sol.wavefunction_on_grid(label, grid)
        assert psi.shape == grid.shape
        assert np.sum(psi**2) * grid.volume_element == pytest.approx(1.0, abs=1e-4)

    def test_density_on_grid(self):
        grid = GridND.cube(half_width=6.0, n_points=64, ndim=2)
        sol = spectrum_for(SeparablePotential([HarmonicWell(omega=1.0)] * 2))
        density = sol.density_on_grid((0, 0), grid)
        assert density.shape == grid.shape
        assert np.all(density >= 0.0)

    def test_grid_dimension_mismatch(self, trap):
        with pytest.raises(ValueError, match="3D but the grid is 1D"):
            trap.wavefunction_on_grid((0, 0, 0), GRID)

    def test_sparse_sampling_matches_dense(self, trap):
        grid = GridND.cube(half_width=5.0, n_points=24, ndim=3)
        sparse = trap.wavefunction_on_grid((1, 0, 2), grid, sparse=True)
        assert sparse.shape == grid.shape
        assert np.allclose(sparse, trap.wavefunction_on_grid((1, 0, 2), grid))

    def test_levels_needs_a_positive_count(self, trap):
        with pytest.raises(ValueError, match="n_levels"):
            trap.levels(0)

    def test_states_needs_a_positive_count(self, trap):
        with pytest.raises(ValueError, match="at least 1"):
            trap.states(0)

    def test_block_name_count_must_match(self):
        blocks = [HarmonicSpectrum(HarmonicWell(omega=1.0))] * 2
        with pytest.raises(ValueError, match="block name"):
            SeparableSpectrum(blocks, names=["only-one"])

    def test_rejects_non_spectrum_blocks(self):
        with pytest.raises(TypeError, match="Spectrum"):
            SeparableSpectrum([HarmonicWell(omega=1.0)])

    def test_needs_at_least_one_block(self):
        with pytest.raises(ValueError, match="at least one block"):
            SeparableSpectrum([])


class TestMixedExactAndNumericalBlocks:
    @pytest.fixture
    def mixed(self):
        potential = SeparablePotential(
            [HarmonicWell(omega=1.0), DoubleWell(a=1.0, b=4.0)], names=["osc", "well"]
        )
        return spectrum_for(potential, grid=GRID, n_states=6)

    def test_is_not_exact(self, mixed):
        assert not mixed.is_exact

    def test_quantum_numbers_use_block_names(self, mixed):
        assert mixed.quantum_numbers == ("n_osc", "n_well")

    def test_energies_ascend(self, mixed):
        energies = mixed.energies(12)
        assert np.all(np.diff(energies) >= -1e-12)

    def test_energies_are_sums_of_block_energies(self, mixed):
        harmonic = HarmonicSpectrum(HarmonicWell(omega=1.0))
        numerical = NumericalSpectrum.solve(DoubleWell(a=1.0, b=4.0), GRID, n_states=6)
        assert mixed.energy((2, 1)) == pytest.approx(
            harmonic.energy((2,)) + numerical.energy((1,))
        )

    def test_bounded_block_stops_producing_states(self, mixed):
        labels = mixed.states(60)
        assert all(label[1] < 6 for label in labels)  # the numerical block has 6 states

    def test_fully_bounded_system_is_finite(self):
        potential = SeparablePotential([DoubleWell(a=1.0, b=4.0)] * 2)
        sol = spectrum_for(potential, grid=GRID, n_states=4)
        assert sol.n_available == 16
        assert len(sol.states(100)) == 16


class TestSpectrumFor:
    def test_known_1d_potential_is_exact(self):
        sol = spectrum_for(HarmonicWell(omega=1.0))
        assert isinstance(sol, HarmonicSpectrum)
        assert sol.is_exact

    def test_unknown_1d_potential_uses_the_grid(self):
        sol = spectrum_for(DoubleWell(a=1.0, b=4.0), grid=GRID, n_states=4)
        assert isinstance(sol, NumericalSpectrum)
        assert sol.energies(4) == pytest.approx(
            solve_eigenstates(GRID, DoubleWell(a=1.0, b=4.0), 4).energies
        )

    def test_unknown_1d_potential_without_a_grid(self):
        with pytest.raises(ValueError, match="pass a grid"):
            spectrum_for(DoubleWell())

    def test_multidimensional_generic_potential(self):
        with pytest.raises(ValueError, match="no analytic solution"):
            spectrum_for(GenericPotential(lambda x, y: x + y, ndim=2))

    def test_a_shared_1d_grid_serves_every_block(self):
        potential = SeparablePotential([DoubleWell(a=1.0, b=4.0)] * 2)
        sol = spectrum_for(potential, grid=GRID, n_states=3)
        assert sol.ndim == 2
        assert not sol.is_exact

    def test_an_nd_grid_is_sliced_per_block(self):
        potential = SeparablePotential([DoubleWell(a=1.0, b=4.0)] * 2)
        nd_grid = GridND.cube(half_width=8.0, n_points=512, ndim=2)
        sliced = spectrum_for(potential, grid=nd_grid, n_states=3)
        shared = spectrum_for(potential, grid=GRID, n_states=3)
        assert sliced.energies(4) == pytest.approx(shared.energies(4))

    def test_exact_blocks_need_no_grid_at_all(self):
        sol = spectrum_for(SeparablePotential([HarmonicWell(1.0), InfiniteSquareWell(2.0)]))
        assert sol.is_exact
        assert sol.energy((0, 1)) == pytest.approx(0.5 + np.pi**2 / 8)


class TestQuantumSystem:
    def test_1d_solve_still_works(self):
        system = QuantumSystem(grid=GRID, potential=HarmonicWell(omega=1.0))
        assert system.ndim == 1
        assert system.solve(n_states=3).energies == pytest.approx([0.5, 1.5, 2.5], abs=1e-3)

    def test_legacy_alias(self):
        assert QuantumSystem1D is QuantumSystem

    def test_1d_evolve_still_works(self):
        from eigora.qm.states import GaussianWavepacket

        system = QuantumSystem(grid=GRID, potential=HarmonicWell(omega=1.0))
        evolution = system.evolve(GaussianWavepacket(x0=-2.0, k0=0.0, sigma=1.0),
                                  t_max=1.0, dt=0.01, n_frames=5)
        assert evolution.n_frames == 5
        assert evolution.norm(-1) == pytest.approx(1.0, abs=1e-4)

    def test_1d_potential_with_a_higher_dimensional_grid(self):
        system = QuantumSystem(grid=GridND.cube(8.0, 32, ndim=2),
                               potential=HarmonicWell(omega=1.0))
        with pytest.raises(ValueError, match="needs a 1D grid"):
            system.solve()

    def test_spectrum_of_a_separable_system(self):
        trap = SeparablePotential([HarmonicWell(omega=1.0)] * 3)
        system = QuantumSystem(grid=GridND.cube(8.0, 64, ndim=3), potential=trap)
        assert system.ndim == 3
        assert [level.degeneracy for level in system.spectrum().levels(3)] == [1, 3, 6]

    def test_solve_rejects_higher_dimensions(self):
        trap = SeparablePotential([HarmonicWell(omega=1.0)] * 2)
        system = QuantumSystem(grid=GridND.cube(8.0, 64, ndim=2), potential=trap)
        with pytest.raises(ValueError, match="'solve' is 1D only"):
            system.solve()

    def test_evolve_rejects_higher_dimensions(self):
        from eigora.qm.states import GaussianWavepacket

        trap = SeparablePotential([HarmonicWell(omega=1.0)] * 2)
        system = QuantumSystem(grid=GridND.cube(8.0, 64, ndim=2), potential=trap)
        with pytest.raises(ValueError, match="'evolve' is 1D only"):
            system.evolve(GaussianWavepacket(), t_max=1.0, dt=0.01)
