# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Systems with a closed-form partition function override `log_z` rather than
summing over their levels. These tests hold the two implementations against
each other: the analytic result must agree with the sweep over the spectrum it
replaces, and the sweep must actually be the thing being compared against.

Energies are absolute throughout -- `HarmonicMode` includes its zero point --
so a free energy that disagrees by a constant is a real disagreement.
"""

import math

import pytest

from eigora.statphys import (
    Box1D,
    CompositeSystem,
    Degenerate,
    HarmonicMode,
    Level,
    NLevel,
    Rotor,
    SpectralSystem,
    Spin,
    System,
    TwoLevel,
    particle_in_box,
)
from eigora.statphys.ensembles import Canonical, Magnetic

BETAS = (0.05, 0.2, 1.0, 3.0, 10.0)


def summed_log_z(system, beta):
    """log Z from the spectrum sweep, bypassing any closed-form override."""
    return SpectralSystem.moments(system, beta).log_z


def magnetic(field):
    """The couplings a `Magnetic(T, field)` ensemble reduces to."""
    return (("magnetisation", field),)


class TestLevel:
    def test_degeneracy_must_be_positive(self):
        with pytest.raises(ValueError, match="degeneracy must be at least 1"):
            Level(0.0, 0)

    def test_energy_must_be_finite(self):
        with pytest.raises(ValueError, match="energy must be finite"):
            Level(math.inf)

    def test_extensive_defaults_to_empty(self):
        assert Level(1.0).extensive == {}


class TestTwoLevel:
    @pytest.fixture
    def system(self):
        return TwoLevel(splitting=1.0)

    def test_log_z_is_the_two_term_sum(self, system):
        assert system.log_z(1.0) == pytest.approx(math.log(1.0 + math.exp(-1.0)))

    def test_has_two_states(self, system):
        assert system.n_states == 2

    def test_ground_energy_is_zero(self, system):
        assert system.ground_energy == 0.0

    def test_splitting_must_be_positive(self):
        with pytest.raises(ValueError, match="splitting must be positive"):
            TwoLevel(splitting=0.0)

    @pytest.mark.parametrize("beta", BETAS)
    def test_energy_lies_between_the_two_levels(self, system, beta):
        assert 0.0 < system.mean_energy(beta) < system.splitting


class TestNLevel:
    def test_sorts_its_levels(self):
        system = NLevel([2.5, 0.0, 1.0], [5, 1, 3])
        assert system.energies == (0.0, 1.0, 2.5)
        assert system.degeneracies == (1, 3, 5)

    def test_degeneracies_default_to_one(self):
        assert NLevel([0.0, 1.0]).degeneracies == (1, 1)

    def test_n_states_counts_degeneracy(self):
        assert NLevel([0.0, 1.0], [1, 3]).n_states == 4

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="expected 2 degeneracy"):
            NLevel([0.0, 1.0], [1])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one energy"):
            NLevel([])

    def test_matches_a_two_level_system(self):
        explicit = NLevel([0.0, 1.0])
        assert explicit.log_z(2.0) == pytest.approx(TwoLevel(1.0).log_z(2.0))


class TestDegenerate:
    def test_log_z_is_log_g_at_every_temperature(self):
        system = Degenerate(5)
        for beta in BETAS:
            assert system.log_z(beta) == pytest.approx(math.log(5))

    def test_closed_form_matches_the_sum(self):
        system = Degenerate(5)
        assert system.log_z(2.0) == pytest.approx(summed_log_z(system, 2.0))

    def test_carries_no_energy(self):
        assert Degenerate(3).mean_energy(1.0) == pytest.approx(0.0)


class TestHarmonicMode:
    @pytest.fixture
    def system(self):
        return HarmonicMode(omega=1.0)

    @pytest.mark.parametrize("beta", BETAS)
    def test_closed_form_matches_the_summed_spectrum(self, system, beta):
        assert system.log_z(beta) == pytest.approx(summed_log_z(system, beta), rel=1e-12)

    @pytest.mark.parametrize("beta", BETAS)
    def test_closed_form_moments_match_the_summed_ones(self, system, beta):
        summed = SpectralSystem.moments(system, beta)
        assert system.mean_energy(beta) == pytest.approx(summed.energy, rel=1e-10)
        assert system.energy_variance(beta) == pytest.approx(
            summed.energy_variance, rel=1e-9
        )

    def test_closed_form_survives_where_the_sum_would_overflow(self, system):
        # sinh(beta w / 2) overflows near beta = 1420; the closed form is
        # written so it does not.
        assert system.log_z(4000.0) == pytest.approx(-2000.0)

    def test_spectrum_is_unbounded(self, system):
        assert system.n_states is None

    def test_energy_tends_to_the_zero_point(self, system):
        assert system.mean_energy(100.0) == pytest.approx(0.5)

    def test_energy_tends_to_equipartition(self, system):
        # <E> -> T + O(1/T): the zero point stops mattering at high T.
        assert system.mean_energy(1e-3) == pytest.approx(1000.0, rel=1e-5)

    def test_omega_must_be_positive(self):
        with pytest.raises(ValueError, match="omega must be positive"):
            HarmonicMode(omega=-1.0)

    def test_energies_scale_with_omega(self):
        assert HarmonicMode(2.0).mean_energy(100.0) == pytest.approx(1.0)


class TestSpin:
    def test_states_carry_their_magnetisation(self):
        levels = list(Spin(1.0).levels())
        assert [level.extensive["magnetisation"] for level in levels] == [-1.0, 0.0, 1.0]

    def test_multiplicity(self):
        assert Spin(1.5).n_states == 4

    def test_rejects_non_half_integer(self):
        with pytest.raises(ValueError, match="multiple of 1/2"):
            Spin(0.3)

    @pytest.mark.parametrize("temperature, field", [(1.0, 0.5), (0.5, 2.0), (2.0, 0.1)])
    def test_half_spin_follows_the_brillouin_function(self, temperature, field):
        moments = Spin(0.5).moments(1.0 / temperature, magnetic(field))
        expected = 0.5 * math.tanh(field / (2.0 * temperature))
        assert moments.means["magnetisation"] == pytest.approx(expected)

    def test_carries_no_energy_of_its_own(self):
        # The Zeeman term lives in the ensemble, not in the spectrum.
        assert Spin(0.5).moments(1.0).energy == pytest.approx(0.0)


class TestRotor:
    @pytest.fixture
    def system(self):
        return Rotor(b=1.0)

    def test_degeneracies_are_two_l_plus_one(self, system):
        levels = [level for _, level in zip(range(4), system.levels())]
        assert [level.degeneracy for level in levels] == [1, 3, 5, 7]
        assert [level.energy for level in levels] == [0.0, 2.0, 6.0, 12.0]

    def test_high_temperature_energy_approaches_equipartition(self, system):
        # The classical limit is <E> = T - b/3 + O(b^2/T): two rotational
        # degrees of freedom, with the leading quantum correction.
        for temperature in (50.0, 200.0):
            energy = system.mean_energy(1.0 / temperature)
            assert energy == pytest.approx(temperature - 1.0 / 3.0, rel=1e-4)

    def test_high_temperature_heat_capacity_approaches_one(self, system):
        beta = 1.0 / 200.0
        assert beta**2 * system.energy_variance(beta) == pytest.approx(1.0, abs=1e-4)

    def test_heat_capacity_dies_at_low_temperature(self, system):
        beta = 1.0 / 0.05
        assert beta**2 * system.energy_variance(beta) < 1e-10

    def test_b_must_be_positive(self):
        with pytest.raises(ValueError, match="b must be positive"):
            Rotor(b=0.0)


class TestBox:
    def test_level_spacing(self):
        assert Box1D(length=1.0, mass=1.0).level_spacing == pytest.approx(
            math.pi**2 / 2.0
        )

    def test_particle_in_box_is_a_product_of_1d_boxes(self):
        box = particle_in_box(length=1.0, ndim=3)
        assert isinstance(box, CompositeSystem)
        assert len(box.blocks) == 3

    def test_three_dimensions_cost_three_times_one(self):
        beta = 1.0
        one = Box1D(length=1.0)
        assert particle_in_box(1.0, ndim=3).log_z(beta) == pytest.approx(
            3.0 * one.log_z(beta)
        )

    def test_rejects_zero_dimensions(self):
        with pytest.raises(ValueError, match="ndim must be at least 1"):
            particle_in_box(1.0, ndim=0)


class TestComposition:
    def test_log_z_adds(self):
        a, b = TwoLevel(1.0), HarmonicMode(2.0)
        assert (a * b).log_z(0.7) == pytest.approx(a.log_z(0.7) + b.log_z(0.7))

    def test_power_equals_repeated_multiplication(self):
        a = TwoLevel(1.0)
        assert (a**4).log_z(0.9) == pytest.approx((a * a * a * a).log_z(0.9))

    def test_energies_and_variances_add(self):
        a, b = TwoLevel(1.0), Rotor(0.5)
        both = a * b
        assert both.mean_energy(0.8) == pytest.approx(
            a.mean_energy(0.8) + b.mean_energy(0.8)
        )
        assert both.energy_variance(0.8) == pytest.approx(
            a.energy_variance(0.8) + b.energy_variance(0.8)
        )

    def test_nesting_is_flattened(self):
        a = TwoLevel(1.0)
        assert len(((a * a) * (a * a)).blocks) == 4

    def test_blocks_must_be_systems(self):
        with pytest.raises(TypeError, match="expected a System"):
            CompositeSystem([TwoLevel(1.0), 3.0])

    def test_rejects_zero_copies(self):
        with pytest.raises(ValueError, match="at least 1"):
            TwoLevel(1.0) ** 0

    def test_is_exact_propagates(self):
        assert (TwoLevel(1.0) * HarmonicMode(1.0)).is_exact

    def test_magnetisation_adds_over_copies(self):
        one = Spin(0.5).moments(1.0, magnetic(0.5)).means["magnetisation"]
        many = (Spin(0.5) ** 50).moments(1.0, magnetic(0.5)).means["magnetisation"]
        assert many == pytest.approx(50.0 * one)


class TestSummation:
    def test_beta_must_be_positive(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            TwoLevel(1.0).log_z(0.0)

    def test_truncation_raises_rather_than_returning_a_wrong_number(self):
        # A spectrum dense enough that the tail never becomes negligible
        # within the cap. Silently truncating would return a plausible float,
        # so this must be an error rather than an approximation.
        class Dense(SpectralSystem):
            def levels(self):
                n = 0
                while True:
                    yield Level(n * 1e-12, 1)
                    n += 1

            @property
            def n_states(self):
                return None

        with pytest.raises(ValueError, match="did not converge"):
            Dense().log_z(1.0)

    def test_descending_levels_are_rejected(self):
        class Backwards(SpectralSystem):
            def levels(self):
                yield Level(1.0)
                yield Level(0.0)

            @property
            def n_states(self):
                return 2

        with pytest.raises(ValueError, match="must ascend in energy"):
            Backwards().log_z(1.0)

    def test_a_system_without_the_freed_variable_is_refused(self):
        # TwoLevel carries no magnetisation, so a magnetic ensemble cannot be
        # applied to it -- better than silently averaging zero.
        with pytest.raises(ValueError, match="magnetisation"):
            TwoLevel(1.0).moments(1.0, magnetic(0.5))

    def test_non_spectral_systems_refuse_freed_variables(self):
        composite = TwoLevel(1.0) * TwoLevel(2.0)
        with pytest.raises(ValueError, match="magnetisation"):
            composite.moments(1.0, magnetic(0.5))


class TestExactVersusNumerical:
    """The generic finite-difference route must reproduce the exact sums."""

    @pytest.mark.parametrize(
        "system", [TwoLevel(1.0), NLevel([0.0, 1.0, 2.5], [1, 3, 5]), Rotor(1.0)]
    )
    @pytest.mark.parametrize("beta", (0.3, 1.0, 2.0))
    def test_mean_energy_matches_the_derivative_of_log_z(self, system, beta):
        numerical = System.moments(system, beta).energy
        assert system.mean_energy(beta) == pytest.approx(numerical, rel=1e-7)

    @pytest.mark.parametrize("system", [TwoLevel(1.0), NLevel([0.0, 1.0, 2.5], [1, 3, 5])])
    @pytest.mark.parametrize("beta", (0.3, 1.0, 2.0))
    def test_energy_variance_matches_the_second_derivative(self, system, beta):
        numerical = System.moments(system, beta).energy_variance
        assert system.energy_variance(beta) == pytest.approx(numerical, rel=1e-5)

    @pytest.mark.parametrize("field", (0.0, 0.5, -2.0))
    def test_freed_variables_survive_the_finite_difference_route(self, field):
        """
        `<M>` and `Var(M)` by differencing, against the exact sweep.

        This is what the generic route buys: a system with nothing but a
        closed-form `log_z` is still a full member of a non-canonical
        ensemble. Before, `System.moments` had to refuse outright.
        """
        spin, beta = Spin(1.5), 0.8
        exact = SpectralSystem.moments(spin, beta, magnetic(field))
        differenced = System.moments(spin, beta, magnetic(field))
        assert differenced.log_z == pytest.approx(exact.log_z, rel=1e-14)
        assert differenced.means["magnetisation"] == pytest.approx(
            exact.means["magnetisation"], rel=1e-7, abs=1e-9
        )
        assert differenced.variances["magnetisation"] == pytest.approx(
            exact.variances["magnetisation"], rel=1e-5
        )

    @pytest.mark.parametrize("field", (0.5, -2.0))
    def test_the_energy_is_differentiated_at_fixed_natural_parameter(self, field):
        """
        The trap this refactor exists to avoid.

        Differentiating log Z in beta at fixed *field* gives `<E - c M>`, not
        `<E>`. The two differ by `c <M>`, both are floats, and nothing would
        flag the swap -- so assert the right one and the size of the gap.
        """
        spin, beta = Spin(1.5), 0.8
        exact = SpectralSystem.moments(spin, beta, magnetic(field))
        differenced = System.moments(spin, beta, magnetic(field))
        assert differenced.energy == pytest.approx(exact.energy, abs=1e-9)

        # What the naive derivative would have returned instead.
        step = 1e-5 * beta
        naive = -(
            spin.log_z(beta + step, magnetic(field))
            - spin.log_z(beta - step, magnetic(field))
        ) / (2.0 * step)
        assert naive == pytest.approx(
            exact.energy - field * exact.means["magnetisation"], abs=1e-7
        )
        assert abs(naive - exact.energy) > 1e-3

    def test_closed_forms_still_guard_beta(self):
        """An override must not quietly accept what the sum would reject."""
        for system in (Degenerate(3), HarmonicMode(1.0)):
            with pytest.raises(ValueError, match="beta must be positive"):
                system.log_z(-1.0)
