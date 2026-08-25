# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
An ensemble is a log-weight on a microstate and nothing more, so these tests
start from that formula and work outwards. The identity that matters most is
that `log_weight_change` really is the difference of two `log_weight` calls:
everything the Monte Carlo layer will do rests on it, and it is the one thing
no physics result would reveal if it were wrong.

The thermodynamics is then checked against closed forms where they exist --
the Schottky anomaly, the Brillouin function, the third law -- and against the
numerical derivatives it is supposed to replace where they do not.
"""

import math

import pytest

from eigora.statphys import (
    Canonical,
    Ensemble,
    Degenerate,
    Field,
    Generalised,
    HarmonicMode,
    Level,
    Magnetic,
    NLevel,
    Rotor,
    Spin,
    SpectralSystem,
    ThermalState,
    TwoLevel,
    equilibrium,
)

# Peak of the two-level heat capacity: x = splitting/T solving e^x = (x+2)/(x-2).
SCHOTTKY_X = 2.399357
SCHOTTKY_C = 0.439229


class TestField:
    def test_sign_must_be_plus_or_minus_one(self):
        with pytest.raises(ValueError, match="sign must be"):
            Field("particles", 1.0, 0)

    def test_needs_a_variable_name(self):
        with pytest.raises(ValueError, match="name of its extensive variable"):
            Field("", 1.0, 1)


class TestEnsembleWeight:
    def test_canonical_weight_is_the_boltzmann_factor(self):
        assert Canonical(temperature=2.0).log_weight(3.0) == pytest.approx(-1.5)

    def test_a_field_trades_against_the_energy(self):
        ensemble = Magnetic(temperature=2.0, magnetic_field=0.5)
        got = ensemble.log_weight(3.0, magnetisation=4.0)
        assert got == pytest.approx(-(3.0 - 0.5 * 4.0) / 2.0)

    def test_negative_sign_adds_rather_than_subtracts(self):
        ensemble = Generalised(1.0, [Field("volume", 2.0, -1, "P")])
        assert ensemble.log_weight(1.0, volume=3.0) == pytest.approx(-(1.0 + 2.0 * 3.0))

    def test_change_is_the_difference_of_two_weights(self):
        # The identity the whole Metropolis loop rests on.
        ensemble = Magnetic(temperature=0.7, magnetic_field=-1.3)
        before = ensemble.log_weight(2.0, magnetisation=1.0)
        after = ensemble.log_weight(2.5, magnetisation=-1.0)
        change = ensemble.log_weight_change(0.5, magnetisation=-2.0)
        assert change == pytest.approx(after - before)

    def test_a_missing_free_variable_is_refused(self):
        with pytest.raises(ValueError, match="needs a value for 'magnetisation'"):
            Magnetic(1.0, 0.5).log_weight(1.0)

    def test_a_value_for_a_fixed_variable_is_refused(self):
        with pytest.raises(ValueError, match="holds"):
            Canonical(1.0).log_weight(1.0, magnetisation=2.0)

    def test_temperature_must_be_positive(self):
        with pytest.raises(ValueError, match="temperature must be positive"):
            Canonical(temperature=0.0)


class TestEnsembleStructure:
    def test_canonical_frees_nothing(self):
        assert Canonical(1.0).free == ()

    def test_magnetic_frees_the_magnetisation(self):
        assert Magnetic(1.0, 0.5).free == ("magnetisation",)

    def test_at_replaces_a_control_value(self):
        ensemble = Magnetic(temperature=1.0, magnetic_field=0.5)
        assert ensemble.at(temperature=3.0).magnetic_field == 0.5
        assert ensemble.at(temperature=3.0).temperature == 3.0

    def test_with_field_is_named_for_the_physics(self):
        assert Magnetic(1.0, 0.5).with_field("magnetisation", 2.0).magnetic_field == 2.0

    def test_asking_for_a_fixed_variables_field_raises(self):
        with pytest.raises(ValueError, match="holds 'particles' fixed"):
            Canonical(1.0).field("particles")

    def test_generalised_takes_arbitrary_conjugates(self):
        ensemble = Generalised(
            1.0, [Field("particles", -0.5, +1, "mu"), Field("magnetisation", 0.2, +1, "h")]
        )
        assert set(ensemble.free) == {"particles", "magnetisation"}

    def test_generalised_refuses_a_repeated_variable(self):
        with pytest.raises(ValueError, match="only once"):
            Generalised(1.0, [Field("particles", 1.0, 1), Field("particles", 2.0, 1)])

    def test_generalised_with_no_conjugates_is_canonical(self):
        assert Generalised(2.0).log_weight(3.0) == Canonical(2.0).log_weight(3.0)


class TestPotentialNaming:
    """
    Each ensemble generates a different potential, and they are all floats.
    Mislabelling one is invisible unless the name is checked.
    """

    def test_canonical_generates_the_free_energy(self):
        state = equilibrium(TwoLevel(1.0), Canonical(1.0))
        assert state.potential_name == "free_energy"
        assert state.free_energy == pytest.approx(state.potential)

    def test_canonical_has_no_grand_potential(self):
        state = equilibrium(TwoLevel(1.0), Canonical(1.0))
        with pytest.raises(ValueError, match="not the grand potential"):
            state.grand_potential

    def test_a_magnetic_ensemble_is_not_a_free_energy(self):
        state = equilibrium(Spin(0.5), Magnetic(1.0, 0.5))
        assert state.potential_name == "magnetic_free_energy"
        with pytest.raises(ValueError, match="not the free energy"):
            state.free_energy

    def test_free_energy_is_minus_t_log_z(self):
        state = equilibrium(TwoLevel(1.0), Canonical(0.5))
        assert state.free_energy == pytest.approx(-0.5 * state.log_z)


class TestCanonicalThermodynamics:
    @pytest.fixture
    def system(self):
        return TwoLevel(splitting=1.0)

    def test_schottky_peak_height(self, system):
        state = equilibrium(system, Canonical(temperature=1.0 / SCHOTTKY_X))
        assert state.heat_capacity == pytest.approx(SCHOTTKY_C, abs=1e-5)

    def test_schottky_peak_is_a_maximum(self, system):
        peak = 1.0 / SCHOTTKY_X
        centre = equilibrium(system, Canonical(peak)).heat_capacity
        for offset in (0.9, 1.1):
            assert equilibrium(system, Canonical(peak * offset)).heat_capacity < centre

    def test_heat_capacity_vanishes_at_both_extremes(self, system):
        assert equilibrium(system, Canonical(0.02)).heat_capacity < 1e-10
        assert equilibrium(system, Canonical(1e4)).heat_capacity < 1e-7

    def test_heat_capacity_matches_the_closed_form(self, system):
        for temperature in (0.2, 0.5, 1.0, 3.0):
            x = 1.0 / temperature
            expected = x**2 * math.exp(x) / (math.exp(x) + 1.0) ** 2
            got = equilibrium(system, Canonical(temperature)).heat_capacity
            assert got == pytest.approx(expected)

    def test_entropy_saturates_at_log_two(self, system):
        assert equilibrium(system, Canonical(1e6)).entropy == pytest.approx(
            math.log(2), abs=1e-10
        )

    def test_entropy_is_beta_u_plus_log_z(self, system):
        state = equilibrium(system, Canonical(0.8))
        assert state.entropy == pytest.approx(state.beta * state.energy + state.log_z)

    @pytest.mark.parametrize(
        "system, ground_degeneracy",
        [
            (TwoLevel(1.0), 1),
            (Degenerate(4), 4),
            (NLevel([0.0, 2.0], [3, 1]), 3),
            (HarmonicMode(1.0), 1),
        ],
    )
    def test_third_law(self, system, ground_degeneracy):
        """S -> log g0 as T -> 0, whatever the spectrum above it looks like."""
        state = equilibrium(system, Canonical(temperature=1e-3))
        assert state.entropy == pytest.approx(math.log(ground_degeneracy), abs=1e-9)

    def test_harmonic_energy_matches_the_closed_form(self):
        state = equilibrium(HarmonicMode(omega=1.0), Canonical(temperature=0.5))
        assert state.energy == pytest.approx(0.5 / math.tanh(1.0))

    def test_extensivity(self):
        """N copies have N times the energy, entropy and heat capacity."""
        one = equilibrium(TwoLevel(1.0), Canonical(0.7))
        many = equilibrium(TwoLevel(1.0) ** 20, Canonical(0.7))
        assert many.energy == pytest.approx(20.0 * one.energy)
        assert many.entropy == pytest.approx(20.0 * one.entropy)
        assert many.heat_capacity == pytest.approx(20.0 * one.heat_capacity)


class TestFreedVariables:
    @pytest.fixture
    def state(self):
        return equilibrium(Spin(0.5), Magnetic(temperature=1.0, magnetic_field=0.5))

    def test_magnetisation_follows_the_brillouin_function(self, state):
        assert state.magnetisation == pytest.approx(0.5 * math.tanh(0.25))

    def test_susceptibility_is_the_magnetisation_fluctuation(self, state):
        assert state.susceptibility == pytest.approx(
            state.beta * state.variance("magnetisation")
        )

    def test_susceptibility_matches_the_numerical_derivative(self):
        """Fluctuation-dissipation, checked against the derivative it replaces."""
        spin, temperature, field, step = Spin(0.5), 1.0, 0.5, 1e-5
        derivative = (
            equilibrium(spin, Magnetic(temperature, field + step)).magnetisation
            - equilibrium(spin, Magnetic(temperature, field - step)).magnetisation
        ) / (2.0 * step)
        exact = equilibrium(spin, Magnetic(temperature, field)).susceptibility
        assert exact == pytest.approx(derivative, rel=1e-7)

    def test_curie_law_at_high_temperature(self):
        """chi -> j(j+1)/(3T) = 1/(4T) for a spin one half."""
        temperature = 500.0
        state = equilibrium(Spin(0.5), Magnetic(temperature, 0.0))
        assert state.susceptibility == pytest.approx(1.0 / (4.0 * temperature), rel=1e-6)

    def test_saturation_at_low_temperature(self):
        state = equilibrium(Spin(1.5), Magnetic(temperature=1e-3, magnetic_field=1.0))
        assert state.magnetisation == pytest.approx(1.5)

    def test_field_for_inverts_the_magnetisation(self):
        state = equilibrium(Spin(0.5), Magnetic(1.0, 0.0))
        field = state.field_for("magnetisation", 0.3)
        assert equilibrium(Spin(0.5), Magnetic(1.0, field)).magnetisation == (
            pytest.approx(0.3)
        )

    def test_field_for_refuses_an_unreachable_target(self):
        # A spin one half saturates at 1/2; no field reaches 2.
        state = equilibrium(Spin(0.5), Magnetic(1.0, 0.0))
        with pytest.raises(ValueError, match="no field reproduces"):
            state.field_for("magnetisation", 2.0)

    def test_entropy_subtracts_the_traded_term(self, state):
        expected = (
            state.beta * (state.energy - 0.5 * state.magnetisation) + state.log_z
        )
        assert state.entropy == pytest.approx(expected)

    def test_magnetisation_needs_a_magnetic_ensemble(self):
        state = equilibrium(Spin(0.5), Canonical(1.0))
        with pytest.raises(ValueError, match="needs an ensemble that frees"):
            state.magnetisation


class Balloon(SpectralSystem):
    """
    A toy system for exercising a field that enters with sign -1.

    Each quantum of the mode costs `omega` and swells the box by `v`, so the
    volume is an extensive variable of the microstate and pressure is conjugate
    to it. Nothing in the catalogue carries a volume yet, and a user-defined
    system is exactly how one is meant to be supplied.
    """

    def __init__(self, omega: float, swell: float, n_max: int) -> None:
        self.omega, self.swell, self.n_max = omega, swell, n_max

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def n_states(self) -> int | None:
        return self.n_max + 1

    def levels(self):
        for n in range(self.n_max + 1):
            yield Level(self.omega * n, 1, {"volume": 1.0 + self.swell * n})


class TestNegativeSignField:
    """
    A field entering as `-beta P V` rather than `+beta h M`.

    Every freed variable in the catalogue so far has sign +1, which hides any
    place the sign is dropped: `<X>` is a plain weighted average and does not
    care, but its *derivative* does. Pressure is the standard case where it
    matters, and it must come out negative -- a gas compresses.
    """

    @pytest.fixture
    def ensemble(self):
        return Generalised(
            temperature=2.0, conjugates=[Field("volume", 0.5, -1, "P")]
        )

    @pytest.fixture
    def state(self, ensemble):
        return equilibrium(Balloon(1.0, 0.3, 40), ensemble)

    def test_the_potential_is_the_gibbs_energy(self, state):
        assert state.potential_name == "gibbs_energy"
        assert state.gibbs_energy == pytest.approx(state.potential)

    def test_the_volume_responds_negatively_to_pressure(self, state):
        """d<V>/dP < 0: raising the pressure shrinks the box."""
        assert state.response("volume") < 0.0
        assert state.response("volume") == pytest.approx(
            -state.beta * state.variance("volume")
        )

    def test_response_matches_the_numerical_derivative(self, ensemble, state):
        """The check that fails if the sign of the field is dropped."""
        step = 1e-5
        derivative = (
            equilibrium(state.system, ensemble.with_field("volume", 0.5 + step))
            .mean("volume")
            - equilibrium(state.system, ensemble.with_field("volume", 0.5 - step))
            .mean("volume")
        ) / (2.0 * step)
        assert state.response("volume") == pytest.approx(derivative, rel=1e-6)

    def test_field_for_inverts_a_decreasing_average(self, state):
        """Brackets have to cope with `<X>` falling as the field rises."""
        pressure = state.field_for("volume", 1.4)
        found = equilibrium(
            state.system, state.ensemble.with_field("volume", pressure)
        )
        assert found.mean("volume") == pytest.approx(1.4)
        # 1.4 is above the volume at P = 0.5, so it takes a *lower* pressure.
        assert pressure < 0.5

    def test_entropy_still_subtracts_the_traded_term(self, state):
        expected = (
            state.beta * (state.energy + 0.5 * state.mean("volume")) + state.log_z
        )
        assert state.entropy == pytest.approx(expected)

    @pytest.mark.parametrize("system", [HarmonicMode(1.0), TwoLevel(1.0)])
    def test_a_system_without_the_variable_is_refused(self, ensemble, system):
        # Refused where the two are joined, not deep inside a sweep, and
        # before any levels are touched.
        with pytest.raises(ValueError, match="frees \\['volume'\\]"):
            equilibrium(system, ensemble)


class TestEquilibrium:
    def test_rejects_a_non_system(self):
        with pytest.raises(TypeError, match="expected a System"):
            equilibrium(3.0, Canonical(1.0))

    def test_rejects_a_non_ensemble(self):
        with pytest.raises(TypeError, match="expected an Ensemble"):
            equilibrium(TwoLevel(1.0), 2.0)

    def test_returns_a_thermal_state(self):
        assert isinstance(equilibrium(TwoLevel(1.0), Canonical(1.0)), ThermalState)

    def test_the_same_system_serves_several_ensembles(self):
        """The reason systems and ensembles are kept apart."""
        spin = Spin(0.5)
        canonical = equilibrium(spin, Canonical(1.0))
        magnetic = equilibrium(spin, Magnetic(1.0, 0.5))
        assert canonical.entropy == pytest.approx(math.log(2))
        assert magnetic.entropy < math.log(2)

    def test_a_temperature_sweep_reuses_the_system(self):
        rotor = Rotor(b=1.0)
        base = Canonical(temperature=1.0)
        energies = [
            equilibrium(rotor, base.at(temperature=t)).energy for t in (1.0, 10.0, 100.0)
        ]
        assert energies == sorted(energies)


class TestGuards:
    """The ensemble and state refusals, none of which the physics tests reach."""

    def test_temperature_must_be_finite(self):
        with pytest.raises(ValueError, match="temperature must be finite"):
            Canonical(temperature=math.inf)

    def test_field_for_rejects_a_non_finite_target(self):
        state = equilibrium(Spin(0.5), Magnetic(1.0, 0.0))
        with pytest.raises(ValueError, match="target must be finite"):
            state.field_for("magnetisation", math.inf)

    def test_generalised_rejects_a_non_field(self):
        with pytest.raises(TypeError, match="expected a Field"):
            Generalised(1.0, ["magnetisation"])

    def test_with_field_needs_the_parameter_to_be_declared(self):
        """
        `with_field` is named for the physics, so it needs the map from
        variable to constructor parameter. An ensemble that frees something
        without declaring where the field lives cannot be re-valued.
        """

        class Undeclared(Ensemble):
            @property
            def fields(self):
                return (Field("magnetisation", 0.5, +1, "h"),)

        with pytest.raises(ValueError, match="does not say which parameter"):
            Undeclared(temperature=1.0).with_field("magnetisation", 2.0)
