# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The classical limit: the ideal gas, its pressure, and the isobaric ensemble.

`IdealGas` is the first system in the catalogue with no levels at all, and the
first with a closed form in two different ensembles. So these tests check two
things at once -- the textbook results, and the machinery that gets to them
without an enumerable spectrum: every moment here comes from the base-class
finite differences at fixed natural parameter, not from a sweep.

Two textbook formulas are *asymptotic* and this file says so rather than
loosening a tolerance around them. Sackur-Tetrode drops the `ln(2 pi N)/2N`
Stirling term, and the high-temperature box drops the Euler-Maclaurin `-1/2`;
the exact results are tested against exact forms, and the asymptotes against
their own convergence.
"""

import math
from dataclasses import dataclass, replace

import pytest

from eigora.statphys import (
    Box1D,
    Canonical,
    Field,
    Generalised,
    IdealGas,
    IsothermalIsobaric,
    Level,
    ParametrisedSystem,
    SpectralSystem,
    equilibrium,
    particle_in_box,
)


def wavelength(temperature, mass=1.0):
    """The thermal de Broglie wavelength, sqrt(2 pi beta / m) with hbar = 1."""
    return math.sqrt(2.0 * math.pi / (temperature * mass))


class TestIdealGasCanonical:
    PARTICLES, VOLUME, TEMPERATURE = 100, 50.0, 2.0

    @pytest.fixture
    def state(self):
        gas = IdealGas(particles=self.PARTICLES, volume=self.VOLUME)
        return equilibrium(gas, Canonical(self.TEMPERATURE))

    def test_log_z_is_the_closed_form(self, state):
        n, v, t = self.PARTICLES, self.VOLUME, self.TEMPERATURE
        expected = n * math.log(v / wavelength(t) ** 3) - math.lgamma(n + 1)
        assert state.log_z == pytest.approx(expected, rel=1e-14)

    def test_equipartition(self, state):
        """U = (3/2) N T: half a T per quadratic degree of freedom."""
        assert state.energy == pytest.approx(
            1.5 * self.PARTICLES * self.TEMPERATURE, rel=1e-9
        )

    @pytest.mark.parametrize("ndim", (1, 2, 3))
    def test_equipartition_in_any_dimension(self, ndim):
        gas = IdealGas(particles=8, volume=20.0, ndim=ndim)
        state = equilibrium(gas, Canonical(1.5))
        assert state.energy == pytest.approx(0.5 * ndim * 8 * 1.5, rel=1e-9)

    def test_the_ideal_gas_law(self, state):
        """PV = NT, from differentiating log Z with respect to the volume."""
        assert state.pressure == pytest.approx(
            self.PARTICLES * self.TEMPERATURE / self.VOLUME, rel=1e-8
        )

    def test_heat_capacity_is_three_halves_n(self, state):
        assert state.heat_capacity == pytest.approx(1.5 * self.PARTICLES, rel=1e-4)

    def test_chemical_potential(self, state):
        """mu = -T log(V / (N lambda^3)), and it is exact: N steps by one."""
        n, v, t = self.PARTICLES, self.VOLUME, self.TEMPERATURE
        expected = -t * math.log(v / (n * wavelength(t) ** 3))
        assert state.chemical_potential == pytest.approx(expected, rel=1e-12)

    def test_chemical_potential_is_a_difference_of_free_energies(self, state):
        gas = IdealGas(particles=self.PARTICLES - 1, volume=self.VOLUME)
        fewer = equilibrium(gas, Canonical(self.TEMPERATURE))
        assert state.chemical_potential == pytest.approx(
            state.free_energy - fewer.free_energy, rel=1e-14
        )

    def test_enthalpy(self, state):
        """H = U + PV = (3/2)NT + NT."""
        assert state.enthalpy == pytest.approx(
            2.5 * self.PARTICLES * self.TEMPERATURE, rel=1e-8
        )

    def test_entropy_is_exact_not_asymptotic(self, state):
        """S = beta(U - F) with the true log N!, no Stirling anywhere."""
        n, v, t = self.PARTICLES, self.VOLUME, self.TEMPERATURE
        expected = (
            n * math.log(v / wavelength(t) ** 3) - math.lgamma(n + 1) + 1.5 * n
        )
        # 1e-8, matching the documented accuracy of the beta difference that
        # supplies U -- log Z itself is exact to roundoff.
        assert state.entropy == pytest.approx(expected, rel=1e-8)

    def test_sackur_tetrode_is_the_asymptote(self, state):
        """
        S/N -> log(V/(N lambda^3)) + 5/2, missing exactly the Stirling tail.

        The gap is not a tolerance to be widened but the discarded terms of
        Stirling's series, `[log(2 pi N)/2 + 1/(12N)] / N`. Asserting it to
        six digits pins that the entropy keeps the true `log N!`; dropping the
        `1/(12N)` term alone already shows up at 2.6e-4 relative, so this is a
        sharp check rather than a decorated one.
        """
        n, v, t = self.PARTICLES, self.VOLUME, self.TEMPERATURE
        sackur_tetrode = math.log(v / (n * wavelength(t) ** 3)) + 2.5
        gap = sackur_tetrode - state.entropy / n
        stirling_tail = (0.5 * math.log(2.0 * math.pi * n) + 1.0 / (12.0 * n)) / n
        assert gap == pytest.approx(stirling_tail, rel=1e-6)

    def test_sackur_tetrode_converges(self):
        """And the gap really does vanish, so the asymptote is the right one."""
        t, density = 2.0, 2.0
        gaps = []
        for n in (10, 100, 1000):
            gas = IdealGas(particles=n, volume=n * density)
            state = equilibrium(gas, Canonical(t))
            sackur_tetrode = math.log(density / wavelength(t) ** 3) + 2.5
            gaps.append(abs(sackur_tetrode - state.entropy / n))
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[-1] < 1e-2


class TestClassicalLimit:
    """
    The quantum box becomes the classical gas, summed against closed form.

    `particle_in_box` is `Box1D ** 3`, so its log Z is three truncated sums;
    `IdealGas(particles=1)` is one closed form. They must meet as the thermal
    wavelength shrinks below the box.
    """

    def test_the_level_sum_approaches_the_closed_form(self):
        length, dimensions = 10.0, 3
        box = particle_in_box(length, ndim=dimensions)
        gas = IdealGas(particles=1, volume=length**dimensions, ndim=dimensions)

        errors = []
        for temperature in (100.0, 10_000.0, 1_000_000.0):
            beta = 1.0 / temperature
            errors.append(abs(box.log_z(beta) - gas.log_z(beta)))
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 1e-2

    def test_the_leading_correction_is_euler_maclaurin(self):
        """
        A one-dimensional box has `Z = L/lambda - 1/2` at high T.

        Summing from n = 1 rather than integrating from 0 costs half the first
        term, so the pressure is `T / (L - lambda/2)` and not `T / L`. Pinning
        the corrected form rather than the leading one turns a 1% discrepancy
        from a loose tolerance into a statement about what is being computed.
        """
        length = 4.0
        for temperature in (500.0, 5000.0):
            state = equilibrium(Box1D(length=length), Canonical(temperature))
            corrected = temperature / (length - wavelength(temperature) / 2.0)
            assert state.pressure == pytest.approx(corrected, rel=1e-7)


class TestIsothermalIsobaric:
    PARTICLES, PRESSURE, TEMPERATURE = 100, 4.0, 2.0

    @pytest.fixture
    def state(self):
        gas = IdealGas(particles=self.PARTICLES)
        return equilibrium(
            gas, IsothermalIsobaric(self.TEMPERATURE, self.PRESSURE)
        )

    def test_the_potential_is_the_gibbs_energy(self, state):
        assert state.potential_name == "gibbs_energy"
        assert state.gibbs_energy == pytest.approx(state.potential)
        with pytest.raises(ValueError, match="not the free energy"):
            state.free_energy

    def test_log_delta_is_the_closed_form(self, state):
        n, p, t = self.PARTICLES, self.PRESSURE, self.TEMPERATURE
        expected = -n * 3.0 * math.log(wavelength(t)) - (n + 1) * math.log(p / t)
        assert state.log_z == pytest.approx(expected, rel=1e-14)

    def test_mean_volume_carries_the_n_plus_one(self, state):
        """
        <V> = (N+1) T / P, not N T / P.

        A real finite-size effect of holding the pressure instead of the
        volume -- the extra 1 comes from the `V^N dV` measure -- so the
        `N + 1` is asserted rather than absorbed into a tolerance.
        """
        expected = (self.PARTICLES + 1) * self.TEMPERATURE / self.PRESSURE
        assert state.mean("volume") == pytest.approx(expected, rel=1e-8)

    def test_volume_fluctuations(self, state):
        """Var(V) = (N+1) (T/P)^2, the variance of a gamma distribution."""
        expected = (self.PARTICLES + 1) * (self.TEMPERATURE / self.PRESSURE) ** 2
        assert state.variance("volume") == pytest.approx(expected, rel=1e-4)

    def test_the_volume_shrinks_under_pressure(self, state):
        """d<V>/dP < 0, the sign that only survives if the field's is carried."""
        assert state.response("volume") < 0.0
        step = 1e-4
        ensemble = state.ensemble
        derivative = (
            equilibrium(state.system, ensemble.at(pressure=self.PRESSURE + step))
            .mean("volume")
            - equilibrium(state.system, ensemble.at(pressure=self.PRESSURE - step))
            .mean("volume")
        ) / (2.0 * step)
        assert state.response("volume") == pytest.approx(derivative, rel=1e-5)

    def test_pressure_is_the_field_it_was_given(self, state):
        assert state.pressure == pytest.approx(self.PRESSURE)

    def test_energy_is_still_equipartition(self, state):
        """
        U = (3/2)NT even though the volume fluctuates.

        The check that the beta-derivative holds `beta P` fixed rather than
        `P`: at fixed P it would pick up `<E> + P<V>`, the enthalpy, which is
        larger by `(N+1)T` and just as plausible a float.
        """
        assert state.energy == pytest.approx(
            1.5 * self.PARTICLES * self.TEMPERATURE, rel=1e-8
        )

    def test_enthalpy_uses_the_mean_volume(self, state):
        expected = state.energy + self.PRESSURE * state.mean("volume")
        assert state.enthalpy == pytest.approx(expected)

    def test_gibbs_approaches_f_plus_pv(self):
        """
        G = F + PV, in the thermodynamic limit and not before.

        The isobaric G and the canonical F evaluated at the mean volume differ
        by O(log N) terms -- the same `N+1` and Stirling remainders as above --
        so the relative gap shrinking with N is the honest statement.
        """
        temperature, pressure = 2.0, 4.0
        gaps = []
        for n in (10, 100, 1000):
            isobaric = equilibrium(
                IdealGas(particles=n), IsothermalIsobaric(temperature, pressure)
            )
            volume = isobaric.mean("volume")
            canonical = equilibrium(
                IdealGas(particles=n, volume=volume), Canonical(temperature)
            )
            legendre = canonical.free_energy + pressure * volume
            gaps.append(abs(isobaric.gibbs_energy - legendre) / abs(legendre))
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[-1] < 1e-2


class TestFixedOrFreeNeverBoth:
    """
    A volume is either a parameter or a coupling. Never both, never neither.
    """

    def test_a_fixed_volume_refuses_an_isobaric_ensemble(self):
        gas = IdealGas(particles=10, volume=5.0)
        assert gas.extensive_variables == frozenset()
        with pytest.raises(ValueError, match="frees \\['volume'\\]"):
            equilibrium(gas, IsothermalIsobaric(1.0, 1.0))

    def test_a_free_volume_refuses_a_canonical_ensemble(self):
        gas = IdealGas(particles=10)
        assert gas.extensive_variables == {"volume"}
        with pytest.raises(ValueError, match="has no volume"):
            equilibrium(gas, Canonical(1.0)).log_z

    def test_a_fixed_volume_is_a_parameter_and_a_free_one_is_not(self):
        assert "volume" in IdealGas(particles=10, volume=5.0).parameters
        assert "volume" not in IdealGas(particles=10).parameters


class TestParameterGuards:
    def test_pressure_needs_a_volume(self):
        from eigora.statphys import TwoLevel

        state = equilibrium(TwoLevel(1.0), Canonical(1.0))
        with pytest.raises(ValueError, match="pressure needs 'volume'"):
            state.pressure

    def test_chemical_potential_needs_a_particle_count(self):
        state = equilibrium(Box1D(1.0), Canonical(1.0))
        with pytest.raises(ValueError, match="chemical_potential needs 'particles'"):
            state.chemical_potential

    def test_chemical_potential_needs_a_particle_to_remove(self):
        state = equilibrium(IdealGas(particles=0, volume=1.0), Canonical(1.0))
        with pytest.raises(ValueError, match="at least one particle"):
            state.chemical_potential

    def test_enthalpy_needs_a_volume(self):
        from eigora.statphys import TwoLevel

        state = equilibrium(TwoLevel(1.0), Canonical(1.0))
        with pytest.raises(ValueError, match="enthalpy needs 'volume'"):
            state.enthalpy

    @pytest.mark.parametrize(
        "build, message",
        [
            (lambda: IdealGas(particles=-1), "particles must be non-negative"),
            (lambda: IdealGas(particles=1, volume=0.0), "volume must be positive"),
            (lambda: IdealGas(particles=1, mass=0.0), "mass must be positive"),
            (lambda: IdealGas(particles=1, ndim=0), "ndim must be at least 1"),
            (lambda: IsothermalIsobaric(1.0, 0.0), "pressure must be positive"),
        ],
    )
    def test_construction_guards(self, build, message):
        with pytest.raises(ValueError, match=message):
            build()

    def test_a_negative_pressure_coupling_is_refused(self):
        with pytest.raises(ValueError, match="needs a positive pressure"):
            IdealGas(particles=2).log_z(1.0, (("volume", 0.5),))

    def test_beta_is_guarded(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            IdealGas(particles=2, volume=1.0).log_z(0.0)

    def test_at_rebuilds_the_gas(self):
        gas = IdealGas(particles=10, volume=5.0)
        assert gas.at(particles=7.0).particles == 7
        assert gas.at(volume=8.0).volume == 8.0

    def test_box_at_renames_volume_to_length(self):
        assert Box1D(length=2.0).at(volume=5.0).length == 5.0
        assert Box1D(length=2.0).parameters == {"volume": 2.0}


class TestConjugateRoutes:
    """
    Both conjugates read as an input when their variable is free.

    `pressure` and `chemical_potential` are outputs when their variable is a
    fixed parameter and inputs when it is a coupling. The second direction has
    no catalogue system until the quantum gases arrive, so it is exercised here
    with a two-orbital system written for the purpose.
    """

    class TwoOrbitals(SpectralSystem):
        """Four occupations of two orbitals, each reporting its particle count."""

        @property
        def is_exact(self):
            return True

        @property
        def n_states(self):
            return 4

        def levels(self):
            for energy, count in [(0.0, 0), (1.0, 1), (2.0, 1), (3.0, 2)]:
                yield Level(energy, 1, {"particles": float(count)})

    def test_chemical_potential_is_the_field_when_particles_are_free(self):
        grand = Generalised(1.0, [Field("particles", -0.4, +1, "mu")])
        state = equilibrium(self.TwoOrbitals(), grand)
        assert state.chemical_potential == pytest.approx(-0.4)

    def test_pressure_is_the_field_when_the_volume_is_free(self):
        state = equilibrium(IdealGas(particles=5), IsothermalIsobaric(1.5, 2.5))
        assert state.pressure == pytest.approx(2.5)

    def test_a_zero_parameter_cannot_be_differentiated(self):
        """A relative step around zero is no step at all, so say so."""

        @dataclass(frozen=True)
        class Flat(SpectralSystem, ParametrisedSystem):
            volume: float = 0.0

            @property
            def n_states(self):
                return 1

            def levels(self):
                yield Level(0.0)

            @property
            def parameters(self):
                return {"volume": self.volume}

            def at(self, **changes):
                return replace(self, **changes)

        state = equilibrium(Flat(), Canonical(1.0))
        with pytest.raises(ValueError, match="cannot differentiate at volume = 0"):
            state.pressure

    def test_the_gas_is_exact(self):
        assert IdealGas(particles=3, volume=1.0).is_exact
