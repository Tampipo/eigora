# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Quantum statistics with the particle number free.

The claim being tested is a factorisation: summing over the 2^K occupations of
K orbitals must give the same numbers as summing over the K orbitals. The first
class here does exactly that, enumerating every configuration by brute force
and holding it against `IdenticalParticles` -- two genuinely different
computations, one exponential and one linear.

After that, the textbook consequences: the Fermi step sharpening as T -> 0, the
Fermi energy landing between the last filled and first empty orbital, Pauli
suppression of fluctuations at a full orbital, bosonic bunching, and both
converging on Boltzmann when the gas is dilute.
"""

import itertools
import math

import pytest

from eigora.statphys import (
    BOLTZMANN,
    BOSE,
    FERMI,
    Canonical,
    Degenerate,
    GrandCanonical,
    HarmonicMode,
    IdenticalParticles,
    NLevel,
    Statistics,
    TwoLevel,
    equilibrium,
    with_degeneracy,
)

LADDER = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

#: The brute force is exponential in the orbital count *and* in the occupancy
#: ceiling, so the configuration sums use a shorter ladder than everything
#: else. Three orbitals at twenty bosons each is 9261 terms; the six-orbital
#: ladder at the same ceiling would be eighty-five million, which is precisely
#: why the factorised form is worth having.
SHORT = [0.5, 1.5, 2.5]


def enumerate_grand(energies, beta, mu, statistics, max_occupancy=20):
    """
    log Xi by brute force over occupation vectors -- the definition.

    Exponential in the number of orbitals, which is the whole reason the
    factorised form exists. Bosons are truncated at `max_occupancy`; with the
    largest fugacity here around 0.29 the discarded tail is ~1e-11, well below
    the tolerances asserted against it.
    """
    ceiling = 1 if statistics is FERMI else max_occupancy
    total = 0.0
    for occupation in itertools.product(range(ceiling + 1), repeat=len(energies)):
        count = sum(occupation)
        energy = sum(n * e for n, e in zip(occupation, energies))
        weight = math.exp(-beta * (energy - mu * count))
        if statistics is BOLTZMANN:
            weight /= math.prod(math.factorial(n) for n in occupation)
        total += weight
    return math.log(total)


class TestFactorisation:
    """
    2^K configurations against K orbitals. The one check worth having.
    """

    @pytest.mark.parametrize("statistics", [FERMI, BOSE, BOLTZMANN])
    def test_log_xi_matches_brute_force(self, statistics):
        beta, mu = 1.25, -0.5
        gas = IdenticalParticles(NLevel(SHORT), statistics)
        assert gas.log_z(beta, (("particles", mu),)) == pytest.approx(
            enumerate_grand(SHORT, beta, mu, statistics), rel=1e-10
        )

    @pytest.mark.parametrize("statistics", [FERMI, BOSE, BOLTZMANN])
    def test_moments_match_brute_force(self, statistics):
        """<N>, <E> and their variances, from the derivative of the brute sum."""
        beta, mu, step = 1.25, -0.5, 1e-5
        gas = IdenticalParticles(NLevel(SHORT), statistics)
        moments = gas.moments(beta, (("particles", mu),))

        def brute(chemical_potential):
            return enumerate_grand(SHORT, beta, chemical_potential, statistics)

        number = (brute(mu + step) - brute(mu - step)) / (2.0 * step * beta)
        variance = (brute(mu + step) - 2.0 * brute(mu) + brute(mu - step)) / (
            (step * beta) ** 2
        )
        assert moments.means["particles"] == pytest.approx(number, rel=1e-6)
        assert moments.variances["particles"] == pytest.approx(variance, rel=1e-4)

    def test_degeneracy_counts_orbitals_not_levels(self):
        """
        Two orbitals at one energy is not one orbital counted twice.

        `NLevel([e], [2])` must equal `NLevel([e, e])` flattened -- which for
        fermions means two particles fit, not one. This is the check that spin
        multiplicity gives Pauli counting for free.
        """
        beta, mu = 1.0, 0.5
        degenerate = IdenticalParticles(NLevel([0.0], [2]), FERMI)
        pair = IdenticalParticles(NLevel([0.0, 1e-12]), FERMI)
        assert degenerate.log_z(beta, (("particles", mu),)) == pytest.approx(
            pair.log_z(beta, (("particles", mu),)), rel=1e-9
        )
        assert degenerate.moments(beta, (("particles", mu),)).means[
            "particles"
        ] == pytest.approx(2.0 / (math.exp(beta * -mu) + 1.0))


class TestFermi:
    @pytest.fixture
    def gas(self):
        return IdenticalParticles(NLevel(LADDER), FERMI)

    def test_occupation_is_fermi_dirac(self, gas):
        beta, mu = 2.0, 2.5
        for energy in LADDER:
            assert gas.occupation(energy, beta, mu) == pytest.approx(
                1.0 / (math.exp(beta * (energy - mu)) + 1.0)
            )

    def test_half_filling_at_the_chemical_potential(self, gas):
        assert gas.occupation(2.5, 3.0, 2.5) == pytest.approx(0.5)

    def test_the_step_sharpens_as_temperature_falls(self, gas):
        """The Fermi step: below mu filled, above mu empty, sharper as T -> 0."""
        mu = 3.0
        widths = []
        for temperature in (1.0, 0.2, 0.02):
            beta = 1.0 / temperature
            below = gas.occupation(2.5, beta, mu)
            above = gas.occupation(3.5, beta, mu)
            widths.append(below - above)
        assert widths[0] < widths[1] < widths[2]
        assert widths[-1] == pytest.approx(1.0, abs=1e-8)

    def test_the_fermi_energy_is_exact_at_half_filling(self, gas):
        """
        mu = 3.0 gives exactly <N> = 3 on a six-orbital ladder, at any T.

        Particle-hole symmetry: the orbitals 0.5, 1.5, 2.5 mirror 3.5, 4.5,
        5.5 about 3.0, so every particle below is matched by a hole above and
        the two errors cancel identically. No thermodynamic limit needed --
        which is what makes it sharp.

        It is *only* exact at half filling. For N = 1 the spectrum is not
        symmetric about the gap, so the midpoint is an approximation, and the
        next test says what does hold in general.
        """
        for temperature in (0.05, 0.5, 2.0):
            state = equilibrium(gas, GrandCanonical(temperature, 3.0))
            assert state.particles == pytest.approx(3.0, abs=1e-12)

    def test_the_fermi_energy_falls_in_the_gap(self, gas):
        """
        mu(N) lies between eps_{N-1} and eps_N -- the general statement.

        At T -> 0 exactly, `<N>` is a staircase and any mu in the gap gives
        the same N, so the inverse is not unique and only the bracket is
        meaningful. At small but finite T it is unique, and this is where it
        lands.
        """
        state = equilibrium(gas, GrandCanonical(temperature=0.02, chemical_potential=0.0))
        for count in (1, 2, 3, 4, 5):
            chemical_potential = state.chemical_potential_for(float(count))
            assert LADDER[count - 1] < chemical_potential < LADDER[count]

    def test_particle_number_is_exact_at_low_temperature(self, gas):
        state = equilibrium(gas, GrandCanonical(temperature=1e-3, chemical_potential=3.0))
        assert state.particles == pytest.approx(3.0, abs=1e-9)

    def test_pauli_suppresses_fluctuations_when_full(self, gas):
        """Var(N) -> 0 as the gas freezes: a filled orbital cannot fluctuate."""
        cold = equilibrium(gas, GrandCanonical(1e-2, 3.0))
        warm = equilibrium(gas, GrandCanonical(1.0, 3.0))
        # n(1-n) ~ exp(-50) at T = 0.01 with the nearest orbital half a unit
        # away, so the floor is 1e-22 and not something smaller.
        assert cold.variance("particles") < 1e-20
        assert warm.variance("particles") > 0.5

    def test_the_potential_is_the_grand_potential(self, gas):
        state = equilibrium(gas, GrandCanonical(0.8, 2.0))
        assert state.potential_name == "grand_potential"
        assert state.grand_potential == pytest.approx(state.potential)
        with pytest.raises(ValueError, match="not the free energy"):
            state.free_energy

    def test_chemical_potential_is_the_field_it_was_given(self, gas):
        state = equilibrium(gas, GrandCanonical(0.8, 2.0))
        assert state.chemical_potential == pytest.approx(2.0)

    def test_entropy_subtracts_mu_n(self, gas):
        """S = beta(U - mu N) + log Xi, the general formula in this ensemble."""
        state = equilibrium(gas, GrandCanonical(0.8, 2.0))
        expected = (
            state.beta * (state.energy - 2.0 * state.particles) + state.log_z
        )
        assert state.entropy == pytest.approx(expected)


class TestBose:
    @pytest.fixture
    def gas(self):
        return IdenticalParticles(NLevel(LADDER), BOSE)

    def test_occupation_is_bose_einstein(self, gas):
        beta, mu = 1.5, -0.5
        for energy in LADDER:
            assert gas.occupation(energy, beta, mu) == pytest.approx(
                1.0 / (math.exp(beta * (energy - mu)) - 1.0)
            )

    def test_bunching_exceeds_the_fermi_fluctuation(self, gas):
        """Var(N) = sum n(1+n) for bosons against n(1-n) for fermions."""
        beta, mu = 1.0, -0.2
        bosons = gas.moments(beta, (("particles", mu),))
        fermions = IdenticalParticles(NLevel(LADDER), FERMI).moments(
            beta, (("particles", mu),)
        )
        assert bosons.variances["particles"] > fermions.variances["particles"]

    def test_occupation_diverges_as_mu_approaches_the_ground_orbital(self, gas):
        beta = 1.0
        occupancies = [
            gas.occupation(LADDER[0], beta, LADDER[0] - gap)
            for gap in (0.1, 0.01, 0.001)
        ]
        assert occupancies[0] < occupancies[1] < occupancies[2]
        assert occupancies[-1] > 900.0

    @pytest.mark.parametrize("chemical_potential", (0.5, 0.6, 10.0))
    def test_condensation_is_refused_by_name(self, gas, chemical_potential):
        """
        mu >= eps_0 makes `log1p(-x)` return nan. Refuse it instead.

        A `nan` would propagate silently through every derived quantity; the
        error says which orbital energy the chemical potential has to stay
        below, and why.
        """
        with pytest.raises(ValueError, match="Bose-Einstein condensation"):
            equilibrium(gas, GrandCanonical(1.0, chemical_potential)).log_z

    def test_fermions_have_no_such_restriction(self):
        gas = IdenticalParticles(NLevel(LADDER), FERMI)
        assert math.isfinite(equilibrium(gas, GrandCanonical(1.0, 10.0)).log_z)


class TestBoltzmannLimit:
    """
    Both statistics become classical when the orbitals are nearly empty.
    """

    @pytest.mark.parametrize("statistics", [FERMI, BOSE])
    def test_dilute_gases_converge_on_boltzmann(self, statistics):
        beta = 1.0
        quantum = IdenticalParticles(NLevel(LADDER), statistics)
        classical = IdenticalParticles(NLevel(LADDER), BOLTZMANN)
        gaps = []
        for mu in (-2.0, -6.0, -12.0):
            couplings = (("particles", mu),)
            gaps.append(
                abs(quantum.log_z(beta, couplings) - classical.log_z(beta, couplings))
                / abs(classical.log_z(beta, couplings))
            )
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[-1] < 1e-4

    def test_fermi_below_boltzmann_below_bose(self):
        """Exclusion suppresses log Xi, bunching enhances it."""
        beta, couplings = 1.0, (("particles", -0.5),)
        orbitals = NLevel(LADDER)
        fermi = IdenticalParticles(orbitals, FERMI).log_z(beta, couplings)
        classical = IdenticalParticles(orbitals, BOLTZMANN).log_z(beta, couplings)
        bose = IdenticalParticles(orbitals, BOSE).log_z(beta, couplings)
        assert fermi < classical < bose

    def test_boltzmann_occupation_is_the_bare_exponential(self):
        gas = IdenticalParticles(NLevel(LADDER), BOLTZMANN)
        assert gas.occupation(2.0, 1.5, 0.5) == pytest.approx(math.exp(-1.5 * 1.5))

    def test_boltzmann_number_fluctuation_is_poisson(self):
        """Var(N) = <N> when occupations are independent."""
        gas = IdenticalParticles(NLevel(LADDER), BOLTZMANN)
        moments = gas.moments(1.0, (("particles", -1.0),))
        assert moments.variances["particles"] == pytest.approx(
            moments.means["particles"]
        )


class TestStatistics:
    def test_the_two_signs_are_opposite(self):
        """
        The distinction a single `sign` field would collapse.

        `grand_sign` and `recursion_sign` disagree for both quantum statistics,
        so one field would be right in the grand canonical sums and wrong in
        the fixed-N recursion, or the reverse.
        """
        for statistics in (FERMI, BOSE):
            assert statistics.grand_sign == -statistics.recursion_sign
        assert BOLTZMANN.grand_sign == BOLTZMANN.recursion_sign == 0

    def test_log_term_has_no_division_by_zero_for_boltzmann(self):
        """s = 0 would divide by zero; the limit is `x = e^-y`."""
        assert BOLTZMANN.log_term(1.2) == pytest.approx(math.exp(-1.2))

    def test_log_term_is_continuous_across_the_boltzmann_limit(self):
        """`log(1+sx)/s -> x` as s -> 0, so tiny x makes all three agree."""
        exponent = 20.0  # x = e^-20 ~ 2e-9
        for statistics in (FERMI, BOSE):
            assert statistics.log_term(exponent) == pytest.approx(
                BOLTZMANN.log_term(exponent), rel=1e-8
            )

    @pytest.mark.parametrize("statistics", [FERMI, BOSE, BOLTZMANN])
    def test_an_empty_orbital_contributes_nothing(self, statistics):
        assert statistics.occupation(800.0) == 0.0
        assert statistics.log_term(800.0) == pytest.approx(0.0, abs=1e-300)

    def test_a_deeply_filled_fermi_orbital_saturates(self):
        """The case that returned `nan` when the fugacity was clamped to inf."""
        assert FERMI.occupation(-800.0) == pytest.approx(1.0)
        assert FERMI.log_term(-800.0) == pytest.approx(800.0)

    def test_boltzmann_diverges_where_fermi_saturates(self):
        assert BOLTZMANN.log_term(-800.0) == math.inf

    @pytest.mark.parametrize(
        "statistics, occupancy, expected",
        [(FERMI, 0.25, 0.1875), (BOSE, 0.25, 0.3125), (BOLTZMANN, 0.25, 0.25)],
    )
    def test_occupation_variance(self, statistics, occupancy, expected):
        assert statistics.occupation_variance(occupancy) == pytest.approx(expected)


class TestSpinDegeneracy:
    """
    Spin enters as an orbital degeneracy, not as a special case.
    """

    def test_two_orbitals_per_level_hold_two_fermions(self):
        orbitals = with_degeneracy(NLevel([0.0, 1.0]), 2)
        gas = IdenticalParticles(orbitals, FERMI)
        state = equilibrium(gas, GrandCanonical(temperature=1e-3, chemical_potential=0.5))
        assert state.particles == pytest.approx(2.0, abs=1e-9)

    def test_the_wrong_ordering_doubles_xi_instead(self):
        """
        The documented foot-gun, asserted so it stays documented.

        Multiplying the *gas* by a degeneracy doubles log Xi -- two independent
        copies -- rather than giving each level two orbitals. Both are legal
        systems; only one is a spin.
        """
        orbitals = NLevel([0.0, 1.0])
        right = IdenticalParticles(with_degeneracy(orbitals, 2), FERMI)
        wrong = IdenticalParticles(orbitals, FERMI) * Degenerate(2)
        beta, couplings = 1.0, (("particles", 0.5),)
        # A two-state block at zero energy multiplies Xi by 2, so log Xi gains
        # log 2 -- one extra bit of entropy bolted on beside the gas, which is
        # not what a spin degree of freedom does to it.
        assert wrong.log_z(beta, couplings) == pytest.approx(
            IdenticalParticles(orbitals, FERMI).log_z(beta, couplings) + math.log(2.0)
        )
        assert right.log_z(beta, couplings) != pytest.approx(
            wrong.log_z(beta, couplings)
        )

    def test_composing_orbitals_is_refused_rather_than_miscounted(self):
        """`base * Degenerate(2)` has no levels, so it cannot supply orbitals."""
        with pytest.raises(TypeError, match="must be a SpectralSystem"):
            IdenticalParticles(NLevel([0.0, 1.0]) * Degenerate(2), FERMI)

    def test_with_degeneracy_multiplies_rather_than_replaces(self):
        doubled = with_degeneracy(NLevel([0.0, 1.0], [2, 3]), 2)
        assert [level.degeneracy for level in doubled.levels()] == [4, 6]
        assert doubled.n_states == 10
        assert doubled.is_exact
        assert with_degeneracy(HarmonicMode(1.0), 3).n_states is None

    def test_with_degeneracy_guards_its_arguments(self):
        with pytest.raises(TypeError, match="base must be a SpectralSystem"):
            with_degeneracy(NLevel([0.0]) * NLevel([1.0]), 2)
        with pytest.raises(ValueError, match="multiplicity must be at least 1"):
            with_degeneracy(NLevel([0.0]), 0)

    def test_particle_fluctuation_is_the_response(self):
        """d<N>/dmu = beta Var(N), the compressibility in this ensemble."""
        gas = IdenticalParticles(NLevel(LADDER), FERMI)
        state = equilibrium(gas, GrandCanonical(0.8, 2.0))
        assert state.particle_fluctuation == pytest.approx(
            state.beta * state.variance("particles")
        )
        assert state.particle_fluctuation > 0.0


class TestGuards:
    def test_orbitals_must_be_enumerable(self):
        from eigora.statphys import IdealGas

        with pytest.raises(TypeError, match="must be a SpectralSystem"):
            IdenticalParticles(IdealGas(particles=1, volume=1.0), FERMI)

    def test_statistics_must_be_statistics(self):
        with pytest.raises(TypeError, match="expected Statistics"):
            IdenticalParticles(NLevel(LADDER), "fermi")

    def test_a_canonical_ensemble_is_refused(self):
        """N is free here, so an ensemble that fixes it is the wrong one."""
        gas = IdenticalParticles(NLevel(LADDER), FERMI)
        with pytest.raises(ValueError, match="needs an ensemble that frees it"):
            equilibrium(gas, Canonical(1.0)).log_z

    def test_beta_is_guarded(self):
        gas = IdenticalParticles(NLevel(LADDER), FERMI)
        with pytest.raises(ValueError, match="beta must be positive"):
            gas.log_z(0.0, (("particles", 1.0),))

    def test_couplings_are_checked_against_the_vocabulary(self):
        gas = IdenticalParticles(NLevel(LADDER), FERMI)
        with pytest.raises(ValueError, match="cannot be summed against"):
            gas.moments(1.0, (("magnetisation", 0.5),))

    def test_a_dense_orbital_sum_raises_rather_than_truncating(self):
        """
        Terms decay in `beta(eps - mu)`, so a huge mu has no decay for a long
        way. A truncated log Xi is a wrong float, not an approximate one.
        """
        gas = IdenticalParticles(HarmonicMode(omega=1e-9), FERMI)
        with pytest.raises(ValueError, match="did not converge"):
            gas.log_z(1.0, (("particles", 1e6),))

    def test_a_deep_orbital_cannot_overflow(self):
        """
        A full orbital gives exactly 1, not `nan`.

        `n = x/(1+x)` with `x = e^-y` overflows to `inf/inf` here; writing it
        as `1/(e^y+1)` never does, which is why the exponent and not the
        fugacity is what gets passed around.
        """
        gas = IdenticalParticles(NLevel([0.0]), FERMI)
        assert gas.occupation(0.0, 1.0, 1e6) == pytest.approx(1.0)

    def test_is_exact_follows_the_orbitals(self):
        assert IdenticalParticles(NLevel(LADDER), FERMI).is_exact

    def test_the_vocabulary_is_the_particle_number(self):
        assert IdenticalParticles(TwoLevel(1.0), BOSE).extensive_variables == {
            "particles"
        }

    def test_statistics_repr_carries_its_name(self):
        assert "fermi" in repr(FERMI)
        assert Statistics("custom", +1, -1).name == "custom"
