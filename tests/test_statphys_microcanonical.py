# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Counting microstates, and the ensembles agreeing about what they mean.

The microcanonical ensemble is the one that counts rather than weights, so its
tests are about exactness of integers rather than tolerance on floats. A
composite's spectrum comes from convolving degeneracies, and for `TwoLevel(w)
** N` that has to give the binomial coefficients *themselves* -- the check is
`==` against `math.comb`, not `approx`.

Then the payoff: a system at fixed energy, and one at the temperature that
produces that mean energy, must agree about the entropy and about the
temperature. They do, to O(log N / N), and this file asserts the convergence
rather than pinning a tolerance -- the rate is the physics.
"""

import math

import pytest

from eigora.statphys import (
    Canonical,
    Degenerate,
    HarmonicMode,
    IdealGas,
    NLevel,
    Spin,
    TwoLevel,
    convolve_levels,
    equilibrium,
    microcanonical,
)
from eigora.statphys.systems.composite import log_omega, total_states


class TestConvolution:
    """
    Energies add, degeneracies multiply -- and the multiplication is exact.
    """

    def test_two_level_powers_are_the_binomial_coefficients(self):
        """
        The sharpest available check: `==`, not `approx`.

        `TwoLevel(w) ** N` puts `C(N, n)` microstates at `E = n w`. Keeping
        degeneracies as Python integers means C(200, 100) -- a 59-digit number
        -- is reproduced exactly, where a float would already have rounded.
        """
        count = 200
        levels = convolve_levels(TwoLevel(1.0) ** count)
        assert len(levels) == count + 1
        for index, level in enumerate(levels):
            assert level.degeneracy == math.comb(count, index)
            assert level.energy == pytest.approx(float(index))

    def test_every_microstate_is_counted_once(self):
        levels = convolve_levels(TwoLevel(1.0) ** 20)
        assert total_states(levels) == 2**20

    def test_unequal_blocks_convolve(self):
        """A two-level times a three-level: 6 states over 4 distinct energies."""
        levels = convolve_levels(TwoLevel(1.0) * NLevel([0.0, 1.0, 2.0]))
        assert total_states(levels) == 6
        assert [level.degeneracy for level in levels] == [1, 2, 2, 1]
        assert [level.energy for level in levels] == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_degeneracies_within_a_block_are_carried(self):
        levels = convolve_levels(Degenerate(3) * TwoLevel(1.0))
        assert total_states(levels) == 6
        assert [level.degeneracy for level in levels] == [3, 3]

    def test_a_lone_spectral_system_passes_through(self):
        levels = convolve_levels(NLevel([0.0, 2.0], [2, 5]))
        assert [(level.energy, level.degeneracy) for level in levels] == [
            (0.0, 2),
            (2.0, 5),
        ]

    def test_log_omega_finds_the_shell(self):
        levels = convolve_levels(TwoLevel(1.0) ** 10)
        assert log_omega(levels, 5.0) == pytest.approx(math.log(math.comb(10, 5)))
        assert log_omega(levels, 2.5) == -math.inf

    def test_an_unbounded_block_is_refused(self):
        with pytest.raises(ValueError, match="unboundedly many states"):
            convolve_levels(HarmonicMode(1.0) ** 3)

    def test_extensive_variables_are_refused(self):
        """Convolution groups by energy alone, so it will not silently drop a
        magnetisation."""
        with pytest.raises(ValueError, match="carries extensive variables"):
            convolve_levels(Spin(0.5) ** 3)

    def test_a_block_without_levels_is_refused(self):
        with pytest.raises(ValueError, match="has none"):
            convolve_levels(IdealGas(particles=1, volume=1.0) ** 2)

    def test_repeated_blocks_do_not_blow_up_even_when_incommensurate(self):
        """
        `A ** n` is cheap whatever `A` is, which is easy to get backwards.

        The joint energies are *multisets* drawn from one block's levels, so
        there are `C(n + d - 1, d - 1)` of them -- polynomial in `n`, not
        exponential. Twenty copies of a three-level block give 231 distinct
        energies, not three-to-the-twenty.
        """
        irrational = NLevel([0.0, math.pi, math.e])
        levels = convolve_levels(irrational**20)
        assert len(levels) == math.comb(22, 2) == 231
        assert total_states(levels) == 3**20

    def test_distinct_incommensurate_blocks_blow_up_and_say_so(self):
        """
        Different blocks are where it does explode: nothing coincides, so
        every microstate lands on its own energy and the count doubles per
        block. This is when convolution stops being worth doing.
        """
        blocks = [NLevel([0.0, math.sqrt(p)]) for p in (2, 3, 5, 7, 11, 13, 17,
                                                        19, 23, 29, 31, 37)]
        product = blocks[0]
        for block in blocks[1:]:
            product = product * block
        with pytest.raises(ValueError, match="distinct energies"):
            convolve_levels(product, max_distinct=1000)


class TestMicrocanonical:
    COUNT = 200

    @pytest.fixture
    def system(self):
        return TwoLevel(1.0) ** self.COUNT

    def test_omega_is_an_exact_integer(self, system):
        state = microcanonical(system, energy=100.0)
        assert state.omega == math.comb(self.COUNT, 100)
        assert isinstance(state.omega, int)

    def test_entropy_is_log_omega(self, system):
        state = microcanonical(system, energy=100.0)
        assert state.entropy == pytest.approx(math.log(math.comb(self.COUNT, 100)))

    def test_the_ground_shell_holds_one_state(self, system):
        assert microcanonical(system, energy=0.0).omega == 1
        assert microcanonical(system, energy=0.0).entropy == 0.0

    def test_a_width_sums_a_range_of_levels(self, system):
        """A shell rather than a single level -- what a dense spectrum needs."""
        narrow = microcanonical(system, energy=100.0)
        wide = microcanonical(system, energy=100.0, width=2.0)
        assert wide.omega == sum(
            math.comb(self.COUNT, n) for n in (99, 100, 101)
        )
        assert wide.omega > narrow.omega

    def test_an_empty_shell_is_refused(self, system):
        """Undefined, not zero -- log 0 is not an entropy."""
        with pytest.raises(ValueError, match="no microstates at energy"):
            microcanonical(system, energy=100.5).entropy

    def test_temperature_comes_out_rather_than_going_in(self, system):
        """`1/T = dS/dE`: derived from the density of states."""
        cold = microcanonical(system, energy=20.0).temperature
        hot = microcanonical(system, energy=90.0).temperature
        assert 0.0 < cold < hot

    def test_the_spectrum_edges_have_no_centred_derivative(self, system):
        for energy in (0.0, float(self.COUNT)):
            with pytest.raises(ValueError, match="no neighbour"):
                microcanonical(system, energy=energy).beta

    def test_rejects_a_non_system(self):
        with pytest.raises(TypeError, match="expected a System"):
            microcanonical(3.0, energy=1.0)

    def test_width_must_be_positive(self, system):
        with pytest.raises(ValueError, match="width must be positive"):
            microcanonical(system, energy=1.0, width=0.0)

    def test_energy_must_be_finite(self, system):
        with pytest.raises(ValueError, match="energy must be finite"):
            microcanonical(system, energy=math.inf)


class TestEnsembleEquivalence:
    """
    The two ensembles agreeing, and the rate at which they do.

    Fix the energy or fix the temperature: for a large system it makes no
    difference, and "large" is quantified here rather than assumed. Both gaps
    close as `N` grows, which is the whole content of ensemble equivalence.
    """

    TEMPERATURE = 1.0

    def _pair(self, count):
        system = TwoLevel(1.0) ** count
        canonical = equilibrium(system, Canonical(self.TEMPERATURE))
        shell = microcanonical(system, energy=float(round(canonical.energy)))
        return canonical, shell

    def test_the_microcanonical_temperature_recovers_the_canonical_one(self):
        """`T_micro(U(T)) -> T`: the inverse trip, closing as N grows."""
        gaps = []
        for count in (50, 100, 200, 400):
            _, shell = self._pair(count)
            gaps.append(abs(shell.temperature - self.TEMPERATURE))
        assert gaps[0] > gaps[1] > gaps[2] > gaps[3]
        assert gaps[-1] < 0.01

    def test_the_entropies_agree_per_particle(self):
        """
        `S_micro` and `S_can` differ by O(log N) -- the canonical entropy
        includes the energy fluctuations the fixed-energy shell forbids -- so
        it is the *per particle* gap that vanishes.
        """
        gaps = []
        for count in (50, 100, 200, 400):
            canonical, shell = self._pair(count)
            gaps.append(abs(shell.entropy - canonical.entropy) / count)
        assert gaps[0] > gaps[1] > gaps[2] > gaps[3]
        assert gaps[-1] < 0.01

    def test_the_canonical_entropy_is_the_larger(self):
        """
        Fixing the energy forbids fluctuations, so it can only remove states.
        """
        canonical, shell = self._pair(200)
        assert shell.entropy < canonical.entropy


class TestShellEdges:
    """The two ways an energy can fail to name a shell."""

    @pytest.fixture
    def system(self):
        return TwoLevel(1.0) ** 20

    def test_the_temperature_needs_a_level_to_sit_on(self, system):
        """
        `dS/dE` is a difference over neighbouring levels, so an energy between
        them has no derivative -- distinct from the shell merely being empty.
        """
        with pytest.raises(ValueError, match="no level at energy"):
            microcanonical(system, energy=10.5).beta

    def test_a_width_still_counts_states_off_a_level(self, system):
        """Counting tolerates an off-level energy; differentiating does not."""
        assert microcanonical(system, energy=10.5, width=1.5).omega > 0


class TestEntropyMaximum:
    def test_the_top_of_the_entropy_curve_has_infinite_temperature(self):
        """
        `dS/dE = 0` at the peak, so `T` diverges. A bounded spectrum always
        has such a point, and it is the boundary past which the temperature
        turns negative -- worth naming rather than dividing by zero.
        """
        system = TwoLevel(1.0) ** 8
        with pytest.raises(ValueError, match="entropy is stationary"):
            microcanonical(system, energy=4.0).temperature
        assert microcanonical(system, energy=4.0).beta == 0.0

    def test_below_and_above_the_peak_the_signs_differ(self):
        system = TwoLevel(1.0) ** 8
        assert microcanonical(system, energy=2.0).beta > 0.0
        assert microcanonical(system, energy=6.0).beta < 0.0
