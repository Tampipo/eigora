# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The link from `eigora.qm` to `eigora.statphys`.

One adapter -- eigenvalues grouped into levels -- and the whole statphys
catalogue applies to anything `qm` can solve. The test that matters is the one
about the **energy origin**: `qm` returns absolute eigenvalues, so a
`qm.discrete.TwoLevel(bias=w)` sits at `-w/2, +w/2` while `statphys.TwoLevel(w)`
sits at `0, w`. Their `log Z` must differ by exactly `beta w / 2` and every
derivative of it must agree exactly, which forces the convention to be explicit
rather than assumed.

The other reason this exists: a Heisenberg chain is *interacting*, so the
orbital factorisation in `identical` cannot touch it. Exact diagonalisation
can, and this is how its thermodynamics gets in.
"""

import math

import pytest

import eigora.qm.discrete as discrete
from eigora.grids import GridND
from eigora.qm.potentials import HarmonicWell
from eigora.qm.spectra.factory import spectrum_for
from eigora.statphys import Canonical, TwoLevel, equilibrium
from eigora.statphys.systems.bridge import (
    from_energies,
    from_hamiltonian,
    from_potential,
    from_spectrum,
)


class TestEnergyOrigin:
    """
    The convention, asserted rather than assumed.

    A free energy is only defined up to the energy origin, so two spectra
    differing by a constant shift have different `log Z` and identical
    everything-that-is-a-derivative. Both halves are checked, because getting
    this wrong silently is easy and the numbers stay plausible.
    """

    SPLITTING = 1.3

    @pytest.fixture
    def pair(self):
        quantum = from_hamiltonian(
            discrete.TwoLevel(bias=self.SPLITTING, coupling=0.0)
        )
        return quantum, TwoLevel(self.SPLITTING)

    def test_the_quantum_spectrum_is_centred_on_zero(self, pair):
        quantum, _ = pair
        assert quantum.energies == pytest.approx(
            (-self.SPLITTING / 2.0, self.SPLITTING / 2.0)
        )

    @pytest.mark.parametrize("beta", (0.3, 0.8, 2.0))
    def test_log_z_differs_by_exactly_the_shift(self, pair, beta):
        quantum, statphys = pair
        assert quantum.log_z(beta) - statphys.log_z(beta) == pytest.approx(
            beta * self.SPLITTING / 2.0, rel=1e-13
        )

    @pytest.mark.parametrize("temperature", (0.5, 1.0, 2.0))
    def test_every_derivative_agrees(self, pair, temperature):
        quantum, statphys = pair
        a = equilibrium(quantum, Canonical(temperature))
        b = equilibrium(statphys, Canonical(temperature))
        assert a.heat_capacity == pytest.approx(b.heat_capacity, rel=1e-12)
        assert a.entropy == pytest.approx(b.entropy, rel=1e-12)
        assert a.energy - b.energy == pytest.approx(-self.SPLITTING / 2.0)


class TestFromHamiltonian:
    def test_an_interacting_chain_keeps_every_state(self):
        """
        `2^n` states, exactly, grouped into total-spin multiplets.

        A Heisenberg chain has no orbital factorisation -- interactions
        destroy it -- so this is the only route to its thermodynamics, and it
        is exact rather than truncated because a matrix has a full spectrum.
        """
        chain = from_hamiltonian(discrete.HeisenbergChain(n_sites=4, coupling=1.0))
        assert chain.n_states == 16
        assert sum(chain.degeneracies) == 16
        # Four spin halves decompose as 5 + 3 + 3 + 3 + 1 + 1.
        assert sorted(chain.degeneracies, reverse=True) == [5, 3, 3, 3, 1, 1]

    def test_it_is_exact_rather_than_truncated(self):
        chain = from_hamiltonian(discrete.HeisenbergChain(n_sites=3, coupling=1.0))
        assert chain.is_exact
        assert chain.n_states == 8

    def test_thermodynamics_of_the_chain_is_finite_everywhere(self):
        chain = from_hamiltonian(discrete.HeisenbergChain(n_sites=4, coupling=1.0))
        for temperature in (0.05, 1.0, 100.0):
            state = equilibrium(chain, Canonical(temperature))
            assert math.isfinite(state.heat_capacity)
            assert state.heat_capacity >= 0.0
        # High T: every state equally likely, so S -> log 16.
        assert equilibrium(chain, Canonical(1e6)).entropy == pytest.approx(
            math.log(16), abs=1e-6
        )

    def test_rejects_a_non_hamiltonian(self):
        with pytest.raises(TypeError, match="expected a Hamiltonian"):
            from_hamiltonian("not a hamiltonian")


class TestFromEnergies:
    def test_degenerate_eigenvalues_are_grouped(self):
        system = from_energies([1.0, 0.0, 1.0, 1.0, 0.0])
        assert system.energies == pytest.approx((0.0, 1.0))
        assert system.degeneracies == (2, 3)

    def test_near_degeneracy_is_grouped_within_tolerance(self):
        """Exact degeneracy is a symmetry statement floats cannot make."""
        system = from_energies([0.0, 1e-12, 1.0], tol=1e-9)
        assert system.degeneracies == (2, 1)

    def test_a_tighter_tolerance_separates_them(self):
        system = from_energies([0.0, 1e-12, 1.0], tol=1e-15)
        assert system.degeneracies == (1, 1, 1)

    def test_energies_stay_absolute(self):
        assert from_energies([-5.0, -3.0]).energies == pytest.approx((-5.0, -3.0))

    def test_rejects_an_empty_spectrum(self):
        with pytest.raises(ValueError, match="at least one energy"):
            from_energies([])

    def test_rejects_a_non_finite_energy(self):
        with pytest.raises(ValueError, match="must be finite"):
            from_energies([0.0, math.inf])


class TestFromSpectrum:
    def test_a_harmonic_well_reproduces_the_ladder(self):
        """
        `spectrum_for` solves the well analytically, so the levels are
        `omega(n + 1/2)` and the statphys twin should agree.
        """
        system = from_spectrum(spectrum_for(HarmonicWell(omega=1.0)), n_levels=40)
        assert system.energies[:3] == pytest.approx((0.5, 1.5, 2.5))
        assert system.degeneracies[:3] == (1, 1, 1)

    def test_truncation_is_good_at_low_temperature(self):
        """
        Forty levels is plenty at T = 0.3 and not at T = 30 -- a truncation,
        not a closed form, so it is only as good as the tail it drops.
        """
        from eigora.statphys import HarmonicMode

        truncated = from_spectrum(spectrum_for(HarmonicWell(omega=1.0)), n_levels=40)
        exact = HarmonicMode(omega=1.0)
        assert truncated.log_z(1.0 / 0.3) == pytest.approx(
            exact.log_z(1.0 / 0.3), rel=1e-12
        )
        assert truncated.log_z(1.0 / 30.0) != pytest.approx(
            exact.log_z(1.0 / 30.0), rel=1e-6
        )

    def test_rejects_a_non_spectrum(self):
        with pytest.raises(TypeError, match="expected a Spectrum"):
            from_spectrum(object())


class TestFromPotential:
    def test_a_potential_becomes_a_system_in_one_step(self):
        system = from_potential(HarmonicWell(omega=2.0), n_levels=20)
        assert system.energies[:2] == pytest.approx((1.0, 3.0))

    def test_a_numerically_solved_potential_needs_a_grid(self):
        grid = GridND.line(-8.0, 8.0, 256)
        system = from_potential(HarmonicWell(omega=1.0), grid, n_levels=5)
        assert len(system.energies) == 5
