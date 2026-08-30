# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sampling, checked against exact answers.

Every model here is small enough that the truth is available some other way --
by enumerating 2^8 spin configurations, or by a closed-form occupation -- which
is the only reason to trust a sampler at all. `Sampling` deliberately shares
its vocabulary with `ThermalState`, so the comparisons are one line rather
than a translation.

The two tests that matter most are not about physics results:

  * the running energy must never drift from a recomputation, because a sign
    error in one `delta_energy` gives a plausible trajectory at the wrong
    temperature;
  * an **asymmetric** proposal must reproduce the Boltzmann distribution, and
    it does so only if `log_bias` is applied. Without it the sampler silently
    degrades from Metropolis-Hastings to Metropolis and converges on the wrong
    answer, confidently.
"""

import itertools
import math

import numpy as np
import pytest

from eigora.statphys import Canonical, Magnetic
from eigora.statphys.monte_carlo import (
    Configuration,
    FermionGas,
    HeisenbergLattice,
    IsingLattice,
    Proposal,
    autocorrelation_time,
    blocked_error,
    metropolis,
)

SITES = 8


def exact_chain(temperature, field=0.0, sites=SITES):
    """Enumerate all 2^sites states of a periodic Ising chain."""
    partition = energy = energy_sq = magnet = magnet_sq = 0.0
    for spins in itertools.product((-1, 1), repeat=sites):
        bonds = -sum(spins[i] * spins[(i + 1) % sites] for i in range(sites))
        total = sum(spins)
        weight = math.exp(-(bonds - field * total) / temperature)
        partition += weight
        energy += weight * bonds
        energy_sq += weight * bonds**2
        magnet += weight * total
        magnet_sq += weight * total**2
    mean_energy = energy / partition
    mean_magnet = magnet / partition
    return {
        "energy": mean_energy,
        "heat_capacity": (energy_sq / partition - mean_energy**2) / temperature**2,
        "magnetisation": mean_magnet,
        "susceptibility": (magnet_sq / partition - mean_magnet**2) / temperature,
    }


class TestIsingAgainstEnumeration:
    """256 microstates, so the sampler has nowhere to hide."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(20260826)

    def test_energy_and_heat_capacity(self, rng):
        lattice = IsingLattice((SITES,), coupling=1.0, rng=rng)
        run = metropolis(lattice, Canonical(2.0), 300_000, burn_in=30_000, rng=rng)
        truth = exact_chain(2.0)
        assert run.energy == pytest.approx(truth["energy"], abs=4 * run.error())
        assert run.heat_capacity == pytest.approx(truth["heat_capacity"], rel=0.05)

    def test_magnetisation_and_susceptibility_in_a_field(self, rng):
        """
        The field lives in the ensemble, not the energy.

        `IsingLattice` reports a magnetisation and never sees `h`; the same
        lattice class serves both this run and the canonical one above.
        """
        lattice = IsingLattice((SITES,), coupling=1.0, rng=rng)
        run = metropolis(lattice, Magnetic(2.0, 0.4), 300_000, burn_in=30_000, rng=rng)
        truth = exact_chain(2.0, field=0.4)
        assert run.magnetisation == pytest.approx(
            truth["magnetisation"], abs=4 * run.error("magnetisation")
        )
        assert run.susceptibility == pytest.approx(truth["susceptibility"], rel=0.10)

    def test_a_two_dimensional_lattice_runs(self, rng):
        lattice = IsingLattice((6, 6), coupling=1.0, rng=rng)
        run = metropolis(lattice, Canonical(3.0), 40_000, burn_in=4_000, rng=rng)
        assert run.heat_capacity > 0.0
        assert 0.0 < run.acceptance < 1.0


class TestBookkeeping:
    """
    The running energy against a recomputation.

    An incremental update is where a sampler goes wrong quietly: the
    trajectory stays plausible and only the temperature is wrong.
    """

    @pytest.mark.parametrize(
        "build",
        [
            lambda rng: IsingLattice((5, 4), coupling=1.3, rng=rng),
            lambda rng: HeisenbergLattice((6,), coupling=0.8, rng=rng),
            lambda rng: FermionGas([0.0, 1.0, 2.0, 3.0, 4.0], 2, rng=rng),
        ],
    )
    def test_no_drift(self, build):
        rng = np.random.default_rng(4)
        configuration = build(rng)
        metropolis(configuration, Canonical(1.5), 30_000, rng=rng)
        assert configuration.energy == pytest.approx(
            configuration.energy_of(), abs=1e-9
        )

    def test_magnetisation_does_not_drift(self):
        rng = np.random.default_rng(5)
        lattice = IsingLattice((6,), coupling=1.0, rng=rng)
        metropolis(lattice, Magnetic(1.5, 0.3), 30_000, rng=rng)
        assert lattice.extensive()["magnetisation"] == pytest.approx(
            float(lattice.spins.sum())
        )


class ThreeState(Configuration):
    """
    A three-state toy whose exact distribution is one line of arithmetic.

    `skew` controls how asymmetric the proposal is: `0.5` picks either
    neighbour equally, anything else prefers one direction, and then the
    proposal ratio `T(x'->x)/T(x->x')` is no longer 1.
    """

    ENERGIES = (0.0, 1.0, 2.0)

    def __init__(self, skew=0.5, biased=True):
        self.state = 0
        self.skew = skew
        self.biased = biased

    @property
    def energy(self):
        return self.ENERGIES[self.state]

    def energy_of(self):
        return self.ENERGIES[self.state]

    def extensive(self):
        return {}

    def propose(self, rng):
        forward = rng.random() < self.skew
        target = (self.state + (1 if forward else -1)) % 3
        ratio = (1.0 - self.skew) / self.skew if forward else self.skew / (
            1.0 - self.skew
        )
        return Proposal(
            delta_energy=self.ENERGIES[target] - self.ENERGIES[self.state],
            log_bias=math.log(ratio) if self.biased else 0.0,
            payload=target,
        )

    def apply(self, proposal):
        self.state = proposal.payload


class TestDetailedBalance:
    """
    The acceptance rule reproduces the Boltzmann distribution -- and only does
    so for an asymmetric proposal when `log_bias` is applied.
    """

    @staticmethod
    def histogram(run):
        counts = np.array(
            [np.sum(np.isclose(run.energies, e)) for e in ThreeState.ENERGIES],
            dtype=float,
        )
        return counts / counts.sum()

    @staticmethod
    def boltzmann(temperature):
        weights = np.exp(-np.array(ThreeState.ENERGIES) / temperature)
        return weights / weights.sum()

    def test_a_symmetric_proposal_needs_no_bias(self):
        rng = np.random.default_rng(11)
        run = metropolis(ThreeState(skew=0.5), Canonical(1.0), 400_000, rng=rng)
        assert self.histogram(run) == pytest.approx(self.boltzmann(1.0), abs=0.004)

    def test_an_asymmetric_proposal_is_corrected_by_the_bias(self):
        rng = np.random.default_rng(12)
        run = metropolis(
            ThreeState(skew=0.85, biased=True), Canonical(1.0), 400_000, rng=rng
        )
        assert self.histogram(run) == pytest.approx(self.boltzmann(1.0), abs=0.006)

    def test_dropping_the_bias_gives_the_wrong_distribution(self):
        """
        The test that stops Metropolis-Hastings degrading to Metropolis.

        Without `log_bias` the same chain converges -- confidently, with small
        error bars -- on a distribution that is not Boltzmann.
        """
        rng = np.random.default_rng(13)
        run = metropolis(
            ThreeState(skew=0.85, biased=False), Canonical(1.0), 400_000, rng=rng
        )
        assert self.histogram(run) != pytest.approx(self.boltzmann(1.0), abs=0.02)


class TestFermionGas:
    """Pauli as the absence of a move, not a special acceptance rule."""

    ENERGIES = (0.0, 1.0, 2.0, 3.0, 4.0)
    PARTICLES = 2

    def exact_occupations(self, temperature):
        """Enumerate the C(5, 2) configurations at fixed particle number."""
        total = np.zeros(len(self.ENERGIES))
        partition = 0.0
        for chosen in itertools.combinations(range(len(self.ENERGIES)), self.PARTICLES):
            weight = math.exp(
                -sum(self.ENERGIES[i] for i in chosen) / temperature
            )
            partition += weight
            for index in chosen:
                total[index] += weight
        return total / partition

    def test_occupations_match_the_enumeration(self):
        rng = np.random.default_rng(21)
        gas = FermionGas(self.ENERGIES, self.PARTICLES, rng=rng)
        temperature, samples = 1.5, 200_000
        tally = np.zeros(len(self.ENERGIES))
        for _ in range(samples):
            metropolis(gas, Canonical(temperature), 1, rng=rng)
            tally += gas.occupations()
        assert tally / samples == pytest.approx(
            self.exact_occupations(temperature), abs=0.01
        )

    def test_hopping_conserves_the_particle_number(self):
        rng = np.random.default_rng(22)
        gas = FermionGas(self.ENERGIES, self.PARTICLES, rng=rng)
        metropolis(gas, Canonical(1.0), 20_000, rng=rng)
        assert gas.occupied.sum() == self.PARTICLES

    def test_a_partly_filled_gas_never_wastes_a_proposal(self):
        """
        The source is drawn from the occupied orbitals and the target from the
        empty ones, so every proposal is a legal move.

        Drawing both uniformly over all orbitals would be legal only with
        probability `(N/K)(1 - N/K)` -- 17% at the filling used in
        `examples/fermi_dirac_mc.py`, and worse the more dilute the gas.
        """
        rng = np.random.default_rng(23)
        gas = FermionGas(self.ENERGIES, self.PARTICLES, rng=rng)
        run = metropolis(gas, Canonical(1.0), 5_000, rng=rng)
        assert run.blocked == 0
        assert run.steps == 5_000
        assert 0.0 < run.acceptance <= 1.0

    @pytest.mark.parametrize("particles", (0, 5))
    def test_blocked_only_when_there_is_nowhere_to_go(self, particles):
        """
        `None` now means what it says: an empty gas has nothing to move, a
        full one has nowhere to put it. Exclusion as the absence of a move.
        """
        rng = np.random.default_rng(24)
        gas = FermionGas(self.ENERGIES, particles, rng=rng)
        run = metropolis(gas, Canonical(1.0), 500, rng=rng)
        assert run.blocked == run.steps
        assert run.accepted == 0
        assert run.acceptance == 0.0

    def test_the_occupied_and_empty_lists_stay_consistent(self):
        """
        The lists are maintained incrementally, so they can drift out of step
        with the mask -- the same class of bug as the running energy.
        """
        rng = np.random.default_rng(25)
        gas = FermionGas(self.ENERGIES, self.PARTICLES, rng=rng)
        metropolis(gas, Canonical(1.0), 20_000, rng=rng)
        assert sorted(gas._filled) == list(np.flatnonzero(gas.occupied))
        assert sorted(gas._empty) == list(np.flatnonzero(~gas.occupied))
        assert len(gas._filled) == self.PARTICLES

    def test_the_proposal_is_symmetric_because_n_is_conserved(self):
        """
        `T(x->x') = 1/(N(K-N))` both ways, so `log_bias` is zero -- and only
        because a hop leaves `N` and `K-N` unchanged. A move that created or
        destroyed a particle would break this and need the bias.
        """
        rng = np.random.default_rng(26)
        gas = FermionGas(self.ENERGIES, self.PARTICLES, rng=rng)
        for _ in range(50):
            proposal = gas.propose(rng)
            assert proposal.log_bias == 0.0
            before = len(gas._filled), len(gas._empty)
            gas.apply(proposal)
            assert (len(gas._filled), len(gas._empty)) == before

    def test_too_many_fermions_is_refused(self):
        with pytest.raises(ValueError, match="will not fit"):
            FermionGas([0.0, 1.0], 3)


class TestErrorBars:
    def test_blocking_shrinks_as_one_over_root_n(self):
        rng = np.random.default_rng(31)
        errors = []
        for steps in (20_000, 80_000, 320_000):
            lattice = IsingLattice((SITES,), coupling=1.0, rng=rng)
            run = metropolis(lattice, Canonical(2.0), steps, burn_in=2_000, rng=rng)
            errors.append(run.error())
        assert errors[0] > errors[1] > errors[2]
        # Sixteen times the samples should be about four times better.
        assert 2.0 < errors[0] / errors[2] < 8.0

    def test_blocking_exceeds_the_naive_estimate_for_a_correlated_chain(self):
        """
        The whole reason blocking exists: consecutive samples are correlated,
        so `sigma/sqrt(n)` is optimistic.
        """
        rng = np.random.default_rng(32)
        lattice = IsingLattice((SITES,), coupling=1.0, rng=rng)
        run = metropolis(lattice, Canonical(1.2), 100_000, burn_in=5_000, rng=rng)
        naive = run.energies.std(ddof=1) / math.sqrt(run.energies.size)
        assert run.error() > 1.5 * naive
        assert run.autocorrelation_time() > 1.0

    def test_white_noise_has_no_correlation_time(self):
        rng = np.random.default_rng(33)
        assert autocorrelation_time(rng.normal(size=5_000)) == pytest.approx(
            0.5, abs=0.3
        )

    def test_degenerate_series_do_not_explode(self):
        assert math.isnan(blocked_error(np.array([1.0])))
        assert autocorrelation_time(np.array([2.0, 2.0, 2.0])) == 0.5


class TestGuards:
    def test_the_ensemble_must_be_reportable(self):
        rng = np.random.default_rng(41)
        gas = FermionGas([0.0, 1.0, 2.0], 1, rng=rng)
        with pytest.raises(ValueError, match="does not report"):
            metropolis(gas, Magnetic(1.0, 0.5), 10, rng=rng)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"steps": 0}, "steps must be at least 1"),
            ({"steps": 10, "thin": 0}, "thin must be at least 1"),
            ({"steps": 10, "burn_in": -1}, "burn_in must be non-negative"),
        ],
    )
    def test_run_length_guards(self, kwargs, message):
        lattice = IsingLattice((4,), rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match=message):
            metropolis(lattice, Canonical(1.0), **kwargs)

    def test_asking_for_an_unrecorded_variable(self):
        rng = np.random.default_rng(42)
        lattice = IsingLattice((4,), rng=rng)
        run = metropolis(lattice, Canonical(1.0), 100, rng=rng)
        with pytest.raises(ValueError, match="this run recorded nothing"):
            run.mean("magnetisation")

    @pytest.mark.parametrize("shape", [(), (1,), (4, 1)])
    def test_lattice_shape_guards(self, shape):
        with pytest.raises(ValueError, match="at least 2"):
            IsingLattice(shape)
        with pytest.raises(ValueError, match="at least 2"):
            HeisenbergLattice(shape)

    def test_thinning_reduces_the_record(self):
        rng = np.random.default_rng(43)
        lattice = IsingLattice((4,), rng=rng)
        run = metropolis(lattice, Canonical(1.0), 1_000, thin=10, rng=rng)
        assert run.energies.size == 100
        assert run.steps == 1_000

    def test_a_default_generator_is_used_when_none_is_given(self):
        lattice = IsingLattice((4,))
        run = metropolis(lattice, Canonical(1.0), 200)
        assert run.energies.size == 200


class TestHeisenberg:
    @pytest.mark.parametrize("temperature", (5.0, 2.0, 1.0, 0.5))
    def test_energy_matches_the_exact_classical_chain(self, temperature):
        """
        The classical Heisenberg chain is solvable: `u/J = -[coth(K) - 1/K]`
        with `K = J/T` (Fisher, 1964).

        A real reference rather than a limit, so the sampler is checked at
        every temperature rather than only where it saturates. The formula is
        for an infinite chain, and eight sites reproduce it to ~0.005 down to
        `T = 0.5`; below that the correlation length outgrows the chain and
        the finite-size deviation becomes the larger error.
        """
        rng = np.random.default_rng(51)
        lattice = HeisenbergLattice((8,), coupling=1.0, rng=rng)
        run = metropolis(
            lattice, Canonical(temperature), 120_000, burn_in=20_000, rng=rng
        )
        coupling = 1.0 / temperature
        exact = -(1.0 / math.tanh(coupling) - 1.0 / coupling)
        assert run.energy / lattice.sites == pytest.approx(exact, abs=0.012)

    def test_the_energy_falls_monotonically_as_it_cools(self):
        rng = np.random.default_rng(53)
        energies = []
        for temperature in (5.0, 1.0, 0.2):
            lattice = HeisenbergLattice((8,), coupling=1.0, rng=rng)
            run = metropolis(
                lattice, Canonical(temperature), 60_000, burn_in=10_000, rng=rng
            )
            energies.append(run.energy / lattice.sites)
        assert energies[0] > energies[1] > energies[2]
        assert energies[-1] > -1.0    # never past the fully aligned floor

    def test_spins_stay_on_the_unit_sphere(self):
        rng = np.random.default_rng(52)
        lattice = HeisenbergLattice((6,), coupling=1.0, rng=rng)
        metropolis(lattice, Canonical(1.0), 20_000, rng=rng)
        assert np.linalg.norm(lattice.spins, axis=-1) == pytest.approx(1.0)


class TestSamplingSurface:
    """The named aliases and the degenerate corners of the estimators."""

    def test_particles_reads_the_recorded_series(self):
        """`Sampling.particles` mirrors `ThermalState.particles`."""
        from eigora.statphys import Field, Generalised

        rng = np.random.default_rng(61)
        gas = FermionGas([0.0, 1.0, 2.0, 3.0], 2, rng=rng)
        grand = Generalised(1.0, [Field("particles", -0.5, +1, "mu")])
        run = metropolis(gas, grand, 2_000, rng=rng)
        # Hopping conserves N, so the average is exactly what it started at.
        assert run.particles == pytest.approx(2.0)
        assert run.response("particles") == pytest.approx(0.0, abs=1e-12)

    def test_autocorrelation_of_a_single_sample_is_undefined(self):
        assert math.isnan(autocorrelation_time(np.array([1.0])))

    def test_lattice_site_counts(self):
        rng = np.random.default_rng(62)
        assert IsingLattice((4, 5), rng=rng).sites == 20
        assert HeisenbergLattice((3, 4), rng=rng).sites == 12
