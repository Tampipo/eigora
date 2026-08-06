# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from scipy.linalg import expm

from physense_qm.discrete import SX, SZ, Hamiltonian, Observable, evolve

OMEGA = 1.7

UP = np.array([1.0, 0.0])


@pytest.fixture
def rabi():
    """Two-level system driven on resonance: H = (omega/2) sigma_x."""
    return Hamiltonian((OMEGA / 2 * SX).matrix)


@pytest.fixture
def random_hamiltonian():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    return Hamiltonian((a + a.conj().T) / 2)


@pytest.fixture
def random_state():
    rng = np.random.default_rng(2)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    return psi / np.linalg.norm(psi)


class TestEvolve:
    def test_rabi_oscillations(self, rabi):
        """P_1(t) = sin^2(omega t / 2), the closed form for a two-level system."""
        times = np.linspace(0.0, 12.0, 200)
        evolution = evolve(rabi, UP, times)
        excited = np.array(
            [evolution.probabilities(n)[1] for n in range(evolution.n_times)]
        )
        assert np.allclose(excited, np.sin(OMEGA * times / 2) ** 2)

    def test_matches_matrix_exponential(self, random_hamiltonian, random_state):
        times = np.array([0.0, 0.31, 2.7, 13.0])
        evolution = evolve(random_hamiltonian, random_state, times)
        for n, t in enumerate(times):
            expected = expm(-1j * random_hamiltonian.matrix * t) @ random_state
            assert np.allclose(evolution.state(n), expected)

    def test_initial_time_returns_initial_state(self, random_hamiltonian, random_state):
        assert np.allclose(evolve(random_hamiltonian, random_state, [0.0]).state(0), random_state)

    def test_norm_conserved(self, random_hamiltonian, random_state):
        """Nothing accumulates: every time is computed from psi(0) directly."""
        evolution = evolve(random_hamiltonian, random_state, np.linspace(0.0, 50.0, 25))
        for n in range(evolution.n_times):
            assert evolution.norm(n) == pytest.approx(1.0, abs=1e-12)

    def test_eigenstate_is_stationary(self, random_hamiltonian):
        """An energy eigenstate only picks up a phase."""
        energy = random_hamiltonian.eigenenergies()[3]
        state = random_hamiltonian.eigenvectors()[3]
        evolution = evolve(random_hamiltonian, state, [0.0, 9.0])
        assert np.allclose(evolution.probabilities(0), evolution.probabilities(1))
        assert np.allclose(evolution.state(1), np.exp(-1j * energy * 9.0) * state)

    def test_unordered_and_negative_times(self, random_hamiltonian, random_state):
        """No stepping, so the times need no structure at all."""
        evolution = evolve(random_hamiltonian, random_state, [5.0, -3.0, 0.0, 100.0])
        assert evolution.n_times == 4
        assert np.allclose(evolution.state(2), random_state)

    def test_backwards_evolution_inverts(self, random_hamiltonian, random_state):
        forward = evolve(random_hamiltonian, random_state, [3.0]).state(0)
        assert np.allclose(evolve(random_hamiltonian, forward, [-3.0]).state(0), random_state)

    def test_initial_state_is_normalised(self, rabi):
        assert evolve(rabi, np.array([3.0, 0.0]), [0.0]).norm(0) == pytest.approx(1.0)

    def test_diagonalises_once_for_all_times(self, monkeypatch):
        calls = []
        real_eigh = np.linalg.eigh
        monkeypatch.setattr(
            np.linalg, "eigh", lambda m: (calls.append(1), real_eigh(m))[1]
        )
        hamiltonian = Hamiltonian(np.diag([1.0, 2.0, 3.0]))
        state = np.ones(3) / np.sqrt(3)
        evolve(hamiltonian, state, np.linspace(0.0, 5.0, 500))
        evolve(hamiltonian, state, np.linspace(0.0, 2.0, 300))
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "psi0, times, match",
        [
            (np.array([1.0, 0.0]), [1.0], "shape"),
            (np.zeros(3), [1.0], "non-zero"),
            (np.array([1.0, 0.0, 0.0]), [[1.0, 2.0]], "one-dimensional"),
            (np.array([1.0, 0.0, 0.0]), [], "at least one"),
        ],
    )
    def test_validation(self, psi0, times, match):
        hamiltonian = Hamiltonian(np.diag([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match=match):
            evolve(hamiltonian, psi0, times)


class TestEvolutionResult:
    @pytest.fixture
    def evolution(self, rabi):
        return evolve(rabi, UP, np.linspace(0.0, 12.0, 200))

    def test_shape(self, evolution):
        assert evolution.psi.shape == (200, 2)
        assert evolution.n_times == 200
        assert evolution.dim == 2

    def test_keeps_its_hamiltonian(self, evolution, rabi):
        assert evolution.hamiltonian is rabi

    def test_probabilities_sum_to_one(self, evolution):
        assert np.sum(evolution.probabilities(50)) == pytest.approx(1.0)

    def test_expectation_of_sigma_z(self, evolution):
        """<sz>(t) = cos(omega t) for a resonantly driven two-level system."""
        assert np.allclose(
            evolution.expectation(SZ), np.cos(OMEGA * evolution.times)
        )

    def test_expectation_agrees_with_pointwise(self, evolution):
        observable = SZ
        tracked = evolution.expectation(observable)
        for n in (0, 17, 199):
            assert tracked[n] == pytest.approx(observable.expectation(evolution.state(n)))

    def test_expectation_rejects_mismatched_observable(self, evolution):
        with pytest.raises(ValueError, match="3D"):
            evolution.expectation(Observable(np.eye(3)))

    @pytest.mark.parametrize("index", [-1, 200])
    def test_state_index_out_of_range(self, evolution, index):
        with pytest.raises(IndexError):
            evolution.state(index)
