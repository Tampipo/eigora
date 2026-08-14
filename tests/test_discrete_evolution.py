# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from scipy.linalg import expm

from eigora.qm.discrete import SX, SZ, Hamiltonian, Observable, evolve

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


class TestCoefficients:
    """c_n(t) = <E_n|psi(t)>, the state read in the energy eigenbasis."""

    @pytest.fixture
    def ladder(self):
        """Already diagonal, so the computational basis is the eigenbasis."""
        return Hamiltonian(np.diag([0.5, 1.5, 2.5, 3.5, 4.5]))

    @pytest.fixture
    def spread(self, ladder):
        """An even superposition of all five levels."""
        psi0 = np.ones(ladder.dim) / np.sqrt(ladder.dim)
        return evolve(ladder, psi0, np.linspace(0.0, 4 * np.pi, 64))

    def test_shape(self, spread):
        assert spread.coefficients.shape == (64, 5)

    def test_only_the_phase_moves(self, spread):
        """|c_n| is fixed by psi(0): H commutes with itself."""
        moduli = np.abs(spread.coefficients)
        assert np.allclose(moduli, 1 / np.sqrt(5))

    def test_each_turns_at_its_own_energy(self, spread, ladder):
        """c_n(t) = c_n(0) exp(-i E_n t), one frequency per level."""
        expected = spread.coefficients[0] * np.exp(
            -1j * np.outer(spread.times, ladder.eigenenergies())
        )
        assert np.allclose(spread.coefficients, expected)

    def test_diagonal_hamiltonian_leaves_the_state_unchanged(self, spread):
        """Nothing to rotate when the basis is already the eigenbasis."""
        assert np.allclose(spread.coefficients, spread.psi)

    def test_rotated_hamiltonian_is_not_the_computational_basis(
        self, random_hamiltonian, random_state
    ):
        evolution = evolve(random_hamiltonian, random_state, [0.0, 2.5])
        assert not np.allclose(evolution.coefficients, evolution.psi)
        # ...but the two bases are related by the eigenvectors, and both are
        # normalised, so no probability is created or lost by the change.
        assert np.sum(np.abs(evolution.coefficients[1]) ** 2) == pytest.approx(1.0)

    def test_agrees_with_projecting_state_by_state(
        self, random_hamiltonian, random_state
    ):
        evolution = evolve(random_hamiltonian, random_state, [0.0, 1.3, 7.0])
        for n, eigenvector in enumerate(random_hamiltonian.eigenvectors()):
            assert np.allclose(
                evolution.coefficients[:, n],
                [np.vdot(eigenvector, evolution.state(i)) for i in range(3)],
            )

    def test_eigenstate_has_a_single_coefficient(self, random_hamiltonian):
        state = random_hamiltonian.eigenvectors()[3]
        coefficients = evolve(random_hamiltonian, state, [0.0, 6.0]).coefficients
        assert np.allclose(np.abs(coefficients[:, 3]), 1.0)
        assert np.allclose(np.delete(np.abs(coefficients), 3, axis=1), 0.0)


class TestOverlap:
    @pytest.fixture
    def evolution(self, random_hamiltonian, random_state):
        return evolve(random_hamiltonian, random_state, np.linspace(0.0, 8.0, 40))

    def test_shape(self, evolution):
        assert evolution.overlap(np.eye(16)[0]).shape == (40,)

    def test_with_an_eigenstate_is_that_coefficient(
        self, evolution, random_hamiltonian
    ):
        for n in (0, 7, 15):
            assert np.allclose(
                evolution.overlap(random_hamiltonian.eigenvectors()[n]),
                evolution.coefficients[:, n],
            )

    def test_with_itself_at_t_zero_is_one(self, evolution, random_state):
        assert evolution.overlap(random_state)[0] == pytest.approx(1.0)

    def test_squared_modulus_is_a_probability(self, evolution):
        probabilities = np.abs(evolution.overlap(np.eye(16)[5])) ** 2
        assert np.allclose(probabilities, evolution.psi[:, 5].real**2 + evolution.psi[:, 5].imag**2)
        assert np.all(probabilities <= 1.0 + 1e-12)

    def test_is_conjugate_linear_in_phi(self, evolution):
        phi = np.eye(16)[2]
        assert np.allclose(evolution.overlap(1j * phi), -1j * evolution.overlap(phi))

    def test_unnormalised_reference_scales_the_amplitude(self, evolution):
        phi = np.eye(16)[4]
        assert np.allclose(evolution.overlap(3 * phi), 3 * evolution.overlap(phi))

    def test_rejects_mismatched_dimension(self, evolution):
        with pytest.raises(ValueError, match=r"shape \(16,\)"):
            evolution.overlap(np.ones(3))
