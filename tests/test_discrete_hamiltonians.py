# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Systems with a closed-form spectrum override `eigensystem` rather than
diagonalising. These tests hold the two implementations against each other:
the analytic eigensystem must agree with `np.linalg.eigh` on the same matrix.
"""

import pytest
import numpy as np

from physense_qm.discrete import (
    SX,
    SY,
    SZ,
    HarmonicLadder,
    Hamiltonian,
    HeisenbergChain,
    Rabi,
    SpinInField,
    TwoLevel,
    commutator,
    embed,
    evolve,
    noninteracting,
)


def assert_closed_form_matches_solver(hamiltonian):
    """The overridden eigensystem must agree with brute-force diagonalisation."""
    exact = hamiltonian.eigensystem
    numerical, _ = np.linalg.eigh(hamiltonian.matrix)

    assert np.allclose(exact.eigenvalues, numerical), "eigenvalues disagree"
    assert np.all(np.diff(exact.eigenvalues) >= -1e-12), "not ascending"
    # Eigenvectors are only fixed up to phase, so check the defining property
    # rather than comparing against eigh's choice directly.
    for i in range(hamiltonian.dim):
        assert np.allclose(
            hamiltonian.matrix @ exact.eigenvectors[i],
            exact.eigenvalues[i] * exact.eigenvectors[i],
        ), f"eigenvector {i} does not satisfy H v = E v"
    assert np.allclose(
        exact.eigenvectors.conj() @ exact.eigenvectors.T, np.eye(hamiltonian.dim)
    ), "eigenvectors not orthonormal"


class TestTwoLevel:
    @pytest.mark.parametrize(
        "bias, coupling",
        [(0.0, 1.0), (1.0, 0.0), (2.0, 3.0), (-1.5, 0.7), (0.3, -2.2), (0.0, 0.0)],
    )
    def test_closed_form_matches_solver(self, bias, coupling):
        assert_closed_form_matches_solver(TwoLevel(bias, coupling))

    @pytest.mark.parametrize(
        "bias, coupling", [(0.0, 1.0), (2.0, 3.0), (-1.5, 0.7), (0.3, -2.2)]
    )
    def test_spectrum_is_the_quadrature_sum(self, bias, coupling):
        """E± = ± sqrt(bias² + coupling²) / 2"""
        gap = np.hypot(bias, coupling)
        assert np.allclose(
            TwoLevel(bias, coupling).eigenenergies(), [-gap / 2, gap / 2]
        )

    def test_does_not_diagonalise(self, monkeypatch):
        """The whole point of the override: eigh is never reached."""
        monkeypatch.setattr(
            np.linalg, "eigh", lambda m: pytest.fail("fell back to diagonalisation")
        )
        gap = np.hypot(1.0, 2.0)
        assert np.allclose(TwoLevel(1.0, 2.0).eigenenergies(), [-gap / 2, gap / 2])

    def test_parameters_are_kept(self):
        hamiltonian = TwoLevel(bias=1.5, coupling=0.5)
        assert hamiltonian.bias == 1.5
        assert hamiltonian.coupling == 0.5

    def test_degenerate_when_both_vanish(self):
        assert np.allclose(TwoLevel(0.0, 0.0).eigenenergies(), [0.0, 0.0])

    def test_avoided_crossing(self):
        """Sweeping the bias through zero, the gap is minimal but never closes."""
        biases = np.linspace(-3.0, 3.0, 61)
        gaps = np.array([np.ptp(TwoLevel(b, 0.4).eigenenergies()) for b in biases])
        assert gaps.min() == pytest.approx(0.4)
        assert np.argmin(gaps) == len(biases) // 2

    def test_pure_bias_is_already_diagonal(self):
        """The basis states are the eigenstates, ordered by energy: down, up."""
        vectors = TwoLevel(bias=2.0, coupling=0.0).eigenvectors()
        assert np.allclose(np.abs(vectors), [[0.0, 1.0], [1.0, 0.0]])


class TestRabi:
    @pytest.mark.parametrize("detuning", [0.0, 0.5, 2.0, -1.3])
    def test_closed_form_matches_solver(self, detuning):
        assert_closed_form_matches_solver(Rabi(detuning, 1.7))

    def test_same_matrix_as_two_level(self):
        assert np.allclose(Rabi(0.4, 1.3).matrix, TwoLevel(0.4, 1.3).matrix)

    def test_generalised_frequency(self):
        assert Rabi(3.0, 4.0).generalised_frequency == pytest.approx(5.0)

    @pytest.mark.parametrize("detuning", [0.0, 0.5, 2.0, -1.3])
    def test_transfer_formula(self, detuning):
        """P(t) = (omega/W)² sin²(Wt/2), against the evolution."""
        hamiltonian = Rabi(detuning, 1.7)
        times = np.linspace(0.0, 15.0, 300)
        evolution = evolve(hamiltonian, np.array([1.0, 0.0]), times)

        width = hamiltonian.generalised_frequency
        expected = hamiltonian.max_transfer * np.sin(width * times / 2) ** 2
        transferred = np.array(
            [evolution.probabilities(n)[1] for n in range(evolution.n_times)]
        )
        assert np.allclose(transferred, expected)

    def test_resonant_drive_inverts_completely(self):
        hamiltonian = Rabi(0.0, 1.7)
        assert hamiltonian.max_transfer == pytest.approx(1.0)
        pulse = evolve(hamiltonian, np.array([1.0, 0.0]), [np.pi / 1.7])
        assert pulse.probabilities(0)[1] == pytest.approx(1.0)

    @pytest.mark.parametrize("detuning", [0.5, 2.0])
    def test_detuned_drive_cannot_invert(self, detuning):
        hamiltonian = Rabi(detuning, 1.7)
        assert hamiltonian.max_transfer < 1.0
        times = np.linspace(0.0, 40.0, 2000)
        evolution = evolve(hamiltonian, np.array([1.0, 0.0]), times)
        peak = max(evolution.probabilities(n)[1] for n in range(evolution.n_times))
        assert peak == pytest.approx(hamiltonian.max_transfer, abs=1e-3)

    def test_no_drive_transfers_nothing(self):
        assert Rabi(0.0, 0.0).max_transfer == 0.0


class TestSpinInField:
    @pytest.mark.parametrize(
        "field",
        [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, -2.0, 3.0), (0.0, 0.0, 0.0)],
    )
    def test_closed_form_matches_solver(self, field):
        assert_closed_form_matches_solver(SpinInField(field))

    @pytest.mark.parametrize(
        "field", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, -2.0, 3.0)]
    )
    def test_spectrum_depends_only_on_magnitude(self, field):
        """E± = ± |B| / 2, whatever direction the field points."""
        magnitude = np.linalg.norm(field)
        assert np.allclose(
            SpinInField(field).eigenenergies(), [-magnitude / 2, magnitude / 2]
        )

    def test_strength(self):
        assert SpinInField((3.0, 4.0, 0.0)).strength == pytest.approx(5.0)

    def test_larmor_precession(self):
        """A spin transverse to B precesses about it at the Larmor frequency |B|."""
        hamiltonian = SpinInField((0.0, 0.0, 1.3))
        times = np.linspace(0.0, 10.0, 200)
        evolution = evolve(hamiltonian, np.array([1.0, 1.0]) / np.sqrt(2), times)
        assert np.allclose(evolution.expectation(SX), np.cos(1.3 * times))
        assert np.allclose(evolution.expectation(SY), np.sin(1.3 * times))

    def test_component_along_the_field_is_conserved(self):
        evolution = evolve(
            SpinInField((0.0, 0.0, 1.3)), np.array([1.0, 1.0]) / np.sqrt(2), [0.0, 4.0]
        )
        tracked = evolution.expectation(SZ)
        assert tracked[1] == pytest.approx(tracked[0])

    def test_eigenstates_point_along_the_field(self):
        """For B along x, the eigenstates are the SX eigenstates."""
        hamiltonian = SpinInField((2.0, 0.0, 0.0))
        for i in range(2):
            overlap = abs(np.vdot(SX.eigenvectors()[i], hamiltonian.eigenvectors()[i]))
            assert overlap == pytest.approx(1.0)

    def test_field_is_stored_as_a_tuple(self):
        assert SpinInField(np.array([1.0, 0.0, 0.0])).field == (1.0, 0.0, 0.0)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="three components"):
            SpinInField((1.0, 0.0))


class TestHarmonicLadder:
    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.4])
    def test_closed_form_matches_solver(self, omega):
        assert_closed_form_matches_solver(HarmonicLadder(6, omega))

    @pytest.mark.parametrize("omega", [0.5, 1.0, 2.4])
    def test_spectrum(self, omega):
        """E_n = omega (n + 1/2)"""
        ladder = HarmonicLadder(6, omega)
        assert np.allclose(ladder.eigenenergies(), omega * (np.arange(6) + 0.5))

    def test_matches_the_continuous_harmonic_well(self):
        """The same spectrum the continuous side reaches by solving an ODE."""
        from physense_qm.potentials import HarmonicWell
        from physense_qm.spectra import HarmonicSpectrum

        omega = 1.3
        exact = HarmonicSpectrum(HarmonicWell(omega=omega))
        assert np.allclose(HarmonicLadder(5, omega).eigenenergies(), exact.energies(5))

    def test_basis_is_the_eigenbasis(self):
        assert np.allclose(HarmonicLadder(4, 1.0).eigenvectors(), np.eye(4))

    def test_ground_energy_is_the_zero_point(self):
        assert HarmonicLadder(4, 2.0).ground_energy == pytest.approx(1.0)

    def test_spacing_is_uniform(self):
        assert np.allclose(np.diff(HarmonicLadder(8, 0.7).eigenenergies()), 0.7)

    @pytest.mark.parametrize(
        "n_levels, omega, match",
        [(0, 1.0, "at least 1"), (-2, 1.0, "at least 1"), (3, 0.0, "positive")],
    )
    def test_validation(self, n_levels, omega, match):
        with pytest.raises(ValueError, match=match):
            HarmonicLadder(n_levels, omega)


class TestHeisenbergChain:
    def test_two_sites_split_into_singlet_and_triplet(self):
        """Pauli dot product has eigenvalues -3 (once) and +1 (three times)."""
        coupling = 0.8
        energies = HeisenbergChain(2, coupling).eigenenergies()
        assert np.allclose(energies, [-3 * coupling, coupling, coupling, coupling])

    def test_singlet_is_the_ground_state_when_antiferromagnetic(self):
        chain = HeisenbergChain(2, coupling=1.0)
        singlet = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2)
        assert chain.ground_energy == pytest.approx(-3.0)
        assert abs(np.vdot(singlet, chain.ground_state)) == pytest.approx(1.0)

    def test_ferromagnetic_coupling_flips_the_ordering(self):
        assert HeisenbergChain(2, coupling=-1.0).ground_energy == pytest.approx(-1.0)

    def test_falls_back_to_diagonalisation(self):
        """No elementary closed form beyond two sites, so no override."""
        assert "eigensystem" not in HeisenbergChain.__dict__

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    def test_conserves_total_magnetisation(self, n_sites):
        """[H, sum_i SZ_i] = 0, so magnetisation is a good quantum number."""
        dims = (2,) * n_sites
        total_sz = embed(SZ, 0, dims)
        for site in range(1, n_sites):
            total_sz = total_sz + embed(SZ, site, dims)

        chain = HeisenbergChain(n_sites, coupling=0.9, field=0.4)
        assert np.allclose(commutator(chain, total_sz).matrix, 0.0)

    def test_dimension_grows_as_two_to_the_n(self):
        assert HeisenbergChain(4).dim == 16

    def test_field_polarises(self):
        """With no exchange, |up...up> sits at +field per site."""
        chain = HeisenbergChain(3, coupling=0.0, field=0.5)
        assert np.max(chain.eigenenergies()) == pytest.approx(1.5)

    def test_bonds(self):
        assert HeisenbergChain(4).bonds == [(0, 1), (1, 2), (2, 3)]
        assert HeisenbergChain(4, periodic=True).bonds == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_periodic_ignored_for_two_sites(self):
        """Closing a two-site chain would double-count its single bond."""
        assert HeisenbergChain(2, periodic=True).bonds == [(0, 1)]

    def test_single_site_is_just_the_field(self):
        assert np.allclose(HeisenbergChain(1, field=0.5).eigenenergies(), [-0.5, 0.5])

    def test_rejects_empty_chain(self):
        with pytest.raises(ValueError, match="at least 1"):
            HeisenbergChain(0)


class TestDegradesUnderArithmetic:
    """A known system that has been operated on is no longer that system."""

    @pytest.mark.parametrize(
        "operation",
        [
            lambda h: 2.0 * h,
            lambda h: -h,
            lambda h: h + h,
            lambda h: h - TwoLevel(0.5, 0.5),
            lambda h: h.dagger(),
            lambda h: embed(h, 0, (2, 2)),
        ],
    )
    def test_becomes_a_plain_hamiltonian(self, operation):
        assert type(operation(TwoLevel(1.0, 2.0))) is Hamiltonian

    def test_embedded_system_diagonalises_its_full_space(self):
        """The 2x2 closed form must not survive onto a 4x4 operator."""
        embedded = embed(TwoLevel(1.0, 2.0), 0, (2, 2))
        assert embedded.dim == 4
        assert len(embedded.eigenenergies()) == 4

    def test_scaled_system_has_scaled_energies(self):
        original = TwoLevel(1.0, 2.0)
        assert np.allclose((3.0 * original).eigenenergies(), 3.0 * original.eigenenergies())


class TestComposition:
    def test_two_independent_two_level_systems(self):
        """Energies of the pair are all sums of the individual ones."""
        a = TwoLevel(bias=1.0, coupling=0.0)
        b = TwoLevel(bias=3.0, coupling=0.0)
        joint = noninteracting([a, b])
        expected = sorted(x + y for x in a.eigenenergies() for y in b.eigenenergies())
        assert np.allclose(np.sort(joint.eigenenergies()), expected)

    def test_sum_of_known_systems_is_still_a_hamiltonian(self):
        total = TwoLevel(1.0, 0.0) + SpinInField((0.5, 0.0, 0.0))
        assert type(total) is Hamiltonian
        assert np.allclose(total.matrix, TwoLevel(1.0, 0.5).matrix)
