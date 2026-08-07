# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from scipy.linalg import expm

from eigora.qm.discrete import (
    Hamiltonian,
    Observable,
    Operator,
    SMINUS,
    SPLUS,
    SX,
    SY,
    SZ,
    anticommutator,
    commutator,
    embed,
    identity,
    noninteracting,
    operations,
    tensor_product,
)


class TestPauliConstants:
    def test_hermitian_ones_are_observables(self):
        assert all(type(pauli) is Observable for pauli in (SX, SY, SZ))

    def test_ladder_operators_are_not_observables(self):
        """S± raise and lower; they are not measurable quantities."""
        assert type(SPLUS) is Operator
        assert not SPLUS.is_hermitian()

    def test_eigenvalues(self):
        for pauli in (SX, SY, SZ):
            assert np.allclose(pauli.eigenvalues(), [-1.0, 1.0])

    def test_square_to_identity(self):
        for pauli in (SX, SY, SZ):
            assert np.allclose((pauli @ pauli).matrix, np.eye(2))

    def test_su2_algebra(self):
        """[SX, SY] = 2i·SZ in the Pauli convention."""
        assert np.allclose(commutator(SX, SY).matrix, 2j * SZ.matrix)
        assert np.allclose(commutator(SY, SZ).matrix, 2j * SX.matrix)
        assert np.allclose(commutator(SZ, SX).matrix, 2j * SY.matrix)

    def test_ladder_from_paulis(self):
        assert np.allclose((0.5 * (SX + 1j * SY)).matrix, SPLUS.matrix)
        assert np.allclose((0.5 * (SX - 1j * SY)).matrix, SMINUS.matrix)

    def test_ladder_operators_are_adjoints(self):
        assert np.allclose(SPLUS.dagger().matrix, SMINUS.matrix)

    def test_matrices_are_read_only(self):
        """Module-level singletons: an in-place write would leak everywhere."""
        with pytest.raises(ValueError):
            SZ.matrix[0, 0] = 99.0


class TestOperations:
    """The array-level primitives."""

    def test_dagger(self):
        assert np.allclose(operations.dagger(SPLUS.matrix), SMINUS.matrix)

    def test_dagger_conjugates(self):
        assert np.allclose(operations.dagger(SY.matrix), SY.matrix)

    def test_is_hermitian(self):
        assert operations.is_hermitian(SZ.matrix)
        assert not operations.is_hermitian(SPLUS.matrix)

    def test_is_hermitian_respects_tolerance(self):
        """np.allclose keeps rtol=1e-5 unless told otherwise, which would
        swamp `tol` for large matrix elements."""
        matrix = np.array([[1e6, 1e6 + 5.0], [1e6, 1e6]])
        assert not operations.is_hermitian(matrix, tol=1e-10)

    @pytest.mark.parametrize("shape", [(1, 3), (2, 3), (3,)])
    def test_non_square_is_not_hermitian(self, shape):
        """Must not broadcast its way to True."""
        assert not operations.is_hermitian(np.ones(shape))
        assert not operations.is_unitary(np.ones(shape))

    def test_is_unitary(self):
        assert operations.is_unitary(SY.matrix)
        assert not operations.is_unitary(SPLUS.matrix)

    def test_commutator(self):
        assert np.allclose(operations.commutator(SX.matrix, SY.matrix), 2j * SZ.matrix)

    def test_anticommutator(self):
        assert np.allclose(
            operations.anticommutator(SX.matrix, SX.matrix), 2 * np.eye(2)
        )

    def test_tensor_product(self):
        product = operations.tensor_product(SZ.matrix, SZ.matrix)
        assert product.shape == (4, 4)
        assert np.allclose(np.diag(product), [1, -1, -1, 1])


class TestOperator:
    def test_accepts_non_hermitian(self):
        """Ladder operators and propagators are not observables."""
        assert Operator(SPLUS.matrix).is_hermitian() is False

    def test_dim(self):
        assert SX.dim == 2

    @pytest.mark.parametrize("bad", [np.ones((2, 3)), np.array([1.0, 2.0, 3.0])])
    def test_rejects_non_square(self, bad):
        with pytest.raises(ValueError, match="square"):
            Operator(bad)

    def test_coerces_to_complex(self):
        assert Operator(np.array([[1, 0], [0, -1]])).matrix.dtype == complex

    def test_accepts_nested_lists(self):
        assert Operator([[1, 0], [0, -1]]).dim == 2

    def test_frozen(self):
        with pytest.raises(Exception):
            Operator(SX.matrix).matrix = np.eye(2)

    def test_dagger(self):
        assert np.allclose(SPLUS.dagger().matrix, SMINUS.matrix)

    def test_matmul(self):
        assert np.allclose((SX @ SX).matrix, np.eye(2))

    def test_add_and_sub(self):
        assert np.allclose((SX + SZ).matrix, SX.matrix + SZ.matrix)
        assert np.allclose((SX - SZ).matrix, SX.matrix - SZ.matrix)

    def test_scalar_multiplication_both_sides(self):
        assert np.allclose((2 * SX).matrix, 2 * SX.matrix)
        assert np.allclose((SX * 2).matrix, 2 * SX.matrix)

    def test_negation(self):
        assert np.allclose((-SX).matrix, -SX.matrix)


class TestObservable:
    def test_rejects_non_hermitian(self):
        with pytest.raises(ValueError, match="Hermitian"):
            Observable(SPLUS.matrix)

    @pytest.mark.parametrize("bad", [np.ones((2, 3)), np.array([1.0, 2.0, 3.0])])
    def test_non_square_reports_the_shape_problem(self, bad):
        """Not 'must be Hermitian' -- a 1-D array is not a matrix at all."""
        with pytest.raises(ValueError, match="square"):
            Observable(bad)

    def test_eigenvalues_ascending(self):
        assert np.allclose(SZ.eigenvalues(), [-1.0, 1.0])

    def test_eigenvectors_are_rows(self):
        """eigh returns columns; row i must be the eigenvector for eigenvalue i."""
        observable = Observable(
            np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.3], [0.0, 0.3, 5.0]])
        )
        values, vectors = observable.eigenvalues(), observable.eigenvectors()
        for i in range(observable.dim):
            assert np.allclose(observable.matrix @ vectors[i], values[i] * vectors[i])

    def test_eigenvectors_orthonormal(self):
        vectors = SX.eigenvectors()
        assert np.allclose(vectors.conj() @ vectors.T, np.eye(2))

    def test_diagonalises_once(self, monkeypatch):
        calls = []
        real_eigh = np.linalg.eigh
        monkeypatch.setattr(
            np.linalg, "eigh", lambda m: (calls.append(1), real_eigh(m))[1]
        )
        observable = Observable(SZ.matrix)
        observable.eigenvalues()
        observable.eigenvectors()
        observable.eigensystem
        assert len(calls) == 1

    def test_expectation(self):
        assert SZ.expectation(np.array([1.0, 0.0])) == pytest.approx(1.0)
        assert SZ.expectation(np.array([1.0, 1.0]) / np.sqrt(2)) == pytest.approx(0.0)

    def test_eigensystem_indexing(self):
        eigensystem = SZ.eigensystem
        assert eigensystem.n_states == 2
        assert np.allclose(eigensystem.eigenvector(0), eigensystem.eigenvectors[0])
        with pytest.raises(IndexError):
            eigensystem.eigenvector(2)


class TestHamiltonian:
    @pytest.fixture
    def hamiltonian(self):
        return Hamiltonian(np.array([[1.0, 0.2], [0.2, 2.0]]))

    def test_eigenenergies_match_numpy(self, hamiltonian):
        assert np.allclose(
            hamiltonian.eigenenergies(), np.linalg.eigvalsh(hamiltonian.matrix)
        )

    def test_ground_state_and_energy(self, hamiltonian):
        assert hamiltonian.ground_energy == pytest.approx(hamiltonian.eigenenergies()[0])
        assert np.allclose(
            hamiltonian.matrix @ hamiltonian.ground_state,
            hamiltonian.ground_energy * hamiltonian.ground_state,
        )

    def test_propagator_matches_expm(self, hamiltonian):
        propagator = hamiltonian.propagator(1.7)
        assert np.allclose(propagator.matrix, expm(-1j * hamiltonian.matrix * 1.7))

    def test_propagator_is_unitary(self, hamiltonian):
        assert hamiltonian.propagator(3.2).is_unitary()

    def test_propagator_at_zero_is_identity(self, hamiltonian):
        assert np.allclose(hamiltonian.propagator(0.0).matrix, np.eye(2))

    def test_propagator_inverse(self, hamiltonian):
        """U(-t) = U(t)†"""
        assert np.allclose(
            hamiltonian.propagator(-1.7).matrix,
            hamiltonian.propagator(1.7).dagger().matrix,
        )

    def test_propagator_is_not_an_observable(self, hamiltonian):
        """Unitary, not Hermitian -- must not claim the Observable guarantee."""
        assert type(hamiltonian.propagator(1.7)) is Operator


class TestIdentityAndEmbed:
    def test_identity_is_an_observable(self):
        assert type(identity(3)) is Observable
        assert np.allclose(identity(3).matrix, np.eye(3))

    def test_identity_is_hermitian_and_unitary(self):
        assert identity(4).is_hermitian()
        assert identity(4).is_unitary()

    @pytest.mark.parametrize("dim", [0, -1])
    def test_identity_rejects_non_positive_dimension(self, dim):
        with pytest.raises(ValueError, match="at least 1"):
            identity(dim)

    def test_embed_is_the_padded_tensor_product(self):
        assert np.allclose(
            embed(SZ, 0, (2, 2)).matrix, np.kron(SZ.matrix, np.eye(2))
        )
        assert np.allclose(
            embed(SZ, 1, (2, 2)).matrix, np.kron(np.eye(2), SZ.matrix)
        )

    def test_embed_preserves_the_class(self):
        """Padding with identities changes neither Hermiticity nor meaning."""
        assert type(embed(SZ, 0, (2, 2))) is Observable
        assert type(embed(Hamiltonian(SZ.matrix), 0, (2, 2))) is Hamiltonian
        assert type(embed(SPLUS, 0, (2, 2))) is Operator

    def test_total_spin_energies_add(self):
        """SZ (x) I + I (x) SZ has eigenvalues +2, 0, 0, -2 -- sums, not products."""
        total = embed(SZ, 0, (2, 2)) + embed(SZ, 1, (2, 2))
        assert type(total) is Observable
        assert np.allclose(np.sort(total.eigenvalues()), [-2.0, 0.0, 0.0, 2.0])

    def test_total_differs_from_the_bare_tensor_product(self):
        total = embed(SZ, 0, (2, 2)) + embed(SZ, 1, (2, 2))
        product = tensor_product(SZ, SZ)
        assert not np.allclose(total.matrix, product.matrix)

    def test_embed_handles_unequal_dimensions(self):
        embedded = embed(identity(3), 1, (2, 3, 2))
        assert embedded.dim == 12

    def test_embed_rejects_empty_dims(self):
        with pytest.raises(ValueError, match="at least one subsystem"):
            embed(SZ, 0, ())

    @pytest.mark.parametrize("site", [-1, 2])
    def test_embed_rejects_site_out_of_range(self, site):
        with pytest.raises(ValueError, match="out of range"):
            embed(SZ, site, (2, 2))

    def test_embed_rejects_dimension_mismatch(self):
        with pytest.raises(ValueError, match="but subsystem"):
            embed(SZ, 0, (3, 2))


class TestClassPreservation:
    """Operations keep the subclass exactly when the maths allows it."""

    def test_sum_of_hamiltonians_is_a_hamiltonian(self):
        """H_free + H_interaction is how an interacting system is built."""
        total = Hamiltonian(np.diag([1.0, 2.0])) + Hamiltonian(0.3 * SX.matrix)
        assert type(total) is Hamiltonian
        assert np.allclose(total.eigenenergies(), np.linalg.eigvalsh(total.matrix))

    def test_difference_and_negation_preserve(self):
        assert type(SZ - SX) is Observable
        assert type(-Hamiltonian(SZ.matrix)) is Hamiltonian

    def test_real_scalar_preserves(self):
        assert type(2.5 * SZ) is Observable

    def test_complex_scalar_does_not_preserve(self):
        """1j * A is anti-Hermitian."""
        scaled = 1j * SZ
        assert type(scaled) is Operator
        assert not scaled.is_hermitian()

    def test_dagger_preserves(self):
        assert type(Hamiltonian(SZ.matrix).dagger()) is Hamiltonian

    def test_common_type_is_symmetric(self):
        assert type(Hamiltonian(SZ.matrix) + SX) is Observable
        assert type(SX + Hamiltonian(SZ.matrix)) is Observable

    def test_mixing_in_a_bare_operator_drops_to_operator(self):
        assert type(SZ + SPLUS) is Operator

    def test_siblings_fall_back_to_their_nearest_common_ancestor(self):
        """Neither derives from the other, but both are still Observables."""

        class Projector(Observable):
            pass

        assert type(Projector(SZ.matrix) + Hamiltonian(SZ.matrix)) is Observable
        assert type(Projector(SZ.matrix) + Projector(SZ.matrix)) is Projector

    def test_commutator_is_not_an_observable(self):
        """[A,B]† = -[A,B]: anti-Hermitian, so returning Observable would raise."""
        result = commutator(SX, SY)
        assert type(result) is Operator
        assert not result.is_hermitian()

    def test_anticommutator_preserves(self):
        """{A,B}† = {A,B}, unlike the commutator."""
        assert type(anticommutator(SX, SZ)) is Observable

    def test_matmul_does_not_preserve(self):
        """Hermitian only when they commute."""
        assert type(SZ @ SX) is Operator

    def test_tensor_product_does_not_preserve(self):
        """Deliberate: H_1 ⊗ H_2 is not the Hamiltonian of the pair."""
        pair = tensor_product(Hamiltonian(SZ.matrix), Hamiltonian(SZ.matrix))
        assert type(pair) is Operator


class TestNoninteracting:
    def test_energies_add(self):
        """Independent subsystems: energies sum, they do not multiply."""
        a = Hamiltonian(np.diag([0.0, 1.5]))
        b = Hamiltonian(np.diag([0.0, 4.0]))
        joint = noninteracting([a, b])
        assert np.allclose(np.sort(joint.eigenenergies()), [0.0, 1.5, 4.0, 5.5])

    def test_is_not_the_kronecker_product(self):
        """np.kron(H1, H2) would give [0, 0, 0, 6] instead."""
        a = Hamiltonian(np.diag([0.0, 1.5]))
        b = Hamiltonian(np.diag([0.0, 4.0]))
        kron = np.linalg.eigvalsh(np.kron(a.matrix, b.matrix))
        assert not np.allclose(
            np.sort(noninteracting([a, b]).eigenenergies()), np.sort(kron)
        )

    def test_dimension_is_the_product(self):
        parts = [Hamiltonian(np.diag([0.0, 1.0]))] * 3
        assert noninteracting(parts).dim == 8

    def test_returns_a_hamiltonian(self):
        assert type(noninteracting([Hamiltonian(SZ.matrix)])) is Hamiltonian

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            noninteracting([])
