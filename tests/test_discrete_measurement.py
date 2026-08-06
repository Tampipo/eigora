# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np

from physense_qm.discrete import SZ, Observable, embed, measurement

PLUS = np.array([1.0, 1.0]) / np.sqrt(2)

# Total S_z for two spins: SZ (x) I + I (x) SZ, eigenvalues +2, 0, 0, -2. The
# zero is two-fold degenerate, which is where measurement written
# per-eigenvector goes wrong.
SZ_TOTAL = embed(SZ, 0, (2, 2)) + embed(SZ, 1, (2, 2))


class TestDegenerateGroups:
    def test_distinct_values_stay_separate(self):
        groups = measurement.degenerate_groups(np.array([-1.0, 1.0]))
        assert groups == [(-1.0, (0,)), (1.0, (1,))]

    def test_equal_values_group(self):
        groups = measurement.degenerate_groups(np.array([-2.0, 0.0, 0.0, 2.0]))
        assert [value for value, _ in groups] == [-2.0, 0.0, 2.0]
        assert groups[1][1] == (1, 2)

    def test_within_tolerance_groups(self):
        groups = measurement.degenerate_groups(np.array([1.0, 1.0 + 1e-12, 5.0]))
        assert len(groups) == 2

    def test_outside_tolerance_does_not_group(self):
        groups = measurement.degenerate_groups(np.array([1.0, 1.0 + 1e-3, 5.0]))
        assert len(groups) == 3


class TestOutcomes:
    def test_superposition_gives_equal_probabilities(self):
        results = SZ.outcomes(PLUS)
        assert [result.value for result in results] == [-1.0, 1.0]
        assert all(result.probability == pytest.approx(0.5) for result in results)

    def test_eigenstate_is_certain(self):
        results = SZ.outcomes(np.array([1.0, 0.0]))
        certain = [result for result in results if result.value == 1.0][0]
        assert certain.probability == pytest.approx(1.0)

    def test_degenerate_value_is_one_outcome(self):
        """Three distinct eigenvalues, not four eigenvectors."""
        results = SZ_TOTAL.outcomes(np.ones(4) / 2.0)
        assert len(results) == 3

    def test_degenerate_probability_sums_over_the_subspace(self):
        state = np.array([0.0, 0.6, 0.8, 0.0])  # entirely inside the zero eigenspace
        results = SZ_TOTAL.outcomes(state)
        zero = [result for result in results if result.value == pytest.approx(0.0)][0]
        assert zero.probability == pytest.approx(1.0)
        assert zero.degeneracy == 2
        assert zero.indices == (1, 2)

    def test_probabilities_sum_to_one(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(6, 6))
        observable = Observable((matrix + matrix.T) / 2)
        state = rng.normal(size=6) + 1j * rng.normal(size=6)
        state /= np.linalg.norm(state)
        total = sum(result.probability for result in observable.outcomes(state))
        assert total == pytest.approx(1.0)

    def test_agrees_with_expectation_value(self):
        """sum(P(lambda) * lambda) must equal <A>, computed a different way."""
        rng = np.random.default_rng(4)
        matrix = rng.normal(size=(6, 6))
        observable = Observable((matrix + matrix.T) / 2)
        state = rng.normal(size=6) + 1j * rng.normal(size=6)
        state /= np.linalg.norm(state)
        weighted = sum(
            result.value * result.probability for result in observable.outcomes(state)
        )
        assert weighted == pytest.approx(observable.expectation(state))

    def test_unnormalised_state_is_normalised(self):
        results = SZ.outcomes(np.array([3.0, 3.0]))
        assert all(result.probability == pytest.approx(0.5) for result in results)

    def test_zero_state_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            SZ.outcomes(np.zeros(2))


class TestCollapse:
    @pytest.fixture
    def observable(self):
        return SZ_TOTAL

    def test_projects_onto_the_subspace_not_one_eigenvector(self, observable):
        """The classic error: keeping only the first eigenvector of the pair."""
        state = np.full(4, 0.5)
        post = measurement.collapse(observable.eigenvectors(), (1, 2), state)
        assert np.allclose(post, [0.0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])
        assert not np.allclose(post, [0.0, 1.0, 0.0, 0.0])

    def test_result_is_normalised(self, observable):
        post = measurement.collapse(observable.eigenvectors(), (1, 2), np.full(4, 0.5))
        assert np.linalg.norm(post) == pytest.approx(1.0)

    def test_measuring_again_is_certain(self, observable):
        """Repeated measurement returns the same value with probability 1."""
        post = measurement.collapse(observable.eigenvectors(), (1, 2), np.full(4, 0.5))
        zero = [r for r in observable.outcomes(post) if r.value == pytest.approx(0.0)][0]
        assert zero.probability == pytest.approx(1.0)

    def test_impossible_outcome_rejected(self, observable):
        """A state with no component in the subspace cannot collapse into it."""
        with pytest.raises(ValueError, match="no component"):
            measurement.collapse(observable.eigenvectors(), (1, 2), np.array([1.0, 0.0, 0.0, 0.0]))


class TestMeasure:
    def test_statistics_follow_the_born_rule(self):
        rng = np.random.default_rng(0)
        observable = SZ
        draws = [observable.measure(PLUS, rng)[0].value for _ in range(4000)]
        assert np.mean(np.array(draws) == 1.0) == pytest.approx(0.5, abs=0.03)

    def test_eigenstate_always_gives_its_own_value(self):
        rng = np.random.default_rng(5)
        observable = SZ
        for _ in range(20):
            outcome, post = observable.measure(np.array([1.0, 0.0]), rng)
            assert outcome.value == 1.0
            assert np.allclose(post, [1.0, 0.0])

    def test_returns_the_collapsed_state(self):
        rng = np.random.default_rng(6)
        observable = SZ
        outcome, post = observable.measure(PLUS, rng)
        assert np.linalg.norm(post) == pytest.approx(1.0)
        repeated = [r for r in observable.outcomes(post) if r.value == outcome.value][0]
        assert repeated.probability == pytest.approx(1.0)

    def test_is_reproducible_with_a_seeded_generator(self):
        observable = SZ
        first = [observable.measure(PLUS, np.random.default_rng(7))[0].value for _ in range(5)]
        second = [observable.measure(PLUS, np.random.default_rng(7))[0].value for _ in range(5)]
        assert first == second

    def test_works_without_an_explicit_generator(self):
        outcome, _ = SZ.measure(PLUS)
        assert outcome.value in (-1.0, 1.0)
