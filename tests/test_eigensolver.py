# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from physense_utils.grids import GridND
from physense_qm.potentials import HarmonicWell, InfiniteSquareWell
from physense_qm.solvers.eigensolver import solve_eigenstates


@pytest.fixture
def harmonic_solution():
    grid = GridND.line(-10.0, 10.0, 512)
    potential = HarmonicWell(omega=1.0)
    return solve_eigenstates(grid, potential, n_states=5)


class TestEigenSolverHarmonic:
    def test_n_states(self, harmonic_solution):
        assert harmonic_solution.n_states == 5

    def test_energies_ascending(self, harmonic_solution):
        assert np.all(np.diff(harmonic_solution.energies) > 0)

    def test_ground_state_energy(self, harmonic_solution):
        """E_0 should be close to 0.5 (hbar*omega/2, atomic units)."""
        assert harmonic_solution.ground_energy == pytest.approx(0.5, rel=1e-2)

    def test_energy_spacing(self, harmonic_solution):
        """Energies should be spaced by omega=1."""
        diffs = np.diff(harmonic_solution.energies)
        assert np.allclose(diffs, 1.0, atol=0.02)

    def test_normalisation(self, harmonic_solution):
        """Each eigenfunction should be normalised."""
        for i in range(harmonic_solution.n_states):
            psi = harmonic_solution.wavefunction(i)
            norm = np.trapezoid(psi**2, harmonic_solution.grid.x)
            assert norm == pytest.approx(1.0, rel=1e-3)

    def test_orthogonality(self, harmonic_solution):
        """Eigenfunctions should be orthogonal."""
        psi0 = harmonic_solution.wavefunction(0)
        psi1 = harmonic_solution.wavefunction(1)
        overlap = np.trapezoid(psi0 * psi1, harmonic_solution.grid.x)
        assert overlap == pytest.approx(0.0, abs=1e-6)

    def test_probability_density_positive(self, harmonic_solution):
        assert np.all(harmonic_solution.probability_density(0) >= 0)

    def test_index_out_of_range(self, harmonic_solution):
        with pytest.raises(IndexError):
            harmonic_solution.wavefunction(10)


class TestEigenSolverValidation:
    def test_invalid_n_states(self):
        grid = GridND.line(-5.0, 5.0, 100)
        with pytest.raises(ValueError):
            solve_eigenstates(grid, HarmonicWell(), n_states=0)

    def test_n_states_too_large(self):
        grid = GridND.line(-5.0, 5.0, 10)
        with pytest.raises(ValueError):
            solve_eigenstates(grid, HarmonicWell(), n_states=10)
