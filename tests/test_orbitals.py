# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from eigora.grids import GridND
from eigora.qm.states.orbitals import SingleAtomState

class TestSingleAtomState:
    def test_basic(self):
        state = SingleAtomState(Z=2, n=1, l=0, m=0)
        assert state.n == 1
        assert state.l == 0
        assert state.m == 0

    def test_invalid_quantum_numbers(self):
        with pytest.raises(ValueError):
            SingleAtomState(Z=2, n=0, l=0, m=0)  # n must be >= 1
        with pytest.raises(ValueError):
            SingleAtomState(Z=2, n=1, l=1, m=0)  # l must be < n
        with pytest.raises(ValueError):
            SingleAtomState(Z=2, n=1, l=0, m=1)  # |m| must be <= l

    def test_orbital_value(self):
        state = SingleAtomState(Z=3, n=1, l=0, m=0)
        r = 1.0
        theta = np.pi / 2
        phi = 0.0
        value = state._orbital(r, theta, phi)
        assert isinstance(value, complex)

        phi2 = np.pi / 4
        value2 = state._orbital(r, theta, phi2)
        assert value == pytest.approx(value2)  # For l=0, m=0, orbital should be independent of phi

        theta2 = np.pi / 3
        value3 = state._orbital(r, theta2, phi)
        assert value == pytest.approx(value3)  # For l=0, m=0, orbital should be independent of theta, but the radial part may vary with r.

        #Testing ratio is the exponential of the difference of the radial parts, which is a function of r only.
        r2 = 2.0
        value4 = state._orbital(r2, theta, phi)
        assert value/value4  == pytest.approx(np.exp(state.Z * (r2 - r) / state.n))  # For l=0, m=0, the ratio of orbitals at different r should follow the exponential decay of the radial part.

    def test_density(self):
        state = SingleAtomState(Z=1, n=1, l=0, m=0)
        r = 1.0
        theta = np.pi / 2
        phi = 0.0
        density = state.density(r, theta, phi)
        assert isinstance(density, float)
        assert density >= 0.0

    def test_density_to_grid(self):
        state = SingleAtomState(Z=1, n=1, l=0, m=0)
        grid = GridND.cube(half_width=5.0, n_points=11, ndim=3)
        density_grid = state.density_on_grid(grid)
        assert density_grid.shape == grid.shape
        assert np.all(density_grid >= 0.0)

    def test_wavefunction(self):
        state = SingleAtomState(Z=1, n=1, l=0, m=0)
        r = 1.0
        theta = np.pi / 2
        phi = 0.0
        psi = state.wavefunction(r, theta, phi)
        assert isinstance(psi, float)
        # 1s has no nodes and Y_0^0 is a positive real constant, so psi > 0 everywhere.
        assert psi > 0.0
        assert psi**2 == pytest.approx(state.density(r, theta, phi))

    def test_wavefunction_to_grid(self):
        # 2p_z (l=1, m=0) has an angular node at theta=pi/2: psi should be
        # positive on one side of the xy-plane and negative on the other —
        # this sign is exactly what the orbital-viewer isosurface colors by.
        state = SingleAtomState(Z=1, n=2, l=1, m=0)
        grid = GridND.cube(half_width=10.0, n_points=11, ndim=3)
        psi_grid = state.wavefunction_on_grid(grid)
        assert psi_grid.shape == grid.shape
        assert psi_grid.dtype == np.float64
        assert np.any(psi_grid > 0.0)
        assert np.any(psi_grid < 0.0)
        assert psi_grid**2 == pytest.approx(state.density_on_grid(grid))