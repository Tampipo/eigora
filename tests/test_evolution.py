# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from eigora.grids import GridND
from eigora.qm.potentials import FreeParticle, RectangularBarrier
from eigora.qm.states.wavepacket import GaussianWavepacket
from eigora.qm.evolution import evolve
from eigora.qm import observables


@pytest.fixture
def grid():
    return GridND.line(-20.0, 20.0, 512)


@pytest.fixture
def wavepacket():
    return GaussianWavepacket(x0=-5.0, k0=2.0, sigma=1.0)


class TestGaussianWavepacket:
    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            GaussianWavepacket(sigma=0.0)

    def test_normalised(self, grid, wavepacket):
        psi = wavepacket.normalised(grid)
        n = np.trapezoid(np.abs(psi) ** 2, grid.x)
        assert n == pytest.approx(1.0, rel=1e-3)

    def test_peak_near_x0(self, grid, wavepacket):
        psi = wavepacket.normalised(grid)
        x_peak = grid.x[np.argmax(np.abs(psi))]
        assert x_peak == pytest.approx(-5.0, abs=0.1)


class TestEvolution:
    def test_norm_conservation(self, grid, wavepacket):
        """Norm should be conserved during free evolution."""
        evo = evolve(
            grid=grid,
            potential=FreeParticle(),
            initial_state=wavepacket,
            t_max=2.0,
            dt=0.01,
            n_frames=10,
        )
        for i in range(evo.n_frames):
            assert evo.norm(i) == pytest.approx(1.0, rel=1e-2)

    def test_n_frames(self, grid, wavepacket):
        evo = evolve(
            grid=grid,
            potential=FreeParticle(),
            initial_state=wavepacket,
            t_max=1.0,
            dt=0.01,
            n_frames=20,
        )
        assert evo.n_frames == 20

    def test_n_frames_exact_with_awkward_step_count(self, grid, wavepacket):
        # t_max/dt doesn't divide evenly by n_frames-1 -- this used to
        # overcount (a fixed save-every stride plus a forced final frame).
        evo = evolve(
            grid=grid,
            potential=FreeParticle(),
            initial_state=wavepacket,
            t_max=9.0,
            dt=0.005,
            n_frames=110,
        )
        assert evo.n_frames == 110

    def test_wavepacket_moves(self, grid, wavepacket):
        """With k0 > 0, the wavepacket should move to the right."""
        evo = evolve(
            grid=grid,
            potential=FreeParticle(),
            initial_state=wavepacket,
            t_max=2.0,
            dt=0.01,
            n_frames=10,
        )
        x_initial = grid.x[np.argmax(evo.probability_density(0))]
        x_final = grid.x[np.argmax(evo.probability_density(-1))]
        assert x_final > x_initial

    def test_invalid_dt(self, grid, wavepacket):
        with pytest.raises(ValueError):
            evolve(grid, FreeParticle(), wavepacket, t_max=1.0, dt=0.0)

    def test_invalid_t_max(self, grid, wavepacket):
        with pytest.raises(ValueError):
            evolve(grid, FreeParticle(), wavepacket, t_max=0.0, dt=0.01)


class TestObservables:
    def test_heisenberg(self, grid, wavepacket):
        """Delta x * Delta p should be >= 0.5."""
        psi = wavepacket.normalised(grid)
        product = observables.heisenberg_product(psi, grid)
        assert product >= 0.5 - 1e-3

    def test_expectation_x(self, grid, wavepacket):
        psi = wavepacket.normalised(grid)
        x_mean = observables.expectation_x(psi, grid)
        assert x_mean == pytest.approx(-5.0, abs=0.1)

    def test_norm(self, grid, wavepacket):
        psi = wavepacket.normalised(grid)
        assert observables.norm(psi, grid) == pytest.approx(1.0, rel=1e-3)
