# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from eigora.grids import GridND
from eigora.qm.potentials import DoubleWell, HarmonicWell, RectangularBarrier
from eigora.qm.states.wavepacket import GaussianWavepacket
from eigora.qm.evolution import (
    classical_trajectory,
    evolve,
    quantum_trajectory,
)

OMEGA = 1.0
PERIOD = 2 * np.pi / OMEGA


@pytest.fixture
def grid():
    return GridND.line(-12.0, 12.0, 512)


@pytest.fixture
def well():
    return HarmonicWell(omega=OMEGA)


def run(grid, potential, packet, t_max=PERIOD, n_frames=80):
    evolution = evolve(
        grid, potential, packet, t_max=t_max, dt=0.002, n_frames=n_frames
    )
    return quantum_trajectory(evolution)


class TestCoherentWidth:
    def test_matches_ground_state_width(self):
        # sigma = 1/sqrt(2 omega)
        assert HarmonicWell(omega=1.0).coherent_width == pytest.approx(1 / np.sqrt(2))
        assert HarmonicWell(omega=2.0).coherent_width == pytest.approx(0.5)

    def test_narrows_as_the_well_stiffens(self):
        assert HarmonicWell(omega=9.0).coherent_width < HarmonicWell(omega=1.0).coherent_width


class TestQuantumTrajectory:
    def test_coherent_state_follows_classical_cosine(self, grid, well):
        """A displaced ground state oscillates as x0*cos(omega t), rigidly."""
        x0 = 2.5
        traj = run(grid, well, GaussianWavepacket(x0=x0, k0=0.0, sigma=well.coherent_width))

        expected = x0 * np.cos(OMEGA * traj.times)
        assert np.max(np.abs(traj.mean_position - expected)) < 1e-3

    def test_coherent_state_does_not_spread(self, grid, well):
        """Delta x is constant in time — this is what makes it 'coherent'."""
        sigma = well.coherent_width
        traj = run(grid, well, GaussianWavepacket(x0=2.5, k0=0.0, sigma=sigma))

        assert np.allclose(traj.spread_position, sigma, atol=1e-3)
        # ... and it sits exactly on the Heisenberg bound the whole way.
        assert np.allclose(traj.uncertainty_product, 0.5, atol=1e-3)

    def test_squeezed_state_breathes_at_twice_omega(self, grid, well):
        """Any other width oscillates in Delta x, at 2*omega."""
        sigma = 0.4  # narrower than the coherent width 0.707
        traj = run(grid, well, GaussianWavepacket(x0=0.0, k0=0.0, sigma=sigma))

        spread = traj.spread_position
        assert spread.max() - spread.min() > 0.2  # genuinely breathing

        # Starting narrow, it is back to its minimum after half a period.
        half = np.argmin(np.abs(traj.times - PERIOD / 2))
        assert spread[half] == pytest.approx(spread[0], abs=1e-2)
        # ... having passed through its widest point a quarter period in.
        quarter = np.argmin(np.abs(traj.times - PERIOD / 4))
        assert spread[quarter] == pytest.approx(spread.max(), abs=1e-2)

    def test_ehrenfest_holds_exactly_in_a_harmonic_well(self, grid, well):
        """<x>(t) coincides with the classical path — even when squeezed."""
        packet = GaussianWavepacket(x0=1.5, k0=1.0, sigma=0.4)
        traj = run(grid, well, packet)
        x_cl, _ = classical_trajectory(well, packet.x0, packet.k0, traj.times)

        assert np.max(np.abs(traj.mean_position - x_cl)) < 5e-3

    def test_energy_is_the_analytic_value(self, grid, well):
        """E = k0^2/2 + 1/(8 sigma^2) + omega^2 (x0^2 + sigma^2)/2."""
        x0, k0, sigma = 2.5, 0.0, well.coherent_width
        traj = run(grid, well, GaussianWavepacket(x0=x0, k0=k0, sigma=sigma))

        expected = (
            0.5 * k0**2
            + 1 / (8 * sigma**2)
            + 0.5 * OMEGA**2 * (x0**2 + sigma**2)
        )
        assert traj.energy == pytest.approx(expected, rel=1e-3)
        # For a coherent state that is omega*(nbar + 1/2) with nbar = omega x0^2/2
        assert traj.energy == pytest.approx(OMEGA * (0.5 * OMEGA * x0**2 + 0.5), rel=1e-3)

    def test_momentum_lags_position_by_a_quarter_period(self, grid, well):
        """<p> = -omega x0 sin(omega t) when <x> = x0 cos(omega t)."""
        x0 = 2.0
        traj = run(grid, well, GaussianWavepacket(x0=x0, k0=0.0, sigma=well.coherent_width))

        expected = -OMEGA * x0 * np.sin(OMEGA * traj.times)
        assert np.max(np.abs(traj.mean_momentum - expected)) < 5e-3

    def test_shapes_and_conservation(self, grid, well):
        traj = run(grid, well, GaussianWavepacket(x0=1.0, k0=0.5, sigma=0.6), n_frames=37)

        assert traj.times.shape == (37,)
        assert traj.mean_position.shape == (37,)
        assert traj.uncertainty_product.shape == (37,)
        assert np.all(traj.uncertainty_product >= 0.5 - 1e-6)


class TestAnharmonicIsWhereTheHarmonicMagicStops:
    """
    The trajectory tools are general; the coherent state is not.

    In V = a x^4 - b x^2 the force is nonlinear, so <V'(x)> != V'(<x>) and
    Ehrenfest no longer closes: the quantum mean drifts away from the
    classical path, and no Gaussian width keeps its shape.
    """

    @pytest.fixture
    def anharmonic(self):
        return DoubleWell(a=1.0, b=4.0)

    def test_tools_still_run(self, grid, anharmonic):
        traj = run(grid, anharmonic, GaussianWavepacket(x0=1.4, k0=0.0, sigma=0.5))
        assert traj.times.shape == traj.mean_position.shape
        assert np.all(np.isfinite(traj.mean_position))
        assert np.all(traj.uncertainty_product >= 0.5 - 1e-6)

    def test_quantum_mean_leaves_the_classical_path(self, grid, anharmonic):
        packet = GaussianWavepacket(x0=1.4, k0=0.0, sigma=0.5)
        traj = run(grid, anharmonic, packet, t_max=4.0)
        x_cl, _ = classical_trajectory(anharmonic, packet.x0, packet.k0, traj.times)

        # Same start, then they separate — unlike the harmonic case above,
        # where the two agree to 5e-3 for the whole run.
        assert abs(traj.mean_position[0] - x_cl[0]) < 1e-2
        assert np.max(np.abs(traj.mean_position - x_cl)) > 0.1

    def test_no_width_stays_constant(self, grid, anharmonic):
        """Every Gaussian deforms here; there is no 'coherent width' to find."""
        for sigma in (0.3, 0.5, 1 / np.sqrt(2), 1.0):
            traj = run(grid, anharmonic, GaussianWavepacket(x0=1.4, k0=0.0, sigma=sigma), t_max=4.0)
            spread = traj.spread_position
            assert spread.max() - spread.min() > 0.05, f"sigma={sigma} held its shape"


class TestClassicalTrajectory:
    def test_harmonic_oscillation(self, well):
        times = np.linspace(0, 2 * PERIOD, 200)
        x, p = classical_trajectory(well, x0=3.0, p0=0.0, times=times)

        assert np.max(np.abs(x - 3.0 * np.cos(OMEGA * times))) < 1e-3
        assert np.max(np.abs(p + 3.0 * OMEGA * np.sin(OMEGA * times))) < 1e-3

    def test_free_particle_moves_at_constant_velocity(self):
        # A wide, flat barrier region far from the particle: force is zero here.
        flat = RectangularBarrier(height=1.0, width=1.0, x0=50.0)
        times = np.linspace(0, 4.0, 40)
        x, p = classical_trajectory(flat, x0=0.0, p0=2.0, times=times)

        assert np.allclose(x, 2.0 * times, atol=1e-6)
        assert np.allclose(p, 2.0, atol=1e-6)

    def test_energy_is_conserved(self, well):
        times = np.linspace(0, 3 * PERIOD, 300)
        x, p = classical_trajectory(well, x0=2.0, p0=1.0, times=times)

        energy = 0.5 * p**2 + 0.5 * OMEGA**2 * x**2
        assert np.max(np.abs(energy - energy[0])) < 1e-3

    def test_empty_times(self, well):
        x, p = classical_trajectory(well, x0=1.0, p0=0.0, times=np.array([]))
        assert x.size == 0 and p.size == 0
