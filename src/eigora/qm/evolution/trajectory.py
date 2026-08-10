# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Trajectories extracted from a time evolution.

`quantum_trajectory` reduces an `Evolution` -- a stack of wavefunctions -- to
the handful of numbers that describe where the packet *is*: <x>(t), <p>(t) and
the spreads around them.

`classical_trajectory` integrates Newton's equations for a point particle in
the same potential. Comparing the two is Ehrenfest's theorem made visible:
d<x>/dt = <p> and d<p>/dt = -<V'(x)>, so whenever the force is linear in x --
the harmonic oscillator, exactly -- <V'(x)> = V'(<x>) and the quantum mean
follows the classical path with no approximation at all. Anywhere else the two
curves separate, and the gap is the genuinely quantum part of the motion.

Atomic units: hbar = m = 1.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eigora.qm.evolution.split_step import Evolution
from eigora.qm.observables import (
    expectation_p,
    expectation_p2,
    expectation_x,
    expectation_x2,
)
from eigora.qm.potentials import Potential


@dataclass(frozen=True)
class QuantumTrajectory:
    """
    Expectation values of a wavepacket over the course of an evolution.

    Attributes
    ----------
    times : NDArray of shape (n_frames,)
        Time of each frame.
    mean_position : NDArray of shape (n_frames,)
        <x>(t).
    mean_momentum : NDArray of shape (n_frames,)
        <p>(t).
    spread_position : NDArray of shape (n_frames,)
        Delta x (t), the standard deviation of position. Constant for a
        coherent state; oscillating at 2*omega for any other Gaussian in a
        harmonic well.
    spread_momentum : NDArray of shape (n_frames,)
        Delta p (t).
    uncertainty_product : NDArray of shape (n_frames,)
        Delta x * Delta p (t). Bounded below by 1/2, and equal to 1/2 for all
        t when the state is coherent.
    energy : float
        <H> = <p^2>/2 + <V>, evaluated on the first frame. Conserved, so one
        number describes the whole run.
    """

    times: NDArray[np.float64]
    mean_position: NDArray[np.float64]
    mean_momentum: NDArray[np.float64]
    spread_position: NDArray[np.float64]
    spread_momentum: NDArray[np.float64]
    uncertainty_product: NDArray[np.float64]
    energy: float


def quantum_trajectory(evolution: Evolution) -> QuantumTrajectory:
    """
    Reduce an evolution to its position/momentum expectation values.

    Parameters
    ----------
    evolution : Evolution
        Result of `evolve`.

    Returns
    -------
    QuantumTrajectory
    """
    grid = evolution.grid
    n = evolution.n_frames

    mean_x = np.empty(n)
    mean_p = np.empty(n)
    spread_x = np.empty(n)
    spread_p = np.empty(n)

    for i in range(n):
        psi = evolution.psi[i]
        x1 = expectation_x(psi, grid)
        x2 = expectation_x2(psi, grid)
        p1 = expectation_p(psi, grid)
        p2 = expectation_p2(psi, grid)
        mean_x[i] = x1
        mean_p[i] = p1
        spread_x[i] = np.sqrt(max(x2 - x1**2, 0.0))
        spread_p[i] = np.sqrt(max(p2 - p1**2, 0.0))

    psi0 = evolution.psi[0]
    v_mean = float(
        np.trapezoid(evolution.potential * np.abs(psi0) ** 2, grid.x)
    )
    energy = 0.5 * expectation_p2(psi0, grid) + v_mean

    return QuantumTrajectory(
        times=np.asarray(evolution.times, dtype=float),
        mean_position=mean_x,
        mean_momentum=mean_p,
        spread_position=spread_x,
        spread_momentum=spread_p,
        uncertainty_product=spread_x * spread_p,
        energy=energy,
    )


def classical_trajectory(
    potential: Potential,
    x0: float,
    p0: float,
    times: NDArray[np.float64],
    dt: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Integrate a point particle in `potential`, sampled at `times`.

    Velocity Verlet on a uniform internal step, then linear interpolation onto
    the requested times. The force is taken as a central difference of V, so
    this works for any potential the library can evaluate, not just the ones
    with an analytic derivative.

    Parameters
    ----------
    potential : Potential
        The same V(x) the wavepacket evolves in.
    x0, p0 : float
        Initial position and momentum (m = 1, so p0 is also the velocity).
    times : NDArray
        Times to report, assumed sorted and starting at or after 0.
    dt : float, optional
        Internal integration step. Defaults to 1/20 of the mean output
        spacing, which keeps the sampling error well below line width.

    Returns
    -------
    x, p : NDArray of shape (len(times),)
        Classical position and momentum at each requested time.
    """
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return np.empty(0), np.empty(0)

    t_end = float(times[-1])
    if dt is None:
        spacing = t_end / max(times.size - 1, 1)
        dt = spacing / 20 if spacing > 0 else 1e-3

    n_steps = max(int(np.ceil(t_end / dt)), 1)
    dt = t_end / n_steps if t_end > 0 else dt

    def force(x: float) -> float:
        h = 1e-5
        probe = np.array([x + h, x - h], dtype=float)
        v_plus, v_minus = potential(probe)
        return float(-(v_plus - v_minus) / (2 * h))

    xs = np.empty(n_steps + 1)
    ps = np.empty(n_steps + 1)
    x, p = float(x0), float(p0)
    f = force(x)
    xs[0], ps[0] = x, p

    for i in range(1, n_steps + 1):
        x = x + p * dt + 0.5 * f * dt**2
        f_next = force(x)
        p = p + 0.5 * (f + f_next) * dt
        f = f_next
        xs[i], ps[i] = x, p

    grid_t = np.linspace(0.0, t_end, n_steps + 1)
    return np.interp(times, grid_t, xs), np.interp(times, grid_t, ps)


__all__ = [
    "QuantumTrajectory",
    "quantum_trajectory",
    "classical_trajectory",
]
