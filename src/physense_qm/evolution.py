# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Time evolution of quantum states using the split-step Fourier method.
Atomic units: hbar = m = 1.

The split-step method approximates exp(-i H dt) as:
    exp(-i V dt/2) * exp(-i T dt) * exp(-i V dt/2)

where T is applied in momentum space.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physense_utils.grids import Grid1D
from physense_utils.fft import fft_frequencies

from physense_qm.potentials import Potential
from physense_qm.wavepacket import InitialState


@dataclass(frozen=True)
class Evolution:
    """
    Result of a time evolution simulation.

    Attributes
    ----------
    psi : NDArray of shape (n_frames, n_points), complex
        Wavefunction at each saved time step.
    times : NDArray of shape (n_frames,)
        Time values corresponding to each frame.
    grid : Grid1D
        Spatial grid.
    potential : NDArray of shape (n_points,)
        The potential evaluated on the grid.
    """

    psi: NDArray[np.complex128]
    times: NDArray[np.float64]
    grid: Grid1D
    potential: NDArray[np.float64]

    @property
    def n_frames(self) -> int:
        return len(self.times)

    def probability_density(self, frame: int) -> NDArray[np.float64]:
        """Return |psi(x, t)|^2 at a given frame index."""
        return np.abs(self.psi[frame]) ** 2

    def norm(self, frame: int) -> float:
        """Return the norm of the wavefunction at a given frame (should stay ~1)."""
        return float(np.trapezoid(self.probability_density(frame), self.grid.x))


def evolve(
    grid: Grid1D,
    potential: Potential,
    initial_state: InitialState,
    t_max: float,
    dt: float,
    n_frames: int = 100,
) -> Evolution:
    """
    Evolve an initial state under a potential using split-step Fourier.

    Parameters
    ----------
    grid : Grid1D
        Spatial grid.
    potential : Potential
        Potential V(x).
    initial_state : InitialState
        Initial wavefunction psi(x, 0).
    t_max : float
        Total evolution time.
    dt : float
        Time step (should be small enough for stability).
    n_frames : int
        Number of frames to save (evenly spaced between 0 and t_max).

    Returns
    -------
    Evolution
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if n_frames < 2:
        raise ValueError(f"n_frames must be at least 2, got {n_frames}")

    x = grid.x
    V = potential(x)
    k = fft_frequencies(grid.n_points, grid.dx)

    # Precompute propagators
    half_V_prop = np.exp(-0.5j * V * dt)
    T_prop = np.exp(-0.5j * k**2 * dt)  # kinetic: hbar=m=1 so T = k^2/2

    psi = initial_state.normalised(grid)

    n_steps = int(t_max / dt)
    # Evenly-spaced target steps, deduplicated (only relevant if n_frames
    # exceeds n_steps+1) -- guarantees exactly n_frames frames, always
    # including the first and last step, unlike a fixed save-every stride
    # which can silently overshoot n_frames when n_steps isn't a multiple of it.
    save_steps = set(np.linspace(0, n_steps, n_frames).round().astype(int).tolist())

    saved_psi = []
    saved_times = []

    for step in range(n_steps + 1):
        if step in save_steps:
            saved_psi.append(psi.copy())
            saved_times.append(step * dt)

        if step < n_steps:
            psi = half_V_prop * psi
            psi_k = np.fft.fftshift(np.fft.fft(psi))
            psi_k = T_prop * psi_k
            psi = np.fft.ifft(np.fft.ifftshift(psi_k))
            psi = half_V_prop * psi

    return Evolution(
        psi=np.array(saved_psi),
        times=np.array(saved_times),
        grid=grid,
        potential=V,
    )


__all__ = ["Evolution", "evolve"]
