# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Scattering-theory quantities for a wavepacket incident on a rectangular barrier.
Atomic units: hbar = m = 1.
"""

import numpy as np
from numpy.typing import NDArray

from physense_qm.potentials import RectangularBarrier
from physense_qm.states.wavepacket import GaussianWavepacket


def momentum_density(
    wavepacket: GaussianWavepacket, k: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    |phi(k)|^2, the normalised momentum-space probability density.

    The Fourier transform of psi(x) = exp(-(x-x0)^2 / (4 sigma^2)) exp(i k0 x)
    is itself Gaussian in k, centred at k0, with momentum-space standard
    deviation delta_k = 1 / (2 sigma) -- a minimum-uncertainty wavepacket
    (delta_x * delta_k = 1/2). x0 doesn't affect |phi(k)|^2, only its phase.
    """
    delta_k = 1.0 / (2.0 * wavepacket.sigma)
    return np.exp(-((k - wavepacket.k0) ** 2) / (2 * delta_k**2)) / (
        np.sqrt(2 * np.pi) * delta_k
    )


def energy_averaged_transmission(
    barrier: RectangularBarrier,
    wavepacket: GaussianWavepacket,
    n_sigma: float = 8.0,
    n_points: int = 2001,
) -> float:
    """
    Predicted asymptotic transmitted probability for a wavepacket scattering
    off a rectangular barrier:

        T_eff = integral of |phi(k)|^2 * T(E(k)) dk,   E(k) = k^2 / 2

    where T is RectangularBarrier.transmission_coefficient. A genuinely
    localised wavepacket is a superposition of many momenta, so what it
    actually transmits is this energy-weighted average -- not T evaluated at
    a single "mean" energy.

    Only meaningful when k0 >> delta_k = 1/(2 sigma), i.e. the wavepacket
    isn't so spatially sharp that its momentum spread swamps k0 (weight at
    k <= 0, where "incident from the left" stops making sense, must be
    negligible). The integral is truncated to [k0 - n_sigma*delta_k, k0 +
    n_sigma*delta_k] (clipped at a small positive k) and renormalised over
    that range.
    """
    delta_k = 1.0 / (2.0 * wavepacket.sigma)
    k_min = max(1e-6, wavepacket.k0 - n_sigma * delta_k)
    k_max = wavepacket.k0 + n_sigma * delta_k
    k = np.linspace(k_min, k_max, n_points)

    weights = momentum_density(wavepacket, k)
    weights = weights / np.trapezoid(weights, k)

    energies = 0.5 * k**2
    transmission = np.array([barrier.transmission_coefficient(float(e)) for e in energies])

    return float(np.trapezoid(weights * transmission, k))


__all__ = ["momentum_density", "energy_averaged_transmission"]
