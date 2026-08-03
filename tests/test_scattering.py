# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

from physense_qm.potentials import RectangularBarrier
from physense_qm.states.wavepacket import GaussianWavepacket
from physense_qm.scattering import momentum_density, energy_averaged_transmission


class TestMomentumDensity:
    def test_normalised(self):
        wp = GaussianWavepacket(x0=0.0, k0=2.0, sigma=1.0)
        k = np.linspace(-20, 20, 20001)
        assert np.trapezoid(momentum_density(wp, k), k) == pytest.approx(1.0, abs=1e-4)

    def test_peaked_at_k0(self):
        wp = GaussianWavepacket(x0=0.0, k0=3.0, sigma=1.5)
        k = np.linspace(-10, 10, 2001)
        density = momentum_density(wp, k)
        assert k[np.argmax(density)] == pytest.approx(3.0, abs=0.02)


class TestEnergyAveragedTransmission:
    def test_converges_to_mean_energy_transmission_as_sigma_grows(self):
        barrier = RectangularBarrier(height=5.0, width=1.0)
        E0 = 2.0**2 / 2
        T_mean = barrier.transmission_coefficient(E0)

        errors = [
            abs(
                energy_averaged_transmission(
                    barrier, GaussianWavepacket(x0=-4.0, k0=2.0, sigma=sigma)
                )
                - T_mean
            )
            for sigma in [1.0, 5.0, 20.0, 100.0]
        ]
        # Error should shrink monotonically as the momentum spread narrows
        assert all(a > b for a, b in zip(errors, errors[1:]))
        assert errors[-1] < 1e-4

    def test_sharp_packet_exceeds_mean_energy_transmission(self):
        # T(E) is convex in the tunnelling regime, so a spread-out wavepacket
        # transmits *more* on average than a monochromatic beam at the mean
        # energy would (Jensen's inequality).
        barrier = RectangularBarrier(height=5.0, width=1.0)
        wp = GaussianWavepacket(x0=-4.0, k0=2.0, sigma=1.0)
        E0 = 2.0**2 / 2
        assert energy_averaged_transmission(barrier, wp) > barrier.transmission_coefficient(E0)

    def test_result_is_a_probability(self):
        barrier = RectangularBarrier(height=5.0, width=1.0)
        wp = GaussianWavepacket(x0=-4.0, k0=2.0, sigma=1.0)
        T_eff = energy_averaged_transmission(barrier, wp)
        assert 0.0 < T_eff < 1.0
