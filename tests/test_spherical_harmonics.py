# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from physense_utils.spherical_harmonics import spherical_harmonic

class TestSphericalHarmonics:
    def test_def(self):
        # Test creation cannot happen with invalid l and m
        with pytest.raises(ValueError):
            spherical_harmonic(-1, 0, 0.0, 0.0)
        with pytest.raises(ValueError):
            spherical_harmonic(1, 2, 0.0, 0.0)

    def test_values(self):
        # Test known values of spherical harmonics
        theta = np.pi / 2  # 90 degrees
        phi = 0.0

        # Y_0^0 should be 1/sqrt(4*pi)
        assert spherical_harmonic(0, 0, theta, phi) == pytest.approx(1 / np.sqrt(4 * np.pi))

        # Y_1^0 should be sqrt(3/(4*pi)) * cos(theta)
        assert spherical_harmonic(1, 0, theta, phi) == pytest.approx(np.sqrt(3 / (4 * np.pi)) * np.cos(theta))

        # Y_1^1 should be -sqrt(3/(8*pi)) * sin(theta) * exp(i*phi)
        expected_value = -np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(1j * phi)
        assert spherical_harmonic(1, 1, theta, phi) == pytest.approx(expected_value)

    def test_symmetry(self):
        # Test symmetry properties of spherical harmonics
        theta = np.pi / 3  # 60 degrees
        phi = np.pi / 4    # 45 degrees

        Y_lm = spherical_harmonic(2, 1, theta, phi)
        Y_lm_conj = spherical_harmonic(2, -1, theta, phi).conjugate()

        assert Y_lm == pytest.approx((-1)**1 * Y_lm_conj)

    def test_parity(self):
        # Test parity property of spherical harmonics
        theta, phi = np.meshgrid(np.arange(0, np.pi, 0.1),
                         np.arange(0, 2*np.pi, 0.1),
                         indexing='ij')

        Y_lm = spherical_harmonic(3, 2, theta, phi)
        Y_lm_parity = spherical_harmonic(3, 2, np.pi - theta, phi + np.pi)

        assert Y_lm == pytest.approx((-1)**3 * Y_lm_parity)