# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from eigora.functions import gaussian, sinc, heaviside


class TestGaussian:
    def test_peak_at_x0(self):
        x = np.linspace(-5, 5, 1000)
        g = gaussian(x, x0=2.0, sigma=1.0)
        assert x[np.argmax(g)] == pytest.approx(2.0, abs=0.02)

    def test_amplitude(self):
        x = np.array([0.0])
        assert gaussian(x, x0=0.0, sigma=1.0, amplitude=3.0)[0] == pytest.approx(3.0)

    def test_width(self):
        """Value at x0 ± sigma should be exp(-0.5) of peak."""
        x = np.array([0.0, 1.0, -1.0])
        g = gaussian(x, x0=0.0, sigma=1.0)
        assert g[1] == pytest.approx(np.exp(-0.5) * g[0])
        assert g[2] == pytest.approx(np.exp(-0.5) * g[0])

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            gaussian(np.array([0.0]), sigma=0.0)

    def test_negative_sigma(self):
        with pytest.raises(ValueError):
            gaussian(np.array([0.0]), sigma=-1.0)


class TestSinc:
    def test_value_at_zero(self):
        assert sinc(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_zeros_at_pi_multiples(self):
        x = np.array([np.pi, 2 * np.pi, 3 * np.pi])
        assert np.allclose(sinc(x), 0.0, atol=1e-14)

    def test_symmetry(self):
        x = np.linspace(0.1, 10.0, 100)
        assert np.allclose(sinc(x), sinc(-x))


class TestHeaviside:
    def test_positive_side(self):
        x = np.array([1.0, 2.0, 3.0])
        assert np.all(heaviside(x) == 1.0)

    def test_negative_side(self):
        x = np.array([-1.0, -2.0, -3.0])
        assert np.all(heaviside(x) == 0.0)

    def test_value_at_zero(self):
        x = np.array([0.0])
        assert heaviside(x, value_at_zero=0.5)[0] == pytest.approx(0.5)

    def test_shift(self):
        x = np.array([2.0, 3.0, 4.0])
        h = heaviside(x, x0=3.0)
        assert h[0] == 0.0
        assert h[2] == 1.0
