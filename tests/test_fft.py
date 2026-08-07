# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from eigora.fft import fft1d, ifft1d, fft_frequencies
from eigora.grids import GridND


class TestFFTFrequencies:
    def test_length(self):
        k = fft_frequencies(n=64, dx=0.1)
        assert len(k) == 64

    def test_sorted(self):
        k = fft_frequencies(n=64, dx=0.1)
        assert np.all(np.diff(k) > 0)

    def test_zero_centred(self):
        k = fft_frequencies(n=64, dx=0.1)
        assert k[32] == pytest.approx(0.0)

    def test_spacing(self):
        n, dx = 64, 0.1
        k = fft_frequencies(n=n, dx=dx)
        dk_expected = 2 * np.pi / (n * dx)
        assert np.diff(k)[0] == pytest.approx(dk_expected, rel=1e-10)


class TestFFT1D:
    def test_roundtrip(self):
        grid = GridND.line(-10.0, 10.0, 256)
        f = np.exp(-0.5 * grid.x**2)
        F = fft1d(f, grid.dx)
        f_reconstructed = ifft1d(F, grid.dx)
        assert np.allclose(f, f_reconstructed.real, atol=1e-10)

    def test_gaussian_transform(self):
        """FFT of a Gaussian should be a Gaussian."""
        n = 512
        grid = GridND.line(-20.0, 20.0, n)
        sigma = 1.0
        f = np.exp(-0.5 * (grid.x / sigma) ** 2)
        F = fft1d(f, grid.dx)
        k = fft_frequencies(n, grid.dx)
        sigma_k = 1.0 / sigma
        F_expected = sigma * np.sqrt(2 * np.pi) * np.exp(-0.5 * (k * sigma) ** 2)
        assert np.allclose(np.abs(F), F_expected, atol=1e-3)

    def test_parsevals_theorem(self):
        """Energy should be conserved: integral |f|^2 dx = integral |F|^2 dk / 2pi."""
        grid = GridND.line(-10.0, 10.0, 256)
        n = grid.n_points
        f = np.exp(-0.5 * grid.x**2)
        F = fft1d(f, grid.dx)
        k = fft_frequencies(n, grid.dx)
        dk = k[1] - k[0]
        energy_x = np.sum(np.abs(f) ** 2) * grid.dx
        energy_k = np.sum(np.abs(F) ** 2) * dk / (2 * np.pi)
        assert energy_x == pytest.approx(energy_k, rel=1e-4)
