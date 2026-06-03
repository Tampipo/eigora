# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
FFT helpers with consistent normalisation and frequency ordering.
All functions return arrays sorted by increasing frequency.
"""

import numpy as np
from numpy.typing import NDArray


def fft1d(f: NDArray[np.float64], dx: float) -> NDArray[np.complex128]:
    """
    Compute the 1D FFT of f with physical normalisation.

    The transform is defined as:
        F(k) = dx * sum_n f(x_n) * exp(-i k x_n)

    Parameters
    ----------
    f : array of shape (N,)
        Input signal sampled on a uniform grid with spacing dx.
    dx : float
        Grid spacing.

    Returns
    -------
    F : array of shape (N,), complex
        FFT coefficients sorted by increasing frequency.
    """
    return np.fft.fftshift(np.fft.fft(f)) * dx


def ifft1d(F: NDArray[np.complex128], dx: float) -> NDArray[np.complex128]:
    """
    Compute the inverse 1D FFT with physical normalisation.

    Parameters
    ----------
    F : array of shape (N,), complex
        FFT coefficients sorted by increasing frequency.
    dx : float
        Grid spacing of the original spatial grid.

    Returns
    -------
    f : array of shape (N,), complex
        Reconstructed signal.
    """
    n = len(F)
    dk = 2 * np.pi / (n * dx)
    return np.fft.ifft(np.fft.ifftshift(F)) * n * dk / (2 * np.pi)


def fft_frequencies(n: int, dx: float) -> NDArray[np.float64]:
    """
    Return the angular frequency array (k) corresponding to an FFT of length n.

    Parameters
    ----------
    n : int
        Number of points.
    dx : float
        Spatial grid spacing.

    Returns
    -------
    k : array of shape (N,)
        Angular frequencies in rad/unit, sorted increasingly.
    """
    return np.fft.fftshift(np.fft.fftfreq(n, d=dx)) * 2 * np.pi

__all__ = ["fft1d", "ifft1d", "fft_frequencies"]