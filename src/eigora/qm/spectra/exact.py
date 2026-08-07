# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Analytic solutions for the 1D potentials that have one.

A registry maps a potential class to its spectrum, rather than each potential
class knowing how to solve itself -- that keeps `potentials` free of any
dependency on `spectra`.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_hermite, gammaln

from eigora.qm.potentials.known import HarmonicWell, InfiniteSquareWell, Potential
from eigora.qm.potentials.separable import Block1D
from eigora.qm.spectra.base import Label, Spectrum


class HarmonicSpectrum(Spectrum):
    """
    Exact eigenstates of V(x) = 0.5 * omega^2 * (x - x0)^2.

        E_n = omega * (n + 1/2),  n = 0, 1, 2, ...
        psi_n(x) = (omega/pi)^(1/4) / sqrt(2^n n!) * H_n(sqrt(omega) u)
                   * exp(-omega u^2 / 2),   u = x - x0

    The tower is unbounded, but `scipy.special.eval_hermite` overflows to inf
    once H_n exceeds the double-precision range: measured against the
    normalisation integral, states are accurate up to n ~ 160 and become nan
    beyond n ~ 170.
    """

    def __init__(self, potential: HarmonicWell) -> None:
        self.potential = potential
        self.omega = potential.omega
        self.x0 = potential.x0

    @property
    def ndim(self) -> int:
        return 1

    @property
    def quantum_numbers(self) -> tuple[str, ...]:
        return ("n",)

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def n_available(self) -> int | None:
        return None

    def energy(self, label: Label) -> float:
        n = self._quantum_number(label)
        return float(self.omega * (n + 0.5))

    def wavefunction(self, label: Label) -> Callable[..., NDArray[np.float64]]:
        n = self._quantum_number(label)
        omega, x0 = self.omega, self.x0
        # log-space prefactor: 2^n n! overflows well before eval_hermite does
        log_norm = 0.25 * np.log(omega / np.pi) - 0.5 * (n * np.log(2) + gammaln(n + 1))

        def psi(x: NDArray[np.float64]) -> NDArray[np.float64]:
            u = np.asarray(x, dtype=np.float64) - x0
            return np.exp(log_norm) * eval_hermite(n, np.sqrt(omega) * u) * np.exp(
                -0.5 * omega * u**2
            )

        return psi

    def states(self, n: int) -> list[Label]:
        _check_count(n)
        return [(i,) for i in range(n)]

    def _quantum_number(self, label: Label) -> int:
        self._check_arity(label)
        n = label[0]
        if n < 0:
            raise ValueError(f"harmonic quantum number n must be >= 0, got {n}")
        return n


class BoxSpectrum(Spectrum):
    """
    Exact eigenstates of an infinite square well of width L centred on x0.

        E_n = n^2 pi^2 / (2 L^2),  n = 1, 2, 3, ...
        psi_n(x) = sqrt(2/L) * sin(n pi (x - x_left) / L)  inside, 0 outside

    `InfiniteSquareWell` approximates the walls with a large finite value; this
    spectrum is the idealised infinite-wall limit, so numerical eigenstates of
    that potential approach these energies from above.
    """

    def __init__(self, potential: InfiniteSquareWell) -> None:
        self.potential = potential
        self.width = potential.width
        self.x_left = potential.x0 - potential.width / 2

    @property
    def ndim(self) -> int:
        return 1

    @property
    def quantum_numbers(self) -> tuple[str, ...]:
        return ("n",)

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def n_available(self) -> int | None:
        return None

    def energy(self, label: Label) -> float:
        n = self._quantum_number(label)
        return float((n * np.pi) ** 2 / (2 * self.width**2))

    def wavefunction(self, label: Label) -> Callable[..., NDArray[np.float64]]:
        n = self._quantum_number(label)
        width, x_left = self.width, self.x_left

        def psi(x: NDArray[np.float64]) -> NDArray[np.float64]:
            u = np.asarray(x, dtype=np.float64) - x_left
            inside = (u >= 0) & (u <= width)
            return np.where(
                inside, np.sqrt(2 / width) * np.sin(n * np.pi * u / width), 0.0
            )

        return psi

    def states(self, n: int) -> list[Label]:
        _check_count(n)
        return [(i,) for i in range(1, n + 1)]

    def _quantum_number(self, label: Label) -> int:
        self._check_arity(label)
        n = label[0]
        if n < 1:
            raise ValueError(f"box quantum number n must be >= 1, got {n}")
        return n


#: Potential class -> the spectrum that solves it analytically.
EXACT_SPECTRA: dict[type, Callable[[Potential], Spectrum]] = {
    HarmonicWell: HarmonicSpectrum,
    InfiniteSquareWell: BoxSpectrum,
}


def unwrap(potential: object) -> object:
    """The underlying 1D potential of a block, or the object itself."""
    return potential.potential if isinstance(potential, Block1D) else potential


def has_exact_spectrum(potential: object) -> bool:
    """True if this potential has an analytic solution in the registry."""
    return type(unwrap(potential)) in EXACT_SPECTRA


def exact_spectrum(potential: object) -> Spectrum:
    """
    The analytic spectrum of a known 1D potential.

    Raises
    ------
    ValueError
        If the potential has no registered analytic solution.
    """
    inner = unwrap(potential)
    try:
        factory = EXACT_SPECTRA[type(inner)]
    except KeyError:
        known = ", ".join(sorted(cls.__name__ for cls in EXACT_SPECTRA))
        raise ValueError(
            f"no exact spectrum for {type(inner).__name__}; known: {known}"
        ) from None
    return factory(inner)


def _check_count(n: int) -> None:
    if n < 1:
        raise ValueError(f"number of states must be at least 1, got {n}")


__all__ = [
    "HarmonicSpectrum",
    "BoxSpectrum",
    "EXACT_SPECTRA",
    "exact_spectrum",
    "has_exact_spectrum",
    "unwrap",
]
