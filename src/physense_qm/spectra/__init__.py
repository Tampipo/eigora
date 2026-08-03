# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Eigenstates of quantum systems, exact or numerical.

Every solution implements the same `Spectrum` interface -- energies,
wavefunctions, energy levels and degeneracies -- so analytic and numerical
blocks compose into the same separable system. `spectrum_for` is the entry
point.
"""

from physense_qm.spectra.base import EnergyLevel, Label, Spectrum
from physense_qm.spectra.exact import (
    EXACT_SPECTRA,
    BoxSpectrum,
    HarmonicSpectrum,
    exact_spectrum,
    has_exact_spectrum,
)
from physense_qm.spectra.numerical import NumericalSpectrum
from physense_qm.spectra.separable import SeparableSpectrum
from physense_qm.spectra.factory import spectrum_for

__all__ = [
    "Spectrum",
    "EnergyLevel",
    "Label",
    "HarmonicSpectrum",
    "BoxSpectrum",
    "EXACT_SPECTRA",
    "exact_spectrum",
    "has_exact_spectrum",
    "NumericalSpectrum",
    "SeparableSpectrum",
    "spectrum_for",
]
