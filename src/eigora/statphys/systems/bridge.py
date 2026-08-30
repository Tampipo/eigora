# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
From quantum mechanics to statistical mechanics: spectra become systems.

"Should a system take a Hamiltonian?" resolves as **no -- it takes a spectrum,
and `eigora.qm` already knows how to produce one.** Everything here is an
adapter: group eigenvalues into levels, hand them to `NLevel`, and the whole
statphys catalogue applies to anything `qm` can solve.

One adapter buys a great deal. `from_hamiltonian(HeisenbergChain(n_sites=4))`
gives exact thermodynamics of an *interacting* spin chain -- the case the
factorised machinery in `identical` explicitly cannot reach, since interactions
destroy the orbital factorisation. Exact diagonalisation pays for it with an
exponential Hilbert space, which is the honest trade and the reason all three
routes coexist:

    identical, non-interacting   IdenticalParticles      O(K) orbitals
    interacting, small           from_hamiltonian        exponential, exact
    interacting, large           Monte Carlo             polynomial, stochastic

**This is the only module in `statphys` that imports `eigora.qm`.** The
dependency runs one way and is visible at a glance; nothing in `qm` knows that
`statphys` exists.

A warning about the energy origin. `qm` returns absolute eigenvalues and so
does this, which means a `qm.discrete.TwoLevel(bias=w)` -- whose levels are
`-w/2, +w/2` -- has a `log Z` differing from `statphys.TwoLevel(w)` by exactly
`beta w / 2`. Every quantity that is a *derivative* of `log Z` agrees; the free
energy does not, because it is only defined up to the origin. Nothing here
shifts a spectrum silently.
"""

import numpy as np

from eigora.grids import GridND
from eigora.qm.discrete.operators import Hamiltonian
from eigora.qm.spectra.base import Spectrum
from eigora.qm.spectra.factory import spectrum_for

from eigora.statphys.systems.known import NLevel

# Eigenvalues closer than this are one level. Degeneracy is exact in theory and
# approximate in floating point, so the grouping needs a tolerance.
_DEGENERACY_TOL = 1e-9


def from_energies(
    energies: "np.ndarray | list[float]", tol: float = _DEGENERACY_TOL
) -> NLevel:
    """
    Group a list of eigenvalues into levels and return a system.

    The primitive the other two adapters share. Eigenvalues within `tol` of
    each other are one level, and how many there are is its degeneracy -- which
    is the only place a numerical spectrum needs a tolerance, since exact
    degeneracy is a statement about symmetry that floating point cannot make.

    Parameters
    ----------
    energies : array-like of float
        Eigenvalues, in any order. Kept absolute.
    tol : float
        Eigenvalues within this are treated as one level.

    Returns
    -------
    NLevel
    """
    values = np.sort(np.asarray(energies, dtype=np.float64))
    if values.size == 0:
        raise ValueError("a system needs at least one energy")
    if not np.all(np.isfinite(values)):
        raise ValueError("every energy must be finite")

    levels: list[float] = [float(values[0])]
    counts: list[int] = [1]
    for value in values[1:]:
        if value - levels[-1] <= tol:
            counts[-1] += 1
        else:
            levels.append(float(value))
            counts.append(1)
    return NLevel(levels, counts)


def from_hamiltonian(
    hamiltonian: Hamiltonian, tol: float = _DEGENERACY_TOL
) -> NLevel:
    """
    The thermodynamics of a finite quantum system, from its Hamiltonian.

    Always exact and always finite: a `Hamiltonian` is a matrix, so its
    spectrum is complete rather than truncated, and no convergence question
    arises. This is the route to interacting systems -- a Heisenberg chain has
    no orbital factorisation, so `IdenticalParticles` cannot touch it, but its
    matrix diagonalises perfectly well up to a dozen or so sites.

    Example
    -------
    >>> from eigora.qm.discrete import HeisenbergChain
    >>> from eigora.statphys import Canonical, equilibrium
    >>> chain = from_hamiltonian(HeisenbergChain(n_sites=4, coupling=1.0))
    >>> equilibrium(chain, Canonical(0.5)).heat_capacity
    ...

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Any `eigora.qm.discrete` Hamiltonian.
    tol : float
        Degeneracy grouping tolerance.

    Returns
    -------
    NLevel
    """
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError(
            f"expected a Hamiltonian, got {type(hamiltonian).__name__}"
        )
    return from_energies(hamiltonian.eigenenergies(), tol)


def from_spectrum(
    spectrum: Spectrum, n_levels: int = 64, tol: float = _DEGENERACY_TOL
) -> NLevel:
    """
    The lowest `n_levels` of a `qm` spectrum, as a system.

    Unlike `from_hamiltonian` this is a **truncation**, so the resulting `log
    Z` is only as good as the tail it discards. That is fine at low
    temperature, where the discarded levels are exponentially suppressed, and
    silently wrong at high temperature -- so ask for more levels than you think
    you need and check the answer stops moving.

    `Spectrum.levels` already returns `EnergyLevel(energy, states)` with a
    `degeneracy`, so the grouping is done for us here.

    Parameters
    ----------
    spectrum : Spectrum
        From `eigora.qm.spectra`.
    n_levels : int
        How many levels to keep.
    tol : float
        Degeneracy grouping tolerance, passed to `Spectrum.levels`.

    Returns
    -------
    NLevel
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError(f"expected a Spectrum, got {type(spectrum).__name__}")
    # `Spectrum.levels` guards `n_levels >= 1` itself and always returns at
    # least one level, so there is nothing to check for here.
    levels = spectrum.levels(n_levels, tol)
    return NLevel(
        [level.energy for level in levels],
        [level.degeneracy for level in levels],
    )


def from_potential(
    potential: object,
    grid: GridND | None = None,
    n_levels: int = 64,
    n_states: int = 64,
) -> NLevel:
    """
    Straight from a potential to its thermodynamics.

    Wraps `spectrum_for`, so a harmonic well, a box, a separable 3D trap or a
    numerically solved potential all arrive here by the same door. Truncating,
    for the reason `from_spectrum` gives.

    Parameters
    ----------
    potential : Potential, PotentialND or SeparablePotential
        The system to solve.
    grid : GridND, optional
        Needed only for blocks with no analytic solution.
    n_levels : int
        Levels to keep in the resulting system.
    n_states : int
        States to compute per numerically solved block.

    Returns
    -------
    NLevel
    """
    return from_spectrum(spectrum_for(potential, grid, n_states), n_levels)


__all__ = [
    "from_energies",
    "from_hamiltonian",
    "from_spectrum",
    "from_potential",
]
