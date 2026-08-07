# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
`spectrum_for`: the single entry point from a potential to its eigenstates.

It walks a separable potential block by block, giving each block the analytic
solution if its potential has one and the numerical solver otherwise, then
composes the results.
"""

from eigora.grids import GridND

from eigora.qm.potentials.base import PotentialND
from eigora.qm.potentials.separable import SeparablePotential, as_block
from eigora.qm.spectra.base import Spectrum
from eigora.qm.spectra.exact import exact_spectrum, has_exact_spectrum
from eigora.qm.spectra.numerical import NumericalSpectrum
from eigora.qm.spectra.separable import SeparableSpectrum


def spectrum_for(
    potential: object,
    grid: GridND | None = None,
    n_states: int = 20,
) -> Spectrum:
    """
    Build the spectrum of a potential.

    Parameters
    ----------
    potential : Potential, PotentialND or SeparablePotential
        The system to solve. A `SeparablePotential` is solved block by block
        and recomposed with `SeparableSpectrum`.
    grid : GridND, optional
        Needed only for blocks without an analytic solution. Either a grid of
        the potential's own dimension (each block gets its own sub-grid) or a
        single 1D grid, used for every 1D block that needs solving.
    n_states : int
        How many states to compute per numerically solved block.

    Returns
    -------
    Spectrum
        `is_exact` reports whether every block was solved analytically.

    Raises
    ------
    ValueError
        If a block has no analytic solution and cannot be solved numerically.
    """
    block = as_block(potential)

    if isinstance(block, SeparablePotential):
        blocks = [
            spectrum_for(inner, _grid_for(grid, block_slice, inner), n_states)
            for inner, block_slice in zip(block.blocks, block.block_slices)
        ]
        return SeparableSpectrum(blocks, names=block.names)

    if has_exact_spectrum(block):
        return exact_spectrum(block)

    if block.ndim != 1:
        raise ValueError(
            f"{type(block).__name__} is {block.ndim}D with no analytic solution, and "
            f"the numerical solver is 1D only; build it as a SeparablePotential "
            f"of solvable blocks instead"
        )
    if grid is None:
        raise ValueError(
            f"{type(_described(block)).__name__} has no analytic solution; pass a "
            f"grid to solve it numerically"
        )
    return NumericalSpectrum.solve(block, grid, n_states=n_states)


def _grid_for(
    grid: GridND | None,
    block_slice: slice,
    block: PotentialND,
) -> GridND | None:
    """The sub-grid a block should be solved on."""
    if grid is None:
        return None
    if grid.ndim == 1:
        # A single 1D grid stands in for every 1D block.
        return grid if block.ndim == 1 else None
    return grid.sub(block_slice.start, block_slice.stop)


def _described(block: PotentialND) -> object:
    """The object to name in an error message (unwrapping 1D adapters)."""
    return getattr(block, "potential", block)


__all__ = ["spectrum_for"]
