# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
High-level interface for quantum mechanics simulations.
"""

from eigora.grids import GridND

from eigora.qm.potentials.base import PotentialND
from eigora.qm.potentials.known import Potential
from eigora.qm.solvers.eigensolver import EigenSolution, solve_eigenstates
from eigora.qm.spectra.base import Spectrum
from eigora.qm.spectra.factory import spectrum_for
from eigora.qm.states.wavepacket import InitialState
from eigora.qm.evolution.split_step import Evolution, evolve


class QuantumSystem:
    """
    A quantum system defined by a spatial grid and a potential.

    `solve` and `evolve` are the 1D numerical methods; `spectrum` works in any
    dimension, as long as the system separates into solvable blocks.

    Example
    -------
    >>> from eigora.grids import GridND
    >>> from eigora.qm import QuantumSystem
    >>> from eigora.qm.potentials import HarmonicWell, SeparablePotential
    >>> from eigora.qm.states import GaussianWavepacket
    >>>
    >>> grid = GridND.line(-10.0, 10.0, 512)
    >>> system = QuantumSystem(grid=grid, potential=HarmonicWell(omega=1.0))
    >>>
    >>> solution = system.solve(n_states=5)
    >>> print(solution.energies)   # should be close to 0.5, 1.5, 2.5, ...
    >>>
    >>> state = GaussianWavepacket(x0=-5.0, k0=2.0, sigma=1.0)
    >>> evolution = system.evolve(state, t_max=10.0, dt=0.01)
    >>>
    >>> trap = SeparablePotential([HarmonicWell(omega=1.0)] * 3)
    >>> levels = QuantumSystem(GridND.cube(8.0, 64), trap).spectrum().levels(4)
    """

    def __init__(self, grid: GridND, potential: "Potential | PotentialND") -> None:
        self.grid = grid
        self.potential = potential

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions of the system."""
        return self.potential.ndim if isinstance(self.potential, PotentialND) else 1

    def solve(self, n_states: int = 10) -> EigenSolution:
        """
        Compute the n_states lowest eigenstates numerically (1D only).

        Parameters
        ----------
        n_states : int
            Number of eigenstates to return.

        Returns
        -------
        EigenSolution
        """
        self._require_1d("solve")
        return solve_eigenstates(self.grid, self.potential, n_states)

    def evolve(
        self,
        initial_state: InitialState,
        t_max: float,
        dt: float,
        n_frames: int = 100,
    ) -> Evolution:
        """
        Evolve an initial state in time using the split-step Fourier method
        (1D only).

        Parameters
        ----------
        initial_state : InitialState
            The initial wavefunction psi(x, 0).
        t_max : float
            Total simulation time.
        dt : float
            Time step.
        n_frames : int
            Number of frames to save for animation.

        Returns
        -------
        Evolution
        """
        self._require_1d("evolve")
        return evolve(self.grid, self.potential, initial_state, t_max, dt, n_frames)

    def spectrum(self, n_states: int = 20) -> Spectrum:
        """
        Eigenstates of the system, analytic where possible (any dimension).

        Parameters
        ----------
        n_states : int
            States to compute for each block that has to be solved numerically.

        Returns
        -------
        Spectrum
            Check `is_exact` to know whether anything was solved numerically.
        """
        return spectrum_for(self.potential, self.grid, n_states)

    def _require_1d(self, operation: str) -> None:
        if self.ndim != 1:
            raise ValueError(
                f"'{operation}' is 1D only, this system is {self.ndim}D; "
                f"use spectrum() for a separable system in higher dimensions"
            )
        if self.grid.ndim != 1:
            raise ValueError(f"'{operation}' needs a 1D grid, got a {self.grid.ndim}D one")


#: The 1D system was the only one for a while -- keep the old name working.
QuantumSystem1D = QuantumSystem


__all__ = ["QuantumSystem", "QuantumSystem1D"]
