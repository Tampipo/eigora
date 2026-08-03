# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
High-level interface for 1D quantum mechanics simulations.
"""

from physense_utils.grids import GridND

from physense_qm.potentials import Potential
from physense_qm.eigensolver import EigenSolution, solve_eigenstates
from physense_qm.wavepacket import InitialState
from physense_qm.evolution import Evolution, evolve


class QuantumSystem1D:
    """
    A 1D quantum system defined by a spatial grid and a potential.

    Example
    -------
    >>> from physense_utils.grids import GridND
    >>> from physense_qm import QuantumSystem1D
    >>> from physense_qm.potentials import HarmonicWell
    >>> from physense_qm.wavepacket import GaussianWavepacket
    >>>
    >>> grid = GridND.line(-10.0, 10.0, 512)
    >>> system = QuantumSystem1D(grid=grid, potential=HarmonicWell(omega=1.0))
    >>>
    >>> solution = system.solve(n_states=5)
    >>> print(solution.energies)   # should be close to 0.5, 1.5, 2.5, ...
    >>>
    >>> state = GaussianWavepacket(x0=-5.0, k0=2.0, sigma=1.0)
    >>> evolution = system.evolve(state, t_max=10.0, dt=0.01)
    """

    def __init__(self, grid: GridND, potential: Potential) -> None:
        self.grid = grid
        self.potential = potential

    def solve(self, n_states: int = 10) -> EigenSolution:
        """
        Compute the n_states lowest eigenstates of the Hamiltonian.

        Parameters
        ----------
        n_states : int
            Number of eigenstates to return.

        Returns
        -------
        EigenSolution
        """
        return solve_eigenstates(self.grid, self.potential, n_states)

    def evolve(
        self,
        initial_state: InitialState,
        t_max: float,
        dt: float,
        n_frames: int = 100,
    ) -> Evolution:
        """
        Evolve an initial state in time using the split-step Fourier method.

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
        return evolve(self.grid, self.potential, initial_state, t_max, dt, n_frames)
