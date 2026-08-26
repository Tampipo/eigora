# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The second supplier of microstates: one state at a time, mutable.

Where a `System` enumerates levels, a `Configuration` *is* a single microstate
and can be perturbed. That is the whole difference between the exact machinery
and the sampled one, and it is why they can share an `Ensemble`: summing
`exp(log w)` over a spectrum gives a partition function, while comparing
`log w` before and after a proposed move gives an acceptance rule.

A `Proposal` describes a move by its **deltas**, never by the state it would
produce. That keeps acceptance O(1) in the system size -- flipping one spin in
a million touches four numbers -- which is the only reason Monte Carlo is worth
doing at all.

`log_bias` is what makes this Metropolis-*Hastings* rather than plain
Metropolis: the proposal asymmetry `log[T(x'->x) / T(x->x')]`. It is zero for a
spin flip and for a uniform relocation, and it is not optional in general --
without it a volume move in the isobaric ensemble silently samples the wrong
distribution, because the `V^N` Jacobian of the coordinate rescaling never
appears in the energy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Proposal:
    """
    A candidate move, described by what it would change.

    Attributes
    ----------
    delta_energy : float
        `E(x') - E(x)`.
    deltas : dict of str to float
        The change in each freed extensive variable, e.g.
        `{"magnetisation": -2.0}`. Must cover exactly what the ensemble frees.
    log_bias : float
        `log[T(x'->x) / T(x->x')]`, zero for a symmetric proposal.
    payload : object
        Whatever `apply` needs to commit the move -- a site index, a particle
        and its destination. Never inspected here.
    """

    delta_energy: float
    deltas: dict[str, float] = field(default_factory=dict)
    log_bias: float = 0.0
    payload: object = None


class Configuration(ABC):
    """
    One microstate, mutable in place.

    Deliberately *not* a frozen dataclass, unlike everything else in the
    package. A Metropolis step has to be O(1) in the system size, and copying
    the configuration to accept a move would make it O(N) -- the immutability
    that makes an `Ensemble` safe to cache would make a sampler useless.

    Subclasses keep `energy` and `extensive` up to date incrementally: `apply`
    adjusts them by the proposal's deltas rather than recomputing from
    scratch. `energy_of` exists so tests can check that the incremental
    bookkeeping has not drifted.
    """

    @property
    @abstractmethod
    def energy(self) -> float:
        """The current energy, maintained incrementally."""

    @abstractmethod
    def extensive(self) -> dict[str, float]:
        """Current values of the extensive variables this state carries."""

    @abstractmethod
    def propose(self, rng) -> "Proposal | None":
        """
        A candidate move, or `None` when there is no legal one.

        `None` is not a rejection: it means the proposal machinery had nothing
        to offer, as when every neighbour of a particle is already occupied.
        Expressing a Pauli block this way keeps it out of the acceptance rule,
        which stays pure Metropolis.
        """

    @abstractmethod
    def apply(self, proposal: Proposal) -> None:
        """Commit a move, updating the energy and extensive variables."""

    @abstractmethod
    def energy_of(self) -> float:
        """
        The energy recomputed from scratch, for checking the running value.

        Incremental updates are where a sampler goes quietly wrong: a sign
        error in one `delta_energy` produces a plausible trajectory at the
        wrong temperature. This is the thing to assert against.
        """


__all__ = ["Configuration", "Proposal"]
