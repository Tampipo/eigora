# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The Metropolis-Hastings loop, and what a run leaves behind.

Three lines of physics:

    proposal = configuration.propose(rng)
    log_accept = ensemble.log_weight_change(**deltas) + proposal.log_bias
    if log_accept >= 0 or rng.random() < exp(log_accept): configuration.apply(...)

and the important word is `ensemble` -- the *same* object the exact machinery
sums over. Canonical, magnetic and grand canonical sampling differ only in
which `Field`s it carries, so there is one sampler rather than one per
ensemble. That was the point of making an ensemble a log-weight in the first
place.

`Sampling` mirrors `ThermalState`: `mean`, `variance`, `response`, `energy`,
`heat_capacity`, `magnetisation`, `susceptibility` under exactly those names,
so a Monte Carlo estimate and an exact result are compared without translating
between two vocabularies. What it adds is `error`, because a sample average
has an uncertainty and a spectral sum does not.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eigora.statphys.ensembles import Ensemble
from eigora.statphys.monte_carlo.base import Configuration
from eigora.statphys.monte_carlo.measure import autocorrelation_time, blocked_error


@dataclass(frozen=True)
class Sampling:
    """
    What a Metropolis run recorded.

    Attributes
    ----------
    ensemble : Ensemble
        The one the chain sampled. Kept so `response` knows the field signs.
    energies : array of float
        Energy at each recorded step.
    series : dict of str to array
        Each freed extensive variable, recorded alongside.
    steps : int
        Proposals attempted, including those with no legal move.
    accepted : int
        Proposals committed.
    blocked : int
        Steps where `propose` returned `None` -- no legal move existed.
    """

    ensemble: Ensemble
    energies: NDArray[np.float64]
    series: dict[str, NDArray[np.float64]]
    steps: int
    accepted: int
    blocked: int

    @property
    def acceptance(self) -> float:
        """Fraction of *offered* moves that were taken."""
        offered = self.steps - self.blocked
        return self.accepted / offered if offered else 0.0

    @property
    def beta(self) -> float:
        return self.ensemble.beta

    # -- the same names `ThermalState` uses --------------------------------

    @property
    def energy(self) -> float:
        return float(self.energies.mean())

    @property
    def energy_variance(self) -> float:
        return float(self.energies.var(ddof=1))

    @property
    def heat_capacity(self) -> float:
        """`C = beta^2 Var(E)`, from the sampled fluctuations."""
        return self.beta**2 * self.energy_variance

    def mean(self, variable: str) -> float:
        return float(self._series(variable).mean())

    def variance(self, variable: str) -> float:
        return float(self._series(variable).var(ddof=1))

    def response(self, variable: str) -> float:
        """
        `d<X>/df = sign * beta * Var(X)`, carrying the field's sign exactly as
        `ThermalState.response` does.
        """
        field = self.ensemble.field(variable)
        return field.sign * self.beta * self.variance(variable)

    @property
    def magnetisation(self) -> float:
        return self.mean("magnetisation")

    @property
    def susceptibility(self) -> float:
        return self.response("magnetisation")

    @property
    def particles(self) -> float:
        return self.mean("particles")

    # -- what a sample has and a sum does not ------------------------------

    def error(self, variable: str | None = None) -> float:
        """
        Standard error of a mean, corrected for the chain's correlations.

        `None` asks about the energy. Blocking rather than `sigma/sqrt(n)`,
        because consecutive samples are not independent and the naive estimate
        is optimistic by roughly `sqrt(2 tau)`.
        """
        return blocked_error(
            self.energies if variable is None else self._series(variable)
        )

    def autocorrelation_time(self, variable: str | None = None) -> float:
        """Steps between effectively independent samples."""
        return autocorrelation_time(
            self.energies if variable is None else self._series(variable)
        )

    def _series(self, variable: str) -> NDArray[np.float64]:
        try:
            return self.series[variable]
        except KeyError:
            known = ", ".join(sorted(self.series)) or "nothing"
            raise ValueError(
                f"this run recorded {known}, not '{variable}'; a variable is "
                f"recorded when the ensemble frees it"
            ) from None


def metropolis(
    configuration: Configuration,
    ensemble: Ensemble,
    steps: int,
    burn_in: int = 0,
    thin: int = 1,
    rng: "np.random.Generator | None" = None,
) -> Sampling:
    """
    Sample a configuration against an ensemble.

    Parameters
    ----------
    configuration : Configuration
        Mutated in place. Pass a fresh one per run.
    ensemble : Ensemble
        Supplies the acceptance rule. Whatever it frees is recorded.
    steps : int
        Proposals to attempt after the burn-in.
    burn_in : int
        Proposals to discard first, letting the chain forget where it started.
    thin : int
        Record one sample every `thin` steps. Reduces storage, not correlation
        -- `error` already accounts for that.
    rng : numpy Generator, optional
        Seed it for reproducibility; defaults to a fresh one.

    Returns
    -------
    Sampling
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if thin < 1:
        raise ValueError(f"thin must be at least 1, got {thin}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative, got {burn_in}")
    if rng is None:
        rng = np.random.default_rng()

    free = ensemble.free
    missing = set(free) - set(configuration.extensive())
    if missing:
        raise ValueError(
            f"{type(ensemble).__name__} frees {sorted(missing)}, which "
            f"{type(configuration).__name__} does not report"
        )

    energies: list[float] = []
    recorded: dict[str, list[float]] = {name: [] for name in free}
    accepted = blocked = 0

    for step in range(burn_in + steps):
        proposal = configuration.propose(rng)
        if proposal is None:
            blocked += 1
        else:
            # A configuration reports everything it carries; the ensemble
            # takes only what it frees. An Ising lattice always has a
            # magnetisation, whether or not this run is trading against it.
            traded = {name: proposal.deltas[name] for name in free}
            change = ensemble.log_weight_change(proposal.delta_energy, **traded)
            log_accept = change + proposal.log_bias
            if log_accept >= 0.0 or rng.random() < math.exp(log_accept):
                configuration.apply(proposal)
                accepted += 1
        if step >= burn_in and (step - burn_in) % thin == 0:
            energies.append(configuration.energy)
            if free:
                current = configuration.extensive()
                for name in free:
                    recorded[name].append(current[name])

    return Sampling(
        ensemble=ensemble,
        energies=np.asarray(energies, dtype=np.float64),
        series={
            name: np.asarray(values, dtype=np.float64)
            for name, values in recorded.items()
        },
        steps=burn_in + steps,
        accepted=accepted,
        blocked=blocked,
    )


__all__ = ["Sampling", "metropolis"]
