# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Indistinguishable particles: quantum statistics on a set of single-particle
orbitals.

The microstates of N identical particles are *occupation numbers*, not tuples
of labelled states, and there are combinatorially many of them. But with the
particle number free the grand partition function factorises over orbitals,

    log Xi = sum_k g_k * (1/s) log(1 + s x_k),   x_k = exp(-beta(eps_k - mu))

so a sum over 2^K configurations becomes a sum over K orbitals. That is why
`IdenticalParticles` is a plain `System` and not a `SpectralSystem`: it has a
cheap partition function and no cheap level stream, which is exactly the case
the layering exists to express.

`s` is the statistics sign, and **it is not the same sign the fixed-N
recursion uses.** They are opposite, which is why `Statistics` carries two:

                grand canonical (1/s) log(1 + s x)      recursion s^(k+1)
    Fermi       s = +1   ->   log(1 + x)                s = -1  ->  + - + -
    Bose        s = -1   ->  -log(1 - x)                s = +1  ->  all +

A single `sign` field would be right in one place and silently wrong in the
other, and both produce plausible floats.

Particle number is free here. Fixing it is a different computation -- the
Borrmann-Franke recursion -- and a different set of microstates, which is the
distinction between a coupling and a parameter.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

from eigora.statphys.systems.base import (
    _MAX_TERMS,
    _QUIET_RUN,
    _TOL,
    Couplings,
    Level,
    Moments,
    SpectralSystem,
    System,
)

# Beyond this the exponential is past the double range. Both branches guarded
# by it have an exact limit, so the guard costs no accuracy.
_LOG_HUGE = 700.0


@dataclass(frozen=True)
class Statistics:
    """
    How indistinguishable particles may share an orbital.

    A frozen dataclass carrying the formulas rather than a tag to branch on:
    every quantity below is one expression valid for all three statistics, so
    there is no `if fermi ... elif bose` anywhere in the package.

    Attributes
    ----------
    name : str
        For messages and repr.
    grand_sign : int
        `s` in `(1/s) log(1 + s x)`. +1 Fermi, -1 Bose, 0 Boltzmann.
    recursion_sign : int
        `s` in the fixed-N recursion `s^(k+1)`. -1 Fermi, +1 Bose, 0 Boltzmann
        (which has no recursion -- `log Z_N = N log Z_1 - log N!`). Opposite to
        `grand_sign`, deliberately.
    """

    name: str
    grand_sign: int
    recursion_sign: int

    def log_term(self, exponent: float) -> float:
        """
        One orbital's contribution to `log Xi`: `(1/s) log(1 + s e^-y)`.

        Takes `y = beta(eps - mu)` rather than the fugacity `x = e^-y`, because
        `x` overflows for a deeply filled orbital while `y` never does. The
        `s = 0` branch is mandatory rather than cosmetic -- the expression
        divides by zero there. Boltzmann is its limit, `log(1 + s x)/s -> x`.
        """
        sign = self.grand_sign
        if exponent < -_LOG_HUGE:
            # e^-y is past the double range. For fermions log(1 + x) -> -y, an
            # orbital so far below mu that it is simply full. Bosons never
            # reach here (mu stays below every orbital) and Boltzmann diverges.
            return -exponent if sign > 0 else math.inf
        x = math.exp(-exponent)
        if sign == 0:
            return x
        return math.log1p(sign * x) / sign

    def occupation(self, exponent: float) -> float:
        """
        Mean occupation of an orbital: `1 / (e^y + s)`, `y = beta(eps - mu)`.

        Algebraically `x / (1 + s x)`, but written by dividing through by `x`
        so that a full orbital gives exactly 1 rather than `inf / inf`. One
        expression gives Fermi-Dirac, Bose-Einstein and Boltzmann for
        s = +1, -1, 0 -- `1/(e^y+1)`, `1/(e^y-1)` and `e^-y`.
        """
        if exponent > _LOG_HUGE:
            return 0.0
        return 1.0 / (math.exp(exponent) + self.grand_sign)

    def occupation_variance(self, occupancy: float) -> float:
        """
        `n (1 - s n)`: the fluctuation of one orbital's occupation.

        Fermi gives `n(1-n)`, vanishing for an empty or full orbital as Pauli
        demands; Bose gives `n(1+n)`, the bunching term; Boltzmann gives `n`,
        which is Poisson.
        """
        return occupancy * (1.0 - self.grand_sign * occupancy)


#: Half-integer spin: at most one particle per orbital.
FERMI = Statistics("fermi", +1, -1)

#: Integer spin: any number per orbital, and a chemical potential below the
#: ground energy.
BOSE = Statistics("bose", -1, +1)

#: The classical limit, where the two agree because occupations are tiny.
BOLTZMANN = Statistics("boltzmann", 0, 0)


@dataclass(frozen=True)
class IdenticalParticles(System):
    """
    Indistinguishable particles on a set of orbitals, with the number free.

    The `orbitals` argument is a **single-particle** system: its levels are the
    orbitals, and their degeneracies are how many orbitals sit at that energy.
    Spin enters through that degeneracy rather than as a special case, so
    Pauli counting falls out of `g_k` for free -- two fermions per orbital
    energy, no special case anywhere::

        IdenticalParticles(with_degeneracy(single, 2), FERMI)   # right

    and two ways of writing it that look similar and are not::

        IdenticalParticles(single * Degenerate(2), FERMI)       # TypeError
        IdenticalParticles(single, FERMI) * Degenerate(2)       # adds log 2

    `single * Degenerate(2)` is a `CompositeSystem`, which has no level stream
    -- so it cannot supply orbitals at all, and is refused rather than
    silently mis-counted. The last line is legal but bolts a two-state block
    onto the gas, adding `log 2` to `log Xi`; it is not a spin.

    **`with_degeneracy` counts capacity, not spin.** It gives `g` copies at the
    same energy carrying the same extensive values, which is exactly right for
    spin-half in zero field and wrong the moment the copies must be told apart.
    In a field the two states split to `eps -+ h/2`, and if you want `<M>` they
    carry magnetisation `+-1/2`; either way they are two *different* levels and
    have to be written as such.

    Example
    -------
    >>> from eigora.statphys import GrandCanonical, NLevel, equilibrium
    >>> orbitals = NLevel([0.0, 1.0, 2.0, 3.0])
    >>> gas = IdenticalParticles(orbitals, FERMI)
    >>> state = equilibrium(gas, GrandCanonical(temperature=0.5, chemical_potential=1.5))
    >>> state.particles
    2.0

    Parameters
    ----------
    orbitals : SpectralSystem
        The single-particle levels. Must be enumerable -- the factorisation is
        over orbitals, so they are the one thing that does get iterated.
    statistics : Statistics
        `FERMI`, `BOSE` or `BOLTZMANN`.
    """

    orbitals: SpectralSystem
    statistics: Statistics

    def __post_init__(self) -> None:
        if not isinstance(self.orbitals, SpectralSystem):
            raise TypeError(
                f"orbitals must be a SpectralSystem so they can be enumerated, "
                f"got {type(self.orbitals).__name__}"
            )
        if not isinstance(self.statistics, Statistics):
            raise TypeError(
                f"expected Statistics, got {type(self.statistics).__name__}"
            )

    @property
    def is_exact(self) -> bool:
        return self.orbitals.is_exact

    @property
    def extensive_variables(self) -> frozenset[str]:
        """The particle number, which is what fluctuates here."""
        return frozenset({"particles"})

    # -- the factorised sums ----------------------------------------------

    def occupation(self, energy: float, beta: float, chemical_potential: float) -> float:
        """
        Mean occupation of one orbital at `energy`.

        The Fermi-Dirac or Bose-Einstein distribution, evaluated directly
        rather than obtained from a derivative.
        """
        return self.statistics.occupation(
            beta * (energy - chemical_potential)
        )

    def log_z(self, beta: float, couplings: Couplings = ()) -> float:
        return self.moments(beta, couplings).log_z

    def moments(self, beta: float, couplings: Couplings = (), tol: float = _TOL) -> Moments:
        """
        Every moment in one pass over the orbitals.

        `log Xi`, `<N>`, `Var(N)`, `<E>` and `Var(E)` are all sums of the same
        per-orbital quantities, so they come out together and exactly -- no
        finite difference anywhere, which makes this the reference the generic
        differencing route is checked against.

        Truncation is judged on `beta(eps - mu)` and not on `beta eps`: a
        Fermi gas with a large chemical potential has *no* decay at all until
        the orbital energy passes mu, so a criterion written in `beta eps`
        would stop while the terms were still growing.
        """
        self._check_couplings(couplings)
        chemical_potential = dict(couplings).get("particles")
        if chemical_potential is None:
            raise ValueError(
                "the particle number of an ideal quantum gas is free, so it "
                "needs an ensemble that frees it -- use GrandCanonical"
            )
        if beta <= 0.0 or not math.isfinite(beta):
            raise ValueError(f"beta must be positive and finite, got {beta}")
        self._check_condensation(beta, chemical_potential)

        statistics = self.statistics
        log_xi = number = number_variance = 0.0
        energy = energy_variance = 0.0
        quiet = 0

        for index, level in enumerate(self.orbitals.levels()):
            weight = level.degeneracy
            exponent = beta * (level.energy - chemical_potential)
            term = statistics.log_term(exponent)
            occupancy = statistics.occupation(exponent)
            spread = statistics.occupation_variance(occupancy)

            log_xi += weight * term
            number += weight * occupancy
            number_variance += weight * spread
            energy += weight * level.energy * occupancy
            energy_variance += weight * level.energy**2 * spread

            contribution = weight * (
                abs(term)
                + occupancy * (1.0 + abs(level.energy) + level.energy**2)
            )
            accumulated = (
                abs(log_xi) + number + number_variance
                + abs(energy) + energy_variance
            )
            if contribution <= tol * max(accumulated, 1.0):
                quiet += 1
                if quiet >= _QUIET_RUN:
                    break
            else:
                quiet = 0

            if index + 1 >= _MAX_TERMS:
                raise ValueError(
                    f"the orbital sum did not converge at beta={beta}, "
                    f"mu={chemical_potential}: after {_MAX_TERMS} orbitals the "
                    f"last still contributes {contribution:.2e}; terms decay in "
                    f"beta(eps - mu), so a large chemical potential needs many "
                    f"orbitals before the decay begins"
                )

        return Moments(
            log_z=log_xi,
            energy=energy,
            energy_variance=energy_variance,
            means={"particles": number},
            variances={"particles": number_variance},
        )

    # -- helpers ----------------------------------------------------------

    def _check_condensation(self, beta: float, chemical_potential: float) -> None:
        """
        Bosons need `mu` below every orbital energy.

        At `mu = eps_0` the ground occupation diverges: that is Bose-Einstein
        condensation, and the ideal-gas formulas stop being the whole story
        rather than merely becoming inaccurate. `log1p(-x)` would return `nan`
        for `x >= 1`, so refuse it by name instead.
        """
        if self.statistics.grand_sign >= 0:
            return
        ground = self.orbitals.ground_energy
        if chemical_potential >= ground:
            raise ValueError(
                f"bosons need a chemical potential below the ground orbital "
                f"energy {ground}, got {chemical_potential}; at or above it the "
                f"ground state occupation diverges (Bose-Einstein condensation)"
            )


@dataclass(frozen=True)
class WithDegeneracy(SpectralSystem):
    """
    Every level of `base`, with its degeneracy multiplied.

    The orbital counterpart of `Degenerate`: that one *is* a system of `g`
    states at zero energy, this one *multiplies* an existing spectrum's
    degeneracies while keeping it enumerable. `base * Degenerate(g)` would give
    the right partition function but a `CompositeSystem`, which has no levels
    and so cannot be handed to `IdenticalParticles` as orbitals.

    The copies are identical in every respect -- same energy, and the same
    extensive values -- so this is a statement about *capacity*: how many
    particles fit, not what distinguishes them. That is right for a spin-half
    in zero field and wrong as soon as the copies differ. Under a field they
    split to `eps -+ h/2`, and carrying a magnetisation means `+-1/2` on each;
    in both cases they are two levels, not one doubled level.

    Parameters
    ----------
    base : SpectralSystem
        The spectrum to multiply.
    multiplicity : int
        How many orbitals sit where `base` has one. `2` is a spin-half in zero
        field -- see above for when that stops being what you want.
    """

    base: SpectralSystem
    multiplicity: int

    def __post_init__(self) -> None:
        if not isinstance(self.base, SpectralSystem):
            raise TypeError(
                f"base must be a SpectralSystem, got {type(self.base).__name__}"
            )
        if self.multiplicity < 1:
            raise ValueError(
                f"multiplicity must be at least 1, got {self.multiplicity}"
            )

    @property
    def is_exact(self) -> bool:
        return self.base.is_exact

    @property
    def n_states(self) -> int | None:
        count = self.base.n_states
        return None if count is None else count * self.multiplicity

    def levels(self) -> Iterator[Level]:
        for level in self.base.levels():
            yield Level(
                level.energy,
                level.degeneracy * self.multiplicity,
                level.extensive,
            )


def with_degeneracy(base: SpectralSystem, multiplicity: int) -> WithDegeneracy:
    """`base` with every level `multiplicity` times as degenerate."""
    return WithDegeneracy(base, multiplicity)


__all__ = [
    "Statistics",
    "WithDegeneracy",
    "with_degeneracy",
    "FERMI",
    "BOSE",
    "BOLTZMANN",
    "IdenticalParticles",
]
