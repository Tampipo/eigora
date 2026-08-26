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

Both paths live here, and which one applies is a matter of construction:

    IdenticalParticles(orbitals, FERMI)                # N free -> grand canonical
    IdenticalParticles(orbitals, FERMI, particles=8)   # N fixed -> canonical

They are genuinely different computations over genuinely different sets of
microstates, which is the distinction between a coupling and a parameter. With
N free the orbitals factorise. With N fixed they do not, and `Z_N` comes from
the Borrmann-Franke recursion -- the `N!`-term permutation sum reorganised by
the cycle structure of each permutation, which is where `recursion_sign` earns
its place beside `grand_sign`.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

from eigora.statphys.systems.base import (
    _MAX_TERMS,
    _QUIET_RUN,
    _TOL,
    Couplings,
    Level,
    Moments,
    ParametrisedSystem,
    SpectralSystem,
)

# Beyond this the exponential is past the double range. Both branches guarded
# by it have an exact limit, so the guard costs no accuracy.
_LOG_HUGE = 700.0

# The fixed-N recursion costs O(N^2) arithmetic and N evaluations of the
# single-particle log Z. This is a guard against an absurd request, not a
# physical limit.
_MAX_RECURSION_N = 1000

# Decimal digits the fermionic recursion may lose to cancellation before the
# answer is noise. Double precision carries about 16.
_MAX_LOST_DIGITS = 10


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
class IdenticalParticles(ParametrisedSystem):
    """
    Indistinguishable particles on a set of single-particle orbitals.

    The particle number is either free or fixed, never both, and saying which
    is how you choose the ensemble -- the same fixed-or-free construction
    `IdealGas` uses for its volume::

        IdenticalParticles(orbitals, FERMI)               # N free, grand canonical
        IdenticalParticles(orbitals, FERMI, particles=8)  # N fixed, canonical

    Free, the grand partition function factorises over orbitals and every
    moment comes from one pass. Fixed, nothing factorises: `Z_N` is the
    permutation sum, evaluated by the cycle recursion in `_fixed_log_z`, and
    the energy moments come from differentiating it.

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
    particles: int | None = None

    def __post_init__(self) -> None:
        if self.particles is not None and self.particles < 0:
            raise ValueError(
                f"particles must be non-negative, got {self.particles}"
            )
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
        """The particle number, when it is the thing that fluctuates."""
        if self.particles is not None:
            return frozenset()
        return frozenset({"particles"})

    @property
    def parameters(self) -> dict[str, float]:
        """The particle number, when it is instead held fixed."""
        if self.particles is None:
            return {}
        return {"particles": float(self.particles)}

    def at(self, **changes: float) -> "IdenticalParticles":
        """
        The same gas at a different particle number.

        What makes the canonical chemical potential exact: `mu = F(N) - F(N-1)`
        is a step of exactly one particle, so it is the definition of the
        derivative rather than an approximation to it.
        """
        if "particles" in changes:
            changes = dict(changes)
            changes["particles"] = int(changes["particles"])
        return replace(self, **changes)

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
        self._check_couplings(couplings)
        if self.particles is not None:
            return self._fixed_log_z(beta)
        return self._grand_moments(beta, couplings, _TOL).log_z

    def moments(
        self, beta: float, couplings: Couplings = (), tol: float = _TOL
    ) -> Moments:
        """
        Moments, from whichever of the two computations applies.

        With the number free the orbital factorisation gives every moment
        exactly in one pass. With it fixed there is no such factorisation, so
        the energy moments come from differentiating the recursion's `log Z` --
        the generic route in `System`, which is why this defers to `super`.
        """
        self._check_couplings(couplings)
        if self.particles is not None:
            return super().moments(beta, couplings)
        return self._grand_moments(beta, couplings, tol)

    def _grand_moments(
        self, beta: float, couplings: Couplings, tol: float
    ) -> Moments:
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

    # -- fixed particle number: the permutation sum -----------------------

    def _fixed_log_z(self, beta: float) -> float:
        """
        `log Z_N` by the Borrmann-Franke recursion.

            Z_N = (1/N) sum_{k=1..N} s^(k+1) Z_1(k beta) Z_(N-k),   Z_0 = 1

        This *is* the permutation sum, reorganised. Symmetrising N identical
        particles means `Z_N = (1/N!) sum_P s^P Tr[P exp(-beta H)]`, and for a
        non-interacting `H` the trace factorises over the **cycles** of `P`: a
        cycle of length `k` threads the single-particle propagator round `k`
        times and so contributes exactly `Z_1(k beta)`. Counting permutations
        by cycle type collapses `N!` terms into `N`, and only ever asks the
        orbitals for `log_z` at `k` different temperatures -- so an infinite
        spectrum with a closed form is no harder than a finite one.

        `s` here is `recursion_sign`, **not** `grand_sign`: a k-cycle carries
        parity `(-1)^(k-1)`, so fermions alternate and bosons do not, which is
        the opposite of the two signs in the grand canonical sums.
        """
        # No coupling can reach here: with N fixed `extensive_variables` is
        # empty, so `_check_couplings` has already refused anything non-empty.
        count = self.particles
        if count == 0:
            return 0.0
        if self.statistics.recursion_sign == 0:
            # Boltzmann is not in the recursion at all -- s = 0 degenerates it.
            # Correct classical counting is a single division by N!.
            return count * self.orbitals.log_z(beta) - math.lgamma(count + 1)
        if count > _MAX_RECURSION_N:
            raise ValueError(
                f"the fixed-N recursion is O(N^2); {count} particles is past "
                f"the {_MAX_RECURSION_N} cap. Use the grand canonical ensemble, "
                f"which has no recursion and no sign problem"
            )
        if self._beyond_capacity(count):
            # Exactly zero, not a small number: there are no configurations.
            return -math.inf

        sign = self.statistics.recursion_sign
        singles = [self.orbitals.log_z(k * beta) for k in range(1, count + 1)]
        log_z = [0.0]  # log Z_0 = 0

        for number in range(1, count + 1):
            magnitudes = [
                singles[k - 1] + log_z[number - k] for k in range(1, number + 1)
            ]
            signs = [1 if k % 2 else sign for k in range(1, number + 1)]
            total_sign, total = _signed_logsumexp(magnitudes, signs)
            self._check_cancellation(magnitudes, total_sign, total, number)
            log_z.append(total - math.log(number))

        return log_z[count]

    def _beyond_capacity(self, count: int) -> bool:
        """True if Pauli leaves no room for this many fermions."""
        if self.statistics.grand_sign <= 0:
            return False
        capacity = self.orbitals.n_states
        return capacity is not None and count > capacity

    def _check_cancellation(
        self,
        magnitudes: list[float],
        total_sign: int,
        total: float,
        number: int,
    ) -> None:
        """
        Refuse an answer the alternating sum has already destroyed.

        The fermionic terms alternate and cancel, which is the sign problem in
        miniature: the partial sums are exponentially larger than the result,
        so significant digits are lost at a rate that grows with N. The loss is
        measurable -- it is the ratio of the largest term to the signed total --
        so the recursion can say when it has stopped meaning anything instead
        of returning noise shaped like a float.
        """
        peak = max(magnitudes)
        if total_sign <= 0 or peak == -math.inf:
            raise ValueError(
                f"the fermionic recursion cancelled to a non-positive Z_{number}; "
                f"it is reliable to a few tens of particles at most -- use "
                f"BOLTZMANN for the classical limit, or the grand canonical "
                f"ensemble, which has no sign problem"
            )
        lost = (peak - total) / math.log(10.0)
        if lost > _MAX_LOST_DIGITS:
            raise ValueError(
                f"the fermionic recursion lost {lost:.0f} of about 16 digits to "
                f"cancellation at N={number}; it is reliable to a few tens of "
                f"particles at most -- use BOLTZMANN for the classical limit, "
                f"or the grand canonical ensemble, which has no sign problem"
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


def _signed_logsumexp(logs: list[float], signs: list[int]) -> tuple[int, float]:
    """
    `sum_i s_i exp(l_i)`, returned as `(sign, log|sum|)`.

    Log-space is not tidiness here, it is the only thing that works: at low
    temperature `Z_N` underflows to zero in plain floats, and rescaling by the
    ground energy rescues bosons but not fermions, whose N-particle ground
    state is the Fermi sea rather than N times the lowest orbital.
    """
    peak = max(logs)
    if peak == -math.inf:
        return 0, -math.inf
    total = sum(sign * math.exp(value - peak) for value, sign in zip(logs, signs))
    if total == 0.0:
        return 0, -math.inf
    return (1 if total > 0.0 else -1), peak + math.log(abs(total))


__all__ = [
    "Statistics",
    "WithDegeneracy",
    "with_degeneracy",
    "FERMI",
    "BOSE",
    "BOLTZMANN",
    "IdenticalParticles",
]
