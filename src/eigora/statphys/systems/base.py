# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The `System` interface: whatever can supply microstates to statistical mechanics.

Statistical mechanics never needs eigenstates, only the partition function, so
the primitive here is `log_z(beta, couplings)` and nothing else. An enumerable
spectrum is a *stronger* guarantee layered on top as `SpectralSystem` -- because
some of the most useful systems have a cheap partition function and no cheap
level stream at all: a product of many subsystems, an ideal gas in the
continuum limit, N identical particles. Making the level stream the primitive
would force those three to enumerate something combinatorial to satisfy an
interface they do not need.

`System` has **two independent refinements**, not a chain -- being enumerable
and having dials are unrelated capabilities, and a system may have either,
both or neither:

    System              a partition function, and derivatives of it
      SpectralSystem    the levels can be enumerated, so moments are exact
      ParametrisedSystem extensive parameters can be varied, so their
                        conjugates (pressure, chemical potential) exist

Both are combined by ordinary multiple inheritance where a system is both:
`class ClosedTrap(SpectralSystem, ParametrisedSystem)`. `SpectralSystem`
alone is the closest analogue of the `Operator` / `Observable` / `Hamiltonian`
ladder in `eigora.qm.discrete`.

A system also owns the **vocabulary**: `extensive_variables` says what its
microstates carry beyond their energy, and an ensemble may free any subset of
that. The check happens once, in `statphys.state`, where the two are joined.

A `Level` carries an energy, a degeneracy, and optionally the other extensive
quantities its microstates hold -- a magnetisation, a particle number. Those
are what an ensemble trades against a field, and carrying them here is what
lets `<M>` and its susceptibility come out of the same sweep that gives `log Z`,
exactly rather than by differencing.

Energies are absolute. `Z = 1/(2 sinh(beta w / 2))` for a harmonic mode
includes the zero-point energy and `1/(1 - exp(-beta w))` does not; both are
legitimate, and a free energy is only defined up to the origin, so mixing the
two silently is the easiest way to get a wrong answer that looks right. Nothing
here shifts an energy without saying so.

Units: k_B = 1, so temperature is an energy and beta = 1/T. Entropy is
dimensionless. This matches the atomic units (hbar = m = 1) used everywhere
else in the package, and means nothing in `statphys` imports `constants`.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field as _field

# Summation never runs past this many levels while looking for convergence.
_MAX_TERMS = 100_000

# Relative contribution below which a level no longer moves the sums.
_TOL = 1e-12

# Consecutive negligible levels required before the tail is called done. One
# would do when the weights decrease monotonically, which they need not once a
# field is trading against something other than the energy.
_QUIET_RUN = 3

# Relative steps for numerical derivatives in beta. The second derivative needs
# a coarser step: its roundoff floor goes as eps/h^2 rather than eps/h, so the
# optimal step is eps^(1/4) rather than eps^(1/3).
_FIRST_STEP = 1e-5
_SECOND_STEP = 1e-4

#: The coefficient of each freed extensive variable in the exponent, as
#: `(variable, sign * field)` pairs -- what an `Ensemble` reduces to once a
#: system needs to sum against it. Empty is the canonical ensemble, which is
#: therefore not a special case but the absence of one.
Couplings = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Level:
    """
    A group of microstates sharing an energy.

    Attributes
    ----------
    energy : float
        Absolute energy of the level.
    degeneracy : int
        Number of microstates at this energy.
    extensive : mapping of str to float, optional
        Other extensive quantities these microstates carry, e.g.
        `{"magnetisation": -1.0}`. Only needed for variables some ensemble
        sets free; anything omitted is simply held fixed.
    """

    energy: float
    degeneracy: int = 1
    extensive: Mapping[str, float] = _field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.degeneracy < 1:
            raise ValueError(f"degeneracy must be at least 1, got {self.degeneracy}")
        if not math.isfinite(self.energy):
            raise ValueError(f"energy must be finite, got {self.energy}")
        for name, value in self.extensive.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"extensive value '{name}' must be finite, got {value}"
                )


@dataclass(frozen=True)
class Moments:
    """
    Everything one sweep over a spectrum yields.

    Attributes
    ----------
    log_z : float
        log of the ensemble's partition function.
    energy : float
        <E>.
    energy_variance : float
        Var(E), which gives the heat capacity as beta^2 Var(E).
    means : dict of str to float
        <X_i> for each extensive variable the ensemble set free.
    variances : dict of str to float
        Var(X_i), which gives the response d<X_i>/df_i as beta Var(X_i).
    """

    log_z: float
    energy: float
    energy_variance: float
    means: dict[str, float] = _field(default_factory=dict)
    variances: dict[str, float] = _field(default_factory=dict)


class System(ABC):
    """
    Abstract base class for anything with a partition function.

    Subclasses implement `log_z` and `is_exact`. Everything else -- the energy
    moments, and the moments of any freed variable -- is derived here by
    differentiating `log_z` numerically, which is the only route available when
    the levels cannot be enumerated. `SpectralSystem` overrides `moments` with
    exact sums.

    There is no canonical special case anywhere in the class: an empty
    `couplings` *is* the canonical ensemble, and every method takes the same
    argument. Having a separate canonical entry point next to a general one
    would mean two definitions of the same physics, and the two silently
    disagreeing is the failure mode -- both return floats.
    """

    @abstractmethod
    def log_z(self, beta: float, couplings: Couplings = ()) -> float:
        """
        Natural log of the partition function of the ensemble weight

            log w(x) = -beta E(x) + beta sum_i c_i X_i(x)

        so `couplings` carries the coefficient `c_i = sign_i * field_i` of each
        freed extensive variable. Empty couplings give the canonical `log Z`.

        Parameters
        ----------
        beta : float
            Inverse temperature, 1/T with k_B = 1. Must be positive.
        couplings : sequence of (str, float)
            One `(variable, coefficient)` pair per freed variable.

        Returns
        -------
        float
        """

    @property
    @abstractmethod
    def is_exact(self) -> bool:
        """True if the partition function is analytic rather than truncated."""

    @property
    def extensive_variables(self) -> frozenset[str]:
        """
        Which extensive quantities this system's microstates carry.

        Everything but the energy, which every microstate has by definition.
        A system can only be put in an ensemble that frees a subset of these,
        and a composite routes each coupling to the blocks that report it --
        so a spin attached to a two-level still has a magnetisation, with the
        two-level contributing zero to it rather than refusing.

        Empty by default, which is right for a system whose only extensive
        quantity is its energy. `SpectralSystem` reads it off the levels.
        """
        return frozenset()

    # -- derived ----------------------------------------------------------

    def moments(self, beta: float, couplings: Couplings = ()) -> Moments:
        """
        Every moment the weight implies, by differentiating `log_z`.

        **The derivatives are taken at fixed natural parameter** `lam_i =
        beta * c_i`, not at fixed field `c_i`, and the distinction is not
        cosmetic:

            d(log Z)/d(beta) at fixed c    = -<E - sum_i c_i X_i>
            d(log Z)/d(beta) at fixed lam  = -<E>

        Differentiating at fixed chemical potential would return `<E - mu N>`
        where every caller expects `<E>`. So `beta` and the couplings move
        together here, which is what `_rescaled` is for. In the same
        parameterisation `<X_i> = dlogZ/dlam_i` and `Var(X_i) =
        d2logZ/dlam_i^2`, so the freed variables need no enumerable spectrum
        either -- a system with nothing but a closed-form `log_z` is still a
        complete member of any ensemble.

        Accuracy is about 1e-10 relative on the first derivatives and 1e-8 on
        the second; `SpectralSystem` overrides with exact sums, and the tests
        hold the two against each other.

        Parameters
        ----------
        beta : float
            Inverse temperature.
        couplings : sequence of (str, float)
            One `(variable, coefficient)` pair per freed variable.

        Returns
        -------
        Moments
        """
        _check_beta(beta)
        couplings = tuple(couplings)
        self._check_couplings(couplings)
        centre = self.log_z(beta, couplings)

        step = _FIRST_STEP * beta
        wide = _SECOND_STEP * beta
        energy = -(
            self._log_z_at(beta, step, couplings)
            - self._log_z_at(beta, -step, couplings)
        ) / (2.0 * step)
        energy_variance = (
            self._log_z_at(beta, wide, couplings)
            - 2.0 * centre
            + self._log_z_at(beta, -wide, couplings)
        ) / wide**2

        means: dict[str, float] = {}
        variances: dict[str, float] = {}
        for index, (name, value) in enumerate(couplings):
            # A coupling may legitimately be zero (a field switched off), so
            # the step is absolute there rather than relative.
            scale = max(abs(value), 1.0)
            step = _FIRST_STEP * scale
            wide = _SECOND_STEP * scale
            means[name] = (
                self.log_z(beta, _bumped(couplings, index, step))
                - self.log_z(beta, _bumped(couplings, index, -step))
            ) / (2.0 * step * beta)
            variances[name] = max(
                (
                    self.log_z(beta, _bumped(couplings, index, wide))
                    - 2.0 * centre
                    + self.log_z(beta, _bumped(couplings, index, -wide))
                )
                / (wide * beta) ** 2,
                0.0,
            )

        return Moments(
            log_z=centre,
            energy=energy,
            energy_variance=energy_variance,
            means=means,
            variances=variances,
        )

    def mean_energy(self, beta: float, couplings: Couplings = ()) -> float:
        """<E>, the mean energy under this weight."""
        return self.moments(beta, couplings).energy

    def energy_variance(self, beta: float, couplings: Couplings = ()) -> float:
        """Var(E), which gives the heat capacity as beta^2 Var(E)."""
        return self.moments(beta, couplings).energy_variance

    def _log_z_at(
        self, beta: float, offset: float, couplings: Couplings
    ) -> float:
        """`log_z` at a shifted beta, holding every `beta * c_i` fixed."""
        moved = beta + offset
        return self.log_z(moved, _rescaled(couplings, beta / moved))

    def _check_couplings(self, couplings: Couplings) -> None:
        """
        Refuse couplings for variables this system's microstates never report.

        Guarded on `couplings` being non-empty so the canonical path costs
        nothing: `extensive_variables` walks to the first level, and a system
        summed with no couplings should not pay for a question nobody asked.
        """
        if not couplings:
            return
        missing = sorted({name for name, _ in couplings} - self.extensive_variables)
        if missing:
            carried = ", ".join(sorted(self.extensive_variables)) or "nothing"
            raise ValueError(
                f"{type(self).__name__} carries {carried}, so it cannot be "
                f"summed against {missing}"
            )

    # -- composition ------------------------------------------------------

    def __mul__(self, other: "System") -> "System":
        """Tensor product with another system, treated as distinguishable."""
        from eigora.statphys.systems.composite import CompositeSystem

        if not isinstance(other, System):
            return NotImplemented
        return CompositeSystem([self, other])

    def __pow__(self, n: int) -> "System":
        """
        `n` distinguishable copies of this system, so `A ** 4 == A * A * A * A`.

        This is *not* n identical particles: the copies are labelled and no
        symmetrisation is applied. For indistinguishable particles use
        `IdenticalParticles`, which asks for the statistics by name.
        """
        from eigora.statphys.systems.composite import CompositeSystem

        if not isinstance(n, int):
            return NotImplemented
        if n < 1:
            raise ValueError(f"number of copies must be at least 1, got {n}")
        return CompositeSystem([self] * n)


class SpectralSystem(System):
    """
    Abstract base class for a system whose levels can be enumerated.

    Subclasses implement `levels` and `n_states`. The partition function and
    every moment then come from one sweep over the spectrum, so they are exact
    up to the truncation tolerance rather than up to a finite-difference step.

    `levels` is a method, not a property: it must return a *fresh* iterator on
    every call, since a stored generator would be exhausted after the first
    sweep.
    """

    @abstractmethod
    def levels(self) -> Iterator[Level]:
        """
        The levels, ascending in energy. May be infinite.

        Returns
        -------
        Iterator of Level
        """

    @property
    @abstractmethod
    def n_states(self) -> int | None:
        """Total number of microstates, or None if the spectrum is unbounded."""

    @property
    def is_exact(self) -> bool:
        """True unless the levels themselves came from a numerical solution."""
        return True

    @property
    def extensive_variables(self) -> frozenset[str]:
        """
        Read off the first level, since every level must carry the same set.

        That requirement is not new: the sweep already refuses a spectrum in
        which some level omits a variable the ensemble freed, because there is
        no honest way to average a quantity that only some microstates report.
        Given that, the first level speaks for all of them.
        """
        return frozenset(next(iter(self.levels())).extensive)

    @property
    def ground_energy(self) -> float:
        """Energy of the lowest level."""
        return next(iter(self.levels())).energy

    @property
    def ground_degeneracy(self) -> int:
        """Number of microstates in the lowest level."""
        return next(iter(self.levels())).degeneracy

    # -- derived ----------------------------------------------------------

    def moments(
        self, beta: float, couplings: Couplings = (), tol: float = _TOL
    ) -> Moments:
        """
        Everything the weight implies, from a single sweep over the levels.

        Freeing a variable changes what is summed rather than how, so averages
        of the freed variables come out of the same pass, exactly, instead of
        by differentiating log Z with respect to its field. That makes this the
        reference the finite-difference route in `System` is checked against.
        """
        couplings = tuple(couplings)
        self._check_couplings(couplings)
        return self._sweep(beta, couplings, tol)

    def log_z(self, beta: float, couplings: Couplings = (), tol: float = _TOL) -> float:
        """
        log Z by summation over the levels.

        The largest weight is factored out before exponentiating, so the sum
        starts at 1 and cannot overflow however large `beta * E` becomes.
        """
        return self._sweep(beta, tuple(couplings), tol).log_z

    # -- helpers ----------------------------------------------------------

    def _sweep(self, beta: float, couplings: Couplings, tol: float) -> Moments:
        """
        One pass over the spectrum accumulating every moment that is wanted.

        Sums are kept relative to the largest log-weight seen so far and
        rescaled whenever a bigger one turns up. That is the streaming form of
        the log-sum-exp trick, and here it is needed rather than merely tidy:
        with a field trading against the energy, the biggest weight is not
        necessarily the first level.

        Energies and extensive values are measured from their value on the
        first level before being squared, which keeps `<X^2> - <X>^2` from
        losing its significant digits when the mean is large and the spread
        small.

        Raises
        ------
        ValueError
            If the sum has not converged after `_MAX_TERMS` levels. A truncated
            partition function is a wrong answer rather than an approximate
            one, and unlike a truncated list of levels the caller cannot see
            that it happened -- so this raises rather than returning quietly.
        """
        _check_beta(beta)
        names = tuple(name for name, _ in couplings)

        shift = -math.inf
        energy_origin = 0.0
        origin: dict[str, float] = {}

        total = 0.0
        energy_sum = energy_squares = 0.0
        sums = dict.fromkeys(names, 0.0)
        squares = dict.fromkeys(names, 0.0)

        quiet = 0
        seen = 0

        for index, level in enumerate(self.levels()):
            self._require_extensive(level, names, index)
            values = {name: float(level.extensive[name]) for name in names}

            if index == 0:
                energy_origin = level.energy
                origin = dict(values)
            elif level.energy < energy_origin - abs(energy_origin) * 1e-12:
                raise ValueError(
                    f"{type(self).__name__}.levels() must ascend in energy, but "
                    f"level {index} at {level.energy} sits below the first level "
                    f"at {energy_origin}"
                )

            traded = sum(value * values[name] for name, value in couplings)
            log_weight = -beta * (level.energy - traded)

            if log_weight > shift:
                # A bigger weight turned up: rescale what has accumulated.
                factor = math.exp(shift - log_weight) if shift > -math.inf else 0.0
                total *= factor
                energy_sum *= factor
                energy_squares *= factor
                for name in names:
                    sums[name] *= factor
                    squares[name] *= factor
                shift = log_weight

            term = level.degeneracy * math.exp(log_weight - shift)
            energy_offset = level.energy - energy_origin
            offsets = {name: values[name] - origin[name] for name in names}

            total += term
            energy_sum += term * energy_offset
            energy_squares += term * energy_offset**2
            for name in names:
                sums[name] += term * offsets[name]
                squares[name] += term * offsets[name] ** 2
            seen += 1

            # The tail must be negligible in every sum, not only in Z: the
            # squared terms grow without bound, so the variances settle last.
            contribution = term * (1.0 + _spread(energy_offset, offsets))
            accumulated = (
                total
                + abs(energy_sum)
                + energy_squares
                + sum(abs(sums[name]) + squares[name] for name in names)
            )
            if contribution <= tol * accumulated:
                quiet += 1
                if quiet >= _QUIET_RUN:
                    break
            else:
                quiet = 0

            if index + 1 >= _MAX_TERMS:
                raise ValueError(
                    f"log_z did not converge at beta={beta}: after {_MAX_TERMS} "
                    f"levels the last one still contributes "
                    f"{contribution / accumulated:.2e} > tol={tol}; the spectrum "
                    f"is too dense at this temperature -- use a closed form or "
                    f"the classical limit"
                )

        if seen == 0 or total == 0.0:
            raise ValueError(f"{type(self).__name__} has no levels to sum over")

        return Moments(
            log_z=math.log(total) + shift,
            energy=energy_origin + energy_sum / total,
            energy_variance=_variance(energy_sum, energy_squares, total),
            means={name: origin[name] + sums[name] / total for name in names},
            variances={
                name: _variance(sums[name], squares[name], total) for name in names
            },
        )

    def _require_extensive(
        self, level: Level, names: tuple[str, ...], index: int
    ) -> None:
        """
        Fail loudly rather than silently averaging a missing variable as zero.

        This is what makes `extensive_variables` trustworthy. That property is
        read off the first level and everything upstream believes it, so the
        one thing it cannot catch is a spectrum that disagrees with itself --
        level 0 reporting a magnetisation and level 7 forgetting it. There is
        no honest average over a quantity only some microstates carry, so the
        sweep checks every level rather than assuming the first spoke for all.
        """
        missing = [name for name in names if name not in level.extensive]
        if missing:
            them = "it" if len(missing) == 1 else "them"
            raise ValueError(
                f"level {index} of {type(self).__name__} does not carry "
                f"{missing}, but level 0 does; every level must report the "
                f"same extensive variables, or {them} cannot be averaged"
            )


class ParametrisedSystem(System):
    """
    Abstract base class for a system with extensive parameters that can vary.

    Pressure and chemical potential are the same operation -- differentiate
    log Z with respect to a parameter of the system rather than a field of the
    ensemble -- so one hook serves both. Concretes are frozen dataclasses, so
    `at` is a one-line `dataclasses.replace`.
    """

    #: Parameters differentiated by a step of exactly 1, where the finite
    #: difference is the definition rather than an approximation to one.
    DISCRETE_PARAMETERS = frozenset({"particles"})

    @property
    @abstractmethod
    def parameters(self) -> dict[str, float]:
        """The extensive parameters this system can be re-made with."""

    @abstractmethod
    def at(self, **changes: float) -> "ParametrisedSystem":
        """A copy of this system with some parameters changed."""


def _rescaled(couplings: Couplings, factor: float) -> Couplings:
    """
    The couplings at a new beta that keep every natural parameter `beta * c`
    fixed, so differentiating in beta gives `<E>` rather than `<E - sum c X>`.
    """
    if not couplings:
        return ()
    return tuple((name, value * factor) for name, value in couplings)


def _bumped(couplings: Couplings, index: int, step: float) -> Couplings:
    """The couplings with one of them moved by `step`, the rest untouched."""
    return tuple(
        (name, value + step) if position == index else (name, value)
        for position, (name, value) in enumerate(couplings)
    )


def _spread(energy_offset: float, offsets: dict[str, float]) -> float:
    """How far a level of unit weight can move the first and second moments."""
    scale = abs(energy_offset) + energy_offset**2
    for offset in offsets.values():
        scale += abs(offset) + offset**2
    return scale


def _variance(first: float, second: float, total: float) -> float:
    """<x^2> - <x>^2 on offset values, clamped against roundoff going negative."""
    return max(second / total - (first / total) ** 2, 0.0)


def _check_beta(beta: float) -> None:
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if not math.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")


__all__ = [
    "Couplings",
    "Level",
    "Moments",
    "System",
    "SpectralSystem",
    "ParametrisedSystem",
]
