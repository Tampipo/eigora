# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
`ThermalState`: a system in an ensemble, and everything that follows.

A system says what the microstates are; an ensemble says how they are weighted;
this is the two of them together, which is the point at which thermodynamics
exists. Keeping them apart until here is deliberate -- the same system can be
studied in several ensembles, and comparing them is most of what equilibrium
statistical mechanics is for.

This module sits above both halves rather than inside either, because it
belongs to neither and because an ensemble has more than one consumer: summed
over a spectrum it gives the thermodynamics here, and compared across a
proposed move it will give the Metropolis acceptance rule in
`statphys.monte_carlo`.

Every quantity is a moment of the ensemble weight or a derivative of log Z, so
three generic methods cover them all:

    mean(variable)       <X_i>
    variance(variable)   Var(X_i)
    response(variable)   d<X_i>/df_i = beta Var(X_i)

The named properties -- energy, entropy, heat capacity, magnetisation,
susceptibility -- are thin aliases over those, and they raise rather than
guess when the ensemble holds the variable they need fixed.

`response` is the fluctuation-dissipation theorem, and it earns its place
twice: it is the cheap way to get a susceptibility, and it proves `<X>` is
monotone in its field, which is what makes `field_for` a safe inversion rather
than a hopeful one.
"""

import math
from dataclasses import dataclass
from functools import cached_property

from scipy.optimize import brentq

from eigora.statphys.ensembles import Ensemble
from eigora.statphys.systems.base import Level, Moments, System
from eigora.statphys.systems.composite import convolve_levels

# How far `field_for` will push a bracket outward before giving up.
_MAX_BRACKET_STEPS = 60

# Relative step for differentiating log Z with respect to a continuous system
# parameter. Coarser than the beta steps because each evaluation rebuilds the
# system, so the usual eps^(1/3) tuning is not the binding constraint.
_PARAMETER_STEP = 1e-6


@dataclass(frozen=True)
class ThermalState:
    """
    A system held in an ensemble, with its equilibrium properties.

    Example
    -------
    >>> from eigora.statphys import Canonical, HarmonicMode, equilibrium
    >>> state = equilibrium(HarmonicMode(omega=1.0), Canonical(temperature=0.5))
    >>> state.energy            # (omega/2) coth(beta omega / 2)
    0.5820...
    >>> state.heat_capacity
    0.1710...

    Parameters
    ----------
    system : System
        Supplies the microstates and their energies.
    ensemble : Ensemble
        Supplies the temperature and any conjugate fields.
    """

    system: System
    ensemble: Ensemble

    def __post_init__(self) -> None:
        """
        Refuse an ensemble that frees something the system cannot report.

        The system owns the vocabulary: what its microstates carry is what can
        be asked of it, and an ensemble may free any subset of that -- freeing
        nothing, the canonical case, is always allowed. Checking here rather
        than inside the sweep means the error names the two objects the caller
        actually wrote, instead of surfacing as a complaint about level 0 of
        some block they never mentioned.
        """
        missing = sorted(set(self.ensemble.free) - self.system.extensive_variables)
        if missing:
            carried = ", ".join(sorted(self.system.extensive_variables)) or "nothing"
            raise ValueError(
                f"{type(self.ensemble).__name__} frees {missing}, but "
                f"{type(self.system).__name__} carries {carried}; a system can "
                f"only be put in an ensemble that frees variables its "
                f"microstates report"
            )

    # -- the single sweep everything else reads ---------------------------

    @cached_property
    def moments(self) -> Moments:
        """
        Every moment the ensemble weight implies, computed once.

        This line is the only place the two halves of the package touch: the
        ensemble is reduced to `(beta, couplings)` -- plain numbers -- so no
        system ever has to know what an `Ensemble` is.
        """
        return self.system.moments(self.ensemble.beta, self.ensemble.couplings)

    @property
    def temperature(self) -> float:
        """Temperature, an energy since k_B = 1."""
        return self.ensemble.temperature

    @property
    def beta(self) -> float:
        """Inverse temperature."""
        return self.ensemble.beta

    @property
    def log_z(self) -> float:
        """log of this ensemble's partition function."""
        return self.moments.log_z

    # -- potentials -------------------------------------------------------

    @property
    def potential(self) -> float:
        """
        `-T log Z`: whichever thermodynamic potential this ensemble generates.

        Helmholtz free energy in the canonical ensemble, grand potential once
        particle number is free, Gibbs energy once volume is. Use
        `potential_name` to find out which, or the matching alias to assert it.
        """
        return -self.temperature * self.log_z

    @property
    def potential_name(self) -> str:
        """The name of what `potential` returns, e.g. 'free_energy'."""
        return self.ensemble.potential_name

    @property
    def free_energy(self) -> float:
        """Helmholtz free energy F = U - TS. Canonical ensemble only."""
        return self._named_potential("free_energy")

    @property
    def grand_potential(self) -> float:
        """Grand potential Omega = F - mu N. Grand canonical ensemble only."""
        return self._named_potential("grand_potential")

    @property
    def gibbs_energy(self) -> float:
        """Gibbs energy G = F + PV. Isothermal-isobaric ensemble only."""
        return self._named_potential("gibbs_energy")

    # -- energy -----------------------------------------------------------

    @property
    def energy(self) -> float:
        """<E>, the mean energy."""
        return self.moments.energy

    @property
    def energy_variance(self) -> float:
        """Var(E)."""
        return self.moments.energy_variance

    @property
    def heat_capacity(self) -> float:
        """
        C = beta^2 Var(E).

        Taken from the energy fluctuations rather than by differentiating
        `<E>`, which is both exact and the more revealing statement: the
        response of the energy to temperature *is* its spread.
        """
        return self.beta**2 * self.energy_variance

    @property
    def entropy(self) -> float:
        """
        S = beta (<E> - sum_i f_i <X_i>) + log Z.

        One formula for every ensemble in the family. It reduces to the
        familiar `beta U + log Z` when nothing but the energy is free, and to
        `beta (U - mu N) + log Xi` once particle number is.
        """
        traded = sum(
            field.sign * field.value * self.mean(field.variable)
            for field in self.ensemble.fields
        )
        return self.beta * (self.energy - traded) + self.log_z

    # -- freed variables --------------------------------------------------

    def mean(self, variable: str) -> float:
        """
        <X>, the ensemble average of a freed extensive variable.

        Parameters
        ----------
        variable : str
            Name of the variable, e.g. "magnetisation".

        Returns
        -------
        float
        """
        self._require_free(variable, "mean")
        return self.moments.means[variable]

    def variance(self, variable: str) -> float:
        """Var(X) for a freed extensive variable."""
        self._require_free(variable, "variance")
        return self.moments.variances[variable]

    def response(self, variable: str) -> float:
        """
        d<X>/df = sign * beta * Var(X), the response of X to its own field.

        Fluctuation-dissipation. The sign of the field in the exponent survives
        differentiation and must be carried: with `log w = -beta(E - s f X)`,
        `dlogZ/df = beta s <X>` and `d2logZ/df2 = beta^2 Var(X)`, so
        `d<X>/df = s beta Var(X)`.

        It is therefore *not* always positive. `<M>` grows with the field that
        drives it (s = +1), but `<V>` falls as the pressure rises (s = -1) --
        which is the same statement as the isothermal compressibility
        `-(1/V) dV/dP` being positive. What is always positive is `s d<X>/df`,
        and that is what makes `field_for` a safe inversion.
        """
        return self.ensemble.field(variable).sign * self.beta * self.variance(variable)

    @property
    def magnetisation(self) -> float:
        """<M>. Needs an ensemble that frees the magnetisation."""
        return self.mean("magnetisation")

    @property
    def susceptibility(self) -> float:
        """chi = d<M>/dh = beta Var(M)."""
        return self.response("magnetisation")

    @property
    def particles(self) -> float:
        """<N>. Needs an ensemble that frees the particle number."""
        return self.mean("particles")

    @property
    def particle_fluctuation(self) -> float:
        """
        d<N>/dmu = beta Var(N).

        Proportional to the isothermal compressibility, and non-negative for
        the same reason every response is: it is a variance. It diverges where
        the compressibility does, which is how a phase transition announces
        itself in this ensemble.
        """
        return self.response("particles")

    def chemical_potential_for(self, count: float) -> float:
        """
        The chemical potential that produces `<N> = count`.

        The inverse of `particles`, and the counterpart of the fixed-N route:
        one sets mu and reads N, the other sets N and reads mu. They agree only
        as N grows, so comparing them measures the finite-size gap rather than
        checking an implementation.
        """
        return self.field_for("particles", count)

    # -- conjugates of fixed parameters -----------------------------------

    @property
    def pressure(self) -> float:
        """
        P = (1/beta) dlogZ/dV, at fixed temperature and particle number.

        Two honest routes, and which one applies is decided by the ensemble
        rather than by a flag. If the volume is *free* the pressure is an
        input, and this returns the field that was set. If the volume is a
        fixed system parameter the pressure is an output, obtained by
        rebuilding the system at `V +- h` and differencing -- `at` changes
        which microstates exist, so this genuinely costs two more partition
        functions rather than re-reading one sweep.
        """
        if self.ensemble.has_field("volume"):
            return self.ensemble.field("volume").value
        return self._parameter_slope("volume", "pressure") / self.beta

    @property
    def chemical_potential(self) -> float:
        """
        mu = F(N) - F(N-1), the free-energy cost of one more particle.

        Exact rather than approximate: particle number is discrete, so a step
        of one *is* the derivative and there is no step size to choose. That
        is what `ParametrisedSystem.DISCRETE_PARAMETERS` marks.

        As with `pressure`, a grand canonical ensemble makes mu an input and
        this returns it -- the two routes are conjugate, and holding them
        against each other is the sharpest check available on either.
        """
        if self.ensemble.has_field("particles"):
            return self.ensemble.field("particles").value
        count = self._parameter("particles", "chemical_potential")
        if count < 1.0:
            raise ValueError(
                f"chemical_potential needs at least one particle to remove, "
                f"got {count:g}"
            )
        fewer = self._rebuilt(particles=count - 1.0)
        return self.potential - fewer.potential

    @property
    def enthalpy(self) -> float:
        """
        H = <E> + P<V>.

        Derived, never generating. `H` is the potential at fixed (S, P) and no
        simple ensemble samples at fixed entropy -- the (T, P, N) ensemble
        generates the Gibbs energy `G = H - TS`. So this is computed from a
        state rather than read off one, and it is not among the aliases of
        `potential`.
        """
        if self.ensemble.has_field("volume"):
            return self.energy + self.pressure * self.mean("volume")
        volume = self._parameter("volume", "enthalpy")
        return self.energy + self.pressure * volume

    def field_for(self, variable: str, target: float) -> float:
        """
        The field value that makes `<X>` equal `target`.

        Inverts `mean`, which is safe because `s d<X>/df = beta Var(X) >= 0`
        makes the average monotone in its field. The bracket is grown outward
        from the current value until it straddles the target, then handed to
        Brent's method.

        Monotone, but not always increasing: for a field entering with `s = -1`
        the average *falls* as the field rises. Multiplying the residual by `s`
        makes it increasing in every case, so one bracketing rule covers both.

        Parameters
        ----------
        variable : str
            The freed variable whose average is being matched.
        target : float
            The value it should take.

        Returns
        -------
        float
            The field conjugate to `variable`.

        Raises
        ------
        ValueError
            If the target lies outside the range the variable can reach -- for
            a bounded quantity such as a spin magnetisation, that is any target
            beyond saturation.
        """
        self._require_free(variable, "field_for")
        if not math.isfinite(target):
            raise ValueError(f"target must be finite, got {target}")

        conjugate = self.ensemble.field(variable)

        def residual(value: float) -> float:
            state = self._with_field(variable, value)
            return conjugate.sign * (state.mean(variable) - target)

        centre = conjugate.value
        low = high = centre
        width = 1.0
        for _ in range(_MAX_BRACKET_STEPS):
            if residual(low) <= 0.0 <= residual(high):
                break
            low, high = centre - width, centre + width
            width *= 2.0
        else:
            raise ValueError(
                f"no field reproduces {variable} = {target}: the bracket grew to "
                f"[{low:.3g}, {high:.3g}] without straddling it, so the target is "
                f"probably outside the range {variable} can reach"
            )
        return float(brentq(residual, low, high))

    # -- helpers ----------------------------------------------------------

    def _parameter(self, name: str, where: str) -> float:
        """The value of a fixed system parameter, or a message saying why not."""
        parameters = getattr(self.system, "parameters", None)
        if parameters is None or name not in parameters:
            held = ", ".join(sorted(parameters or {})) or "none"
            raise ValueError(
                f"{where} needs '{name}' among the parameters of "
                f"{type(self.system).__name__}, which has {held}"
            )
        return parameters[name]

    def _rebuilt(self, **changes: float) -> "ThermalState":
        """The same ensemble applied to the system rebuilt at other parameters."""
        return type(self)(self.system.at(**changes), self.ensemble)

    def _parameter_slope(self, name: str, where: str) -> float:
        """
        dlogZ/d(parameter), by central difference on a rebuilt system.

        The step is relative to the parameter, since a volume has no natural
        scale of its own.
        """
        value = self._parameter(name, where)
        step = _PARAMETER_STEP * abs(value)
        if step == 0.0:
            raise ValueError(f"{where} cannot differentiate at {name} = 0")
        couplings = self.ensemble.couplings
        beta = self.beta
        up = self.system.at(**{name: value + step}).log_z(beta, couplings)
        down = self.system.at(**{name: value - step}).log_z(beta, couplings)
        return (up - down) / (2.0 * step)

    def _with_field(self, variable: str, value: float) -> "ThermalState":
        """The same system in the same ensemble with one field re-valued."""
        return type(self)(self.system, self.ensemble.with_field(variable, value))

    def _named_potential(self, name: str) -> float:
        if self.potential_name != name:
            raise ValueError(
                f"{type(self.ensemble).__name__} generates the "
                f"{self.potential_name.replace('_', ' ')}, not the "
                f"{name.replace('_', ' ')}; use `.potential` if you want the "
                f"number whatever it is called"
            )
        return self.potential

    def _require_free(self, variable: str, where: str) -> None:
        if not self.ensemble.has_field(variable):
            free = ", ".join(self.ensemble.free) or "nothing"
            raise ValueError(
                f"{where}('{variable}') needs an ensemble that frees "
                f"'{variable}', but {type(self.ensemble).__name__} frees {free}"
            )


def equilibrium(system: System, ensemble: Ensemble) -> ThermalState:
    """
    Put a system in an ensemble.

    The single entry point from the two halves of the package to the
    thermodynamics, named to echo `spectrum_for` in `eigora.qm.spectra`.

    Parameters
    ----------
    system : System
        Any system, from the catalogue or built by composition.
    ensemble : Ensemble
        Canonical, magnetic, or any combination of conjugate fields.

    Returns
    -------
    ThermalState

    Example
    -------
    >>> from eigora.statphys import Canonical, TwoLevel, equilibrium
    >>> sweep = [
    ...     equilibrium(TwoLevel(splitting=1.0), Canonical(temperature=t))
    ...     for t in (0.1, 0.4, 1.0)
    ... ]
    >>> [round(state.heat_capacity, 4) for state in sweep]
    [0.0045, 0.4392, 0.2100]
    """
    if not isinstance(system, System):
        raise TypeError(f"expected a System, got {type(system).__name__}")
    if not isinstance(ensemble, Ensemble):
        raise TypeError(f"expected an Ensemble, got {type(ensemble).__name__}")
    return ThermalState(system, ensemble)


__all__ = [
    "ThermalState",
    "MicrocanonicalState",
    "equilibrium",
    "microcanonical",
]


@dataclass(frozen=True)
class MicrocanonicalState:
    """
    A system at fixed energy: the ensemble that counts rather than weights.

    Not an `Ensemble`, and deliberately so. Every ensemble in that family is a
    log-weight `-beta[E - sum f X]` with a temperature going *in*; this one has
    no temperature at all. It fixes the energy, gives every microstate at that
    energy the same weight, and the temperature comes **out**:

        S = log Omega,      1/T = dS/dE

    So there is no `log Z`, no `-T log Z`, and no field to trade against. Its
    potential *is* the entropy, carrying no `-T` factor -- which is why forcing
    it into `ThermalState` would mean a temperature that is an input on one
    branch and an output on the other.

    Counting is exact. `Omega` is a Python integer from the convolution, so
    `TwoLevel(w) ** 200` at half filling gives `C(200, 100)` itself -- a
    59-digit number whose logarithm is exact and whose float is not.

    Example
    -------
    >>> from eigora.statphys import TwoLevel, microcanonical
    >>> state = microcanonical(TwoLevel(1.0) ** 200, energy=100.0)
    >>> state.omega                         # C(200, 100), exactly
    90548514656103281165404177077484163874504589675413336841320
    >>> round(state.entropy, 6)
    135.753236
    >>> round(state.temperature, 6)         # comes out, was not put in
    ...

    Parameters
    ----------
    system : System
        Must have a countable spectrum: a finite `SpectralSystem`, or a
        composite of them, which is convolved.
    energy : float
        The shell energy. With `width` unset it must match a level.
    width : float, optional
        Shell thickness. Unset means an exact level match, which is the honest
        discrete microcanonical ensemble; a width sums every level inside
        `energy +- width/2`, which is what a continuum spectrum needs.
    tol : float
        Tolerance for matching a level energy.
    """

    system: System
    energy: float
    width: float | None = None
    tol: float = 1e-9

    def __post_init__(self) -> None:
        if self.width is not None and self.width <= 0.0:
            raise ValueError(f"width must be positive, got {self.width}")
        if not math.isfinite(self.energy):
            raise ValueError(f"energy must be finite, got {self.energy}")

    @cached_property
    def levels(self) -> tuple[Level, ...]:
        """The system's spectrum, convolved if it is a composite."""
        return convolve_levels(self.system)

    @property
    def omega(self) -> int:
        """
        The exact number of microstates in the shell.

        An integer, not a float: the whole reason the convolution keeps
        degeneracies exact is that this number routinely exceeds the double
        range while its logarithm is perfectly ordinary.
        """
        return sum(level.degeneracy for level in self._shell())

    @property
    def entropy(self) -> float:
        """
        `S = log Omega` -- Boltzmann's, and the definition rather than a
        derivative of something else.
        """
        count = self.omega
        if count == 0:
            raise ValueError(
                f"no microstates at energy {self.energy}: the shell is empty, "
                f"so the entropy is undefined rather than zero. Give a width, "
                f"or pick an energy the spectrum actually has"
            )
        return math.log(count)

    @property
    def temperature(self) -> float:
        """
        `1/T = dS/dE`, by a centred difference over neighbouring shells.

        The direction that makes this ensemble worth having: temperature is
        derived from the density of states, not supplied. It agrees with the
        canonical temperature that produces the same mean energy, to O(1/N) --
        which is the statement that the ensembles are equivalent.
        """
        inverse = self.beta
        if inverse == 0.0:
            raise ValueError(
                f"the entropy is stationary at energy {self.energy}, so 1/T = 0 "
                f"and the temperature is infinite; this is the top of the "
                f"entropy curve, where a bounded spectrum turns over into "
                f"negative temperature"
            )
        return 1.0 / inverse

    @property
    def beta(self) -> float:
        """Inverse temperature, `dS/dE`."""
        index = self._level_index()
        levels = self.levels
        if index == 0 or index == len(levels) - 1:
            edge = "ground" if index == 0 else "highest"
            raise ValueError(
                f"the {edge} level has no neighbour below and above, so dS/dE "
                f"cannot be centred there; the temperature diverges at the "
                f"spectrum's edges"
            )
        below, above = levels[index - 1], levels[index + 1]
        return (
            math.log(above.degeneracy) - math.log(below.degeneracy)
        ) / (above.energy - below.energy)

    # -- helpers ----------------------------------------------------------

    def _shell(self) -> tuple[Level, ...]:
        """Every level inside the shell."""
        if self.width is None:
            return tuple(
                level
                for level in self.levels
                if abs(level.energy - self.energy) <= self.tol
            )
        half = 0.5 * self.width
        return tuple(
            level
            for level in self.levels
            if abs(level.energy - self.energy) <= half + self.tol
        )

    def _level_index(self) -> int:
        """Position of the matching level, for the entropy derivative."""
        for index, level in enumerate(self.levels):
            if abs(level.energy - self.energy) <= self.tol:
                return index
        raise ValueError(
            f"no level at energy {self.energy}; dS/dE is a difference over "
            f"neighbouring levels, so it needs one to sit on"
        )


def microcanonical(
    system: System,
    energy: float,
    width: float | None = None,
) -> MicrocanonicalState:
    """
    Put a system at fixed energy.

    The counterpart of `equilibrium`, and asymmetric with it on purpose: the
    second argument is a number rather than an `Ensemble`, because fixing the
    energy is not a choice of weights.

    Parameters
    ----------
    system : System
        Must have a countable spectrum.
    energy : float
        The shell energy.
    width : float, optional
        Shell thickness; unset means an exact level match.

    Returns
    -------
    MicrocanonicalState
    """
    if not isinstance(system, System):
        raise TypeError(f"expected a System, got {type(system).__name__}")
    return MicrocanonicalState(system, energy, width)
