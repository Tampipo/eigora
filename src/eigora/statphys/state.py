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
from eigora.statphys.systems.base import Moments, System

# How far `field_for` will push a bracket outward before giving up.
_MAX_BRACKET_STEPS = 60


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


__all__ = ["ThermalState", "equilibrium"]
