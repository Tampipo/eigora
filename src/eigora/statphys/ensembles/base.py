# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The `Ensemble` interface: a log-weight on a microstate.

An ensemble says which extensive variables are held fixed and which are set
free in exchange for an intensive field conjugate to them. That is the whole
content of it, and it fits in one line:

    log w(x) = -beta [ E(x) - sum_i f_i X_i(x) ]

`E(x)` is the energy of the microstate and each `X_i(x)` is some other
extensive quantity the microstate carries -- its particle number, its
magnetisation, its volume -- paired with the field `f_i` that fixes its mean.
A variable that appears in no `Field` is simply held fixed, so "fixed volume"
needs no special class: it is any ensemble without a volume field.

This one function has two consumers, which is why it sits at the centre of the
package rather than off to the side:

  * sum `exp(log w)` over an enumerable spectrum and you have the partition
    function, and with it every equilibrium quantity;
  * compare `log w` before and after a proposed move and you have the
    Metropolis acceptance rule.

So the exact machinery and the Monte Carlo machinery read the same object, and
canonical, grand canonical and magnetic sampling differ only in which `Field`s
the ensemble carries -- not in the sampler.

Units: k_B = 1, so temperature is an energy and beta = 1/T.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

#: Free variable -> the name of the potential `-T log Z` generates.
#: Each field added to an ensemble is one more Legendre transform, and the
#: result stops being the Helmholtz free energy. Reporting a grand potential as
#: a free energy is an error no test would catch, since both are floats.
_POTENTIAL_NAMES: dict[frozenset[str], str] = {
    frozenset(): "free_energy",
    frozenset({"particles"}): "grand_potential",
    frozenset({"volume"}): "gibbs_energy",
    frozenset({"magnetisation"}): "magnetic_free_energy",
}


@dataclass(frozen=True)
class Field:
    """
    An extensive variable set free, and the intensive field conjugate to it.

    Attributes
    ----------
    variable : str
        Name of the extensive quantity, as a microstate reports it:
        "particles", "volume", "magnetisation".
    value : float
        The conjugate field: mu, P, h.
    sign : int
        Its sign in the exponent, +1 for `+beta mu N`, -1 for `-beta P V`.
    symbol : str
        Short label for the field, used in messages and plots.
    """

    variable: str
    value: float
    sign: int
    symbol: str = ""

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign}")
        if not self.variable:
            raise ValueError("a field needs the name of its extensive variable")


@dataclass(frozen=True)
class Ensemble(ABC):
    """
    Abstract base class for an ensemble in the exponential family.

    Subclasses declare their conjugate `fields`; everything else is derived
    here. Being frozen dataclasses, they are re-parameterised with `at`, so
    scanning a temperature is a comprehension rather than a rebuild.

    Parameters
    ----------
    temperature : float
        Must be positive. With k_B = 1 this is an energy.
    """

    temperature: float

    #: Freed variable -> the constructor parameter holding its field. Lets a
    #: field be re-set by the name of the physics rather than of the argument,
    #: which is what `ThermalState.field_for` needs to invert an average.
    FIELD_ATTRIBUTES = {}

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not math.isfinite(self.temperature):
            raise ValueError(f"temperature must be finite, got {self.temperature}")

    @property
    @abstractmethod
    def fields(self) -> tuple[Field, ...]:
        """The conjugate pairs this ensemble sets free. Empty for canonical."""

    # -- derived ----------------------------------------------------------

    @property
    def beta(self) -> float:
        """Inverse temperature, 1/T with k_B = 1."""
        return 1.0 / self.temperature

    @property
    def free(self) -> tuple[str, ...]:
        """Extensive variables allowed to fluctuate, energy aside."""
        return tuple(field.variable for field in self.fields)

    @property
    def couplings(self) -> tuple[tuple[str, float], ...]:
        """
        `(variable, sign * field)` pairs: this ensemble as a system sees it.

        A system sums against numbers, not against an `Ensemble` -- which is
        what keeps `statphys.systems` from importing `statphys.ensembles` at
        all. This property is the whole of the translation between them, and
        the canonical ensemble is simply the empty tuple.
        """
        return tuple(
            (field.variable, field.sign * field.value) for field in self.fields
        )

    @property
    def potential_name(self) -> str:
        """
        Name of the potential this ensemble's `-T log Z` actually is.

        Helmholtz free energy for the canonical ensemble, grand potential once
        particle number is free, Gibbs energy once volume is.
        """
        return _POTENTIAL_NAMES.get(frozenset(self.free), "potential")

    def field(self, variable: str) -> Field:
        """
        The `Field` conjugate to `variable`.

        Raises
        ------
        ValueError
            If this ensemble holds that variable fixed.
        """
        for field in self.fields:
            if field.variable == variable:
                return field
        known = ", ".join(self.free) if self.free else "none"
        raise ValueError(
            f"{type(self).__name__} holds '{variable}' fixed, so it has no "
            f"conjugate field for it; free variables here: {known}"
        )

    def has_field(self, variable: str) -> bool:
        """True if `variable` is free in this ensemble."""
        return any(field.variable == variable for field in self.fields)

    def at(self, **changes: float) -> "Ensemble":
        """
        A copy with some control values changed, e.g. `.at(temperature=2.0)`.

        Example
        -------
        >>> sweep = [ensemble.at(temperature=t) for t in (0.1, 0.5, 1.0)]
        """
        return replace(self, **changes)

    def with_field(self, variable: str, value: float) -> "Ensemble":
        """
        A copy with the field conjugate to `variable` set to `value`.

        Named for the extensive variable rather than the parameter, so a caller
        can say `with_field("magnetisation", 0.3)` without knowing that the
        constructor calls it `magnetic_field`.
        """
        self.field(variable)  # raises with a good message if it is held fixed
        try:
            attribute = self.FIELD_ATTRIBUTES[variable]
        except KeyError:
            raise ValueError(
                f"{type(self).__name__} does not say which parameter carries the "
                f"field for '{variable}'; set FIELD_ATTRIBUTES on it"
            ) from None
        return replace(self, **{attribute: value})

    def log_weight(self, energy: float, **extensive: float) -> float:
        """
        log w(x) = -beta [E - sum_i f_i X_i], the unnormalised weight.

        Parameters
        ----------
        energy : float
            E(x), the energy of the microstate.
        **extensive : float
            One value per free variable, e.g. `particles=3, magnetisation=-1.0`.

        Returns
        -------
        float
        """
        return -self.beta * (energy - self._traded(extensive, "log_weight"))

    def log_weight_change(self, delta_energy: float, **deltas: float) -> float:
        """
        The change in log w under a move, from its deltas alone.

        This is the Metropolis acceptance exponent, and it never touches the
        whole configuration -- which is what keeps a sampling step O(1) in the
        system size. It is exactly the difference of two `log_weight` calls,
        an identity the tests assert directly.

        Parameters
        ----------
        delta_energy : float
            E(x') - E(x).
        **deltas : float
            One change per free variable, e.g. `magnetisation=-2.0`.

        Returns
        -------
        float
        """
        return -self.beta * (delta_energy - self._traded(deltas, "log_weight_change"))

    # -- helpers ----------------------------------------------------------

    def _traded(self, values: dict[str, float], where: str) -> float:
        """sum_i sign_i * f_i * X_i, checking that every free variable is given."""
        total = 0.0
        for field in self.fields:
            if field.variable not in values:
                raise ValueError(
                    f"{where} needs a value for '{field.variable}', which is free "
                    f"in {type(self).__name__}; got {sorted(values) or 'nothing'}"
                )
            total += field.sign * field.value * values[field.variable]

        unknown = set(values) - set(self.free)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} holds {sorted(unknown)} fixed, so passing "
                f"a value for them to {where} has no meaning; free variables here: "
                f"{list(self.free) or 'none'}"
            )
        return total


__all__ = ["Field", "Ensemble"]
