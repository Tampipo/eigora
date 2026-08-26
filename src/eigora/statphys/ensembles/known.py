# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The named ensembles.

Each is a thin declaration of which conjugate fields it carries; all the
machinery lives in `Ensemble`. `Generalised` is the escape hatch for a
combination with no standard name -- fixed volume with a magnetic field, two
chemical potentials for a mixture, and so on.

The potential that `-T log Z` gives is different in each of them, and
`potential_name` says which:

    Canonical           Z = sum exp(-beta E)              Helmholtz F
    GrandCanonical      Xi = sum exp(-beta(E - mu N))     grand potential, = -PV
    IsothermalIsobaric  Delta = sum exp(-beta(E + P V))   Gibbs G
    Magnetic            Z = sum exp(-beta(E - h M))       magnetic free energy

Adding a field is one Legendre transform. Note that enthalpy is *not* in this
list: H = U + PV is the potential at fixed (S, P), and no simple ensemble
samples at fixed entropy -- the (T, P, N) ensemble generates the Gibbs energy.
Enthalpy is a derived quantity, computed from a state rather than generating
one.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from eigora.statphys.ensembles.base import Ensemble, Field


@dataclass(frozen=True)
class Canonical(Ensemble):
    """
    Fixed temperature, everything else held: the (T, V, N) ensemble.

    Only the energy fluctuates, so there are no conjugate fields at all and
    the weight is the bare Boltzmann factor `exp(-beta E)`.

    Example
    -------
    >>> from eigora.statphys import HarmonicMode, equilibrium
    >>> state = equilibrium(HarmonicMode(omega=1.0), Canonical(temperature=0.5))
    >>> state.energy
    0.5820...

    Parameters
    ----------
    temperature : float
        Must be positive.
    """

    @property
    def fields(self) -> tuple[Field, ...]:
        return ()


@dataclass(frozen=True)
class Magnetic(Ensemble):
    """
    Fixed temperature and magnetic field: magnetisation fluctuates.

    The weight is `exp(-beta(E - h M))`. Nothing stops you folding `-h M` into
    the energy instead and calling it canonical -- the two give identical
    physics. Keeping the field separate means a microstate's energy does not
    change when `h` does, so scanning a field is free, and `magnetisation` and
    `susceptibility` become available as derivatives with respect to `h`.

    Parameters
    ----------
    temperature : float
        Must be positive.
    magnetic_field : float
        The field `h`, conjugate to the total magnetisation.
    """

    magnetic_field: float

    FIELD_ATTRIBUTES = {"magnetisation": "magnetic_field"}

    @property
    def fields(self) -> tuple[Field, ...]:
        return (Field("magnetisation", self.magnetic_field, +1, "h"),)


@dataclass(frozen=True)
class IsothermalIsobaric(Ensemble):
    """
    Fixed temperature and pressure: the volume fluctuates. The (T, P, N) ensemble.

    The weight is `exp(-beta(E + P V))`, so the volume enters with sign -1 --
    raising the pressure makes large volumes *less* likely, and `<V>` therefore
    falls as P rises. That sign is carried through every derivative:
    `d<V>/dP = -beta Var(V)`, negative, which is the isothermal compressibility
    being positive.

    `-T log Delta` is the Gibbs energy. Note that enthalpy is *not* what this
    ensemble generates: `H = U + PV` is the potential at fixed (S, P), and no
    simple ensemble samples at fixed entropy. `ThermalState.enthalpy` computes
    it from a state rather than generating one.

    Parameters
    ----------
    temperature : float
        Must be positive.
    pressure : float
        The pressure `P`, conjugate to the volume. Must be positive.
    """

    pressure: float

    FIELD_ATTRIBUTES = {"volume": "pressure"}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pressure <= 0.0:
            raise ValueError(f"pressure must be positive, got {self.pressure}")

    @property
    def fields(self) -> tuple[Field, ...]:
        return (Field("volume", self.pressure, -1, "P"),)


@dataclass(frozen=True)
class Generalised(Ensemble):
    """
    An ensemble with an arbitrary set of conjugate fields.

    For combinations the named classes do not cover: a mixture with one
    chemical potential per species, a magnet at fixed volume, a lattice gas
    with both `mu` and `h`.

    Example
    -------
    >>> Generalised(
    ...     temperature=1.0,
    ...     conjugates=[
    ...         Field("particles", -0.5, +1, "mu"),
    ...         Field("magnetisation", 0.2, +1, "h"),
    ...     ],
    ... )

    Parameters
    ----------
    temperature : float
        Must be positive.
    conjugates : sequence of Field
        The variables set free. An empty sequence is the canonical ensemble.
    """

    conjugates: tuple[Field, ...] = ()

    def __init__(self, temperature: float, conjugates: Sequence[Field] = ()) -> None:
        conjugates = tuple(conjugates)
        for conjugate in conjugates:
            if not isinstance(conjugate, Field):
                raise TypeError(f"expected a Field, got {type(conjugate).__name__}")
        names = [conjugate.variable for conjugate in conjugates]
        if len(set(names)) != len(names):
            raise ValueError(f"each variable may be freed only once, got {names}")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "conjugates", conjugates)
        self.__post_init__()

    @property
    def fields(self) -> tuple[Field, ...]:
        return self.conjugates

    def with_field(self, variable: str, value: float) -> "Generalised":
        """A copy with one conjugate field re-valued, the rest untouched."""
        self.field(variable)
        return type(self)(
            self.temperature,
            tuple(
                Field(c.variable, value, c.sign, c.symbol)
                if c.variable == variable
                else c
                for c in self.conjugates
            ),
        )


__all__ = ["Canonical", "Magnetic", "IsothermalIsobaric", "Generalised"]
