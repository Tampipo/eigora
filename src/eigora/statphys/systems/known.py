# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The catalogue of systems with a spectrum worth knowing by name.

Each is built from its physical parameters rather than from a list of numbers,
and the ones with a closed-form partition function override `log_z` instead of
summing -- the same arrangement as `KnownHamiltonian` overriding `eigensystem`
in `eigora.qm.discrete`. Both paths stay reachable, so the tests can hold the
closed form against the sum it replaces.

`TwoLevel` and `HarmonicMode` share their names with two-level systems in
`eigora.qm.discrete`, deliberately: these are the thermodynamic
parameterisations of the same physics. `qm.discrete.TwoLevel(bias, coupling)`
places its levels at +/- sqrt(bias^2 + coupling^2)/2, while `TwoLevel(splitting)`
here places them at 0 and `splitting`. The two differ by a shift of the energy
origin, so their free energies differ by a constant and every other
thermodynamic quantity agrees exactly. `HarmonicMode` is the untruncated tower,
where `qm.discrete.HarmonicLadder` keeps only its lowest few states.

Units: k_B = 1, and atomic units elsewhere (hbar = m = 1).
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from eigora.statphys.systems.base import (
    _TOL,
    Couplings,
    Level,
    Moments,
    SpectralSystem,
)
from eigora.statphys.systems.composite import CompositeSystem


@dataclass(frozen=True)
class NLevel(SpectralSystem):
    """
    A system given as an explicit list of energies and degeneracies.

    The escape hatch for anything the catalogue does not name: measured level
    schemes, hand-built toy models, the output of a diagonalisation.

    Example
    -------
    >>> NLevel([0.0, 1.0, 1.0, 2.5])                 # degeneracies default to 1
    >>> NLevel([0.0, 1.0, 2.5], [1, 3, 5])           # grouped, with degeneracies

    Parameters
    ----------
    energies : sequence of float
        Level energies. Sorted on construction, so the caller need not.
    degeneracies : sequence of int, optional
        Microstates per level. Defaults to 1 for every level.
    """

    energies: tuple[float, ...]
    degeneracies: tuple[int, ...] = ()

    def __init__(
        self,
        energies: Sequence[float],
        degeneracies: Sequence[int] | None = None,
    ) -> None:
        energies = tuple(float(e) for e in energies)
        if not energies:
            raise ValueError("a level scheme needs at least one energy")

        if degeneracies is None:
            degeneracies = (1,) * len(energies)
        else:
            degeneracies = tuple(int(g) for g in degeneracies)
        if len(degeneracies) != len(energies):
            raise ValueError(
                f"expected {len(energies)} degeneracy value(s), got {len(degeneracies)}"
            )
        if any(g < 1 for g in degeneracies):
            raise ValueError(f"degeneracies must be at least 1, got {degeneracies}")

        order = sorted(range(len(energies)), key=lambda i: energies[i])
        object.__setattr__(self, "energies", tuple(energies[i] for i in order))
        object.__setattr__(self, "degeneracies", tuple(degeneracies[i] for i in order))

    def levels(self) -> Iterator[Level]:
        return (
            Level(energy, degeneracy)
            for energy, degeneracy in zip(self.energies, self.degeneracies)
        )

    @property
    def n_states(self) -> int:
        return sum(self.degeneracies)


@dataclass(frozen=True)
class TwoLevel(SpectralSystem):
    """
    A ground state at 0 and one excited state at `splitting`.

    The smallest system with any thermodynamics at all, and the origin of the
    Schottky anomaly: the heat capacity vanishes at both temperature extremes
    -- nothing to excite at low T, both states equally likely at high T -- and
    peaks in between at T = 0.4168 * splitting.

    Parameters
    ----------
    splitting : float
        Energy gap. Must be positive.
    """

    splitting: float

    def __post_init__(self) -> None:
        if self.splitting <= 0.0:
            raise ValueError(f"splitting must be positive, got {self.splitting}")

    def levels(self) -> Iterator[Level]:
        yield Level(0.0, 1)
        yield Level(self.splitting, 1)

    @property
    def n_states(self) -> int:
        return 2


@dataclass(frozen=True)
class Degenerate(SpectralSystem):
    """
    A single level at zero energy with `degeneracy` microstates.

    Thermodynamically inert on its own -- log Z = log g at every temperature,
    so it carries no energy and no heat capacity -- but it is how an internal
    multiplicity enters a bigger system. A spin-1/2 is `Degenerate(2)`, and
    `orbital * Degenerate(2)` gives each orbital two spin states, which is
    exactly the counting Fermi statistics needs.

    Parameters
    ----------
    degeneracy : int
        Number of microstates. Must be at least 1.
    """

    degeneracy: int

    def __post_init__(self) -> None:
        if self.degeneracy < 1:
            raise ValueError(f"degeneracy must be at least 1, got {self.degeneracy}")

    def levels(self) -> Iterator[Level]:
        yield Level(0.0, self.degeneracy)

    @property
    def n_states(self) -> int:
        return self.degeneracy

    def log_z(self, beta: float, tol: float = 1e-12) -> float:
        """log g, independent of temperature."""
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        return math.log(self.degeneracy)


@dataclass(frozen=True)
class HarmonicMode(SpectralSystem):
    """
    One harmonic oscillator mode: E_n = omega (n + 1/2), n = 0, 1, 2, ...

    The tower is infinite, which is the case that rules out enumerating a
    spectrum as the only interface. Summing it converges quickly at low
    temperature and slowly at high temperature, but it never has to be summed:

        Z    = 1 / (2 sinh(beta omega / 2))
        <E>  = (omega/2) coth(beta omega / 2)
        C    = (beta omega / 2)^2 / sinh^2(beta omega / 2)

    all in closed form. `levels` is still implemented, so the summation in
    `SpectralSystem` remains reachable and the tests can hold the two against
    each other.

    Note the energies include the zero point, so `<E> -> omega/2` as T -> 0
    rather than 0. Dropping it would change the free energy by `omega/2` and
    leave every other quantity alone.

    Parameters
    ----------
    omega : float
        Angular frequency. Must be positive.
    """

    omega: float

    def __post_init__(self) -> None:
        if self.omega <= 0.0:
            raise ValueError(f"omega must be positive, got {self.omega}")

    def levels(self) -> Iterator[Level]:
        n = 0
        while True:
            yield Level(self.omega * (n + 0.5), 1)
            n += 1

    @property
    def n_states(self) -> None:
        return None

    def moments(
        self, beta: float, couplings: Couplings = (), tol: float = _TOL
    ) -> Moments:
        """
        The three closed forms, in place of summing the tower.

        Overriding `moments` rather than `log_z` is what makes the closed forms
        reachable through a `ThermalState`: everything downstream reads
        `moments`, so an override on `log_z` alone would be bypassed and the
        infinite sum taken instead -- silently, and at high temperature it does
        not even converge within the term cap.

        `log Z` is written as `-(x + log1p(-exp(-2x)))` with `x = beta omega/2`
        rather than `-log(2 sinh x)`: `sinh(x)` overflows around x = 710, but
        this form is just `-x` plus a correction that decays to zero.
        """
        if couplings:
            free = [name for name, _ in couplings]
            raise ValueError(
                f"a harmonic mode carries no extensive variable besides its "
                f"energy, so it cannot be summed against {free}"
            )
        x = self._half_gap(beta)
        return Moments(
            log_z=-(x + math.log1p(-math.exp(-2.0 * x))),
            energy=0.5 * self.omega / math.tanh(x),
            # sinh overflows before its reciprocal underflows, so guard the tail.
            energy_variance=(
                0.0 if x > 350.0 else (0.5 * self.omega / math.sinh(x)) ** 2
            ),
        )

    def _half_gap(self, beta: float) -> float:
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        return 0.5 * beta * self.omega


@dataclass(frozen=True)
class Spin(SpectralSystem):
    """
    A spin `j` whose 2j+1 states carry a magnetisation as well as an energy.

    States are labelled by m = -j, -j+1, ..., +j, with magnetisation m and
    energy zero. The Zeeman energy is deliberately *not* built in: a `Magnetic`
    ensemble supplies the field `h` and trades it against the magnetisation
    each state reports, so scanning `h` never rebuilds the spectrum, and
    `magnetisation` and `susceptibility` become averages over the same sweep
    that gives log Z.

    That gives the Brillouin function for free. For j = 1/2 it reduces to

        <M> = (1/2) tanh(beta h / 2)

    and the susceptibility follows Curie's law, chi -> 1/(4T), at high
    temperature.

    Parameters
    ----------
    j : float
        Spin quantum number: a non-negative multiple of 1/2.
    """

    j: float

    def __post_init__(self) -> None:
        if self.j < 0.0:
            raise ValueError(f"j must be non-negative, got {self.j}")
        if abs(2.0 * self.j - round(2.0 * self.j)) > 1e-12:
            raise ValueError(f"j must be a multiple of 1/2, got {self.j}")

    def levels(self) -> Iterator[Level]:
        # All degenerate in energy; they differ only in magnetisation, so the
        # ordering is by m and the energy ascent check is trivially satisfied.
        count = int(round(2.0 * self.j)) + 1
        for index in range(count):
            m = -self.j + index
            yield Level(0.0, 1, {"magnetisation": m})

    @property
    def n_states(self) -> int:
        return int(round(2.0 * self.j)) + 1


@dataclass(frozen=True)
class Rotor(SpectralSystem):
    """
    A rigid rotor: E_l = b l(l+1) with degeneracy 2l+1, l = 0, 1, 2, ...

    The one system in the catalogue with no closed-form partition function, so
    it is what actually exercises the summation path. Both limits are still
    known and make good tests: the gap to the first excited state is 2b, so the
    heat capacity dies exponentially as T -> 0, while at high temperature the
    sum becomes an integral and gives

        z -> T/b,   <E> -> T,   C -> 1

    which is equipartition for the two rotational degrees of freedom.

    Parameters
    ----------
    b : float
        Rotational constant. Must be positive.
    """

    b: float

    def __post_init__(self) -> None:
        if self.b <= 0.0:
            raise ValueError(f"b must be positive, got {self.b}")

    def levels(self) -> Iterator[Level]:
        l = 0
        while True:
            yield Level(self.b * l * (l + 1), 2 * l + 1)
            l += 1

    @property
    def n_states(self) -> None:
        return None


@dataclass(frozen=True)
class Box1D(SpectralSystem):
    """
    One particle in a 1D box of length L: E_n = pi^2 n^2 / (2 m L^2), n >= 1.

    The building block of the ideal gas. A box in higher dimensions is a
    product of these -- `Box1D(...) ** 3` -- and that is not a convenience but
    a necessity: enumerated as a single 3D spectrum the level count needed for
    convergence goes as the cube of the count needed in 1D, which at ordinary
    temperatures runs past any sane cap. As a product it is three short sums.

    Parameters
    ----------
    length : float
        Box length. Must be positive.
    mass : float
        Particle mass, in atomic units. Must be positive.
    """

    length: float
    mass: float = 1.0

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass}")

    @property
    def level_spacing(self) -> float:
        """The n = 1 energy, pi^2 / (2 m L^2), which sets the whole tower."""
        return math.pi**2 / (2.0 * self.mass * self.length**2)

    def levels(self) -> Iterator[Level]:
        spacing = self.level_spacing
        n = 1
        while True:
            yield Level(spacing * n * n, 1)
            n += 1

    @property
    def n_states(self) -> None:
        return None


def particle_in_box(
    length: float, mass: float = 1.0, ndim: int = 3
) -> "CompositeSystem":
    """
    A particle in an `ndim`-dimensional cubic box, as a product of `Box1D`.

    Built by composition rather than by enumerating the joint spectrum, for the
    reason spelled out in `Box1D`.

    Parameters
    ----------
    length : float
        Edge length of the box.
    mass : float
        Particle mass.
    ndim : int
        Number of dimensions. Must be at least 1.

    Returns
    -------
    CompositeSystem
    """
    if ndim < 1:
        raise ValueError(f"ndim must be at least 1, got {ndim}")
    return Box1D(length=length, mass=mass) ** ndim


__all__ = [
    "NLevel",
    "TwoLevel",
    "Degenerate",
    "HarmonicMode",
    "Spin",
    "Rotor",
    "Box1D",
    "particle_in_box",
]
