# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Configurations worth sampling.

Each is small enough to have an exact answer somewhere in the package, which
is the point: a sampler that cannot be checked against something is a sampler
nobody should trust. `IsingLattice` on eight sites has 256 microstates and can
be enumerated outright; `FermionGas` has its occupations in closed form from
`IdenticalParticles`.

Note where the magnetic field is *not*. `IsingLattice` reports a magnetisation
and leaves `h` to the ensemble, so scanning a field never rebuilds the lattice
and `<M>` comes out of the same run as `<E>`. Folding `-h M` into the energy
would work too and gives identical physics; keeping them apart is what lets
one sampler serve every ensemble.
"""

import math

import numpy as np

from eigora.statphys.monte_carlo.base import Configuration, Proposal
from eigora.statphys.systems.identical import BOLTZMANN, BOSE, FERMI, Statistics


class IsingLattice(Configuration):
    """
    Ising spins on a periodic lattice of any dimension.

        E = -J sum_<ij> s_i s_j,      M = sum_i s_i

    A move flips one spin, so `delta_energy = 2 J s_i sum_neighbours` and
    `delta_magnetisation = -2 s_i` -- both O(1) in the lattice size, and both
    symmetric, so `log_bias` is zero.

    Example
    -------
    >>> from eigora.statphys import Canonical
    >>> from eigora.statphys.monte_carlo import metropolis
    >>> lattice = IsingLattice((8,), coupling=1.0, rng=np.random.default_rng(0))
    >>> run = metropolis(lattice, Canonical(2.0), steps=20000, burn_in=2000)
    >>> run.energy
    ...

    Parameters
    ----------
    shape : tuple of int
        Lattice dimensions. `(8,)` is a chain, `(8, 8)` a square lattice.
    coupling : float
        `J`. Positive is ferromagnetic.
    rng : numpy Generator, optional
        Used only to draw the initial spins.
    """

    def __init__(self, shape, coupling=1.0, rng=None):
        self.shape = tuple(shape)
        if not self.shape or any(side < 2 for side in self.shape):
            raise ValueError(
                f"every lattice side must be at least 2, got {self.shape}"
            )
        self.coupling = float(coupling)
        generator = np.random.default_rng() if rng is None else rng
        self.spins = generator.choice((-1, 1), size=self.shape).astype(np.int8)
        self._energy = self.energy_of()
        self._magnetisation = float(self.spins.sum())

    @property
    def sites(self) -> int:
        return int(self.spins.size)

    @property
    def energy(self) -> float:
        return self._energy

    def extensive(self) -> dict[str, float]:
        return {"magnetisation": self._magnetisation}

    def energy_of(self) -> float:
        """Recomputed from scratch: each bond once, over all axes."""
        total = 0.0
        for axis in range(self.spins.ndim):
            total += float((self.spins * np.roll(self.spins, -1, axis=axis)).sum())
        return -self.coupling * total

    def propose(self, rng) -> Proposal:
        site = tuple(int(rng.integers(side)) for side in self.shape)
        spin = int(self.spins[site])
        neighbours = 0
        for axis, side in enumerate(self.shape):
            for offset in (-1, 1):
                shifted = list(site)
                shifted[axis] = (site[axis] + offset) % side
                neighbours += int(self.spins[tuple(shifted)])
        return Proposal(
            delta_energy=2.0 * self.coupling * spin * neighbours,
            deltas={"magnetisation": -2.0 * spin},
            payload=site,
        )

    def apply(self, proposal: Proposal) -> None:
        self.spins[proposal.payload] *= -1
        self._energy += proposal.delta_energy
        self._magnetisation += proposal.deltas["magnetisation"]


class HeisenbergLattice(Configuration):
    """
    Classical Heisenberg spins: unit vectors on a periodic chain or lattice.

        E = -J sum_<ij> S_i . S_j,     M = sum_i S_i^z

    Classical rather than quantum -- the spins are three-vectors, not
    operators, so there is no sign problem and no Hilbert space. The quantum
    chain of the same name lives in `eigora.qm.discrete` and reaches the
    package through `statphys.systems.bridge`; comparing the two is comparing
    two different models, not two methods.

    A move rotates one spin to a uniformly random direction, which is
    symmetric, so `log_bias` is zero.

    Parameters
    ----------
    shape : tuple of int
        Lattice dimensions.
    coupling : float
        `J`. Positive is ferromagnetic.
    rng : numpy Generator, optional
        Used to draw the initial directions.
    """

    def __init__(self, shape, coupling=1.0, rng=None):
        self.shape = tuple(shape)
        if not self.shape or any(side < 2 for side in self.shape):
            raise ValueError(
                f"every lattice side must be at least 2, got {self.shape}"
            )
        self.coupling = float(coupling)
        generator = np.random.default_rng() if rng is None else rng
        self.spins = _random_directions(generator, self.shape)
        self._energy = self.energy_of()
        self._magnetisation = float(self.spins[..., 2].sum())

    @property
    def sites(self) -> int:
        return int(np.prod(self.shape))

    @property
    def energy(self) -> float:
        return self._energy

    def extensive(self) -> dict[str, float]:
        return {"magnetisation": self._magnetisation}

    def energy_of(self) -> float:
        total = 0.0
        for axis in range(len(self.shape)):
            rolled = np.roll(self.spins, -1, axis=axis)
            total += float((self.spins * rolled).sum())
        return -self.coupling * total

    def propose(self, rng) -> Proposal:
        site = tuple(int(rng.integers(side)) for side in self.shape)
        current = self.spins[site]
        candidate = _random_directions(rng, ())
        field = np.zeros(3)
        for axis, side in enumerate(self.shape):
            for offset in (-1, 1):
                shifted = list(site)
                shifted[axis] = (site[axis] + offset) % side
                field += self.spins[tuple(shifted)]
        return Proposal(
            delta_energy=-self.coupling * float((candidate - current) @ field),
            deltas={"magnetisation": float(candidate[2] - current[2])},
            payload=(site, candidate),
        )

    def apply(self, proposal: Proposal) -> None:
        site, candidate = proposal.payload
        self.spins[site] = candidate
        self._energy += proposal.delta_energy
        self._magnetisation += proposal.deltas["magnetisation"]


class Gas(Configuration):
    """
    Identical particles hopping between orbitals, either statistics.

    A microstate is an occupation number per orbital. A move picks a particle
    uniformly -- so the source orbital comes up with probability `n_s / N` --
    and offers it a target. What differs between the statistics is only which
    targets exist:

        fermions   an empty orbital, since a filled one has no room
        bosons     any other orbital, since there is no limit

    **The proposal is asymmetric for bosons, and that is physics.** Picking a
    particle uniformly favours crowded orbitals, and the reverse move is
    offered with a different probability than the forward one:

        T(x -> x')  = (n_s / N) (1 / A),    T(x' -> x) = ((n_t + 1) / N) (1 / A)

    so `log_bias = log((n_t + 1) / n_s)`, with the occupations read *before*
    the move. The number of available targets `A` cancels: a hop conserves the
    particle number, so `A` is the same on both sides.

    For fermions `n_s = 1` and `n_t = 0`, the bias is exactly zero, and the
    sampler reduces to plain Metropolis. For bosons it does not, and dropping
    it is not a small error -- on a three-orbital, three-boson system the
    stationary distribution shifts by 0.25 in absolute probability. Bunching is
    precisely what the bias encodes.
    """

    #: Set by the subclass. Decides capacity, and therefore which targets exist.
    STATISTICS: Statistics = BOLTZMANN

    def __init__(self, energies, particles, rng=None):
        self.energies = np.asarray(energies, dtype=np.float64)
        if self.energies.size < 2:
            raise ValueError(
                f"a gas needs at least two orbitals to move between, got "
                f"{self.energies.size}"
            )
        if particles < 0:
            raise ValueError(f"particles must be non-negative, got {particles}")
        self.particles = int(particles)
        generator = np.random.default_rng() if rng is None else rng

        placed = self._initial_placement(generator)
        self.occupation = np.bincount(placed, minlength=self.energies.size)
        # Particle -> orbital, so a uniform draw over particles gives a source
        # orbital with probability n_s / N. Which particle is irrelevant --
        # they are identical -- but the *counts* must come out right.
        self._where = [int(k) for k in placed]
        self._energy = self.energy_of()
        self._prepare()

    def _initial_placement(self, rng):
        raise NotImplementedError

    def _prepare(self):
        """Any bookkeeping the subclass needs alongside the occupations."""

    def _draw_target(self, rng, source):
        """`(target, slot)`, or `None` when there is nowhere to go."""
        raise NotImplementedError

    def _commit(self, source, target, slot):
        """Update the subclass's bookkeeping after a move."""

    @property
    def statistics(self) -> Statistics:
        return self.STATISTICS

    @property
    def energy(self) -> float:
        return self._energy

    def extensive(self) -> dict[str, float]:
        return {"particles": float(self.particles)}

    def energy_of(self) -> float:
        return float(np.dot(self.energies, self.occupation))

    def occupations(self) -> np.ndarray:
        """Occupation of each orbital: 0 or 1 for fermions, any integer above."""
        return self.occupation.astype(np.float64)

    def propose(self, rng) -> "Proposal | None":
        if self.particles == 0:
            return None
        index = int(rng.integers(self.particles))
        source = self._where[index]
        drawn = self._draw_target(rng, source)
        if drawn is None:
            return None
        target, slot = drawn
        # Occupations *before* the move, which is what the ratio needs.
        crowd = int(self.occupation[source])
        room = int(self.occupation[target])
        return Proposal(
            delta_energy=float(self.energies[target] - self.energies[source]),
            deltas={"particles": 0.0},
            log_bias=math.log((room + 1) / crowd),
            payload=(index, source, target, slot),
        )

    def apply(self, proposal: Proposal) -> None:
        index, source, target, slot = proposal.payload
        self.occupation[source] -= 1
        self.occupation[target] += 1
        self._where[index] = target
        self._energy += proposal.delta_energy
        self._commit(source, target, slot)

    @staticmethod
    def of_spin(energies, particles, spin, rng=None) -> "Gas":
        """
        Build a gas of spin-`s` particles, letting the physics choose.

        The spin fixes both halves at once: `2s + 1` states per orbital energy,
        and half-integer spin means fermions while integer spin means bosons.
        That is the spin-statistics theorem doing the dispatch, rather than the
        caller being asked twice for the same fact.

        Example
        -------
        >>> Gas.of_spin([0.0, 1.0], particles=2, spin=0.5)   # 4 orbitals, fermions
        >>> Gas.of_spin([0.0, 1.0], particles=2, spin=1)     # 6 orbitals, bosons
        """
        multiplicity = round(2.0 * spin + 1.0)
        if abs(2.0 * spin + 1.0 - multiplicity) > 1e-9 or multiplicity < 1:
            raise ValueError(f"spin must be a non-negative multiple of 1/2, got {spin}")
        orbitals = np.repeat(np.asarray(energies, dtype=np.float64), multiplicity)
        kind = FermionGas if multiplicity % 2 == 0 else BosonGas
        return kind(orbitals, particles, rng=rng)


class FermionGas(Gas):
    """
    Fermions: one particle per orbital, so a target must be empty.

    The empty orbitals are kept as a list, so a proposal draws a legal move
    directly instead of guessing and being refused. Drawing both source and
    target uniformly over all orbitals would be legal only with probability
    `(N/K)(1 - N/K)` -- 17% for twelve fermions in fifty-four orbitals, and
    worse the more dilute the gas.

    `propose` returns `None` only when there is genuinely nowhere to go: an
    empty gas has nothing to move, a full one has nowhere to put it. That is
    exclusion expressed as the *absence of a move*, which leaves the
    acceptance rule pure Metropolis -- and with `n_s = 1`, `n_t = 0`, the bias
    is exactly zero.
    """

    STATISTICS = FERMI

    def _initial_placement(self, rng):
        if self.particles > self.energies.size:
            raise ValueError(
                f"{self.particles} fermions will not fit in "
                f"{self.energies.size} orbitals"
            )
        return rng.choice(self.energies.size, size=self.particles, replace=False)

    def _prepare(self):
        self._empty = [int(k) for k in np.flatnonzero(self.occupation == 0)]

    def _draw_target(self, rng, source):
        if not self._empty:
            return None
        slot = int(rng.integers(len(self._empty)))
        return self._empty[slot], slot

    def _commit(self, source, target, slot):
        # The two orbitals trade places: the target is now full, the source
        # empty. O(1), because the slot came along in the payload.
        self._empty[slot] = source

    @property
    def occupied(self) -> np.ndarray:
        """Which orbitals are filled, as a boolean mask."""
        return self.occupation > 0


class BosonGas(Gas):
    """
    Bosons: any number per orbital, so any other orbital is a target.

    The first system in the package for which `log_bias` is not zero. Picking
    a particle uniformly favours already-crowded orbitals, and the reverse hop
    is offered with probability `(n_t + 1)/n_s` times the forward one -- which
    is exactly the bunching that makes bosons bosons. Sampling without it
    converges, confidently, on the wrong distribution.

    The source orbital is excluded from the target draw. A null move would be
    harmless but its bias would not be: `log((n_s + 1)/n_s)` is not zero,
    while the reverse of doing nothing is doing nothing.
    """

    STATISTICS = BOSE

    def _initial_placement(self, rng):
        return rng.choice(self.energies.size, size=self.particles, replace=True)

    def _draw_target(self, rng, source):
        size = self.energies.size
        target = int(rng.integers(size - 1))
        if target >= source:
            target += 1
        return target, None


def _random_directions(rng, shape):
    """Uniform points on the unit sphere -- not uniform in the angles."""
    z = rng.uniform(-1.0, 1.0, size=shape)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    radial = np.sqrt(1.0 - z**2)
    return np.stack((radial * np.cos(phi), radial * np.sin(phi), z), axis=-1)


__all__ = [
    "BosonGas",
    "FermionGas",
    "Gas",
    "HeisenbergLattice",
    "IsingLattice",
]
