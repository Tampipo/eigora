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


class FermionGas(Configuration):
    """
    Fermions hopping between orbitals, with Pauli enforced by refusal.

    A microstate is which orbitals are occupied. A move picks an occupied
    orbital and an empty one; if the draw lands on an occupied target there is
    **no legal move**, and `propose` returns `None`. That is the honest way to
    express exclusion: it is not a rejected move with an unusual acceptance
    rule, it is the absence of a move, and keeping it out of the acceptance
    rule leaves that rule pure Metropolis.

    The particle number is reported, so a grand canonical ensemble can free it
    -- though a hop conserves it, so `<N>` is whatever it started at unless a
    move that creates or destroys is added.

    Parameters
    ----------
    energies : sequence of float
        The single-particle orbital energies.
    particles : int
        How many fermions. Conserved by hopping.
    rng : numpy Generator, optional
        Used to choose the initial occupation.
    """

    def __init__(self, energies, particles, rng=None):
        self.energies = np.asarray(energies, dtype=np.float64)
        if particles < 0 or particles > self.energies.size:
            raise ValueError(
                f"{particles} fermions will not fit in {self.energies.size} "
                f"orbitals"
            )
        self.particles = int(particles)
        generator = np.random.default_rng() if rng is None else rng
        self.occupied = np.zeros(self.energies.size, dtype=bool)
        self.occupied[
            generator.choice(self.energies.size, size=self.particles, replace=False)
        ] = True
        self._energy = self.energy_of()

    @property
    def energy(self) -> float:
        return self._energy

    def extensive(self) -> dict[str, float]:
        return {"particles": float(self.occupied.sum())}

    def energy_of(self) -> float:
        return float(self.energies[self.occupied].sum())

    def occupations(self) -> np.ndarray:
        """Which orbitals are filled, as ones and zeros."""
        return self.occupied.astype(np.float64)

    def propose(self, rng) -> "Proposal | None":
        size = self.energies.size
        source = int(rng.integers(size))
        target = int(rng.integers(size))
        if not self.occupied[source] or self.occupied[target]:
            return None       # nothing to move, or nowhere to put it
        return Proposal(
            delta_energy=float(self.energies[target] - self.energies[source]),
            deltas={"particles": 0.0},
            payload=(source, target),
        )

    def apply(self, proposal: Proposal) -> None:
        source, target = proposal.payload
        self.occupied[source] = False
        self.occupied[target] = True
        self._energy += proposal.delta_energy


def _random_directions(rng, shape):
    """Uniform points on the unit sphere -- not uniform in the angles."""
    z = rng.uniform(-1.0, 1.0, size=shape)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    radial = np.sqrt(1.0 - z**2)
    return np.stack((radial * np.cos(phi), radial * np.sin(phi), z), axis=-1)


__all__ = ["FermionGas", "HeisenbergLattice", "IsingLattice"]
