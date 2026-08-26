# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Where the 1/N! comes from, and where it must not appear.

Take the same atoms twice. In a **gas** they roam a volume and any permutation
of them is the same microstate, so the partition function is `z^N / N!`. In a
**crystal** each sits on its own lattice site, and the site labels them: swap
two and you have a genuinely different microstate, so the partition function is
`z^N` with no division at all.

The 1/N! is therefore not a property of the particles. It is a property of
whether they can exchange places, and getting it wrong is not a small error:

  * omit it in the gas and the entropy stops being extensive -- S/N grows
    without bound, and merging two identical boxes appears to create entropy
    from nothing. That is the Gibbs paradox;
  * include it in the crystal and the entropy becomes *sub*-extensive, which is
    just as wrong in the opposite direction.

The third panel is the paradox in its sharpest form. Removing a partition
between two boxes of the *same* gas must cost nothing -- nothing happened. But
between two *different* gases it really does cost `2N log 2`. The only thing
standing between "nothing happened" and "something happened" is the N!.

Run with `python examples/gibbs_paradox.py`; writes `gibbs_paradox.png`
next to itself.

Units: k_B = 1, so temperature is an energy and entropy is dimensionless.
"""

import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from eigora.statphys import (
    BOLTZMANN,
    Canonical,
    HarmonicMode,
    IdealGas,
    IdenticalParticles,
    equilibrium,
)

TEMPERATURE = 10.0
DENSITY = 0.1          # N/V, dilute enough that the classical treatment holds
OMEGA = 1.0            # crystal mode frequency
COUNTS = np.unique(np.round(np.logspace(0, 2.7, 26)).astype(int))


def wavelength(temperature, mass=1.0):
    """Thermal de Broglie wavelength, sqrt(2 pi beta / m) with hbar = 1."""
    return math.sqrt(2.0 * math.pi / (temperature * mass))


def entropy(system):
    """S of a system in the canonical ensemble."""
    return equilibrium(system, Canonical(TEMPERATURE)).entropy


def gas(count, volume, exchangeable=True):
    """
    `count` classical atoms in `volume`, exchangeable or not.

    Both branches are library systems and the factorial is never written by
    hand: `IdenticalParticles(..., BOLTZMANN, particles=N)` *is* `z^N/N!` --
    correct Boltzmann counting -- while `z ** N` is the tensor product of N
    labelled copies, `z^N`. Their log Z differ by exactly `log N!`, which is
    the entire content of this example.
    """
    single = IdealGas(particles=1, volume=volume)
    if exchangeable:
        return IdenticalParticles(single, BOLTZMANN, particles=int(count))
    return single ** int(count)


def crystal(count, exchangeable=False):
    """
    `count` atoms pinned to lattice sites, three modes each.

    An atom on a site is `HarmonicMode ** 3`, and the crystal is `atom ** N`:
    a product, because the sites label the atoms and products are exactly what
    distinguishable subsystems compose into. Passing `exchangeable=True`
    applies the factorial anyway, which is the mistake the middle panel shows.
    """
    atom = HarmonicMode(omega=OMEGA) ** 3
    if exchangeable:
        return IdenticalParticles(atom, BOLTZMANN, particles=int(count))
    return atom ** int(count)


def panel_gas(ax):
    """
    S/N for a gas at fixed density. Extensive only with the N!.

    With it, S/N is flat and sits on Sackur-Tetrode. Without it, S/N grows like
    log N forever: doubling the system more than doubles its entropy, which no
    thermodynamic quantity is allowed to do.
    """
    proper = [entropy(gas(n, n / DENSITY)) / n for n in COUNTS]
    naive = [entropy(gas(n, n / DENSITY, exchangeable=False)) / n for n in COUNTS]
    sackur_tetrode = math.log(1.0 / (DENSITY * wavelength(TEMPERATURE) ** 3)) + 2.5

    ax.semilogx(COUNTS, proper, "o-", ms=4, color="C0", label="$z^N/N!$  (gas)")
    ax.semilogx(COUNTS, naive, "s--", ms=4, color="C3", label="$z^N$  (no $N!$)")
    ax.axhline(sackur_tetrode, ls=":", lw=1.2, color="k",
               label=f"Sackur-Tetrode = {sackur_tetrode:.3f}")
    ax.annotate("grows as $\\log N$\nnot extensive", xy=(COUNTS[-1], naive[-1]),
                xytext=(20, naive[-1] - 1.6), fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="C3"))
    ax.set(xlabel="$N$", ylabel="$S/N$", title="Gas: the $N!$ is required")
    ax.legend(fontsize=8, loc="upper left")


def panel_crystal(ax):
    """
    S/N for a crystal. Extensive only *without* the N!.

    The same atoms, now localised. Sites are labels, so the microstates are
    already distinguishable and dividing by N! removes a symmetry that was
    never there -- S/N then falls like -log N.
    """
    proper = [entropy(crystal(n)) / n for n in COUNTS]
    naive = [entropy(crystal(n, exchangeable=True)) / n for n in COUNTS]
    single = entropy(crystal(1))

    ax.semilogx(COUNTS, proper, "o-", ms=4, color="C0", label="$z^N$  (crystal)")
    ax.semilogx(COUNTS, naive, "s--", ms=4, color="C3", label="$z^N/N!$  (spurious $N!$)")
    ax.axhline(single, ls=":", lw=1.2, color="k",
               label=f"3 modes per atom = {single:.3f}")
    ax.annotate("falls as $-\\log N$\nsub-extensive", xy=(COUNTS[-1], naive[-1]),
                xytext=(3, naive[-1] + 2.0), fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="C3"))
    ax.set(xlabel="$N$", ylabel="$S/N$", title="Crystal: the $N!$ is forbidden")
    ax.legend(fontsize=8, loc="lower left")


def panel_mixing(ax):
    """
    Remove the partition between two boxes. The paradox itself.

    Same gas both sides: nothing physically happened, so `Delta S` must vanish
    -- and it does, per particle, only because of the N!. Different gases:
    each species really does expand into twice the volume, and `2 log 2` per
    particle is a real entropy of mixing. Both boxes hold N atoms of one
    species, so nothing distinguishes the two cases except whether the atoms
    can be told apart, which is precisely what the N! encodes.
    """
    volume = np.array([n / DENSITY for n in COUNTS])

    same_proper = [
        (entropy(gas(2 * n, 2 * v)) - 2 * entropy(gas(n, v))) / n
        for n, v in zip(COUNTS, volume)
    ]
    same_naive = [
        (
            entropy(gas(2 * n, 2 * v, exchangeable=False))
            - 2 * entropy(gas(n, v, exchangeable=False))
        )
        / n
        for n, v in zip(COUNTS, volume)
    ]
    # Two species: each keeps its own N! and each expands from V to 2V.
    different = [
        2.0 * (entropy(gas(n, 2 * v)) - entropy(gas(n, v))) / n
        for n, v in zip(COUNTS, volume)
    ]

    # The two upper curves lie exactly on top of each other, which is the
    # whole point -- so draw one fat and pale beneath the other rather than
    # letting the second hide the first.
    ax.semilogx(COUNTS, same_naive, "-", lw=5, alpha=0.35, color="C3",
                label="same gas, no $N!$")
    ax.semilogx(COUNTS, different, "^-", ms=4, lw=1.2, color="C2",
                label="different gases")
    ax.semilogx(COUNTS, same_proper, "o-", ms=4, color="C0",
                label="same gas, with $N!$")
    ax.axhline(2.0 * math.log(2.0), ls=":", lw=1.2, color="k",
               label="$2\\log 2$")
    ax.axhline(0.0, lw=0.8, color="grey")
    ax.annotate(
        "without the $N!$, identical gases\nmix exactly like different ones\n"
        "-- that is the paradox",
        xy=(COUNTS[len(COUNTS) // 2], 2.0 * math.log(2.0)),
        xytext=(1.6, 0.72), fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="grey"),
    )
    ax.set(xlabel="$N$", ylabel="$\\Delta S / N$ on merging",
           title="Gibbs paradox: identical vs different")
    ax.legend(fontsize=8, loc="center right")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for panel, ax in zip((panel_gas, panel_crystal, panel_mixing), axes):
        panel(ax)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "The $1/N!$ belongs to exchangeable particles, not to particles",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=130)
    print(f"wrote {output}")

    # The numbers behind the third panel, at the largest N drawn.
    count = int(COUNTS[-1])
    volume = count / DENSITY
    same = entropy(gas(2 * count, 2 * volume)) - 2 * entropy(gas(count, volume))
    naive = entropy(gas(2 * count, 2 * volume, exchangeable=False)) - 2 * entropy(
        gas(count, volume, exchangeable=False)
    )
    print(f"N = {count}:")
    print(f"  same gas, with N!   dS = {same:>12.4f}   (= log(pi N)/2 = "
          f"{0.5 * math.log(math.pi * count):.4f}, a Stirling remainder)")
    print(f"  same gas, no N!     dS = {naive:>12.4f}   (= 2N log 2 = "
          f"{2 * count * math.log(2.0):.4f})")


if __name__ == "__main__":
    main()
