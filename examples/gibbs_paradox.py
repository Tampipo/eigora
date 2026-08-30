# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Where the 1/N! comes from, and where it must not appear.

The factorial is usually introduced as a repair: `z^N` overcounts, so divide by
`N!` and move on. That is backwards. Proper counting never divides by anything
-- the microstates of N identical particles are occupation numbers, and there
are exactly as many as there are. The `1/N!` is what that counting *becomes*
when the gas is dilute enough that no two particles want the same orbital.

The first panel shows it happening: `Z_N` for fermions and for bosons, both
converging on `z^N/N!` as the temperature rises, while `z^N` sits a constant
`log N!` above and never converges on anything.

The rest is the consequence. The same `z^N` that is wrong for a gas is *right*
for a crystal, where the lattice sites label the atoms, and applying the
factorial there is as wrong as omitting it in the gas. The last panel is the
Gibbs paradox: merging two boxes of the same gas must cost nothing, merging two
boxes of different gases really does cost `2N log 2`, and the only thing
standing between them is the factorial.

Every system here is built by the library. Nothing applies a factorial by hand:

    IdenticalParticles(orbitals, FERMI, particles=N)      exact, no factorial
    IdenticalParticles(orbitals, BOLTZMANN, particles=N)  z^N/N!
    orbitals ** N                                         z^N, labelled copies

Run with `python examples/gibbs_paradox.py`; writes `gibbs_paradox.png` next
to itself.

Units: k_B = 1, so temperature is an energy and entropy is dimensionless.
"""

import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from eigora.statphys import (
    BOLTZMANN,
    BOSE,
    FERMI,
    Canonical,
    HarmonicMode,
    IdealGas,
    IdenticalParticles,
    equilibrium,
)

TEMPERATURE = 10.0
DENSITY = 0.1          # N/V, dilute enough that the classical treatment holds
OMEGA = 1.0            # trap and crystal mode frequency
TRAPPED = 6            # particles in the trap for the crossover panel
COUNTS = np.unique(np.round(np.logspace(0, 2.7, 26)).astype(int))


def wavelength(temperature, mass=1.0):
    """Thermal de Broglie wavelength, sqrt(2 pi beta / m) with hbar = 1."""
    return math.sqrt(2.0 * math.pi / (temperature * mass))


def entropy(system, temperature=TEMPERATURE):
    return equilibrium(system, Canonical(temperature)).entropy


def gas(count, volume, exchangeable=True):
    """
    `count` classical atoms in `volume`.

    `IdealGas(particles=N)` carries the `1/N!`; a product of `N` one-particle
    gases does not. Their `log Z` differ by exactly `log N!` and by nothing
    else, which is the whole subject of this file.
    """
    single = IdealGas(particles=1, volume=volume)
    if exchangeable:
        return IdealGas(particles=int(count), volume=volume)
    return single ** int(count)


def crystal(count, exchangeable=False):
    """
    `count` atoms on lattice sites, one mode each.

    The correct crystal is a *product*: sites label the atoms, so the blocks
    are distinguishable and products are what distinguishable subsystems
    compose into. `exchangeable=True` applies the factorial anyway -- the
    mistake the third panel shows.
    """
    mode = HarmonicMode(omega=OMEGA)
    if exchangeable:
        return IdenticalParticles(mode, BOLTZMANN, particles=int(count))
    return mode ** int(count)


def panel_crossover(ax):
    """
    The factorial *emerging*, rather than being imposed.

    `N` particles in a harmonic trap, measured against `z^N/N!`. Fermions sit
    below it and bosons above -- exclusion suppresses, bunching enhances -- and
    both converge on it as the trap empties out. So `z^N/N!` is not a repair
    bolted onto `z^N`; it is what exact quantum counting *becomes* when the
    occupations are small.

    `z^N` converges on nothing. It is a flat `log N!` above the classical
    curve at every temperature, which is what "overcounting" means.

    The fermionic curve stops around T = 1: below that the alternating
    recursion has spent its significant digits, and the library refuses rather
    than returning noise. That boundary is physics, not a plotting choice.
    """
    temperatures = np.logspace(0, 2.3, 60)
    orbitals = HarmonicMode(omega=OMEGA)
    classical = IdenticalParticles(orbitals, BOLTZMANN, particles=TRAPPED)

    for index, statistics in enumerate((FERMI, BOSE)):
        gas_system = IdenticalParticles(orbitals, statistics, particles=TRAPPED)
        shown, values = [], []
        for temperature in temperatures:
            beta = 1.0 / temperature
            try:
                values.append(gas_system.log_z(beta) - classical.log_z(beta))
                shown.append(temperature)
            except ValueError:
                continue     # the sign problem, refused rather than faked
        ax.semilogx(shown, values, lw=1.8, color=f"C{index}",
                    label=f"${statistics.name}$")

    labelled = [
        (orbitals ** TRAPPED).log_z(1.0 / t) - classical.log_z(1.0 / t)
        for t in temperatures
    ]
    ax.semilogx(temperatures, labelled, ls="--", lw=1.5, color="C3",
                label="$z^N$ (labelled)")
    ax.axhline(math.lgamma(TRAPPED + 1), ls=":", lw=1, color="grey")
    ax.axhline(0.0, lw=1, color="k")
    ax.text(1.2, math.lgamma(TRAPPED + 1) + 0.4, f"$\\log {TRAPPED}! $",
            fontsize=9, color="grey")
    ax.text(30, 0.6, "$z^N/N!$", fontsize=9)
    ax.annotate("recursion refuses:\nsign problem", xy=(1.0, -9.0),
                xytext=(1.6, -8.0), fontsize=7.5, color="C0",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="C0"))

    ax.set(xlabel="$T/\\omega$", ylabel="$\\log Z_N - \\log(z^N/N!)$",
           ylim=(-11, 9), title=f"The $N!$ emerges ({TRAPPED} in a trap)")
    ax.legend(fontsize=8, loc="lower right")


def panel_gas(ax):
    """
    S/N for a gas at fixed density. Extensive only with the factorial.

    With it, S/N is flat on Sackur-Tetrode. Without it, S/N grows like log N
    forever: doubling the system more than doubles its entropy, which no
    thermodynamic quantity may do.
    """
    proper = [entropy(gas(n, n / DENSITY)) / n for n in COUNTS]
    naive = [entropy(gas(n, n / DENSITY, exchangeable=False)) / n for n in COUNTS]
    sackur_tetrode = math.log(1.0 / (DENSITY * wavelength(TEMPERATURE) ** 3)) + 2.5

    ax.semilogx(COUNTS, proper, "o-", ms=4, color="C0", label="$z^N/N!$")
    ax.semilogx(COUNTS, naive, "s--", ms=4, color="C3", label="$z^N$")
    ax.axhline(sackur_tetrode, ls=":", lw=1.2, color="k",
               label=f"Sackur-Tetrode = {sackur_tetrode:.3f}")
    ax.annotate("grows as $\\log N$\nnot extensive", xy=(COUNTS[-1], naive[-1]),
                xytext=(18, naive[-1] - 1.7), fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="C3"))
    ax.set(xlabel="$N$", ylabel="$S/N$", title="Gas: the $N!$ is required")
    ax.legend(fontsize=8, loc="upper left")


def panel_crystal(ax):
    """
    S/N for a crystal. Extensive only *without* the factorial.

    The same atoms, now localised. Sites are labels, so the microstates are
    already distinguishable, and dividing by N! removes a symmetry that was
    never there.
    """
    proper = [entropy(crystal(n)) / n for n in COUNTS]
    naive = [entropy(crystal(n, exchangeable=True)) / n for n in COUNTS]
    single = entropy(crystal(1))

    ax.semilogx(COUNTS, proper, "o-", ms=4, color="C0", label="$z^N$")
    ax.semilogx(COUNTS, naive, "s--", ms=4, color="C3", label="$z^N/N!$")
    ax.axhline(single, ls=":", lw=1.2, color="k",
               label=f"one mode per atom = {single:.3f}")
    ax.annotate("falls as $-\\log N$\nsub-extensive", xy=(COUNTS[-1], naive[-1]),
                xytext=(2.5, naive[-1] + 1.6), fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="C3"))
    ax.set(xlabel="$N$", ylabel="$S/N$", title="Crystal: the $N!$ is forbidden")
    ax.legend(fontsize=8, loc="lower left")


def panel_mixing(ax):
    """
    Remove the partition between two boxes. The paradox itself.

    Same gas both sides: nothing physically happened, so `Delta S` must vanish
    per particle -- and it does, only because of the factorial. Different
    gases: each species really does expand into twice the volume, and
    `2 log 2` per particle is a real entropy of mixing.

    The two upper curves coincide exactly, and that coincidence *is* the
    paradox: without the factorial, identical gases mix like different ones.
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
    different = [
        2.0 * (entropy(gas(n, 2 * v)) - entropy(gas(n, v))) / n
        for n, v in zip(COUNTS, volume)
    ]

    ax.semilogx(COUNTS, same_naive, "-", lw=5, alpha=0.35, color="C3",
                label="same gas, no $N!$")
    ax.semilogx(COUNTS, different, "^-", ms=4, lw=1.2, color="C2",
                label="different gases")
    ax.semilogx(COUNTS, same_proper, "o-", ms=4, color="C0",
                label="same gas, with $N!$")
    ax.axhline(2.0 * math.log(2.0), ls=":", lw=1.2, color="k", label="$2\\log 2$")
    ax.axhline(0.0, lw=0.8, color="grey")
    ax.annotate(
        "without the $N!$, identical gases\nmix exactly like different ones",
        xy=(COUNTS[len(COUNTS) // 2], 2.0 * math.log(2.0)),
        xytext=(1.6, 0.75), fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="grey"),
    )
    ax.set(xlabel="$N$", ylabel="$\\Delta S / N$ on merging",
           title="Gibbs paradox: identical vs different")
    ax.legend(fontsize=8, loc="center right")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    panels = (panel_crossover, panel_gas, panel_crystal, panel_mixing)
    for panel, ax in zip(panels, axes.flat):
        panel(ax)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "The $1/N!$ is the dilute limit of quantum counting, not a repair to $z^N$",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=130)
    print(f"wrote {output}")

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
