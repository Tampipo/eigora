# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ensemble equivalence at three sizes: N = 2, 10, 100.

Heat capacity and chemical potential for N bosons on a ladder of 2.5N
orbitals, each computed three ways -- counting at fixed energy, summing the
permutation recursion at fixed temperature, factorising at fixed chemical
potential. They disagree by tens of percent at N = 2 and coincide at N = 100.

Bosons because the Borrmann-Franke recursion alternates in sign for fermions
and loses all sixteen digits by N = 100; for bosons every term is positive and
it is unconditionally stable.

Two things the library does not do, done here instead:

  * `Omega(E, N)` is a count with no factorisation, so it is built by dynamic
    programming over orbitals.
  * `field_for` brackets outward and walks into the bosonic condensation
    guard, so `mu` is solved by bisection below the ground orbital.

Run with `python examples/ensemble_equivalence.py`.
Units: k_B = 1.
"""

import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from eigora.statphys import (
    BOSE,
    Canonical,
    GrandCanonical,
    IdenticalParticles,
    NLevel,
    equilibrium,
)

NUMBERS = (2, 10, 100)


def orbitals(number):
    return NLevel([float(k) for k in range(max(5, round(2.5 * number)))])


def density_of_states(number, count):
    """Omega(E, count) for bosons on the ladder, by dynamic programming."""
    width = len(orbitals(number).energies)
    top = count * (width - 1)
    table = np.zeros((count + 1, top + 1))
    table[0, 0] = 1.0
    for energy in range(width):
        for occupied in range(1, count + 1):
            table[occupied, energy:] += table[occupied - 1, : top + 1 - energy]
    return table[count]


OMEGA = {
    n: {c: density_of_states(n, c) for c in (n - 1, n, n + 1)} for n in NUMBERS
}


def canonical(number, temperature, count=None):
    gas = IdenticalParticles(
        orbitals(number), BOSE, particles=count if count else number
    )
    return equilibrium(gas, Canonical(temperature))


def canonical_mu(number, temperature):
    below = canonical(number, temperature, number - 1).free_energy
    above = canonical(number, temperature, number + 1).free_energy
    return 0.5 * (above - below)


def grand_mu(number, temperature):
    """Bisect for mu strictly below the ground orbital, where <N> diverges."""
    gas = IdenticalParticles(orbitals(number), BOSE)

    def mean(value):
        return equilibrium(gas, GrandCanonical(temperature, value)).mean("particles")

    high, low = -1e-10, -1.0
    while mean(low) > number:
        low *= 2.0
    for _ in range(120):
        middle = 0.5 * (low + high)
        if mean(middle) > number:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def grand_heat(number, temperature):
    """(at fixed mu, at fixed N) -- the plausible answer and the right one."""
    beta = 1.0 / temperature
    mu = grand_mu(number, temperature)
    gas = IdenticalParticles(orbitals(number), BOSE)
    state = equilibrium(gas, GrandCanonical(temperature, mu))
    step = 0.02 * abs(mu)
    covariance = (
        equilibrium(gas, GrandCanonical(temperature, mu + step)).energy
        - equilibrium(gas, GrandCanonical(temperature, mu - step)).energy
    ) / (2.0 * step * beta)
    conditional = state.energy_variance - covariance**2 / state.variance("particles")
    return beta**2 * state.energy_variance, beta**2 * conditional


def micro(number):
    """(T, C, mu) at each shell, from the counted density of states."""
    here, below, above = (OMEGA[number][number + d] for d in (0, -1, 1))
    shells = np.nonzero(here)[0]
    width = max(1, min(8, len(shells) // 14))
    logs = np.log(here[shells])

    def beta(index):
        return (logs[index + width] - logs[index - width]) / (
            shells[index + width] - shells[index - width]
        )

    out = []
    for index in range(width, len(shells) - width):
        b = beta(index)
        if b <= 0.0:
            continue
        temperature = 1.0 / b
        step = beta(index - width) - beta(index + width)
        heat = None if step <= 0 else (
            b * b * (shells[index + width] - shells[index - width]) / step
        )
        energy = shells[index]
        chemical = None
        if energy < len(above) and below[energy] >= 1 and above[energy] >= 1:
            chemical = -temperature * 0.5 * (
                math.log(above[energy]) - math.log(below[energy])
            )
        out.append((temperature, heat, chemical))
    return out[:: max(1, len(out) // 14)]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for column, number in enumerate(NUMBERS):
        grid = np.linspace(0.25 * number + 0.4, 1.6 * number + 1.5, 55)
        sampled = grid[::6]
        points = micro(number)
        top = float(grid[-1]) * 1.05

        ax = axes[0, column]
        heat = [canonical(number, t).heat_capacity for t in grid]
        ax.plot(grid, heat, lw=4.5, alpha=0.32, color="k", label="canonical")
        wrong, right = zip(*(grand_heat(number, t) for t in sampled))
        ax.plot(sampled, right, "o", ms=5.5, mfc="none", mew=1.5, color="C0",
                label="grand canonical, fixed $N$")
        ax.plot(sampled, wrong, "x--", ms=5, lw=1.0, color="C3",
                label="grand canonical, fixed $\\mu$")
        seen = [(t, c) for t, c, _ in points if c and 0 < c and t < top]
        if seen:
            ax.plot([t for t, _ in seen], [c for _, c in seen], "s", ms=5,
                    mfc="none", mew=1.3, color="C2", label="microcanonical")
        ax.set(ylabel="$C$" if column == 0 else "", ylim=(0, 2.4 * max(heat)),
               title=f"$N = {number}$")
        ax.grid(alpha=0.15)
        if column == 0:
            ax.legend(fontsize=8, loc="upper right")

        ax = axes[1, column]
        ax.plot(grid, [canonical_mu(number, t) for t in grid], lw=4.5,
                alpha=0.32, color="k", label="canonical")
        ax.plot(sampled, [grand_mu(number, t) for t in sampled], "o", ms=5.5,
                mfc="none", mew=1.5, color="C0", label="grand canonical")
        seen = [(t, m) for t, _, m in points if m is not None and t < top]
        if seen:
            ax.plot([t for t, _ in seen], [m for _, m in seen], "s", ms=5,
                    mfc="none", mew=1.3, color="C2", label="microcanonical")
        ax.set(xlabel="$T$", ylabel="$\\mu$" if column == 0 else "")
        ax.grid(alpha=0.15)
        if column == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Three ensembles, three sizes: agreement is an $N$ effect",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    output = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=130)
    print(f"wrote {output}")

    print(f"{'N':>5} {'T':>6} {'C_can':>9} {'C_grand':>9} {'gap%':>6}"
          f" {'mu_can':>10} {'mu_grand':>10} {'gap%':>6}")
    for number in NUMBERS:
        t = 0.6 * number + 1.0
        c1 = canonical(number, t).heat_capacity
        c2 = grand_heat(number, t)[1]
        m1, m2 = canonical_mu(number, t), grand_mu(number, t)
        print(f"{number:>5} {t:>6.1f} {c1:>9.4f} {c2:>9.4f}"
              f" {100 * abs(c2 - c1) / c1:>6.1f} {m1:>10.4f} {m2:>10.4f}"
              f" {100 * abs(m2 - m1) / abs(m1):>6.1f}")


if __name__ == "__main__":
    main()
