# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Fermi-Dirac occupations by Monte Carlo, against the analytic form.

Fermions in a cubic box: orbitals `(nx, ny, nz)` with energy proportional to
`nx^2 + ny^2 + nz^2`, each doubled for spin. A Metropolis move relocates one
particle, and lands on nothing if the target is already occupied -- Pauli
exclusion as the *absence of a move* rather than a special acceptance rule.

Three independent sources for the same two numbers, which is the point:

  * **analytic** -- the Fermi-Dirac formula `n = 1/(exp(beta(eps-mu))+1)` with
    `mu` found by bisecting `sum_k n(eps_k) = N` in a dozen lines here, using
    nothing from the library;
  * **library** -- `IdenticalParticles(orbitals, FERMI)` in a grand canonical
    ensemble: `chemical_potential_for(N)` inverts `<N>`, and `occupation`
    evaluates the distribution;
  * **Monte Carlo** -- the measured mean occupation of each orbital, plus the
    `mu` you get by least-squares fitting the Fermi-Dirac form to it, which is
    what you must do when the sampler is all you have.

The first two must agree to roundoff -- they are the same mathematics by
different routes -- and the third to within its error bars. Having the analytic
and library curves means the sampler is being *validated* rather than merely
summarised.

One honest gap: the sampler conserves the particle number, so it is canonical,
while the Fermi-Dirac form is the grand canonical result. They differ at
O(1/N), and at a dozen particles that is visible.

Run with `python examples/fermi_dirac_mc.py`.
Units: k_B = 1.
"""

import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from eigora.statphys import (
    FERMI,
    Canonical,
    GrandCanonical,
    IdenticalParticles,
    NLevel,
    equilibrium,
)
from eigora.statphys.monte_carlo import FermionGas, metropolis

QUANTA = 3          # nx, ny, nz each run 1..QUANTA
PARTICLES = 12
TEMPERATURES = (4.0, 8.0, 16.0)
SAMPLES = 4000
SEED = 20260826


def box_orbitals():
    """Every `(nx, ny, nz)` up to `QUANTA`, doubled for spin."""
    energies = [
        float(x * x + y * y + z * z)
        for x in range(1, QUANTA + 1)
        for y in range(1, QUANTA + 1)
        for z in range(1, QUANTA + 1)
        for _ in (0, 1)              # spin up and down: two orbitals, one energy
    ]
    return np.array(sorted(energies))


ORBITALS = box_orbitals()


def level_system():
    """The same orbitals as a spectral system, for the exact route."""
    values, counts = np.unique(ORBITALS, return_counts=True)
    return NLevel([float(v) for v in values], [int(c) for c in counts])


def analytic_chemical_potential(temperature):
    """
    Bisect `sum_k 1/(exp((eps_k - mu)/T) + 1) = N` by hand.

    Deliberately uses nothing from the library -- this is the calculation you
    would write on paper, and it exists so the library has something
    independent to be checked against.
    """
    def occupied(mu):
        return float(np.sum(fermi_dirac(ORBITALS, mu, temperature)))

    low = float(ORBITALS.min()) - 60.0 * temperature
    high = float(ORBITALS.max()) + 60.0 * temperature
    for _ in range(200):
        middle = 0.5 * (low + high)
        if occupied(middle) < PARTICLES:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def library_gas():
    return IdenticalParticles(level_system(), FERMI)


def library_chemical_potential(temperature):
    """The same mu, inverted from `<N>` by the library."""
    seed = equilibrium(library_gas(), GrandCanonical(temperature, 0.0))
    return seed.chemical_potential_for(float(PARTICLES))


def sample_occupations(temperature, rng):
    """Mean occupation of every orbital, and its standard error."""
    gas = FermionGas(ORBITALS, PARTICLES, rng=rng)
    sweep = ORBITALS.size
    metropolis(gas, Canonical(temperature), 20 * sweep, rng=rng)   # burn in
    history = np.empty((SAMPLES, ORBITALS.size))
    for index in range(SAMPLES):
        metropolis(gas, Canonical(temperature), sweep, rng=rng)
        history[index] = gas.occupations()
    return history.mean(axis=0), history.std(axis=0, ddof=1) / math.sqrt(SAMPLES)


def fermi_dirac(energy, chemical_potential, temperature):
    return 1.0 / (np.exp((energy - chemical_potential) / temperature) + 1.0)


def fit_chemical_potential(energies, occupations, temperature):
    """The old way: least squares for mu against the measured occupations."""
    guess = float(np.interp(0.5, occupations[::-1], energies[::-1]))
    fitted, _ = curve_fit(
        lambda e, mu: fermi_dirac(e, mu, temperature),
        energies,
        occupations,
        p0=[guess],
    )
    return float(fitted[0])


def main():
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    grid = np.linspace(ORBITALS.min() - 1.0, ORBITALS.max() + 1.0, 300)
    levels = np.unique(ORBITALS)
    gas = library_gas()
    summary = []

    ax = axes[0]
    for index, temperature in enumerate(TEMPERATURES):
        measured, errors = sample_occupations(temperature, rng)
        analytic = analytic_chemical_potential(temperature)
        library = library_chemical_potential(temperature)
        fitted = fit_chemical_potential(ORBITALS, measured, temperature)
        summary.append((temperature, analytic, library, fitted))

        colour = f"C{index}"
        ax.plot(grid, fermi_dirac(grid, analytic, temperature), lw=5,
                alpha=0.28, color=colour)
        ax.plot(levels,
                [gas.occupation(e, 1.0 / temperature, library) for e in levels],
                "^", ms=7, mfc="none", mew=1.4, color="k", alpha=0.75)
        ax.errorbar(ORBITALS, measured, yerr=errors, fmt="o", ms=4, mfc="none",
                    mew=1.1, capsize=2, color=colour,
                    label=f"$T = {temperature:g}$")

    ax.plot([], [], lw=5, alpha=0.28, color="grey", label="analytic (by hand)")
    ax.plot([], [], "^", ms=7, mfc="none", mew=1.4, color="k", alpha=0.75,
            label="library, exact")
    ax.plot([], [], "o", ms=4, mfc="none", mew=1.1, color="grey",
            label="Monte Carlo")
    ax.set(xlabel=r"orbital energy $\varepsilon$",
           ylabel=r"$\langle n \rangle$", ylim=(-0.05, 1.08),
           title=f"{PARTICLES} fermions in a box, {ORBITALS.size} orbitals")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.15)

    ax = axes[1]
    fine = np.linspace(min(TEMPERATURES) * 0.8, max(TEMPERATURES) * 1.15, 40)
    ax.plot(fine, [analytic_chemical_potential(t) for t in fine], lw=5,
            alpha=0.28, color="k", label="analytic (by hand)")
    ax.plot([t for t, *_ in summary], [m for _, _, m, _ in summary], "^",
            ms=9, mfc="none", mew=1.6, color="C0", label="library, exact")
    ax.plot([t for t, *_ in summary], [m for *_, m in summary], "o", ms=9,
            mfc="none", mew=1.6, color="C3", label="fitted to the MC")
    for temperature, analytic, library, fitted in summary:
        ax.annotate(
            f"library {abs(library - analytic):.2e}\nMC {abs(fitted - analytic):.3f}",
            xy=(temperature, fitted), xytext=(7, -22),
            textcoords="offset points", fontsize=7.5, color="grey",
        )
    ax.set(xlabel="$T$", ylabel=r"$\mu$",
           title="Chemical potential, three ways (deviation from analytic)")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.15)

    fig.suptitle(
        "Fermi-Dirac occupations: analytic, library, and Metropolis",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=130)
    print(f"wrote {output}")

    print(f"{'T':>6} {'analytic':>10} {'library':>10} {'MC fit':>10}"
          f" {'lib gap':>10} {'MC gap':>9}")
    for temperature, analytic, library, fitted in summary:
        print(f"{temperature:>6.1f} {analytic:>10.5f} {library:>10.5f}"
              f" {fitted:>10.5f} {abs(library - analytic):>10.2e}"
              f" {abs(fitted - analytic):>9.4f}")


if __name__ == "__main__":
    main()
