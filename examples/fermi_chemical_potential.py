# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Chemical potential of a free-fermion gas, sampled against its closed forms.

A reworking of *Simulation de la distribution de Fermi-Dirac* (Hemmen &
Marsault) on top of `eigora.statphys`. Electrons in a periodic box, states
labelled `(nx, ny, nz, s)` with

    E = (hbar^2 / 2m) (2 pi)^2 [ (nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2 ]

are handed to `FermionGas` and sampled with Metropolis. The mean occupation of
each orbital is measured, averaged over states of equal energy, and the
chemical potential is read off by least-squares fitting

    n(E) = 1 / (exp((E - mu)/k_B T) + 1)

with `T` known and `mu` free -- which is the paper's method, and the only one
available when the sampler is all you have.

Four panels, two sweeps, everything in three dimensions:

    A  distributions at several T          where mu(T) comes from
    B  mu(T)   Sommerfeld  mu = Ef [1 - (pi^2/12)(T/Tf)^2]
               Maxwell-Boltzmann  mu = (3/2) T log(4 pi Tf / (6pi^2)^(2/3) T)
    C  distributions at several N          where mu(N) comes from
    D  mu(N) at fixed T   Sommerfeld at Ef(N) = c N^(2/3)

The two limits in B bracket the crossover from between them: Sommerfeld is an
expansion in `(T/Tf)^2` and fails above `T ~ 0.4 Tf`, Maxwell-Boltzmann is the
classical limit and only takes over near `T ~ 2 Tf`. Nothing closed-form covers
the middle in three dimensions, which is the reason to simulate it.

Energies are quoted as temperatures throughout (`E/k_B`, in kelvin), which is
also the library's convention with `k_B = 1`.

Runs the sweep points in parallel, one process per point. Statistics are
deliberately modest -- error bars are visible and that is the honest picture.

Run with `python examples/fermi_chemical_potential.py`.
"""

import math
import pathlib
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from eigora.statphys import Canonical
from eigora.statphys.monte_carlo import FermionGas, metropolis

HBAR = 1.054571817e-34
ELECTRON_MASS = 9.1093837015e-31
BOLTZMANN = 1.380649e-23

BOX = 5.0e-9                 # cubic box edge, metres
PARTICLES = 100
SAMPLES = 2500               # occupation snapshots per point
BURN_SWEEPS = 300
CUTOFF = 6.0                 # keep states up to Ef + CUTOFF * T


def kelvin_scale(length):
    """The energy quantum `(hbar 2pi/L)^2 / 2m` of one axis, in kelvin."""
    return (HBAR**2 / (2.0 * ELECTRON_MASS * BOLTZMANN)) * (2.0 * math.pi / length) ** 2


def fermi_temperature(particles, lengths):
    """`Ef / k_B` for a three-dimensional box."""
    volume = lengths[0] * lengths[1] * lengths[2]
    return (HBAR**2 / (2.0 * ELECTRON_MASS * BOLTZMANN)) * (
        3.0 * math.pi**2 * particles / volume
    ) ** (2.0 / 3.0)


def orbital_energies(lengths, particles, temperature):
    """
    Every `(nx, ny, nz, s)` inside the cutoff, as a flat list of energies.

    The cutoff keeps states up to `Ef + CUTOFF * T`, with a floor that always
    leaves room for the Fermi sea plus a margin -- at low temperature the
    thermal criterion alone would barely hold the particles.
    """
    scales = [kelvin_scale(length) for length in lengths]
    fermi = fermi_temperature(particles, lengths)
    ceiling = fermi + CUTOFF * temperature
    limits = []
    for scale in scales:
        thermal = int(math.sqrt(max(ceiling, 0.0) / scale))
        room = int(math.ceil(math.sqrt(fermi / scale))) + 2
        limits.append(max(thermal, room) if scale < ceiling else 0)

    ranges = [np.arange(-limit, limit + 1) for limit in limits]
    grid = np.meshgrid(*ranges, indexing="ij")
    energies = sum(scale * axis**2 for scale, axis in zip(scales, grid)).ravel()
    return np.repeat(np.sort(energies), 2)        # two spin states per orbital


def fermi_dirac(energy, chemical_potential, temperature):
    return 1.0 / (np.exp((energy - chemical_potential) / temperature) + 1.0)


def simulate(point):
    """
    One (lengths, particles, temperature) point: sample, group, fit `mu`.

    Runs in its own process, so everything it needs travels in `point`.
    """
    label, lengths, particles, temperature, seed = point
    energies = orbital_energies(lengths, particles, temperature)
    rng = np.random.default_rng(seed)
    gas = FermionGas(energies, particles, rng=rng)
    ensemble = Canonical(temperature)

    metropolis(gas, ensemble, BURN_SWEEPS * particles, rng=rng)
    total = np.zeros(energies.size)
    squares = np.zeros(energies.size)
    for _ in range(SAMPLES):
        metropolis(gas, ensemble, particles, rng=rng)
        occupation = gas.occupations()
        total += occupation
        squares += occupation
    mean = total / SAMPLES
    error = np.sqrt(np.maximum(squares / SAMPLES - mean**2, 0.0) / SAMPLES)

    # Average over states of equal energy, as the paper does: the plot is a
    # function of energy, so degenerate orbitals have to be pooled.
    levels, inverse, counts = np.unique(
        np.round(energies, 6), return_inverse=True, return_counts=True
    )
    pooled = np.bincount(inverse, weights=mean) / counts
    pooled_error = np.sqrt(np.bincount(inverse, weights=error**2)) / counts

    fermi = fermi_temperature(particles, lengths)
    fitted, covariance = curve_fit(
        lambda e, mu: fermi_dirac(e, mu, temperature),
        levels,
        pooled,
        p0=[fermi],
        sigma=np.maximum(pooled_error, 1e-4),
        absolute_sigma=True,
    )
    return {
        "label": label,
        "temperature": temperature,
        "particles": particles,
        "fermi": fermi,
        "mu": float(fitted[0]),
        "mu_error": float(math.sqrt(covariance[0, 0])),
        "levels": levels,
        "occupation": pooled,
        "occupation_error": pooled_error,
    }


def sommerfeld(temperature, fermi):
    """Low-temperature expansion, valid for `T << Tf`."""
    return fermi * (1.0 - (math.pi**2 / 12.0) * (temperature / fermi) ** 2)


def maxwell_boltzmann(temperature, fermi):
    """The classical limit, valid for `T >> Tf`. The `6 pi^2` carries the spin."""
    return 1.5 * temperature * np.log(
        4.0 * math.pi * fermi / ((6.0 * math.pi**2) ** (2.0 / 3.0) * temperature)
    )


def main():
    cube = (BOX, BOX, BOX)
    reference = fermi_temperature(PARTICLES, cube)
    fixed = round(0.25 * reference)
    counts = (20, 40, 70, 100, 140, 190, 250, 320)

    sweep_t = [
        ("T", cube, PARTICLES, round(fraction * reference), 100 + index)
        for index, fraction in enumerate((0.10, 0.20, 0.35, 0.5, 0.7, 0.9,
                                          1.2, 1.6, 2.2, 3.0))
    ]
    sweep_n = [
        ("N", cube, count, fixed, 200 + index)
        for index, count in enumerate(counts)
    ]

    with Pool() as pool:
        results = pool.map(simulate, sweep_t + sweep_n)
    by_temperature = [r for r in results if r["label"] == "T"]
    by_number = [r for r in results if r["label"] == "N"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))

    # -- A: distributions at several temperatures -------------------------
    ax = axes[0, 0]
    for index, run in enumerate(by_temperature[::3]):
        colour = f"C{index}"
        ax.errorbar(run["levels"] / run["fermi"], run["occupation"],
                    yerr=run["occupation_error"], fmt="o", ms=3.5, mfc="none",
                    mew=1.0, capsize=1.5, color=colour, alpha=0.8,
                    label=f"$T/T_F = {run['temperature'] / run['fermi']:.2f}$")
        grid = np.linspace(0, run["levels"].max(), 300)
        ax.plot(grid / run["fermi"],
                fermi_dirac(grid, run["mu"], run["temperature"]),
                lw=1.6, color=colour)
        ax.axvline(run["mu"] / run["fermi"], ls=":", lw=1, color=colour, alpha=0.6)
    ax.set(xlabel="$E / E_F$", ylabel=r"$\langle n \rangle$",
           xlim=(0, 3.2), ylim=(-0.05, 1.08),
           title=r"A: distributions against $T$  ($N = 100$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    # -- B: mu(T) ----------------------------------------------------------
    ax = axes[0, 1]
    scaled = np.array([r["temperature"] / reference for r in by_temperature])
    ax.errorbar(scaled, [r["mu"] / reference for r in by_temperature],
                yerr=[r["mu_error"] / reference for r in by_temperature],
                fmt="o", ms=6, mfc="none", mew=1.5, capsize=3, color="C0",
                label="Monte Carlo")
    fine = np.linspace(0.05, 3.2, 300)
    ax.plot(fine, sommerfeld(fine * reference, reference) / reference, lw=4,
            alpha=0.3, color="k", label="Sommerfeld")
    hot = fine[fine > 0.8]
    ax.plot(hot, maxwell_boltzmann(hot * reference, reference) / reference,
            ls="--", lw=1.6, color="C3", label="Maxwell-Boltzmann")
    ax.axhline(0.0, lw=0.8, color="grey")
    ax.set(xlabel="$T / T_F$", ylabel=r"$\mu / E_F$", ylim=(-3.2, 1.4),
           title=f"B: $\\mu(T)$   ($T_F = {reference:.0f}$ K)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.15)

    # -- C: distributions at several particle numbers ---------------------
    ax = axes[1, 0]
    for index, run in enumerate(by_number[::2]):
        colour = f"C{index}"
        ax.errorbar(run["levels"], run["occupation"],
                    yerr=run["occupation_error"], fmt="o", ms=3.5, mfc="none",
                    mew=1.0, capsize=1.5, color=colour, alpha=0.8,
                    label=f"$N = {run['particles']}$")
        grid = np.linspace(0, run["levels"].max(), 300)
        ax.plot(grid, fermi_dirac(grid, run["mu"], run["temperature"]),
                lw=1.6, color=colour)
        ax.axvline(run["mu"], ls=":", lw=1, color=colour, alpha=0.6)
    ax.set(xlabel="$E$ (K)", ylabel=r"$\langle n \rangle$",
           ylim=(-0.05, 1.08), xlim=(0, 12_000),
           title=f"C: distributions against $N$  ($T = {fixed:.0f}$ K)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    # -- D: mu(N) ----------------------------------------------------------
    ax = axes[1, 1]
    numbers = np.array([r["particles"] for r in by_number])
    ax.errorbar(numbers, [r["mu"] for r in by_number],
                yerr=[r["mu_error"] for r in by_number], fmt="o", ms=6,
                mfc="none", mew=1.5, capsize=3, color="C0", label="Monte Carlo")
    span = np.linspace(numbers.min() * 0.8, numbers.max() * 1.1, 200)
    fermi_of_n = np.array([fermi_temperature(float(n), cube) for n in span])
    ax.plot(span, sommerfeld(fixed, fermi_of_n), lw=4, alpha=0.3, color="k",
            label=r"Sommerfeld at $E_F(N) = cN^{2/3}$")
    ax.plot(span, fermi_of_n, ls=":", lw=1.4, color="C2",
            label=r"$E_F(N)$, i.e. $T = 0$")
    ax.set(xlabel="$N$", ylabel=r"$\mu$ (K)",
           title=f"D: $\\mu(N)$ at $T = {fixed:.0f}$ K")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.15)

    fig.suptitle(
        "Chemical potential of a three-dimensional free-fermion gas: "
        "Metropolis against the closed forms",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    output = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=130)
    print(f"wrote {output}")

    print(f"mu(T) at N = {PARTICLES},  T_F = {reference:.0f} K")
    print(f"{'T/T_F':>7} {'mu/E_F':>10} {'+-':>8} {'Sommerfeld':>12} {'M-B':>10}")
    for run in by_temperature:
        ratio = run["temperature"] / reference
        print(f"{ratio:>7.2f} {run['mu'] / reference:>10.4f}"
              f" {run['mu_error'] / reference:>8.4f}"
              f" {sommerfeld(run['temperature'], reference) / reference:>12.4f}"
              f" {maxwell_boltzmann(run['temperature'], reference) / reference:>10.4f}")

    print(f"\nmu(N) at T = {fixed:.0f} K")
    print(f"{'N':>5} {'T/T_F(N)':>9} {'mu (K)':>10} {'+-':>7}"
          f" {'Sommerfeld':>12} {'E_F(N)':>10}")
    for run in by_number:
        expected = sommerfeld(fixed, run["fermi"])
        print(f"{run['particles']:>5} {fixed / run['fermi']:>9.3f}"
              f" {run['mu']:>10.1f} {run['mu_error']:>7.1f}"
              f" {expected:>12.1f} {run['fermi']:>10.1f}")


if __name__ == "__main__":
    main()
