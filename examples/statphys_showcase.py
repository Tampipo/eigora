# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
A tour of `eigora.statphys`, plotted against the closed forms it should match.

Every panel puts a curve computed by the library next to an analytic result
derived independently of it, so the figure is a check and not only a picture.
Where no closed form exists (the rigid rotor) the asymptotic limit is drawn
instead.

Run with `python examples/statphys_showcase.py`; it writes
`statphys_showcase.png` next to itself.

Units: k_B = 1, so temperature is an energy and beta = 1/T.
"""

import itertools
import math
import pathlib
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np

from eigora.statphys import (
    Canonical,
    Degenerate,
    Field,
    Generalised,
    HarmonicMode,
    Level,
    Magnetic,
    NLevel,
    ParametrisedSystem,
    Rotor,
    SpectralSystem,
    Spin,
    TwoLevel,
    equilibrium,
)


def panel_schottky(ax):
    """
    Two-level heat capacity against the closed form.

    The bump is the Schottky anomaly: a gapped system has nowhere to put
    energy at low T and is saturated at high T, so C has to peak in between.
    """
    splitting = 1.0
    system = TwoLevel(splitting=splitting)
    temperatures = np.linspace(0.05, 3.0, 400)

    computed = [
        equilibrium(system, Canonical(t)).heat_capacity for t in temperatures
    ]
    x = splitting / temperatures
    exact = x**2 * np.exp(x) / (np.exp(x) + 1.0) ** 2

    ax.plot(temperatures, exact, lw=4, alpha=0.3, color="k", label="closed form")
    ax.plot(temperatures, computed, lw=1.5, color="C3", label="statphys")
    ax.axhline(0.439229, ls=":", lw=1, color="grey")
    ax.axvline(0.416766, ls=":", lw=1, color="grey")
    ax.annotate(
        "$C_{max}=0.4392$\nat $T=0.4168\\,\\Delta$",
        xy=(0.417, 0.439),
        xytext=(1.05, 0.33),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="grey"),
    )
    ax.set(xlabel="$T/\\Delta$", ylabel="$C$", title="Schottky anomaly")
    ax.legend(fontsize=8)


def panel_harmonic(ax):
    """
    Harmonic mode: the closed form against the truncated level sum.

    `HarmonicMode` overrides `moments` with `-log(2 sinh(beta w/2))` and
    friends, but keeps `levels()` implemented, so the infinite tower can still
    be summed term by term. The two must agree -- and they do to ~1e-14, which
    is why the residual is on its own axis.
    """
    system = HarmonicMode(omega=1.0)
    temperatures = np.logspace(-1, 1.5, 300)

    closed = np.array([system.mean_energy(1.0 / t) for t in temperatures])
    summed = np.array(
        [SpectralSystem.moments(system, 1.0 / t).energy for t in temperatures]
    )

    ax.plot(temperatures, closed, lw=4, alpha=0.3, color="k", label="closed form")
    ax.plot(temperatures, summed, lw=1.5, color="C0", label="level sum")
    ax.plot(temperatures, temperatures, ls="--", lw=1, color="C1",
            label="classical $\\langle E\\rangle=T$")
    ax.axhline(0.5, ls=":", lw=1, color="grey")
    ax.text(0.11, 0.56, "zero point $\\omega/2$", fontsize=8, color="grey")

    residual = ax.twinx()
    residual.semilogy(
        temperatures, np.abs(closed - summed) + 1e-18, lw=0.8, color="C4", alpha=0.7
    )
    residual.set_ylabel("|difference|", color="C4", fontsize=8)
    residual.tick_params(axis="y", labelcolor="C4", labelsize=7)
    residual.set_ylim(1e-18, 1e-6)

    ax.set(xscale="log", xlabel="$T/\\omega$", ylabel="$\\langle E\\rangle$",
           title="Harmonic mode: two routes")
    ax.legend(fontsize=8, loc="upper left")


def panel_brillouin(ax):
    """
    Magnetisation against field for several spins -- the Brillouin functions.

    `Spin` carries no Zeeman energy: its levels sit at E = 0 and report a
    magnetisation. The field lives in the ensemble, so sweeping h never
    rebuilds the spectrum, and <M> comes out of the same sweep as log Z.
    """
    fields = np.linspace(-6.0, 6.0, 300)
    for index, j in enumerate((0.5, 1.5, 2.5)):
        system = Spin(j)
        computed = [
            equilibrium(system, Magnetic(temperature=1.0, magnetic_field=h)).magnetisation
            for h in fields
        ]
        exact = [_brillouin(j, h) for h in fields]
        ax.plot(fields, exact, lw=4, alpha=0.25, color="k")
        ax.plot(fields, computed, lw=1.5, color=f"C{index}", label=f"$j={j}$")
        ax.axhline(j, ls=":", lw=0.8, color=f"C{index}", alpha=0.5)

    ax.plot([], [], lw=4, alpha=0.25, color="k", label="Brillouin")
    ax.set(xlabel="$h/T$", ylabel="$\\langle M\\rangle$",
           title="Paramagnet: saturation")
    ax.legend(fontsize=8, loc="upper left")


def _brillouin(j, h, temperature=1.0):
    """(2j+1)/2 coth((2j+1)x/2) - 1/2 coth(x/2), with x = h/T."""
    x = h / temperature
    if abs(x) < 1e-9:
        return j * (j + 1.0) * x / 3.0
    a = 2.0 * j + 1.0
    return 0.5 * (a / math.tanh(0.5 * a * x) - 1.0 / math.tanh(0.5 * x))


def panel_fluctuation(ax):
    """
    Fluctuation-dissipation: chi = beta Var(M) against d<M>/dh.

    The library gets the susceptibility from the magnetisation *fluctuations*
    in a single sweep. The dots are a central difference of <M> in h -- a
    different computation entirely, and the one the identity claims to
    replace. Curie's law is the high-T asymptote.
    """
    temperatures = np.logspace(-0.7, 2.0, 200)
    system = Spin(0.5)

    fluctuation = [
        equilibrium(system, Magnetic(t, 0.0)).susceptibility for t in temperatures
    ]
    step = 1e-5
    derivative = [
        (
            equilibrium(system, Magnetic(t, step)).magnetisation
            - equilibrium(system, Magnetic(t, -step)).magnetisation
        )
        / (2.0 * step)
        for t in temperatures[::12]
    ]

    ax.loglog(temperatures, fluctuation, lw=2, color="C2",
              label="$\\beta\\,\\mathrm{Var}(M)$")
    ax.loglog(temperatures[::12], derivative, "o", ms=5, mfc="none", color="k",
              label="$d\\langle M\\rangle/dh$")
    ax.loglog(temperatures, 1.0 / (4.0 * temperatures), ls="--", lw=1, color="C1",
              label="Curie $1/4T$")
    ax.set(xlabel="$T$", ylabel="$\\chi$", title="Fluctuation-dissipation")
    ax.legend(fontsize=8)


def panel_entropy(ax):
    """
    Entropy from T = 0 to T = infinity, for four different spectra.

    Two limits are being checked at once. At low T, S -> log(ground
    degeneracy) -- the third law, and the reason the degenerate system does
    not go to zero. At high T every state is equally likely, so S -> log(total
    number of states).
    """
    systems = {
        "TwoLevel(1)": (TwoLevel(1.0), 1, 2),
        "Degenerate(3)": (Degenerate(3), 3, 3),
        "NLevel, $g_0=2$": (NLevel([0.0, 1.0, 2.0], [2, 3, 4]), 2, 9),
        "Spin(3/2)": (Spin(1.5), 4, 4),
    }
    temperatures = np.logspace(-2, 2.5, 300)

    for index, (label, (system, g0, total)) in enumerate(systems.items()):
        entropy = [equilibrium(system, Canonical(t)).entropy for t in temperatures]
        ax.semilogx(temperatures, entropy, lw=1.5, color=f"C{index}", label=label)
        ax.axhline(math.log(g0), ls=":", lw=0.8, color=f"C{index}", alpha=0.6)
        ax.axhline(math.log(total), ls="--", lw=0.8, color=f"C{index}", alpha=0.4)

    ax.set(xlabel="$T$", ylabel="$S$",
           title="Third law ($\\log g_0$) to saturation ($\\log \\Omega$)")
    ax.legend(fontsize=7, loc="upper left")


def panel_composition(ax):
    """
    Composition: `A * B` and `A ** n` add log Z, so everything is extensive.

    A composite has no level stream at all -- its moments are added over the
    blocks -- so `HarmonicMode ** 3000` costs 3000 closed forms rather than a
    sum over an unimaginable spectrum. The dashed line is the single-copy
    result scaled by n.

    The block is a spin times a two-level, and it exercises the coupling
    routing: a `Spin` has no energy at all -- its levels sit at E = 0 and
    carry only a magnetisation -- while a `TwoLevel` has energy and no
    magnetisation. The composite reports both, each block answering only for
    what its own microstates carry and contributing zero to the rest.
    """
    counts = np.arange(1, 41)
    block = Spin(1.5) * TwoLevel(1.0)
    ensemble = Magnetic(0.7, 0.6)
    states = [equilibrium(block ** int(n), ensemble) for n in counts]

    for index, (name, series) in enumerate(
        [
            ("$S$", [state.entropy for state in states]),
            ("$\\langle M\\rangle$", [state.magnetisation for state in states]),
            ("$C$", [state.heat_capacity for state in states]),
        ]
    ):
        ax.plot(counts, series, "o", ms=3, color=f"C{index}", label=name)
        ax.plot(counts, counts * series[0], ls="--", lw=1, color=f"C{index}",
                alpha=0.6)

    ax.set(xlabel="number of copies $n$", ylabel="extensive quantity",
           title="Extensivity of $(\\mathrm{Spin}\\times\\mathrm{TwoLevel})^n$")
    ax.legend(fontsize=8, loc="upper left")


def panel_rotor(ax):
    """
    Rigid rotor: the one system in the catalogue with no closed form.

    Its levels are b l(l+1) with degeneracy 2l+1, so the partition function
    has to be summed -- this panel is the honest exercise of the truncation
    machinery. At high T the sum tends to the classical equipartition result
    C -> 1; at low T the gap 2b freezes it out exponentially.
    """
    temperatures = np.logspace(-1, 1.7, 300)
    for index, b in enumerate((0.5, 1.0, 2.0)):
        system = Rotor(b)
        heat = [equilibrium(system, Canonical(t)).heat_capacity for t in temperatures]
        ax.semilogx(temperatures, heat, lw=1.5, color=f"C{index}", label=f"$b={b}$")

    ax.axhline(1.0, ls="--", lw=1, color="k", alpha=0.5,
               label="classical $C=1$")
    ax.set(xlabel="$T$", ylabel="$C$", title="Rigid rotor (summed, no closed form)")
    ax.legend(fontsize=8, loc="upper left")


# -- a user-defined system: fermions in a trap --------------------------------


ORBITALS = 6
LADDER = [k + 0.5 for k in range(ORBITALS)]


class OpenTrap(SpectralSystem):
    """
    Every occupation of the ladder; the particle number is free.

    Note what is *not* written here. Putting `{"particles": n}` on the levels
    is the whole of it: `extensive_variables` is read off the first level, so
    this system announces itself as usable in any ensemble freeing at most the
    particle number, and `equilibrium` checks that before summing anything.
    Omit the key and a grand canonical ensemble is refused by name rather than
    silently averaging zero.
    """

    @property
    def is_exact(self):
        return True

    @property
    def n_states(self):
        return 2**ORBITALS

    def levels(self):
        states = sorted(
            (sum(n * e for n, e in zip(occ, LADDER)), sum(occ))
            for occ in itertools.product((0, 1), repeat=ORBITALS)
        )
        for energy, particles in states:
            yield Level(energy, 1, {"particles": float(particles)})


@dataclass(frozen=True)
class ClosedTrap(SpectralSystem, ParametrisedSystem):
    """
    Only the occupations holding exactly `particles` fermions.

    Spectral *and* parametrised -- the two are independent refinements of
    `System`, so a system may be either, both or neither. Its levels carry no
    extensive variable at all, because every microstate here has the same
    particle number: N is what defines the system rather than something its
    microstates vary, which is what makes it a parameter and not a coupling.
    """

    particles: int

    @property
    def is_exact(self):
        return True

    @property
    def n_states(self):
        return math.comb(ORBITALS, self.particles)

    @property
    def parameters(self):
        return {"particles": float(self.particles)}

    def at(self, **changes):
        return replace(self, **changes)

    def levels(self):
        for energy in sorted(
            sum(LADDER[k] for k in occupied)
            for occupied in itertools.combinations(range(ORBITALS), self.particles)
        ):
            yield Level(energy)


def _grand(temperature, mu):
    return Generalised(temperature, [Field("particles", mu, +1, "mu")])


def panel_fermi(ax):
    """
    A user-defined system: <N> against mu, at three temperatures.

    Nothing here is in the library -- `OpenTrap` is 15 lines in this file,
    supplying `levels()` that report a particle number. The grand canonical
    ensemble is one `Field`. The staircase is the T -> 0 limit, where <N>
    jumps by one each time mu crosses an orbital energy.
    """
    system = OpenTrap()
    mus = np.linspace(-1.0, 7.0, 400)

    for index, temperature in enumerate((0.08, 0.4, 1.5)):
        computed = [
            equilibrium(system, _grand(temperature, mu)).mean("particles") for mu in mus
        ]
        beta = 1.0 / temperature
        exact = [
            sum(1.0 / (math.exp(beta * (e - mu)) + 1.0) for e in LADDER) for mu in mus
        ]
        ax.plot(mus, exact, lw=4, alpha=0.25, color="k")
        ax.plot(mus, computed, lw=1.5, color=f"C{index}", label=f"$T={temperature}$")

    for energy in LADDER:
        ax.axvline(energy, ls=":", lw=0.7, color="grey", alpha=0.6)
    ax.plot([], [], lw=4, alpha=0.25, color="k", label="Fermi-Dirac sum")
    ax.set(xlabel="$\\mu$", ylabel="$\\langle N\\rangle$",
           title="Fermions in a trap (user-defined)")
    ax.legend(fontsize=8, loc="upper left")


def panel_chemical_potential(ax):
    """
    The same mu, as a coupling and as a parameter.

    Coupling route: N is free, solve <N> = n for mu (`field_for`). Parameter
    route: N is fixed, build both systems and take mu = F(n) - F(n-1). They
    are different questions -- one fixes the mean, the other the value -- and
    agree only in the thermodynamic limit, so the gap here at small n is
    physics, not error.

    The coupling route stops one short of a full ladder: <N> reaches 6 out of
    6 orbitals only as mu -> infinity, so `field_for` refuses it rather than
    returning a large finite number. The parameter route has no such trouble,
    since a full shell is just another system.
    """
    temperature = 0.8
    counts = range(1, ORBITALS)

    coupling_route = [
        equilibrium(OpenTrap(), _grand(temperature, 0.0)).field_for(
            "particles", float(n)
        )
        for n in counts
    ]
    free_energy = [
        equilibrium(ClosedTrap(particles=n), Canonical(temperature)).free_energy
        for n in range(0, ORBITALS + 1)
    ]
    parameter_route = [free_energy[n] - free_energy[n - 1] for n in counts]

    ax.plot(list(counts), coupling_route, "o-", ms=5, color="C0",
            label="coupling: $\\langle N\\rangle = n$")
    ax.plot(list(counts), parameter_route, "s--", ms=5, color="C3",
            label="parameter: $F(n)-F(n{-}1)$")
    for energy in LADDER:
        ax.axhline(energy, ls=":", lw=0.7, color="grey", alpha=0.6)
    ax.set(xlabel="$n$", ylabel="$\\mu$",
           title="Chemical potential, two routes")
    ax.legend(fontsize=8, loc="upper left")


PANELS = [
    panel_schottky,
    panel_harmonic,
    panel_brillouin,
    panel_fluctuation,
    panel_entropy,
    panel_composition,
    panel_rotor,
    panel_fermi,
    panel_chemical_potential,
]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for panel, ax in zip(PANELS, axes.flat):
        panel(ax)
        ax.grid(alpha=0.15)
    fig.suptitle(
        "eigora.statphys -- library curves against independent closed forms",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output = pathlib.Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=130)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
