# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Watching a 2D Ising lattice order itself.

Three lattices, identical apart from temperature, started from the same random
configuration and swept with Metropolis. The exact critical temperature of the
square-lattice Ising model is `T_c = 2 / log(1 + sqrt(2)) = 2.2692`, and the
three columns sit below it, on it, and above it.

What to look for:

  * **below `T_c`** -- domains coarsen. Small islands are eaten by larger ones
    and the lattice drifts towards one sign, because a domain wall costs
    energy and there is not enough temperature to pay for it.
  * **at `T_c`** -- structure at every size at once. No characteristic domain
    scale, which is what a diverging correlation length looks like, and the
    magnetisation wanders instead of settling.
  * **above `T_c`** -- noise. Correlations die within a couple of sites and
    the magnetisation stays near zero.

Nothing here is special-cased for two dimensions: `IsingLattice` takes any
shape, and the sampler is the same one the tests check against exact
enumeration on eight sites.

Run with `python examples/ising_evolution.py`; writes `ising_evolution.gif`
and a final still, `ising_evolution.png`, next to itself.

Units: k_B = 1, so temperature is an energy.
"""

import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from eigora.statphys import Canonical
from eigora.statphys.monte_carlo import IsingLattice, metropolis

SIDE = 48
CRITICAL = 2.0 / math.log(1.0 + math.sqrt(2.0))
TEMPERATURES = (0.70 * CRITICAL, CRITICAL, 1.55 * CRITICAL)
LABELS = ("below $T_c$", "at $T_c$", "above $T_c$")
FRAMES = 90
SWEEPS_PER_FRAME = 2
SEED = 20260826


def build():
    """Three lattices from the *same* random start, so only T differs."""
    lattices = []
    for _ in TEMPERATURES:
        lattice = IsingLattice(
            (SIDE, SIDE), coupling=1.0, rng=np.random.default_rng(SEED)
        )
        lattices.append(lattice)
    return lattices


def onsager_magnetisation(temperature):
    """Onsager's exact spontaneous magnetisation, or 0 above `T_c`."""
    if temperature >= CRITICAL:
        return 0.0
    return (1.0 - math.sinh(2.0 / temperature) ** -4) ** 0.125


def main():
    lattices = build()
    rng = np.random.default_rng(SEED + 1)
    steps = SWEEPS_PER_FRAME * SIDE * SIDE
    history = [[] for _ in TEMPERATURES]

    fig = plt.figure(figsize=(12.6, 7.6))
    grid = fig.add_gridspec(2, 3, height_ratios=(2.5, 1.15), hspace=0.05,
                            top=0.90, bottom=0.09, left=0.07, right=0.98)
    images, titles = [], []
    for column, (temperature, label) in enumerate(zip(TEMPERATURES, LABELS)):
        ax = fig.add_subplot(grid[0, column])
        images.append(
            ax.imshow(
                lattices[0].spins, cmap="RdBu", vmin=-1, vmax=1,
                interpolation="nearest",
            )
        )
        ax.set_xticks([])
        ax.set_yticks([])
        titles.append(ax.set_title(f"$T = {temperature:.2f}$   {label}", fontsize=10))

    trace = fig.add_subplot(grid[1, :])
    lines = [
        trace.plot([], [], lw=1.6, color=f"C{index}",
                   label=f"$T = {t:.2f}$")[0]
        for index, t in enumerate(TEMPERATURES)
    ]
    critical_energy = -math.sqrt(2.0)
    trace.axhline(critical_energy, ls=":", lw=1.2, color="C1")
    trace.text(
        3, critical_energy + 0.06,
        f"Onsager $E/N = -\\sqrt{{2}} = {critical_energy:.3f}$ at $T_c$",
        fontsize=8, color="C1",
    )
    trace.set(
        xlim=(0, FRAMES * SWEEPS_PER_FRAME), ylim=(-2.05, 0.05),
        xlabel="Metropolis sweeps", ylabel="$E/N$",
    )
    trace.grid(alpha=0.15)
    trace.legend(fontsize=8, loc="lower right", ncol=3)

    banner = fig.suptitle("", fontsize=12, y=0.965)

    def draw(frame):
        for index, (lattice, temperature) in enumerate(zip(lattices, TEMPERATURES)):
            if frame:
                metropolis(lattice, Canonical(temperature), steps, rng=rng)
            images[index].set_data(lattice.spins)
            magnet = lattice.extensive()["magnetisation"] / lattice.sites
            history[index].append(lattice.energy / lattice.sites)
            lines[index].set_data(
                np.arange(len(history[index])) * SWEEPS_PER_FRAME, history[index]
            )
            titles[index].set_text(
                f"$T = {temperature:.2f}$   {LABELS[index]}\n"
                f"$m = {magnet:+.3f}$,  $E/N = {lattice.energy / lattice.sites:+.3f}$"
            )
        banner.set_text(
            f"{SIDE}x{SIDE} Ising, Metropolis -- sweep "
            f"{frame * SWEEPS_PER_FRAME}"
        )
        return images + titles + lines + [banner]

    animation = FuncAnimation(fig, draw, frames=FRAMES, interval=90, blit=False)
    output = pathlib.Path(__file__).with_suffix(".gif")
    animation.save(output, writer=PillowWriter(fps=11))
    print(f"wrote {output}")

    fig.savefig(pathlib.Path(__file__).with_suffix(".png"), dpi=120)
    print(f"wrote {pathlib.Path(__file__).with_suffix('.png')}")

    for lattice, temperature, label in zip(lattices, TEMPERATURES, LABELS):
        magnet = abs(lattice.extensive()["magnetisation"]) / lattice.sites
        print(
            f"  T = {temperature:5.2f} ({label:>12}):  |m| = {magnet:.3f}  "
            f"(Onsager {onsager_magnetisation(temperature):.3f}),  "
            f"E/N = {lattice.energy / lattice.sites:+.3f}"
        )
    print(
        "\nBelow T_c the lattice is still coarsening: two domains of opposite\n"
        "sign nearly cancel, so |m| is far below Onsager's equilibrium value.\n"
        "Domains grow like sqrt(sweeps), so spanning a 48x48 lattice takes\n"
        "thousands -- the slowness is the physics, not the sampler."
    )


if __name__ == "__main__":
    main()
