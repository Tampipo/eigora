<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Eigora" src="assets/logo-light.svg" width="240">
  </picture>
</p>

# eigora

Physics simulation library for the [Eigora](https://eigora.tampipo.fr) platform.

Small enough to read end to end, which is the point — it exists to be understood, not to be complete. Pure Python, no web dependencies.

> **Units:** atomic units throughout (ℏ = m = 1).

---

## Installation

```bash
pip install git+https://github.com/Tampipo/eigora
```

For development:

```bash
git clone https://github.com/Tampipo/eigora
cd eigora
pip install -e ".[dev]"
```

---

## Two ways to do quantum mechanics

The package covers the same physics from both directions, and the contrast is deliberate.

**On a grid** — a particle in a potential, where the state is a wavefunction ψ(x) sampled on points and the Hamiltonian is a differential operator. Solved by finite differences and evolved by Trotter-splitting, so time evolution has to march in steps of `dt`.

```python
from eigora import GridND
from eigora.qm import QuantumSystem
from eigora.qm.potentials import HarmonicWell

system = QuantumSystem(grid=GridND.line(-8.0, 8.0, 512), potential=HarmonicWell(omega=1.0))
system.solve(n_states=5).energies      # [0.5, 1.5, 2.5, 3.5, 4.5]
```

**In a finite-dimensional Hilbert space** — a spin or a qubit, where the state is a vector and the Hamiltonian is a matrix. Nothing is discretised because nothing was continuous, so the Hamiltonian can simply be diagonalised, and evolution is exact at any time with no `dt` and no stability criterion.

```python
import numpy as np
from eigora.qm.discrete import Rabi, SZ, evolve

system = Rabi(detuning=0.0, rabi_frequency=1.0)
system.eigenenergies()                 # closed form, no diagonalisation

times = np.linspace(0, 12, 200)
evolve(system, np.array([1.0, 0.0]), times).expectation(SZ)   # cos(omega t)
```

📖 **[Continuous systems →](docs/continuous.md)** · **[Discrete systems →](docs/discrete.md)**

---

## Structure

```
src/eigora/
  grids.py                # GridND — uniform grids in any dimension
  fft.py                  # FFT helpers and frequency axes
  functions.py            # gaussian, sinc, heaviside
  constants.py            # physical constants
  spherical_harmonics.py  # Y_lm
  qm/
    potentials/  spectra/  solvers/  states/  evolution/
    observables.py  scattering.py  system.py
    discrete/             # finite-dimensional Hilbert spaces
```

Each sub-package re-exports its public names, so `from eigora.qm.potentials import HarmonicWell` and `from eigora.qm.discrete import Hamiltonian` work without reaching into the concrete module.

---

## Exact solutions, checked against numerics

Where a system has a closed-form solution, it is implemented directly and the numerical solver is held against it in the test suite. `HarmonicWell` knows its own spectrum; so does `TwoLevel`. Where no closed form exists — `FiniteSquareWell`, a Heisenberg chain past two sites — the same interface falls back to diagonalisation, and `is_exact` tells you which you got.

That agreement is a property the tests enforce, not a claim in the docs:

```python
from eigora.qm.discrete import TwoLevel
import numpy as np

h = TwoLevel(bias=1.0, coupling=2.0)
np.allclose(h.eigenenergies(), np.linalg.eigvalsh(h.matrix))   # True
```

---

## Scope

What this deliberately does **not** do, so you can tell early whether it fits:

- **Closed systems only.** No density matrices, no Lindblad dynamics, no open-system evolution.
- **Dense matrices.** A spin chain is 2ⁿ, so exact diagonalisation runs out of room around a dozen spins.
- **`solve` and `evolve` on a grid are 1D.** `spectrum` generalises to any dimension when the system separates into solvable blocks.
- **No symmetry sectors, no tensor networks, no GPU.**

If you need open systems or large-scale exact diagonalisation, use [QuTiP](https://qutip.org) or [QuSpin](https://quspin.github.io/QuSpin/) — they are better at it and always will be. This one is for seeing how the pieces work.

---

## Running tests

```bash
pytest
```

---

## Dependencies

- `numpy >= 2.0`
- `scipy >= 1.13`
