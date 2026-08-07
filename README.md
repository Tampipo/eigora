# eigora

Quantum mechanics simulation package for the [Eigora](https://eigora.tampipo.fr) platform.

Solves the 1D Schrödinger equation numerically, computes eigenstates, and evolves wavepackets in time. Potentials and their spectra generalise to any number of dimensions when the system is separable. Designed to be used as a pure Python library — no web dependencies.

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

## Quick start

```python
from eigora.grids import GridND
from eigora.qm import QuantumSystem1D
from eigora.qm.potentials import HarmonicWell, RectangularBarrier
from eigora.qm.states import GaussianWavepacket

# Define a grid and a system
grid = GridND.line(-8.0, 8.0, 512)
system = QuantumSystem1D(grid=grid, potential=HarmonicWell(omega=1.0))

# Solve for the 5 lowest eigenstates
solution = system.solve(n_states=5)
print(solution.energies)  # [0.5, 1.5, 2.5, 3.5, 4.5]

# Evolve a wavepacket over a barrier
grid2 = GridND.line(-20.0, 20.0, 1024)
system2 = QuantumSystem1D(grid=grid2, potential=RectangularBarrier(height=2.0, width=2.0))
state = GaussianWavepacket(x0=-8.0, k0=1.5, sigma=1.5)
evolution = system2.evolve(state, t_max=10.0, dt=0.005, n_frames=80)
```

---

## Structure

```
src/eigora.qm/
  potentials/
    known.py        # V(x) catalogue — well, barrier, harmonic, etc.
    base.py         # PotentialND — any dimension
    generic.py      # GenericPotential — arbitrary callable, no structure
    separable.py    # SeparablePotential — sum over disjoint coordinate blocks
  spectra/
    base.py         # Spectrum interface, EnergyLevel
    exact.py        # Analytic solutions + registry
    numerical.py    # Eigensolver-backed spectrum
    separable.py    # Product of block spectra
    factory.py      # spectrum_for — potential → spectrum
  solvers/
    eigensolver.py  # Finite difference Hamiltonian + sparse eigensolver
  states/
    wavepacket.py   # Initial states (GaussianWavepacket)
    orbitals.py     # Hydrogen-like atomic orbitals (SingleAtomState)
  evolution/
    split_step.py   # Split-step Fourier time evolution
  scattering.py     # Momentum density, energy-averaged transmission
  observables.py    # ⟨x⟩, ⟨p⟩, Δx·Δp, norm
  system.py         # QuantumSystem — high-level facade
```

Each sub-package re-exports its public names, so `from eigora.qm.potentials
import HarmonicWell` and `from eigora.qm.evolution import evolve` work
without reaching into the concrete module.

---

## Potentials

| Class | Description | Key parameters |
|---|---|---|
| `FreeParticle` | V(x) = 0 | — |
| `HarmonicWell` | V(x) = ½ω²(x−x₀)² | `omega`, `x0` |
| `InfiniteSquareWell` | V=0 inside, ∞ outside | `width`, `x0` |
| `FiniteSquareWell` | V=−depth inside, 0 outside | `depth`, `width`, `x0` |
| `RectangularBarrier` | V=height inside, 0 outside | `height`, `width`, `x0` |
| `PotentialStep` | V=0 before, height after | `height`, `x0` |
| `DoubleWell` | V = ax⁴ − bx² | `a`, `b` |

Potentials can be combined with `+` :

```python
combined = HarmonicWell(omega=1.0) + RectangularBarrier(height=1.0, width=0.5)
```

---

## Potentials in any dimension

Two ways to build one. `GenericPotential` wraps an arbitrary callable and carries no structure, so it can only ever be evaluated:

```python
from eigora.qm.potentials import GenericPotential

V = GenericPotential(lambda x, y: 0.5 * (x**2 + y**2) + 0.1 * x * y, ndim=2)
V.on_grid(GridND.cube(5.0, 128, ndim=2))     # shape (128, 128)
```

`SeparablePotential` is the interesting one: a sum of sub-system potentials over **disjoint coordinate blocks**, each of its own dimension — a 1D well, a 3D central potential, one of several non-interacting particles. Coordinates are split contiguously in block order:

```python
from eigora.qm.potentials import SeparablePotential, HarmonicWell

trap = SeparablePotential([HarmonicWell(omega=1.0)] * 3)          # 3D isotropic trap
pair = SeparablePotential([one_particle, one_particle],           # two non-interacting
                          names=["a", "b"])                       # 3D particles → ndim 6
```

The blocks are kept as objects rather than collapsed into a callable, because that structure is what makes the solution composable.

---

## Spectra

Every solution — analytic or numerical — implements the same `Spectrum` interface, so blocks compose regardless of how each was solved. `spectrum_for` is the entry point:

```python
from eigora.qm.spectra import spectrum_for

sol = spectrum_for(trap)
sol.is_exact                 # True — every block had an analytic solution
sol.quantum_numbers          # ('n_1', 'n_2', 'n_3')
sol.energy((0, 0, 0))        # 1.5
sol.wavefunction((1, 0, 2))  # callable(X, Y, Z), the product of block states
sol.label_text((1, 0, 2))    # 'n_1=1, n_2=0, n_3=2'
```

Energies add, wavefunctions multiply, quantum numbers concatenate — so degeneracies come out of the enumeration:

```python
[(lvl.energy, lvl.degeneracy) for lvl in sol.levels(4)]
# [(1.5, 1), (2.5, 3), (3.5, 6), (4.5, 10)]   — the (n+1)(n+2)/2 tower
```

A block whose potential has no analytic solution is solved numerically on a grid, behind the same interface. `is_exact` then reports what you actually got:

```python
mixed = SeparablePotential([HarmonicWell(omega=1.0), DoubleWell(a=1.0, b=4.0)],
                           names=["osc", "well"])
sol = spectrum_for(mixed, grid=GridND.line(-8.0, 8.0, 512), n_states=6)
sol.is_exact                 # False
sol.quantum_numbers          # ('n_osc', 'n_well')
sol.energies(5)              # still ascending, still summed block by block
```

| | analytic | notes |
|---|---|---|
| `HarmonicWell` | ✅ `HarmonicSpectrum` | Eₙ = ω(n+½), n ≥ 0 |
| `InfiniteSquareWell` | ✅ `BoxSpectrum` | Eₙ = n²π²/2L², n ≥ 1 |
| anything else 1D | numerical | needs a grid; `FiniteSquareWell` is transcendental |
| anything else N-D | ✗ | build it as a `SeparablePotential` of solvable blocks |

> Only `spectrum` generalises past 1D. `solve` and `evolve` remain 1D numerical methods and raise on a higher-dimensional system.

---

## Eigenstates

The Hamiltonian H = −½ d²/dx² + V(x) is discretised as a sparse tridiagonal matrix via finite differences. The lowest `n_states` eigenpairs are computed using `scipy.sparse.linalg.eigsh` in shift-invert mode, which ensures robust convergence for deep potentials.

```python
solution = system.solve(n_states=6)

solution.energies          # shape (n_states,)
solution.wavefunctions     # shape (n_states, n_points)
solution.ground_state      # psi_0(x)
solution.ground_energy     # E_0
solution.probability_density(n)  # |psi_n(x)|²
```

---

## Time evolution

Uses the **split-step Fourier method** (Strang splitting):

```
ψ(x, t+dt) ≈ e^{-iV dt/2} · IFFT[ e^{-ik²dt/2} · FFT[e^{-iV dt/2} ψ] ]
```

O(N log N) per timestep, unconditionally unitary, works for any potential.

```python
evolution = system.evolve(
    initial_state=GaussianWavepacket(x0=-5.0, k0=2.0, sigma=1.0),
    t_max=10.0,
    dt=0.005,
    n_frames=100,
)

evolution.psi                        # shape (n_frames, n_points), complex
evolution.times                      # shape (n_frames,)
evolution.probability_density(i)     # |ψ(x, tᵢ)|²
evolution.norm(i)                    # should remain ≈ 1.0
```

---

## Observables

```python
from eigora.qm import observables

observables.expectation_x(psi, grid)     # ⟨x⟩
observables.expectation_p(psi, grid)     # ⟨p⟩
observables.uncertainty_x(psi, grid)    # Δx
observables.uncertainty_p(psi, grid)    # Δp
observables.heisenberg_product(psi, grid)  # Δx·Δp ≥ 0.5
observables.norm(psi, grid)             # ‖ψ‖²
```

---

## Running tests

```bash
pytest
```

---

## Dependencies

- `numpy >= 2.0`
- `scipy >= 1.13`
- `eigora`
