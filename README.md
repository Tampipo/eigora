# physense-qm

Quantum mechanics simulation package for the [Physense](https://physense.tampipo.fr) platform.

Solves the 1D Schrödinger equation numerically, computes eigenstates, and evolves wavepackets in time. Designed to be used as a pure Python library — no web dependencies.

> **Units:** atomic units throughout (ℏ = m = 1).

---

## Installation

```bash
pip install git+https://github.com/Tampipo/physense-qm
```

For development:

```bash
git clone https://github.com/Tampipo/physense-qm
cd physense-qm
pip install -e ".[dev]"
```

---

## Quick start

```python
from physense_utils.grids import GridND
from physense_qm import QuantumSystem1D
from physense_qm.potentials import HarmonicWell, RectangularBarrier
from physense_qm.states import GaussianWavepacket

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
src/physense_qm/
  potentials/
    known.py        # V(x) catalogue — well, barrier, harmonic, etc.
  solvers/
    eigensolver.py  # Finite difference Hamiltonian + sparse eigensolver
  states/
    wavepacket.py   # Initial states (GaussianWavepacket)
    orbitals.py     # Hydrogen-like atomic orbitals (SingleAtomState)
  evolution/
    split_step.py   # Split-step Fourier time evolution
  scattering.py     # Momentum density, energy-averaged transmission
  observables.py    # ⟨x⟩, ⟨p⟩, Δx·Δp, norm
  system.py         # QuantumSystem1D — high-level facade
```

Each sub-package re-exports its public names, so `from physense_qm.potentials
import HarmonicWell` and `from physense_qm.evolution import evolve` work
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
from physense_qm import observables

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
- `physense-utils`
