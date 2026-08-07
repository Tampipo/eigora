# Discrete systems

Finite-dimensional Hilbert spaces — spins, qubits, two-level systems, truncated ladders. A state is a complex vector, an operator is a matrix, and there is no grid anywhere.

The contrast with [the continuous side](continuous.md) is the point. There, the Hamiltonian is a differential operator that has to be discretised and the propagator has to be Trotter-split. Here, `H` is already a matrix, so it can simply be diagonalised — which makes time evolution exact at any time, with no timestep and no stability criterion.

> **Units:** atomic units (ℏ = 1). Spin operators use the **Pauli convention**, so `SZ` has eigenvalues ±1 and `[SX, SY] = 2i·SZ`.

---

## Three kinds of operator

Everything is built on a hierarchy where each layer adds a guarantee the one below cannot make:

| | what it promises | examples |
|---|---|---|
| `Operator` | a square matrix, nothing more | `SPLUS`, `SMINUS`, propagators |
| `Observable` | Hermitian: real eigenvalues, orthonormal eigenbasis | `SX`, `SY`, `SZ`, `identity(n)` |
| `Hamiltonian` | generates time translation | `TwoLevel`, `Rabi`, `HeisenbergChain` |

The split is not decoration. Ladder operators are genuinely not observables — `SPLUS` is not Hermitian and cannot be measured — and `np.linalg.eigh` is only valid on the middle layer. Constructing an `Observable` from a non-Hermitian matrix raises.

```python
from eigora.qm.discrete import SX, SY, SZ, SPLUS, Observable, commutator

SZ.eigenvalues()            # [-1., 1.]
SPLUS.is_hermitian()        # False
Observable(SPLUS.matrix)    # ValueError: an observable must be Hermitian
```

### Operations keep the class only when the maths allows

```python
commutator(SX, SY)          # Operator — [A,B]† = -[A,B], anti-Hermitian
SX + SZ                     # Observable — sums of Hermitian are Hermitian
2.0 * SZ                    # Observable
1j * SZ                     # Operator — a complex multiple is not Hermitian
SZ @ SX                     # Operator — a product is Hermitian only if they commute
```

So `commutator(SX, SY)` cannot silently claim to be measurable. It is `2i·SZ`, which is anti-Hermitian; the observable is `i[A,B]`.

Arithmetic on the operators themselves works as expected — `@`, `+`, `-`, unary `-`, and scalar multiplication from either side.

---

## Building bigger systems

`identity(dim)` and `embed` lift single-subsystem operators onto a product space. `embed(A, site, dims)` is `I ⊗ … ⊗ A ⊗ … ⊗ I`.

```python
from eigora.qm.discrete import SZ, embed

dims = (2, 2)                                    # two spins
total_sz = embed(SZ, 0, dims) + embed(SZ, 1, dims)
total_sz.eigenvalues()                           # [-2., 0., 0., 2.]
```

Note the eigenvalues **add**. This is why `embed` exists rather than reaching for `tensor_product` directly — `tensor_product(SZ, SZ)` multiplies them, giving a completely different operator that is not the total spin of anything.

`embed` preserves the class, because padding with identities changes neither Hermiticity nor meaning. `tensor_product` deliberately does not.

For independent subsystems there is a shortcut:

```python
from eigora.qm.discrete import TwoLevel, noninteracting

pair = noninteracting([TwoLevel(bias=1.0, coupling=0.0),
                       TwoLevel(bias=3.0, coupling=0.0)])
pair.eigenenergies()        # [-2., -1., 1., 2.] — every sum of the two spectra
```

---

## Known systems

Each is built from physical parameters rather than a matrix, and those with a closed form override `eigensystem` to return it directly instead of diagonalising.

| Class | Hamiltonian | Exact spectrum |
|---|---|---|
| `TwoLevel(bias, coupling)` | (bias/2)·SZ + (coupling/2)·SX | ±√(bias² + coupling²)/2 |
| `Rabi(detuning, rabi_frequency)` | same, named for the problem | ±W/2, W = √(Ω² + Δ²) |
| `SpinInField(field)` | (B·**σ**)/2 | ±\|B\|/2 |
| `HarmonicLadder(n_levels, omega)` | ω(n + ½) | those energies exactly |
| `HeisenbergChain(n_sites, ...)` | J Σ **σ**ᵢ·**σ**ⱼ + h Σ SZᵢ | only for two sites: −3J, +J (×3) |

```python
from eigora.qm.discrete import Rabi

drive = Rabi(detuning=0.5, rabi_frequency=1.7)
drive.generalised_frequency   # W = sqrt(1.7² + 0.5²)
drive.max_transfer            # (Ω/W)² — peak excited population, 1 only on resonance
```

`HeisenbergChain` has no elementary closed form past two sites, so it does not override `eigensystem` and falls back to diagonalisation. It commutes with the total SZ at any length, so magnetisation stays a good quantum number.

### Operating on a known system degrades it

```python
from eigora.qm.discrete import TwoLevel, Hamiltonian

type(TwoLevel(1.0, 2.0))            # TwoLevel
type(2.0 * TwoLevel(1.0, 2.0))      # Hamiltonian
```

A scaled `TwoLevel` is a *different* two-level system, and an embedded one is not two-level at all — so neither may inherit the closed-form spectrum. The result stays a perfectly good `Hamiltonian`; it just diagonalises like any other.

---

## Time evolution

`evolve` expands the initial state in the energy eigenbasis once and applies the phase e^(−iEₙt) at every requested time:

```
|ψ(t)⟩ = Σₙ ⟨Eₙ|ψ(0)⟩ e^(−iEₙt) |Eₙ⟩
```

One diagonalisation (cached on the Hamiltonian) plus a single matrix product covers every time you ask for.

```python
import numpy as np
from eigora.qm.discrete import Rabi, SZ, evolve

times = np.linspace(0.0, 12.0, 200)
ev = evolve(Rabi(0.0, 1.7), np.array([1.0, 0.0]), times)

ev.psi                  # (n_times, dim), complex
ev.state(50)            # the state at times[50]
ev.probabilities(50)    # |ψᵢ|² over the basis
ev.norm(50)             # 1.0, to machine precision
ev.expectation(SZ)      # ⟨SZ⟩ at every time — here, cos(omega t)
```

Because nothing is stepped, the times need no structure at all — unevenly spaced, out of order, and negative all work, and each is computed from ψ(0) directly so no error accumulates.

```python
evolve(h, psi0, [5.0, -3.0, 0.0, 100.0])     # fine
```

`Hamiltonian.propagator(t)` gives U(t) = exp(−iHt) as an `Operator` if you want the matrix itself, for composing gates or inspecting it. For pushing a state through time, prefer `evolve` — it reuses one decomposition across all times.

---

## Measurement

Outcomes are grouped by **distinct eigenvalue**, not by eigenvector. A degenerate eigenvalue is one outcome whose probability sums over its whole eigenspace, and measuring it projects onto that subspace rather than onto any single eigenvector.

```python
from eigora.qm.discrete import SZ
import numpy as np

plus = np.array([1.0, 1.0]) / np.sqrt(2)
for outcome in SZ.outcomes(plus):
    print(outcome.value, outcome.probability, outcome.degeneracy)
# -1.0  0.5  1
#  1.0  0.5  1
```

With degeneracy, the difference matters:

```python
from eigora.qm.discrete import embed

dims = (2, 2)
total_sz = embed(SZ, 0, dims) + embed(SZ, 1, dims)   # eigenvalues -2, 0, 0, +2

len(total_sz.outcomes(np.full(4, 0.5)))              # 3 outcomes, not 4
```

Writing this per-eigenvector instead is the classic way to get plausible-looking but wrong answers — the probability of 0 would be split in half, and the collapsed state would be one arbitrary basis vector of the pair rather than their superposition.

`measure` draws an outcome and collapses:

```python
rng = np.random.default_rng(0)
outcome, collapsed = SZ.measure(plus, rng)

SZ.outcomes(collapsed)      # the same value now has probability 1
```

Measuring again returns the same value with certainty, which is what makes measurement usable to *prepare* a state.

`expectation` gives the ensemble average directly, and it agrees with summing over outcomes:

```python
SZ.expectation(plus)                                        # 0.0
sum(o.value * o.probability for o in SZ.outcomes(plus))     # 0.0
```

---

## Layers

The module is arranged so dependencies point one way:

```
operations.py    plain matrix algebra — dagger, commutator, tensor_product
measurement.py   Born rule over (eigenvalues, eigenvectors, psi) arrays
operators.py     Operator / Observable / Hamiltonian, Pauli constants
hamiltonians.py  known systems
evolution.py     evolve(hamiltonian, psi0, times)
```

`operations` and `measurement` are class-free, which is what lets `Operator` and `Observable` call into them without a circular import.

The array-level functions share names with their operator-level counterparts, so they stay behind a namespace:

```python
from eigora.qm.discrete import commutator, operations

commutator(SX, SY)                              # takes Operators, returns an Operator
operations.commutator(SX.matrix, SY.matrix)     # takes arrays, returns an array
```

Use the operator-level names unless you are working with raw matrices.
