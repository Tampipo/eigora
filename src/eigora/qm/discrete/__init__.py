# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Specific package for discrete Hilbert spaces (qubits, spin systems, etc.).

The names exported here are the operator-level API. The array-level
primitives share their names (`commutator`, `tensor_product`, ...) and stay
behind the `operations` namespace: `discrete.operations.commutator(a, b)`
takes matrices, `discrete.commutator(A, B)` takes `Operator`s.
"""

from eigora.qm.discrete import measurement, operations
from eigora.qm.discrete.operations import Matrix
from eigora.qm.discrete.measurement import Outcome
from eigora.qm.discrete.operators import (
    Operator,
    Eigensystem,
    Observable,
    Hamiltonian,
    commutator,
    anticommutator,
    tensor_product,
    identity,
    embed,
    SX,
    SY,
    SZ,
    SPLUS,
    SMINUS,
)
from eigora.qm.discrete.hamiltonians import (
    HarmonicLadder,
    HeisenbergChain,
    KnownHamiltonian,
    Rabi,
    SpinInField,
    TwoLevel,
    noninteracting,
)
from eigora.qm.discrete.evolution import Evolution, evolve

__all__ = [
    "measurement",
    "operations",
    "Matrix",
    "Outcome",
    "Operator",
    "Eigensystem",
    "Observable",
    "Hamiltonian",
    "commutator",
    "anticommutator",
    "tensor_product",
    "KnownHamiltonian",
    "TwoLevel",
    "Rabi",
    "SpinInField",
    "HarmonicLadder",
    "HeisenbergChain",
    "noninteracting",
    "Evolution",
    "evolve",
    "identity",
    "embed",
    "SX",
    "SY",
    "SZ",
    "SPLUS",
    "SMINUS",
]
