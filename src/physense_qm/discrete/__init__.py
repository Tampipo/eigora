# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Specific package for discrete Hilbert spaces (qubits, spin systems, etc.).

The names exported here are the operator-level API. The array-level
primitives share their names (`commutator`, `tensor_product`, ...) and stay
behind the `operations` namespace: `discrete.operations.commutator(a, b)`
takes matrices, `discrete.commutator(A, B)` takes `Operator`s.
"""

from physense_qm.discrete import measurement, operations
from physense_qm.discrete.operations import Matrix
from physense_qm.discrete.measurement import Outcome
from physense_qm.discrete.operators import (
    Operator,
    Eigensystem,
    Observable,
    Hamiltonian,
    commutator,
    anticommutator,
    tensor_product,
)
from physense_qm.discrete.hamiltonians import noninteracting
from physense_qm.discrete.evolution import Evolution, evolve

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
    "noninteracting",
    "Evolution",
    "evolve",
]
