# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Known discrete Hamiltonians, and ways of combining them.
"""

import math
from collections.abc import Sequence
from functools import reduce

import numpy as np

from physense_qm.discrete.operations import Matrix, tensor_product
from physense_qm.discrete.operators import Hamiltonian


def noninteracting(hamiltonians: Sequence[Hamiltonian]) -> Hamiltonian:
    """
    Combine independent subsystems into one Hamiltonian.

    H = sum_k I ⊗ ... ⊗ H_k ⊗ ... ⊗ I, acting on the tensor product of the
    subsystem spaces. Energies of the composite system are sums of subsystem
    energies -- not products, which is what a bare H_1 ⊗ H_2 would give.

    Parameters
    ----------
    hamiltonians : sequence of Hamiltonian
        The subsystem Hamiltonians, in tensor-factor order.

    Returns
    -------
    Hamiltonian
        Hamiltonian on the product space, of dimension prod(h.dim).
    """
    if not hamiltonians:
        raise ValueError("at least one Hamiltonian is required")

    dims = [h.dim for h in hamiltonians]
    total_dim = math.prod(dims)
    total = np.zeros((total_dim, total_dim), dtype=complex)
    for k, h in enumerate(hamiltonians):
        factors: list[Matrix] = [np.eye(d, dtype=complex) for d in dims]
        factors[k] = h.matrix
        total += reduce(tensor_product, factors)
    return Hamiltonian(total)


__all__ = ["noninteracting"]
