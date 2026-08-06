# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Known discrete Hamiltonians, and ways of combining them.
"""

from collections.abc import Sequence

from physense_qm.discrete.operators import Hamiltonian, embed


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
    total = embed(hamiltonians[0], 0, dims)
    for site, hamiltonian in enumerate(hamiltonians[1:], start=1):
        total = total + embed(hamiltonian, site, dims)
    return total


__all__ = ["noninteracting"]
