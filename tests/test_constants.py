# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
from physense_utils import constants as C


def test_hbar_positive():
    assert C.HBAR > 0


def test_hbar_equals_h_over_2pi():
    assert abs(C.HBAR - C.H / (2 * 3.141592653589793)) < 1e-43


def test_speed_of_light():
    assert C.C == 299792458.0


def test_elementary_charge_positive():
    assert C.E > 0


def test_boltzmann_positive():
    assert C.K_B > 0


def test_electron_mass_positive():
    assert C.M_E > 0


def test_bohr_radius_positive():
    assert C.A_0 > 0


def test_fine_structure_close_to_1_over_137():
    assert abs(C.ALPHA - 1 / 137) < 1e-4


def test_ev_equals_elementary_charge():
    assert C.EV == C.E
