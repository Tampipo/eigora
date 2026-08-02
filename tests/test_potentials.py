# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from physense_qm.potentials import (
    FreeParticle,
    HarmonicWell,
    InfiniteSquareWell,
    FiniteSquareWell,
    RectangularBarrier,
    PotentialStep,
    DoubleWell,
)

x = np.linspace(-5, 5, 100)


class TestFreeParticle:
    def test_zero_everywhere(self):
        V = FreeParticle()(x)
        assert np.all(V == 0.0)


class TestHarmonicWell:
    def test_minimum_at_x0(self):
        V = HarmonicWell(omega=1.0, x0=2.0)(x)
        assert x[np.argmin(V)] == pytest.approx(2.0, abs=0.2)

    def test_zero_at_minimum(self):
        well = HarmonicWell(omega=1.0, x0=0.0)
        assert well(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_invalid_omega(self):
        with pytest.raises(ValueError):
            HarmonicWell(omega=0.0)

    def test_symmetry(self):
        well = HarmonicWell(omega=1.0, x0=0.0)
        assert np.allclose(well(x), well(-x))


class TestInfiniteSquareWell:
    def test_zero_inside(self):
        well = InfiniteSquareWell(width=2.0, x0=0.0)
        x_inside = np.array([-0.9, 0.0, 0.9])
        assert np.all(well(x_inside) == 0.0)

    def test_wall_outside(self):
        well = InfiniteSquareWell(width=2.0, x0=0.0, wall=1e6)
        x_outside = np.array([-2.0, 2.0])
        assert np.all(well(x_outside) == 1e6)

    def test_invalid_width(self):
        with pytest.raises(ValueError):
            InfiniteSquareWell(width=0.0)


class TestFiniteSquareWell:
    def test_negative_inside(self):
        well = FiniteSquareWell(depth=5.0, width=2.0, x0=0.0)
        assert well(np.array([0.0]))[0] == pytest.approx(-5.0)

    def test_zero_outside(self):
        well = FiniteSquareWell(depth=5.0, width=2.0, x0=0.0)
        assert well(np.array([3.0]))[0] == pytest.approx(0.0)

    def test_invalid_depth(self):
        with pytest.raises(ValueError):
            FiniteSquareWell(depth=-1.0)


class TestRectangularBarrier:
    def test_height_inside(self):
        barrier = RectangularBarrier(height=2.0, width=1.0, x0=0.0)
        assert barrier(np.array([0.0]))[0] == pytest.approx(2.0)

    def test_zero_outside(self):
        barrier = RectangularBarrier(height=2.0, width=1.0, x0=0.0)
        assert barrier(np.array([2.0]))[0] == pytest.approx(0.0)

    def test_invalid_height(self):
        with pytest.raises(ValueError):
            RectangularBarrier(height=0.0)

    def test_transmission_plus_reflection_is_one(self):
        barrier = RectangularBarrier(height=5.0, width=1.0)
        for E in [0.5, 1.0, 2.0, 3.2, 5.0, 6.0, 9.0]:
            T = barrier.transmission_coefficient(E)
            R = barrier.reflection_coefficient(E)
            assert T + R == pytest.approx(1.0)

    def test_transmission_increases_with_energy(self):
        barrier = RectangularBarrier(height=5.0, width=1.0)
        Ts = [barrier.transmission_coefficient(E) for E in [0.5, 2.0, 4.0, 6.0, 9.0]]
        assert all(a < b for a, b in zip(Ts, Ts[1:]))

    def test_transmission_continuous_at_barrier_height(self):
        barrier = RectangularBarrier(height=5.0, width=1.0)
        below = barrier.transmission_coefficient(4.9999)
        at = barrier.transmission_coefficient(5.0)
        above = barrier.transmission_coefficient(5.0001)
        assert below == pytest.approx(at, abs=1e-4)
        assert above == pytest.approx(at, abs=1e-4)

    def test_transmission_matches_thin_barrier_reference(self):
        # Reference value from the exact formula T = [1 + sinh^2(sqrt(sigma(1-eps))) / (4 eps(1-eps))]^-1
        barrier = RectangularBarrier(height=5.0, width=1.0)
        assert barrier.transmission_coefficient(3.2) == pytest.approx(0.079835, abs=1e-5)

    def test_invalid_energy(self):
        barrier = RectangularBarrier(height=5.0, width=1.0)
        with pytest.raises(ValueError):
            barrier.transmission_coefficient(0.0)


class TestPotentialStep:
    def test_zero_before(self):
        step = PotentialStep(height=1.0, x0=0.0)
        assert step(np.array([-1.0]))[0] == pytest.approx(0.0)

    def test_height_after(self):
        step = PotentialStep(height=1.0, x0=0.0)
        assert step(np.array([1.0]))[0] == pytest.approx(1.0)


class TestDoubleWell:
    def test_two_minima(self):
        well = DoubleWell(a=1.0, b=4.0)
        assert well.minima == pytest.approx(np.sqrt(2.0))

    def test_barrier_height(self):
        well = DoubleWell(a=1.0, b=4.0)
        assert well.barrier_height == pytest.approx(4.0)

    def test_invalid_a(self):
        with pytest.raises(ValueError):
            DoubleWell(a=0.0, b=4.0)

    def test_composite(self):
        well = HarmonicWell(omega=1.0)
        barrier = RectangularBarrier(height=1.0, width=1.0)
        composite = well + barrier
        V = composite(x)
        assert V.shape == x.shape
