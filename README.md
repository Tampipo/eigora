# physense-utils

Shared utilities for the [Physense](https://github.com/you/physense-web) simulation platform.

## Contents

- **constants** — Fundamental physical constants (CODATA 2022, SI units)
- **grids** — Uniform simulation grids of any dimension (`GridND`), plus spherical helpers
- **fft** — FFT helpers with physical normalisation
- **functions** — Common mathematical functions (Gaussian, sinc, Heaviside)

## Installation

```bash
pip install git+https://github.com/you/physense-utils
```

For development:

```bash
git clone https://github.com/you/physense-utils
cd physense-utils
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Usage

```python
from physense_utils import GridND, gaussian, HBAR, fft1d, fft_frequencies

grid = GridND.line(-10.0, 10.0, 512)
psi = gaussian(grid.x, x0=0.0, sigma=1.0)
Psi = fft1d(psi, grid.dx)
k = fft_frequencies(grid.n_points, grid.dx)
```

A grid is a tuple of independent uniform `Axis` objects, so every dimension is
the same type — `x`, `dx`, `n_points` and `length` are 1D conveniences, and
`coordinates()`, `shape`, `spacing` and `volume_element` are the general form:

```python
grid = GridND.uniform([(-5.0, 5.0), (-5.0, 5.0)], (128, 128))   # or GridND.cube(5.0, 128, ndim=3)
X, Y = grid.coordinates()          # 'ij' meshgrid, shape (128, 128)
grid.volume_element                # dV for integrals over the grid
grid.sub(1, 2)                     # sub-grid of one coordinate block

r, theta, phi = to_spherical(grid3d)   # 3D grids only
```

## Design principles

- Pure numpy/scipy — no web dependencies
- Immutable grid objects
- Physical normalisation conventions documented explicitly
- 100% test coverage
