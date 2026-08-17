# Experiment and output map

The experiment numbering follows the manuscript and `gdss_experiments.py`.

| Experiment | Purpose | Main output |
|---|---|---|
| 1 | Fourier recovery verification on 128² and 256² grids | `outputs/exp1_recovery/` |
| 2--3 | Baseline evolution, invariant balance, and long-wave diagnostics | `outputs/exp2_baseline/` |
| 4 | Time-step convergence | `outputs/exp4_timestep/` |
| 5 | Spatial resolution and spectral-tail study | `outputs/exp5_spatial/` |
| 6 | Periodic-boundary contamination | `outputs/exp6_boundary/` |
| 7 | FFT cost scaling | `outputs/exp7_cost/` |
| 8 | Babaoglu--Erbay exact travelling-wave temporal benchmark | `outputs/exp8_babaoglu_exact/` |
| 8b | Babaoglu--Erbay spatial convergence | `outputs/exp8b_babaoglu_spatial/` |

`gdss_plots.py` reads these directories and writes PDF and PNG files under
`outputs/figures/`. `gdss_paraview_export.py` converts Experiment 8 snapshots
and time series to legacy VTK, PVD, and optional CSV products.

## Manuscript parameter set

- Grid: `Nx = Ny = 128`
- Domain: `Lx = Ly = 40`
- Base step: `dt = 1e-3`
- Baseline and exact-benchmark horizon: `T = 2`
- Convergence horizon: `T = 1`
- Cost-study horizon: `T = 0.1`
- Model: `alpha = beta = gamma = xi = psi = eta = 1`, `phi = 2`,
  `chi = 0.5`, and structural `theta = sqrt((phi-psi)(eta-chi))`
- Density dealiasing: enabled for the main experiments
- Full long-wave recovery: enabled

Experiment 8 deliberately uses its benchmark-specific coefficients
`gamma = 2` and `xi = -4`; these are written to its `metadata.json`.
