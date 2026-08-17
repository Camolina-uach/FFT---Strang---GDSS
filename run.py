"""
run.py — Entry point for the GDSS numerical experiments.

Place this file in the same folder as:
    gdss_solver.py
    gdss_experiments.py
    gdss_plots.py
    gdss_paraview_export.py

All outputs (data, figures, and ParaView files) are saved in an "outputs/"
subfolder next to this file, regardless of from where you launch the script.

Time horizons per experiment (all set explicitly in run_all_experiments below)
------------------------------------------------------------------------------
t_final_baseline   = 2.0  →  Exps 2/3, 6
    Wave packet reaches ≈ (0.8, -0.4) at t=2; B∂_max stays at machine
    precision.  (T=5 tested: conservation fine — RE_M≈1.78e-12 — but
    B∂_max≈3.67e-2 weakens the periodic-FFT argument.)

t_final_benchmark  = 2.0  →  Exp 8  (Babaoglu temporal)
    Same domain-safety reasoning; the sech wave travels ~0.8 units in x.

t_final_convergence = 1.0  →  Exps 4, 5, 8b  (convergence studies)
    Short horizon: error dominated by truncation, not by integration length.
    All dt values in the refinement sequence divide 1.0 exactly.

t_final_cost = 0.1  →  Exp 7  (computational cost scaling)
    Only Nt=100 steps needed for stable wall-clock timing; using t_final=2
    would repeat 2 000 steps × 3 times × 2 modes × 3 grids unnecessarily.

save_every_baseline = 20
    With dt=1e-3 and t_final=2 → 2 000 steps → 100 saved diagnostic rows.
"""

from pathlib import Path

from gdss_solver import GDSSParams
from gdss_experiments import run_all_experiments
from gdss_plots import plot_all
from gdss_paraview_export import export_exp8_paraview

# ── Model and numerical parameters ───────────────────────────────────────────
params = GDSSParams(
    Nx=128, Ny=128,
    Lx=40.0, Ly=40.0,
    dt=1.0e-3,
    t_final=2.0,               # fallback default; each experiment uses its own horizon below
    alpha=1.0, beta=1.0,
    gamma=1.0, xi=1.0,
    psi=1.0, eta=1.0, phi=2.0, chi=0.5,
    theta=None,                # structural value sqrt((phi-psi)(eta-chi))
    dealias_density=True,
    recover_full_longwave=True,
)

# ── Output directory: always next to this file ────────────────────────────────
output_root = Path(__file__).parent / "outputs"

# ── Run experiments ──────────────────────────────────────────────────────────
print("Running experiments...")
run_all_experiments(
    params,
    output_root=output_root,
    # ── Time horizons ────────────────────────────────────────────────────────
    t_final_baseline=2.0,       # Exps 2/3, 6  — boundary-clean main run
    t_final_benchmark=2.0,      # Exp 8        — Babaoglu temporal
    t_final_convergence=1.0,    # Exps 4, 5, 8b — convergence studies
    t_final_cost=0.1,           # Exp 7        — only ~100 steps needed for stable timing
    # ── Snapshot times (must lie in [0, t_final_baseline]) ───────────────────
    snapshot_times=[0.0, 0.5, 1.0, 1.5, 2.0],
    # ── Diagnostic row density for Exp 2/3 ──────────────────────────────────
    save_every_baseline=20,     # 2 000 steps / 20 = 100 rows
    # ── Babaoglu exact benchmark ─────────────────────────────────────────────
    run_exact_benchmark=True,
    exact_benchmark_time_series_frames=21,
    run_spatial_benchmark=True,
)

# ── Generate paper figures ───────────────────────────────────────────────────
print("\nGenerating figures...")
plot_all(output_root=output_root)

# ── Export Exp. 8 fields for ParaView animation ──────────────────────────────
print("\nExporting ParaView files for the Babaoglu benchmark...")
export_exp8_paraview(output_root=output_root, export_time_series=True)

print(f"\nDone. Results saved to:\n  {output_root.resolve()}")
