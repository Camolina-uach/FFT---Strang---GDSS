"""
gdss_long_horizon.py -- long-horizon drift run (revision experiment, added
after v1.0.0; the archived scientific sources are imported unchanged).

The baseline Gaussian packet of the manuscript is integrated to T = 20 on the
enlarged domain [-80, 80)^2 with N = 512 points per direction, i.e. with the
same grid spacing (0.3125) as the baseline run.  The invariants, the boundary
band amplitude and the spectral tail ratio are recorded every ``--every``
time units.  Running the script for several time steps shows how the energy
drift accumulates in time and how it scales with dt.

Usage:
    python gdss_long_horizon.py --dt 2e-3 [--T 20] [--every 0.1] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np

from gdss_solver import (
    GDSSParams,
    boundary_amplitude,
    energy_longwave_quadratic,
    energy_reduced,
    gaussian_initial_data,
    make_grid,
    mass,
    momentum,
    precompute_multipliers,
    recover_longwave,
    spectral_tail_ratio,
    strang_step,
)

HERE = Path(__file__).resolve().parent


def long_horizon_params(dt: float, T: float, L: float = 160.0, N: int = 512) -> GDSSParams:
    return GDSSParams(Nx=N, Ny=N, Lx=L, Ly=L, dt=dt, t_final=T,
                      alpha=1.0, beta=1.0, gamma=1.0, xi=1.0,
                      psi=1.0, eta=1.0, phi=2.0, chi=0.5, theta=None,
                      dealias_density=True,
                      # Q-only inside the time loop; w, v are recovered at the
                      # output times for the energy diagnostics.
                      recover_full_longwave=False)


def diagnostics(u, g, p, m) -> dict:
    lw = recover_longwave(u, g, p, m, full=True)
    jx, jy = momentum(u, g)
    return {
        "M": mass(u, g),
        "Jx": jx,
        "Jy": jy,
        "E_red": energy_reduced(u, lw["Q"], g, p),
        "E_lw": energy_longwave_quadratic(u, lw, g, p),
        "boundary": boundary_amplitude(u, band=4),
        "tail": spectral_tail_ratio(u, g),
        "max_abs_u": float(np.max(np.abs(u))),
    }


def run(dt: float, T: float, every: float, out: Path) -> Path:
    p = long_horizon_params(dt, T)
    g = make_grid(p)
    m = precompute_multipliers(p, g)
    u = gaussian_initial_data(g, amplitude=1.0, width=4.0, kx0=0.2, ky0=-0.1)

    nsteps = int(round(T / dt))
    stride = int(round(every / dt))
    d0 = diagnostics(u, g, p, m)
    rows = []

    def record(n: int, d: dict) -> None:
        row = {"t": n * dt, "step": n}
        for k in ("M", "Jx", "Jy", "E_red", "E_lw"):
            row[k] = d[k]
            row[f"rel_{k}"] = abs(d[k] - d0[k]) / max(abs(d0[k]), 1e-14)
        row.update({k: d[k] for k in ("boundary", "tail", "max_abs_u")})
        rows.append(row)

    record(0, d0)
    t0 = time.perf_counter()
    for n in range(1, nsteps + 1):
        u, _ = strang_step(u, g, p, m)
        if n % stride == 0 or n == nsteps:
            record(n, diagnostics(u, g, p, m))
            if (n // stride) % 20 == 0:
                print(f"dt={dt:g}  t={n * dt:7.3f}  rel_E={rows[-1]['rel_E_red']:.3e}  "
                      f"elapsed={time.perf_counter() - t0:7.1f}s", flush=True)
    wall = time.perf_counter() - t0

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"long_horizon_dt{dt:g}.csv"
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.10e}" if isinstance(v, float) else v) for k, v in r.items()})
    (out / f"long_horizon_dt{dt:g}.meta.txt").write_text(
        f"dt {dt}\nT {T}\nN {p.Nx}\nL {p.Lx}\nsteps {nsteps}\nwall_seconds {wall:.1f}\n"
        f"python {platform.python_version()}\nnumpy {np.__version__}\n"
        f"platform {platform.platform()}\n")
    print(f"wrote {path} ({wall:.0f} s)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-horizon drift run for the GDSS baseline.")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--every", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "revision")
    a = ap.parse_args()
    run(a.dt, a.T, a.every, a.out)


if __name__ == "__main__":
    main()
