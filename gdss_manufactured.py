"""
gdss_manufactured.py -- verification of the coupled long-wave recovery
(revision experiment, added after v1.0.0; the archived scientific sources
are imported unchanged).

Two tests are provided.

1. Manufactured solution of the long-wave subsystem (``manufactured``).
   Write the subsystem as  L (w, v)^T = grad(rho)  with

       L = [[a, b], [b, c]],   a = psi d_xx + eta d_yy,
                                b = theta d_xy,
                                c = phi d_xx + chi d_yy.

   For any smooth periodic potential G, the adjugate construction

       (w, v)^T = adj(L) grad(G) = (c G_x - b G_y,  -b G_x + a G_y)

   satisfies  L (w, v)^T = det(L) grad(G) = grad(rho)  with
   rho = det(L) G = (a c - b^2) G.  Explicitly,

       w   = phi G_xxx + (chi - theta) G_xyy
       v   = (psi - theta) G_xxy + eta G_yyy
       Q   = w_x + v_y = phi G_xxxx + (psi + chi - 2 theta) G_xxyy + eta G_yyyy
       rho = psi phi G_xxxx + (psi chi + eta phi - theta^2) G_xxyy + eta chi G_yyyy.

   Both long-wave fields are nonzero, the cross-coupling theta enters every
   field, and w, v are mean-free because they are derivatives of a periodic
   function.  G is a sum of separable products of exp(A cos(k x + p)), which
   is analytic but not band-limited, so the recovery error is not trivially
   zero: it decays spectrally with N.  The short-wave field fed to the solver
   is u = sqrt(rho + C) with C > -min(rho), so that the full code path
   |u|^2 -> filter -> multipliers -> (w, v, Q) of gdss_solver.recover_longwave
   is exercised; the constant C only changes the zero mode, which the
   recovery removes.

2. Refined-reference convergence of (w, v, Q) in the dynamic baseline run
   (``reference``).  The Gaussian baseline of the manuscript is integrated to
   T with a fixed time step on nested grids; w, v, Q recovered at T on each
   grid are compared, at the coarse grid points, with those of the finest
   grid.

Usage:
    python gdss_manufactured.py manufactured [--out DIR]
    python gdss_manufactured.py reference    [--out DIR]
    python gdss_manufactured.py all          [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from gdss_solver import (
    GDSSParams,
    gaussian_initial_data,
    longwave_residuals,
    make_grid,
    precompute_multipliers,
    recover_longwave,
    strang_step,
)

HERE = Path(__file__).resolve().parent


# ════════════════════════════════════════════════════ analytic potential G


@dataclass(frozen=True)
class Mode1D:
    """h(s) = A cos(k s + p); the 1-D factor is exp(h(s))."""
    A: float
    k: float
    p: float

    def h_derivs(self, s: np.ndarray) -> List[np.ndarray]:
        """[h, h', h'', h''', h''''] with h^(n) = A k^n cos(k s + p + n pi/2)."""
        return [self.A * self.k**n * np.cos(self.k * s + self.p + n * np.pi / 2)
                for n in range(5)]

    def derivs(self, s: np.ndarray) -> List[np.ndarray]:
        """[f, f', f'', f''', f''''] for f = exp(h) (Faa di Bruno)."""
        h0, h1, h2, h3, h4 = self.h_derivs(s)
        f = np.exp(h0)
        return [
            f,
            h1 * f,
            (h2 + h1**2) * f,
            (h3 + 3 * h1 * h2 + h1**3) * f,
            (h4 + 4 * h1 * h3 + 3 * h2**2 + 6 * h1**2 * h2 + h1**4) * f,
        ]


def potential_terms(Lx: float, Ly: float) -> List[Tuple[Mode1D, Mode1D]]:
    """
    G(x, y) = sum_j f_j(x) g_j(y).  Two terms with different wavenumbers and
    phases, so that G has no symmetry that could hide a sign error in the
    mixed derivative terms.
    """
    kx1, ky1 = 2 * np.pi * 1 / Lx, 2 * np.pi * 1 / Ly
    kx2, ky2 = 2 * np.pi * 2 / Lx, 2 * np.pi * 3 / Ly
    return [
        (Mode1D(A=2.0, k=kx1, p=0.3), Mode1D(A=1.5, k=ky1, p=-np.pi / 2)),
        (Mode1D(A=1.0, k=kx2, p=1.1), Mode1D(A=0.8, k=ky2, p=0.4)),
    ]


def G_derivative(terms, X, Y, i: int, j: int) -> np.ndarray:
    """d^i/dx^i d^j/dy^j G evaluated on the grid (X, Y)."""
    out = np.zeros_like(X)
    for fx, gy in terms:
        out += fx.derivs(X)[i] * gy.derivs(Y)[j]
    return out


def manufactured_fields(p: GDSSParams, X, Y) -> Dict[str, np.ndarray]:
    """Exact w, v, Q, rho of the adjugate construction."""
    T = potential_terms(p.Lx, p.Ly)
    D = lambda i, j: G_derivative(T, X, Y, i, j)  # noqa: E731
    th = p.theta
    w = p.phi * D(3, 0) + (p.chi - th) * D(1, 2)
    v = (p.psi - th) * D(2, 1) + p.eta * D(0, 3)
    Q = p.phi * D(4, 0) + (p.psi + p.chi - 2 * th) * D(2, 2) + p.eta * D(0, 4)
    rho = (p.psi * p.phi * D(4, 0)
           + (p.psi * p.chi + p.eta * p.phi - th**2) * D(2, 2)
           + p.eta * p.chi * D(0, 4))
    return {"w": w, "v": v, "Q": Q, "rho": rho}


def coupling_share(p: GDSSParams, g, ex: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Relative size of the cross-coupling term in each long-wave equation,
        ||theta v_xy|| / ||psi w_xx + eta w_yy||,
        ||theta w_xy|| / ||phi v_xx + chi v_yy||,
    evaluated with spectral derivatives of the exact fields on a fine grid.
    """
    wh, vh = np.fft.fft2(ex["w"]), np.fft.fft2(ex["v"])
    KX, KY = g.KX, g.KY
    diag_w = -(p.psi * KX**2 + p.eta * KY**2) * wh
    diag_v = -(p.phi * KX**2 + p.chi * KY**2) * vh
    cross_w = -p.theta * KX * KY * vh
    cross_v = -p.theta * KX * KY * wh
    return (float(np.linalg.norm(cross_w) / np.linalg.norm(diag_w)),
            float(np.linalg.norm(cross_v) / np.linalg.norm(diag_v)))


# ════════════════════════════════════════════════════ error measures


def rel_l2(num: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm((num - ref).ravel()) / np.linalg.norm(ref.ravel()))


def rel_max(num: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(num - ref)) / np.max(np.abs(ref)))


# ════════════════════════════════════════════════════ test 1: manufactured


PARAMETER_SETS = {
    # Baseline long-wave parameters of the manuscript (EEE, theta = 1/sqrt 2).
    "baseline": dict(psi=1.0, eta=1.0, phi=2.0, chi=0.5, theta=None),
    # Babaoglu--Erbay benchmark parameters (theta = 1).
    "babaoglu": dict(psi=1.0, eta=2.0, phi=2.0, chi=1.0, theta=1.0),
}


def run_manufactured(out: Path, grids: Sequence[int] = (16, 24, 32, 48, 64, 96, 128, 192, 256)) -> List[dict]:
    rows: List[dict] = []
    for set_name, lw in PARAMETER_SETS.items():
        for dealias in (False, True):
            for N in grids:
                p = GDSSParams(Nx=N, Ny=N, Lx=40.0, Ly=40.0, dealias_density=dealias,
                               recover_full_longwave=True, **lw)
                g = make_grid(p)
                m = precompute_multipliers(p, g)
                ex = manufactured_fields(p, g.X, g.Y)
                C = 1.0 - float(np.min(ex["rho"]))
                u = np.sqrt(ex["rho"] + C).astype(np.complex128)
                rec = recover_longwave(u, g, p, m, full=True)
                res_w, res_v = longwave_residuals(rec, g, p)

                # Sensitivity check: the same recovery with the cross-coupling
                # switched off (theta = 0) must NOT reproduce the exact fields.
                p0 = GDSSParams(Nx=N, Ny=N, Lx=40.0, Ly=40.0, dealias_density=dealias,
                                recover_full_longwave=True,
                                **{**lw, "theta": 0.0})
                rec0 = recover_longwave(u, g, p0, precompute_multipliers(p0, g), full=True)

                share_w, share_v = coupling_share(p, g, ex)
                rows.append({
                    "parameter_set": set_name,
                    "theta": p.theta,
                    "coupling_share_w": share_w,
                    "coupling_share_v": share_v,
                    "dealias": int(dealias),
                    "N": N,
                    "e_w_l2": rel_l2(rec["w"], ex["w"]),
                    "e_v_l2": rel_l2(rec["v"], ex["v"]),
                    "e_Q_l2": rel_l2(rec["Q"], ex["Q"]),
                    "e_w_max": rel_max(rec["w"], ex["w"]),
                    "e_v_max": rel_max(rec["v"], ex["v"]),
                    "e_Q_max": rel_max(rec["Q"], ex["Q"]),
                    "res_w": res_w,
                    "res_v": res_v,
                    "e_w_l2_theta0": rel_l2(rec0["w"], ex["w"]),
                    "e_v_l2_theta0": rel_l2(rec0["v"], ex["v"]),
                    "norm_w": float(np.sqrt(g.dx * g.dy) * np.linalg.norm(ex["w"])),
                    "norm_v": float(np.sqrt(g.dx * g.dy) * np.linalg.norm(ex["v"])),
                })
    _write_csv(out / "manufactured_longwave.csv", rows)
    return rows


# ════════════════════════════════════════════════════ test 2: refined reference


def _baseline_params(N: int, dt: float, T: float) -> GDSSParams:
    return GDSSParams(Nx=N, Ny=N, Lx=40.0, Ly=40.0, dt=dt, t_final=T,
                      alpha=1.0, beta=1.0, gamma=1.0, xi=1.0,
                      psi=1.0, eta=1.0, phi=2.0, chi=0.5, theta=None,
                      dealias_density=True, recover_full_longwave=True)


def _evolve_and_recover(N: int, dt: float, T: float) -> Dict[str, np.ndarray]:
    p = _baseline_params(N, dt, T)
    g = make_grid(p)
    m = precompute_multipliers(p, g)
    u = gaussian_initial_data(g, amplitude=1.0, width=4.0, kx0=0.2, ky0=-0.1)
    nsteps = int(round(T / dt))
    for _ in range(nsteps):
        u, _ = strang_step(u, g, p, m)
    lw = recover_longwave(u, g, p, m, full=True)
    return {"w": lw["w"], "v": lw["v"], "Q": lw["Q"], "u": u}


def run_reference(out: Path, grids: Sequence[int] = (32, 48, 64, 96, 128, 192, 256),
                  N_ref: int = 768, dt: float = 1.0e-3, T: float = 1.0) -> List[dict]:
    t0 = time.perf_counter()
    ref = _evolve_and_recover(N_ref, dt, T)
    rows: List[dict] = []
    for N in grids:
        if N_ref % N:
            raise ValueError(f"N_ref={N_ref} is not a multiple of N={N}")
        s = N_ref // N
        num = _evolve_and_recover(N, dt, T)
        sub = {k: v[::s, ::s] for k, v in ref.items()}
        rows.append({
            "N": N, "N_ref": N_ref, "dt": dt, "T": T,
            "e_w_l2": rel_l2(num["w"], sub["w"]),
            "e_v_l2": rel_l2(num["v"], sub["v"]),
            "e_Q_l2": rel_l2(num["Q"], sub["Q"]),
            "e_u_l2": rel_l2(num["u"], sub["u"]),
            "norm_ratio_v_w": float(np.linalg.norm(sub["v"]) / np.linalg.norm(sub["w"])),
        })
    _write_csv(out / "reference_longwave_convergence.csv", rows)
    print(f"reference test: {time.perf_counter() - t0:.1f} s")
    return rows


# ════════════════════════════════════════════════════ utilities


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {path}")


def _write_environment(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.txt").write_text(
        f"python {platform.python_version()}\nnumpy {np.__version__}\n"
        f"platform {platform.platform()}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("test", choices=["manufactured", "reference", "all"])
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "revision")
    a = ap.parse_args()
    _write_environment(a.out)
    if a.test in ("manufactured", "all"):
        run_manufactured(a.out)
    if a.test in ("reference", "all"):
        run_reference(a.out)


if __name__ == "__main__":
    main()
