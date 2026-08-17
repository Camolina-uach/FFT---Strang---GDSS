"""
gdss_plots.py — Figure generation for the GDSS paper.

One function per figure in Section sec:numerical_experiments.
All figures are saved as both PDF and PNG.

Usage:
    from gdss_plots import plot_all
    plot_all(output_root="outputs")
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Anchor output paths to this module's directory.
_HERE = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CMAP_POS  = "viridis"   # non-negative fields: rho, |u|
CMAP_SIGN = "RdBu_r"    # signed fields: Q, w, v
DPI = 150
FIGW = 6.5   # single-column width (inches)
FIGH = 5.0
SURFACE_MAX_POINTS = 140  # max grid points per direction for 3D surfaces


def _savefig(fig, outdir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _contourf(ax, X, Y, fld, cmap, fig, n_levels=40):
    vmax = float(np.max(np.abs(fld)))
    vmin = 0.0 if cmap == CMAP_POS else -vmax
    cf = ax.contourf(X, Y, fld, levels=n_levels, cmap=cmap,
                     vmin=vmin, vmax=max(vmax, 1e-30))
    fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    ax.set_xlabel("$x$", fontsize=10)
    ax.set_ylabel("$y$", fontsize=10)
    return cf


def _surface_stride(fld: np.ndarray, max_points: int = SURFACE_MAX_POINTS) -> tuple[int, int]:
    """
    Return safe strides for 3D surface plots.

    Large FFT grids can make matplotlib 3D surfaces slow and visually crowded.
    This keeps the plotted mesh readable without changing the saved numerical data.
    """
    nx, ny = fld.shape
    sx = max(1, int(math.ceil(nx / max_points)))
    sy = max(1, int(math.ceil(ny / max_points)))
    return sx, sy


def _surface3d(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    fld: np.ndarray,
    cmap: str,
    fig,
    zlabel: str,
    max_points: int = SURFACE_MAX_POINTS,
    view_elev: float = 28.0,
    view_azim: float = -135.0,
):
    """
    Add a downsampled 3D surface plot to an existing 3D axis.
    """
    fld = np.asarray(fld, dtype=float)
    sx, sy = _surface_stride(fld, max_points=max_points)
    Xs = X[::sx, ::sy]
    Ys = Y[::sx, ::sy]
    Fs = fld[::sx, ::sy]

    vmax = float(np.nanmax(np.abs(Fs))) if Fs.size else 0.0
    vmax = max(vmax, 1e-30)
    vmin = 0.0 if cmap == CMAP_POS else -vmax

    surf = ax.plot_surface(
        Xs,
        Ys,
        Fs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08)
    ax.set_xlabel("$x$", fontsize=9, labelpad=5)
    ax.set_ylabel("$y$", fontsize=9, labelpad=5)
    ax.set_zlabel(zlabel, fontsize=9, labelpad=5)
    ax.view_init(elev=view_elev, azim=view_azim)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.55))
    except Exception:
        # set_box_aspect is unavailable in very old matplotlib versions.
        pass
    return surf


def _abs_u_from_npz(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Recover |u| from a saved npz file using the most common field names.
    """
    keys = set(data.files)
    if "abs_u" in keys:
        return np.asarray(data["abs_u"])
    if "u" in keys:
        return np.abs(data["u"])
    if "rho" in keys:
        return np.sqrt(np.maximum(np.asarray(data["rho"]), 0.0))
    raise KeyError("The npz file must contain one of: 'abs_u', 'u', or 'rho'.")


# ─── Figure 1: Recovery fields (fig:fourier_recovery_fields) ─────────────────

def plot_recovery_fields(
    npz_path: str | Path,
    output_dir: str | Path,
    label: str = "recovery_fields",
) -> None:
    """
    Four-panel contour figure: rho, w, v, Q from exp1_recovery.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")

    panels = [
        (data["rho"], r"$\rho = |u_0|^2$", CMAP_POS),
        (data["w"],   r"$w$",               CMAP_SIGN),
        (data["v"],   r"$v$",               CMAP_SIGN),
        (data["Q"],   r"$Q = w_x + v_y$",   CMAP_SIGN),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (fld, title, cmap) in zip(axes.ravel(), panels):
        _contourf(ax, X, Y, fld, cmap, fig)
        ax.set_title(title, fontsize=12)

    fig.suptitle("Fourier long-wave recovery", fontsize=13)
    fig.tight_layout()
    _savefig(fig, outdir, label)



def plot_recovery_surfaces_3d(
    npz_path: str | Path,
    output_dir: str | Path,
    label: str = "recovery_surfaces3d",
) -> None:
    """
    Three-panel 3D surface figure for |u_0| and the long-wave potentials v, w.

    This is intended as a 3D companion to plot_recovery_fields().  If the recovery
    file only stores rho = |u_0|^2, then |u_0| is reconstructed as sqrt(rho).
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    abs_u0 = _abs_u_from_npz(data)

    panels = [
        (abs_u0,   r"$|u_0(x,y)|$", r"$|u_0|$", CMAP_POS),
        (data["v"], r"Potential $v(x,y)$", r"$v$", CMAP_SIGN),
        (data["w"], r"Potential $w(x,y)$", r"$w$", CMAP_SIGN),
    ]

    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    for j, (fld, title, zlabel, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        _surface3d(ax, X, Y, fld, cmap, fig, zlabel=zlabel)
        ax.set_title(title, fontsize=11)

    fig.suptitle(r"3D surfaces: short-wave amplitude and long-wave potentials", fontsize=13)
    _savefig(fig, outdir, label)


# ─── Figures 2a & 2b: Snapshots (fig:baseline_u_snapshots, fig:baseline_longwave_snapshots) ─

def plot_snapshots(
    npz_path: str | Path,
    output_dir: str | Path,
    n_panels: int = 4,
) -> None:
    """
    Two multi-panel figures: |u| snapshots and Q snapshots at selected times.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    times  = data["times"]
    abs_u  = data["abs_u"]
    Q_arr  = data["Q"]

    n = min(n_panels, len(times))
    idx = np.linspace(0, len(times) - 1, n, dtype=int)

    for arr, stem, cmap, suptitle in [
        (abs_u, "snapshots_u", CMAP_POS,  r"Short-wave envelope $|u(x,y,t)|$"),
        (Q_arr, "snapshots_Q", CMAP_SIGN, r"Long-wave coupling $Q(x,y,t)$"),
    ]:
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5),
                                  constrained_layout=True)
        if n == 1:
            axes = [axes]
        for ax, i in zip(axes, idx):
            _contourf(ax, X, Y, arr[i], cmap, fig, n_levels=30)
            ax.set_title(f"$t = {times[i]:.3g}$", fontsize=10)
        fig.suptitle(suptitle, fontsize=12)
        _savefig(fig, outdir, stem)



def plot_snapshot_surfaces_3d(
    npz_path: str | Path,
    output_dir: str | Path,
    n_panels: int = 3,
) -> None:
    """
    3D surface snapshots for |u| and, when available, the potentials v and w.

    Expected arrays in snapshots.npz:
        x, y, times, abs_u
    Optional arrays for potential snapshots:
        v, w

    If v and w are not present, the function still writes the |u| 3D figure and
    prints a short note explaining that potential snapshots were skipped.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    keys = set(data.files)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    times = data["times"]

    fields = [
        (data["abs_u"], "snapshots3d_absu", CMAP_POS,
         r"3D snapshots of the short-wave envelope $|u(x,y,t)|$", r"$|u|$"),
    ]
    if "re_u" in keys:
        fields.append((data["re_u"], "snapshots3d_reu", CMAP_SIGN,
                       r"3D snapshots of $\mathrm{Re}(u(x,y,t))$", r"$\mathrm{Re}(u)$"))
    if "v" in keys:
        fields.append((data["v"], "snapshots3d_v", CMAP_SIGN,
                       r"3D snapshots of potential $v(x,y,t)$", r"$v$"))
    if "w" in keys:
        fields.append((data["w"], "snapshots3d_w", CMAP_SIGN,
                       r"3D snapshots of potential $w(x,y,t)$", r"$w$"))

    missing = [name for name in ("re_u", "v", "w") if name not in keys]
    if missing:
        print(
            "plot_snapshot_surfaces_3d: skipped fields not present in "
            f"snapshots.npz: {missing}."
        )

    n = min(n_panels, len(times))
    idx = np.linspace(0, len(times) - 1, n, dtype=int)

    for arr, stem, cmap, suptitle, zlabel in fields:
        fig = plt.figure(figsize=(4.8 * n, 4.8), constrained_layout=True)
        for j, i in enumerate(idx, start=1):
            ax = fig.add_subplot(1, n, j, projection="3d")
            _surface3d(ax, X, Y, arr[i], cmap, fig, zlabel=zlabel)
            ax.set_title(f"$t = {times[i]:.3g}$", fontsize=10)
        fig.suptitle(suptitle, fontsize=12)
        _savefig(fig, outdir, stem)


# ─── Figure 3: Drift curves (fig:invariant_drift_curves) ─────────────────────

def plot_drift_curves(
    csv_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Four-panel semi-log figure: RE_M, AD_Jx, AD_Jy, RE_E versus time.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items() if v != ""})

    t   = np.array([r["t"]   for r in rows])
    M   = np.array([r["M"]   for r in rows])
    Jx  = np.array([r["Jx"]  for r in rows])
    Jy  = np.array([r["Jy"]  for r in rows])
    E   = np.array([r["E_reduced"] for r in rows])

    eps_M = 1e-14 * max(1.0, abs(M[0]))
    eps_E = 1e-14 * max(1.0, abs(E[0]))

    RE_M  = np.abs(M  - M[0])  / (abs(M[0])  + eps_M)
    AD_Jx = np.abs(Jx - Jx[0])
    AD_Jy = np.abs(Jy - Jy[0])
    RE_E  = np.abs(E  - E[0])  / (abs(E[0])  + eps_E)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    panels = [
        (axes[0, 0], RE_M,  r"$\mathrm{RE}_M$",     "C0"),
        (axes[0, 1], AD_Jx, r"$\mathrm{AD}_{J_x}$", "C1"),
        (axes[1, 0], AD_Jy, r"$\mathrm{AD}_{J_y}$", "C2"),
        (axes[1, 1], RE_E,  r"$\mathrm{RE}_E$",     "C3"),
    ]
    for ax, arr, ylabel, color in panels:
        ax.semilogy(t, np.maximum(arr, 1e-20), color=color, linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel("$t$", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Invariant drift diagnostics", fontsize=13)
    fig.tight_layout()
    _savefig(fig, outdir, "drift_curves")


# ─── Figure 4: Time-step convergence (fig:timestep_refinement_plot) ──────────

def plot_timestep_convergence(
    json_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Log-log plot of ||e_dt||_h vs dt with a reference slope-2 line.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        results = json.load(f)

    valid = [r for r in results
             if r.get("error_h") is not None and r["error_h"] > 0.0]
    dt_arr = np.array([r["dt"]      for r in valid])
    e_arr  = np.array([r["error_h"] for r in valid])

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    ax.loglog(dt_arr, e_arr, "o-", label=r"$\|e_{\Delta t}\|_h$", linewidth=1.5)

    if len(dt_arr) >= 2:
        ref_dt = np.array([dt_arr[0], dt_arr[-1]])
        ref_e  = e_arr[0] * (ref_dt / dt_arr[0]) ** 2
        ax.loglog(ref_dt, ref_e, "--", color="gray", label="slope 2", linewidth=1.2)

    ax.set_xlabel(r"$\Delta t$", fontsize=12)
    ax.set_ylabel(r"$\|e_{\Delta t}\|_h$", fontsize=12)
    ax.set_title("Time-step refinement — FFT–Strang baseline", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, outdir, "timestep_convergence")


# ─── Figure 5: Spectral resolution (fig:spectral_resolution_plot) ────────────

def plot_spectrum_resolution(
    exp5_dir: str | Path,
    nx_ny_pairs: list,
    output_dir: str | Path,
) -> None:
    """
    Semi-log plot of |u_hat| along the kx-midline for different grid sizes.
    exp5_dir: directory containing uhat_NxNy.npy files from exp 5.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(exp5_dir)

    fig, ax = plt.subplots(figsize=(FIGW * 1.4, FIGH))
    for Nx, Ny in nx_ny_pairs:
        path = src / f"uhat_{Nx}x{Ny}.npy"
        if not path.exists():
            continue
        uhat = np.load(path)          # shape (Nx, Ny), magnitude
        mid  = uhat[:, 0]             # midline along kx (k_y = 0, zero-frequency row)
        norm = mid[0] if mid[0] > 0 else 1.0
        k_idx = np.arange(len(mid))
        ax.semilogy(k_idx[: Nx // 2], mid[: Nx // 2] / norm,
                    label=f"$N_x = N_y = {Nx}$")

    ax.set_xlabel("Mode index $p$", fontsize=12)
    ax.set_ylabel(r"$|\hat{u}_{p,0}| / |\hat{u}_{0,0}|$", fontsize=12)
    ax.set_title("Fourier spectrum — $k_x$ midline", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, outdir, "spectrum_resolution")


# ─── Figure 6: Cost scaling (fig:computational_scaling) ──────────────────────

def plot_cost_scaling(
    json_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Log-log plot of C_step vs Nx*Ny for both recovery modes,
    with a reference N log N line.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        results = json.load(f)

    modes: Dict[str, Dict] = {}
    for r in results:
        mode = r["recovery_mode"]
        if mode not in modes:
            modes[mode] = {"N": [], "C_step": []}
        modes[mode]["N"].append(r["Nx"] * r["Ny"])
        modes[mode]["C_step"].append(r["C_step"])

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    colors = {"Q_only": "C0", "full_wvQ": "C1"}
    labels = {"Q_only": "Q-only (6 FFTs/step)",
              "full_wvQ": r"Full $w,v,Q$ (8 FFTs/step)"}

    for mode, d in modes.items():
        N = np.array(sorted(d["N"]))
        C = np.array([d["C_step"][d["N"].index(n)] for n in N])
        ax.loglog(N, C, "o-", color=colors.get(mode, "C2"),
                  label=labels.get(mode, mode), linewidth=1.5)

    # Reference N log N
    if modes:
        d0 = list(modes.values())[0]
        N_ref = np.array(sorted(d0["N"]))
        if len(N_ref) >= 2:
            C_ref0 = d0["C_step"][d0["N"].index(int(N_ref[0]))]
            N0 = int(N_ref[0])
            C_ref = C_ref0 * (N_ref / N0) * (np.log(N_ref) / math.log(max(N0, 2)))
            ax.loglog(N_ref, C_ref, "--", color="gray",
                      label=r"$\mathcal{O}(N\log N)$", linewidth=1.2)

    ax.set_xlabel("$N_x N_y$", fontsize=12)
    ax.set_ylabel("$C_{\\mathrm{step}}$ (s)", fontsize=12)
    ax.set_title("Computational cost per step", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, outdir, "cost_scaling")


# ─────────────────── Figure 7: Babaoglu exact benchmark ─────────────────────

def plot_babaoglu_exact_convergence(
    json_path: str | Path,
    output_dir: str | Path,
    label: str = "fig7_babaoglu_exact_convergence",
) -> None:
    """
    Plot phase-aligned exact-solution error against dt for Exp 8.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        return

    cases = sorted({r["case"] for r in rows})
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    for case in cases:
        data = sorted([r for r in rows if r["case"] == case], key=lambda r: r["dt"])
        dt = np.array([float(r["dt"]) for r in data])
        err = np.array([float(r["u_phase_aligned_L2_relative_error"]) for r in data])
        positive = err > 0.0
        if np.any(positive):
            ax.loglog(dt[positive], err[positive], "o-", linewidth=1.5, label=case)

    ax.set_xlabel(r"$\Delta t$", fontsize=12)
    ax.set_ylabel(r"phase-aligned relative $L^2$ error", fontsize=12)
    ax.set_title("Babaoglu--Erbay exact benchmark", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, outdir, label)


def plot_babaoglu_final_comparison(
    npz_path: str | Path,
    output_dir: str | Path,
    label: str,
) -> None:
    """
    Four-panel final-state comparison for one Exp 8 case.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")

    fields = [
        (data["abs_u_num"], r"$|u_h(T)|$", CMAP_POS),
        (data["abs_u_exact"], r"$|u_{\rm ex}(T)|$", CMAP_POS),
        (data["abs_u_error"], r"$|u_h|-|u_{\rm ex}|$", CMAP_SIGN),
        (data["Q_error"], r"$Q_h-Q_{\rm ex}$", CMAP_SIGN),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(2 * FIGW, 2 * FIGH), constrained_layout=True)
    for ax, (fld, title, cmap) in zip(axs.ravel(), fields):
        _contourf(ax, X, Y, np.asarray(fld), cmap, fig)
        ax.set_title(title, fontsize=11)
    _savefig(fig, outdir, label)



def _field_from_npz_with_fallback(
    data: np.lib.npyio.NpzFile,
    primary_key: str,
    fallback_key: str | None,
    shape_like: np.ndarray,
) -> np.ndarray:
    """
    Load a field from an Exp 8 final_comparison.npz file.

    Newer Exp 8 outputs contain numerical potentials v_num and w_num.  Older
    outputs may only contain the raw exact travelling-wave references
    v_raw_exact and w_raw_exact.  This helper makes the plotting routine
    backward compatible instead of silently skipping the 3D figure.
    """
    keys = set(data.files)
    if primary_key in keys:
        return np.asarray(data[primary_key], dtype=float)
    if fallback_key is not None and fallback_key in keys:
        return np.asarray(data[fallback_key], dtype=float)
    print(
        f"Warning: '{primary_key}' not found"
        + (f" and fallback '{fallback_key}' not found" if fallback_key else "")
        + "; using zeros for this panel."
    )
    return np.zeros_like(shape_like, dtype=float)


def plot_babaoglu_final_surfaces_3d(
    npz_path: str | Path,
    output_dir: str | Path,
    label: str,
    case_label: Optional[str] = None,
) -> None:
    """
    Three-panel 3D final-state figure for Exp 8: |u|, active potential(s), Q.

    For axis-aligned stripe cases one long-wave potential is identically zero:
    - x_stripe (k1=1, k2=0): v_num ≡ 0  → replaced by Q_num
    - y_stripe (k1=0, k2=1): w_num ≡ 0  → replaced by Q_num
    The replacement is automatic: any potential whose max-abs is below 1e-5 of
    the field scale is treated as trivial and swapped for Q_num.

    case_label controls the 3D viewing angle:
    - "x_stripe": azim=-90  (cross-section perpendicular to x-ridge)
    - "y_stripe": azim=0    (cross-section perpendicular to y-ridge)
    - None / other: azim=-135 (default diagonal)
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    keys = set(data.files)

    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")

    if "abs_u_num" in keys:
        abs_u = np.asarray(data["abs_u_num"], dtype=float)
    elif "u_num" in keys:
        abs_u = np.abs(data["u_num"])
    else:
        raise KeyError(
            f"{npz_path} must contain 'abs_u_num' or 'u_num' to build the 3D Babaoglu plot."
        )

    v_field = _field_from_npz_with_fallback(data, "v_num", "v_raw_exact", abs_u)
    w_field = _field_from_npz_with_fallback(data, "w_num", "w_raw_exact", abs_u)

    # Decide which potentials are non-trivial.
    u_scale = max(float(np.max(abs_u)), 1e-30)
    tol = 1e-5 * u_scale
    potential_panels: list = []
    if float(np.max(np.abs(w_field))) > tol:
        potential_panels.append((w_field, r"$w_h(T)$", r"$w$", CMAP_SIGN))
    if float(np.max(np.abs(v_field))) > tol:
        potential_panels.append((v_field, r"$v_h(T)$", r"$v$", CMAP_SIGN))

    # Fill any remaining slot with Q_num (always non-trivial for stripe waves).
    if len(potential_panels) < 2:
        Q_field: Optional[np.ndarray] = None
        for q_key in ("Q_num", "Q_exact"):
            if q_key in keys:
                Q_field = np.asarray(data[q_key], dtype=float)
                break
        if Q_field is not None:
            potential_panels.append(
                (Q_field, r"$Q_h(T) = w_x + v_y$", r"$Q$", CMAP_SIGN)
            )

    panels = [
        (abs_u, r"$|u_h(T)|$", r"$|u|$", CMAP_POS),
        *potential_panels[:2],
    ]

    # View angle: look perpendicular to the stripe so the sech profile is visible.
    if case_label == "x_stripe":
        view_azim = -90.0   # looking from -y: shows x cross-section
    elif case_label == "y_stripe":
        view_azim = 0.0     # looking from +x: shows y cross-section
    else:
        view_azim = -135.0  # default diagonal
    view_elev = 28.0

    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    for j, (fld, title, zlabel, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        _surface3d(ax, X, Y, fld, cmap, fig, zlabel=zlabel,
                   view_elev=view_elev, view_azim=view_azim)
        ax.set_title(title, fontsize=11)

    fig.suptitle(r"Babaoglu--Erbay benchmark: 3D final-state fields", fontsize=13)
    _savefig(fig, outdir, label)
    print(f"Saved Babaoglu 3D surface figure: {outdir / (label + '.pdf')} and .png")


# ──────────── Figure 8b: Babaoglu spatial convergence (fig:babaoglu_spatial) ─

def plot_babaoglu_spatial_convergence(
    json_path: str | Path,
    output_dir: str | Path,
    label: str = "fig8b_babaoglu_spatial_convergence",
) -> None:
    """
    Log-linear plot of phase-aligned L2 error vs N_x for the Exp 8b study.

    Spectral (exponential) convergence appears as a nearly straight line on the
    log-linear scale.  An inset table with the approximate polynomial rate is
    printed to the figure to quantify the steep decrease.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        return

    cases = sorted({r["case"] for r in rows})
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))

    # First pass: plot all curves and collect per-case data.
    case_data = {}
    for case in cases:
        case_rows = sorted(
            [r for r in rows if r["case"] == case],
            key=lambda r: int(r["Nx"]),
        )
        N   = np.array([int(r["Nx"])   for r in case_rows])
        err = np.array([float(r["u_phase_aligned_L2_relative_error"]) for r in case_rows])
        pos = err > 0.0
        if np.any(pos):
            ax.semilogy(N[pos], err[pos], "o-", linewidth=1.5, markersize=5,
                        label=case)
            case_data[case] = (N, err, pos, case_rows)

    # Second pass: annotate each N point only once to avoid label overlap
    # when multiple cases share the same (N, error) coordinates.
    annotated_N: set = set()
    for case in cases:
        if case not in case_data:
            continue
        N, err, pos, case_rows = case_data[case]
        for i, r in enumerate(case_rows):
            rate = r.get("observed_spatial_rate")
            n_val = int(r["Nx"])
            if rate is not None and pos[i] and n_val not in annotated_N:
                annotated_N.add(n_val)
                ax.annotate(
                    f"  $\\approx${rate:.1f}",
                    xy=(N[i], err[i]),
                    fontsize=7,
                    va="center",
                )

    ax.set_xlabel(r"$N_x = N_y$", fontsize=12)
    ax.set_ylabel(r"phase-aligned relative $L^2$ error", fontsize=12)
    ax.set_title("Babaoglu--Erbay exact benchmark: spatial convergence", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _savefig(fig, outdir, label)


# ──────────────────────────────── Convenience wrapper ────────────────────────

def plot_all(
    output_root=_HERE / "outputs",
    nx_ny_pairs_for_spectrum: Optional[list] = None,
) -> None:
    """
    Generate all figures from saved experiment outputs.
    Expects the directory structure produced by run_all_experiments().
    """
    root   = Path(output_root)
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Fig 1: Recovery fields
    npz1 = root / "exp1_recovery" / "fields_baseline.npz"
    if npz1.exists():
        plot_recovery_fields(npz1, figdir, "fig1_recovery_fields")
        plot_recovery_surfaces_3d(npz1, figdir, "fig1b_recovery_surfaces3d")

    # Figs 2a & 2b: Snapshots
    npz2 = root / "exp2_baseline" / "snapshots.npz"
    if npz2.exists():
        plot_snapshots(npz2, figdir)
        plot_snapshot_surfaces_3d(npz2, figdir)

    # Fig 3: Drift curves
    csv3 = root / "exp2_baseline" / "history.csv"
    if csv3.exists():
        plot_drift_curves(csv3, figdir)

    # Fig 4: Time-step convergence
    json4 = root / "exp4_timestep" / "results.json"
    if json4.exists():
        plot_timestep_convergence(json4, figdir)

    # Fig 5: Spectral resolution
    json5 = root / "exp5_spatial" / "results.json"
    if json5.exists():
        with open(json5) as f:
            res5 = json.load(f)
        pairs = nx_ny_pairs_for_spectrum or [(r["Nx"], r["Ny"]) for r in res5]
        plot_spectrum_resolution(root / "exp5_spatial", pairs, figdir)

    # Fig 6: Cost scaling
    json7 = root / "exp7_cost" / "results.json"
    if json7.exists():
        plot_cost_scaling(json7, figdir)

    # Fig 7: Babaoglu--Erbay exact benchmark
    # Supports both structures:
    #   output_root/exp8_babaoglu_exact/results.json
    #   output_root/results.json  (when output_root points directly to Exp 8)
    exp8_candidates = []
    nested_exp8 = root / "exp8_babaoglu_exact"
    if (nested_exp8 / "results.json").exists():
        exp8_candidates.append(nested_exp8)
    if (root / "results.json").exists() and any((root / d / "final_comparison.npz").exists() for d in ("x_stripe", "y_stripe")):
        exp8_candidates.append(root)

    for exp8_dir in exp8_candidates:
        json8 = exp8_dir / "results.json"
        if json8.exists():
            suffix = "" if exp8_dir.name == "exp8_babaoglu_exact" else f"_{exp8_dir.name}"
            plot_babaoglu_exact_convergence(json8, figdir, f"fig7_babaoglu_exact_convergence{suffix}")

        for case_dir in sorted(p for p in exp8_dir.iterdir() if p.is_dir()):
            npz8 = case_dir / "final_comparison.npz"
            if npz8.exists():
                safe_label = case_dir.name.replace(" ", "_")
                plot_babaoglu_final_comparison(
                    npz8, figdir, f"fig7b_babaoglu_{safe_label}_comparison")
                plot_babaoglu_final_surfaces_3d(
                    npz8, figdir, f"fig7c_babaoglu_{safe_label}_surfaces3d",
                    case_label=case_dir.name)

    if not exp8_candidates:
        print("Exp 8 figures skipped: no Babaoglu results folder found under", root)

    # Fig 8b: Babaoglu spatial convergence
    json8b = root / "exp8b_babaoglu_spatial" / "results.json"
    if json8b.exists():
        plot_babaoglu_spatial_convergence(json8b, figdir)

    print(f"All figures saved to {figdir}")


if __name__ == "__main__":
    plot_all(output_root=_HERE / "outputs")
