"""
gdss_experiments.py — Orchestration of the numerical experiments.

Maps to sections in the paper:
    Exp 1  §fourier_recovery_verification
    Exp 2  §baseline_gdss_simulation            (snapshot saving)
    Exp 3  §invariant_drift_diagnostics         (same run as Exp 2)
    Exp 4  §time_step_refinement
    Exp 5  §spatial_resolution_study
    Exp 6  §boundary_periodicity_diagnostics
    Exp 7  §computational_cost
    Exp 8  §babaoglu_exact_benchmark

Usage:
    from gdss_experiments import run_all_experiments
    run_all_experiments(params, output_root=_HERE / "outputs")
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Anchor all output paths to this module's directory so that
# results are saved next to the code files regardless of where
# the script is launched from.
_HERE = Path(__file__).resolve().parent

from gdss_solver import (
    Array, GDSSParams,
    make_grid, precompute_multipliers,
    gaussian_initial_data,
    recover_longwave, strang_step,
    mass, momentum, energy_reduced,
    longwave_interaction_residual, zero_mode_residual, hermitian_defect,
    longwave_residuals, boundary_amplitude, spectral_tail_ratio,
    diagnostics_row, summarize_history,
)


# ───────────────────────────────────────────────────────────── helpers ───────

def _u0(g, amplitude=1.0, width=4.0, kx0=0.2, ky0=-0.1):
    """Default Gaussian initial profile used across all experiments."""
    return gaussian_initial_data(g, amplitude=amplitude, width=width,
                                  kx0=kx0, ky0=ky0)


def _eps_I(v0):
    """Regularization: 1e-14 * max(1, |I_0|), consistent with the manuscript."""
    return 1e-14 * max(1.0, abs(v0))


def _normalized_cost(cpu, Nt, Nx, Ny):
    """C_norm = T_CPU / (Nt * Nx * Ny * log(Nx * Ny))."""
    denom = Nt * Nx * Ny * math.log(max(Nx * Ny, 2))
    return cpu / denom if denom > 0 else float("nan")


def _write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(str(k) for k in keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def _fmt_float(x, digits: int = 6) -> str:
    """Compact scientific notation for CSV/console/table values."""
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return str(xf)
    return f"{xf:.{digits}e}"


def _fmt_latex_sci(x, digits: int = 3) -> str:
    """Return a math-mode LaTeX scientific-notation entry."""
    if x is None:
        return r"--"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return str(xf)
    if xf == 0.0:
        return r"$0$"
    mantissa, exponent = f"{xf:.{digits}e}".split("e")
    exp_int = int(exponent)
    return rf"${mantissa}\times 10^{{{exp_int}}}$"


def _fmt_latex_case(label: str) -> str:
    """Return a \\texttt{}-wrapped case label with escaped underscores."""
    return r"\texttt{" + str(label).replace("_", r"\_") + "}"


def _write_table_csv(rows, path: str | Path) -> None:
    """Write a list of dictionaries with proper CSV escaping."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _invariant_balance_rows(rows: list) -> list:
    """
    Build one row per saved time level with invariant values and drifts.

    Full data are written to CSV.  A representative subset is later written to
    LaTeX so the paper does not contain hundreds of time rows.
    """
    if not rows:
        return []

    first = rows[0]
    M0 = float(first["M"])
    Jx0 = float(first["Jx"])
    Jy0 = float(first["Jy"])
    E0 = float(first["E_reduced"])
    E_lw0 = float(first["E_longwave_quadratic"]) if "E_longwave_quadratic" in first else None

    out = []
    for r in rows:
        M = float(r["M"])
        Jx = float(r["Jx"])
        Jy = float(r["Jy"])
        E = float(r["E_reduced"])
        row = {
            "n": int(r["n"]),
            "t": float(r["t"]),
            "M": M,
            "AD_M": abs(M - M0),
            "RE_M": abs(M - M0) / (abs(M0) + _eps_I(M0)),
            "Jx": Jx,
            "AD_Jx": abs(Jx - Jx0),
            "RE_Jx": abs(Jx - Jx0) / (abs(Jx0) + _eps_I(Jx0)),
            "Jy": Jy,
            "AD_Jy": abs(Jy - Jy0),
            "RE_Jy": abs(Jy - Jy0) / (abs(Jy0) + _eps_I(Jy0)),
            "E_reduced": E,
            "AD_E_reduced": abs(E - E0),
            "RE_E_reduced": abs(E - E0) / (abs(E0) + _eps_I(E0)),
        }
        if E_lw0 is not None and "E_longwave_quadratic" in r:
            E_lw = float(r["E_longwave_quadratic"])
            row.update({
                "E_longwave_quadratic": E_lw,
                "AD_E_longwave_quadratic": abs(E_lw - E_lw0),
                "RE_E_longwave_quadratic": abs(E_lw - E_lw0) / (abs(E_lw0) + _eps_I(E_lw0)),
            })
        for key in ("R_lw", "Z_lw", "H_defect", "res_w", "res_v", "tail_ratio", "boundary_max"):
            if key in r:
                row[key] = float(r[key])
        out.append(row)
    return out


def _representative_rows(rows: list, max_rows: int = 8) -> list:
    """Select evenly spaced rows for a compact LaTeX table."""
    if len(rows) <= max_rows:
        return rows
    idx = np.linspace(0, len(rows) - 1, max_rows, dtype=int)
    # np.linspace can repeat indices for small lists; preserve order and uniqueness.
    seen = set()
    selected = []
    for i in idx:
        if int(i) not in seen:
            selected.append(rows[int(i)])
            seen.add(int(i))
    return selected


def _write_invariant_summary_table(summary: dict, table_dir: str | Path) -> None:
    """
    Write the paper-ready compact invariant-drift table.

    Output:
        invariant_balance_summary.csv
        invariant_balance_summary.tex
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    invariant_specs = [
        ("M", "Mass", r"$M_h$"),
        ("Jx", "Momentum x", r"$J_{x,h}$"),
        ("Jy", "Momentum y", r"$J_{y,h}$"),
        ("E_reduced", "Reduced energy", r"$E_h^{\mathrm{red}}$"),
        ("E_longwave_quadratic", "Long-wave quadratic energy", r"$E_h^{\mathrm{lw}}$"),
    ]

    csv_rows = []
    tex_rows = []
    for key, plain_label, latex_label in invariant_specs:
        k0 = f"{key}_0"
        kad = f"AD_{key}_max"
        kre = f"RE_{key}_max"
        if k0 not in summary:
            continue
        csv_rows.append({
            "invariant": plain_label,
            "initial_value": summary.get(k0),
            "max_absolute_drift": summary.get(kad),
            "max_relative_drift": summary.get(kre),
        })
        tex_rows.append(
            f"{latex_label} & "
            f"{_fmt_latex_sci(summary.get(k0))} & "
            f"{_fmt_latex_sci(summary.get(kad))} & "
            f"{_fmt_latex_sci(summary.get(kre))} \\\\"
        )

    _write_table_csv(csv_rows, table_dir / "invariant_balance_summary.csv")

    tex = r"""% chktex-file 8
% chktex-file 24
\begin{table}[!ht]
\centering
\caption{Maximum invariant-balance defects over the baseline GDSS simulation.}
\label{tab:invariant_balance_summary}
\begin{tabular}{lccc}
\toprule
Invariant & Initial value & Max.\ absolute defect & Max.\ relative defect \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (table_dir / "invariant_balance_summary.tex").write_text(tex, encoding="utf-8")


def _write_invariant_time_table(balance_rows: list, table_dir: str | Path, max_rows: int = 8) -> None:
    """
    Write full time-level balance data as CSV and a compact LaTeX sample table.

    The CSV file contains all saved diagnostic times.  The LaTeX file contains
    a representative subset suitable for the manuscript.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    _write_table_csv(balance_rows, table_dir / "invariant_balance_times.csv")

    tex_rows = []
    for r in _representative_rows(balance_rows, max_rows=max_rows):
        tex_rows.append(
            f"{int(r['n'])} & "
            f"{_fmt_latex_sci(r['t'])} & "
            f"{_fmt_latex_sci(r['RE_M'])} & "
            f"{_fmt_latex_sci(r['AD_Jx'])} & "
            f"{_fmt_latex_sci(r['AD_Jy'])} & "
            f"{_fmt_latex_sci(r['RE_E_reduced'])} \\\\"
        )

    tex = r"""% chktex-file 8
% chktex-file 24
\begin{table}[!ht]
\centering
\caption{Representative time-level invariant-balance defects for the baseline GDSS simulation.}
\label{tab:invariant_balance_time_levels}
\begin{tabular}{cccccc}
\toprule
$n$ & $t_n$ & $\mathrm{RE}_{M}$ & $\mathrm{AD}_{J_x}$ & $\mathrm{AD}_{J_y}$ & $\mathrm{RE}_{E}$ \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (table_dir / "invariant_balance_times.tex").write_text(tex, encoding="utf-8")


def _write_longwave_balance_table(summary: dict, table_dir: str | Path) -> None:
    """
    Write a compact table for auxiliary long-wave/balance diagnostics.

    These quantities are not invariants, but they support the invariant study by
    checking recovery consistency, zero-mode enforcement, Hermitian symmetry,
    boundary contamination, and spectral resolution.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("R_lw_max", "Long-wave interaction residual", r"$R_{\mathrm{lw}}$"),
        ("Z_lw_max", "Zero-mode residual", r"$Z_{\mathrm{lw}}$"),
        ("H_defect_max", "Hermitian symmetry defect", r"$H_f$"),
        ("res_w_max", "First long-wave equation residual", r"$\mathrm{res}_w$"),
        ("res_v_max", "Second long-wave equation residual", r"$\mathrm{res}_v$"),
        ("tail_ratio_max", "Spectral tail ratio", r"$T_u$"),
        ("boundary_max", "Boundary-band amplitude", r"$B_{\partial}$"),
    ]

    csv_rows = []
    tex_rows = []
    for key, plain_label, latex_label in specs:
        if key not in summary:
            continue
        csv_rows.append({"diagnostic": plain_label, "symbol": latex_label, "maximum_value": summary[key]})
        tex_rows.append(f"{latex_label} & {plain_label} & {_fmt_latex_sci(summary[key])} \\\\")

    if not csv_rows:
        return

    _write_table_csv(csv_rows, table_dir / "longwave_balance_summary.csv")
    tex = r"""% chktex-file 8
% chktex-file 24
\begin{table}[!ht]
\centering
\caption{Auxiliary numerical-balance diagnostics for the baseline GDSS simulation.}
\label{tab:longwave_balance_summary}
\begin{tabular}{llc}
\toprule
Symbol & Diagnostic & Maximum value \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (table_dir / "longwave_balance_summary.tex").write_text(tex, encoding="utf-8")


def _write_invariant_balance_outputs(rows: list, summary: dict, output_dir: str | Path) -> None:
    """
    Generate all table data for the invariant-balance subsection.

    Files are written to output_dir/tables when called from the baseline run.
    """
    table_dir = Path(output_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    balance_rows = _invariant_balance_rows(rows)
    _write_invariant_summary_table(summary, table_dir)
    _write_invariant_time_table(balance_rows, table_dir)
    _write_longwave_balance_table(summary, table_dir)


def _check_divisibility(t_final, dt):
    """Return Nt = round(t_final/dt), raise if not an integer."""
    Nt = int(round(t_final / dt))
    if abs(Nt * dt - t_final) > 1e-10 * t_final:
        Nt_base = int(round(t_final / dt))
        suggestion = [t_final / (Nt_base * 2**k) for k in range(4)]
        raise ValueError(
            f"t_final={t_final} is not exactly divisible by dt={dt} "
            f"(t_final/dt = {t_final/dt:.6g}, rounded Nt={Nt}, "
            f"residual={abs(Nt*dt - t_final):.2e}).\n"
            "Generate dt_values as t_final/(Nt_base * 2^k) to guarantee "
            "divisibility. Example for your parameters:\n"
            f"  dt_values_for_refinement={suggestion}"
        )
    return Nt


def _default_dt_sequence(t_final: float, dt_base: float, n_levels: int = 4) -> list:
    """
    Generate a dt refinement sequence guaranteed to divide t_final exactly.
    Uses dt_k = t_final / (Nt_base * 2^k), k = 0, ..., n_levels-1.
    dt_base is the coarsest step (k=0); each subsequent step halves dt.
    """
    Nt_base = int(round(t_final / dt_base))
    if abs(Nt_base * dt_base - t_final) > 1e-10 * t_final:
        raise ValueError(
            f"dt_base={dt_base} does not divide t_final={t_final} exactly. "
            "Choose a dt_base such that t_final/dt_base is an integer."
        )
    return [t_final / (Nt_base * 2**k) for k in range(n_levels)]


# ──────────────────────────── Experiment 1: Recovery verification ────────────

def run_recovery_verification(
    p_base,
    refine_factor=2,
    output_dir=_HERE / "outputs/exp1_recovery",
):
    """
    Static test of the Fourier long-wave recovery (no time stepping).
    Runs on the base grid and on a grid with refine_factor times more modes.
    Produces table data for tab:fourier_recovery_diagnostics and
    fields for fig:fourier_recovery_fields.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for factor, label in [(1, "baseline"), (refine_factor, f"refined_x{refine_factor}")]:
        p = replace(p_base, Nx=p_base.Nx * factor, Ny=p_base.Ny * factor)
        g = make_grid(p)
        m = precompute_multipliers(p, g)
        u = _u0(g)

        lw = recover_longwave(u, g, p, m, full=True)

        R_lw = longwave_interaction_residual(lw, g, p)
        Z_lw = zero_mode_residual(lw, g)
        H_def = hermitian_defect(lw)
        res = longwave_residuals(lw, g, p)

        row = {
            "label": label,
            "Nx": p.Nx,
            "Ny": p.Ny,
            "R_lw": R_lw,
            "Z_lw": Z_lw,
            "H_defect": H_def,
            "res_w": res[0] if res else None,
            "res_v": res[1] if res else None,
        }
        results.append(row)

        np.savez_compressed(
            outdir / f"fields_{label}.npz",
            x=g.x, y=g.y,
            rho=lw["rho"], Q=lw["Q"], w=lw["w"], v=lw["v"],
        )

    with open(outdir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    _write_recovery_table(results, outdir)

    print(f"[Exp 1] Recovery verification -> {outdir}")
    return {"results": results, "output_dir": str(outdir)}


def _write_recovery_table(results: list, output_dir: str | Path) -> None:
    """Write CSV + LaTeX table for the Fourier long-wave recovery diagnostics."""
    outdir = Path(output_dir)
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_table_csv(results, table_dir / "recovery_diagnostics.csv")

    tex_rows = []
    for r in results:
        Nx, Ny = r["Nx"], r["Ny"]
        tex_rows.append(
            f"${Nx}\\times{Ny}$ & "
            f"{_fmt_latex_sci(r['R_lw'])} & "
            f"{_fmt_latex_sci(r['Z_lw'])} & "
            f"{_fmt_latex_sci(r['H_defect'])} & "
            f"{_fmt_latex_sci(r.get('res_w'))} & "
            f"{_fmt_latex_sci(r.get('res_v'))} \\\\"
        )

    tex = r"""% chktex-file 8
% chktex-file 24
\begin{table}[!ht]
\centering
\caption{Fourier long-wave recovery diagnostics at $t=0$.
All entries are relative residuals; values at machine precision confirm
spectral exactness of the recovery kernel.}
\label{tab:recovery_diagnostics}
\begin{tabular}{lccccc}
\toprule
Grid & $\mathcal{R}_{\mathrm{lw}}$ & $\mathcal{Z}_{\mathrm{lw}}$ & $\mathcal{H}$ & $\mathrm{res}_w$ & $\mathrm{res}_v$ \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (table_dir / "recovery_diagnostics.tex").write_text(tex, encoding="utf-8")


# ─────────────── Experiments 2 & 3: Baseline simulation with snapshots ───────

def run_simulation_with_snapshots(
    p,
    snapshot_times=None,
    output_dir=_HERE / "outputs/exp2_baseline",
    save_every=1,
):
    """
    Runs the baseline GDSS simulation and saves:
      - history.csv  : diagnostic row at every save_every steps
      - summary.json : max drift values  (tab:invariant_drift_summary)
      - snapshots.npz: full fields at snapshot_times  (figs 2a, 2b)
      - final_fields.npz

    snapshot_times: list of physical times; if None, saves only at t=0 and t=T.
    Covers Experiments 2 (snapshots) and 3 (invariant drift).
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    g = make_grid(p)
    m = precompute_multipliers(p, g)
    u = _u0(g)
    Nt = _check_divisibility(p.t_final, p.dt)

    if snapshot_times is None:
        snapshot_times = [0.0, p.t_final]
    snap_steps = {int(round(ts / p.dt)) for ts in snapshot_times}

    rows = []
    snapshots = []

    def _snap(t_val, u_arr, lw_arr):
        return {
            "t": float(t_val),
            "abs_u": np.abs(u_arr).copy(),
            "re_u":  np.real(u_arr).copy(),   # real part for 3D surface plots
            "Q": lw_arr["Q"].copy(),
            "w": lw_arr.get("w", np.zeros_like(lw_arr["Q"])).copy(),
            "v": lw_arr.get("v", np.zeros_like(lw_arr["Q"])).copy(),
        }

    lw = recover_longwave(u, g, p, m, full=p.recover_full_longwave)
    rows.append(diagnostics_row(0, 0.0, u, lw, g, p))
    if 0 in snap_steps:
        snapshots.append(_snap(0.0, u, lw))

    start = time.perf_counter()
    for n in range(1, Nt + 1):
        u, _ = strang_step(u, g, p, m)
        need_diag = (n % save_every == 0) or (n == Nt) or (n in snap_steps)
        if need_diag:
            lw = recover_longwave(u, g, p, m, full=p.recover_full_longwave)
            if n % save_every == 0 or n == Nt:
                rows.append(diagnostics_row(n, n * p.dt, u, lw, g, p))
            if n in snap_steps:
                snapshots.append(_snap(n * p.dt, u, lw))
    cpu = time.perf_counter() - start

    summary = summarize_history(rows)
    summary["cpu_time_seconds"] = cpu
    summary["Nt"] = Nt
    summary["C_step"] = cpu / Nt
    summary["C_norm"] = _normalized_cost(cpu, Nt, p.Nx, p.Ny)

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _write_csv(rows, outdir / "history.csv")
    _write_invariant_balance_outputs(rows, summary, outdir / "tables")

    if snapshots:
        np.savez_compressed(
            outdir / "snapshots.npz",
            times=np.array([s["t"] for s in snapshots]),
            abs_u=np.stack([s["abs_u"] for s in snapshots]),
            re_u =np.stack([s["re_u"]  for s in snapshots]),
            Q=np.stack([s["Q"] for s in snapshots]),
            w=np.stack([s["w"] for s in snapshots]),
            v=np.stack([s["v"] for s in snapshots]),
            x=g.x, y=g.y,
        )

    final_lw = recover_longwave(u, g, p, m, full=True)
    np.savez_compressed(
        outdir / "final_fields.npz",
        x=g.x, y=g.y, u=u, abs_u=np.abs(u),
        rho=np.abs(u) ** 2,
        Q=final_lw["Q"], w=final_lw["w"], v=final_lw["v"],
    )

    print(f"[Exp 2/3] Baseline simulation -> {outdir}")
    return {"summary": summary, "rows": rows, "output_dir": str(outdir)}


# ─────────────────────────── Experiment 4: Time-step refinement ──────────────

def run_time_step_study(
    p_base,
    dt_values,
    output_dir=_HERE / "outputs/exp4_timestep",
):
    """
    Runs the solver for each dt (same spatial grid, same t_final).
    Reference: smallest dt (finest).
    Computes e_dt = ||u_dt(T) - u_ref(T)||_h and observed order p_i.

    dt_values: in any order; sorted internally coarsest-to-finest.
    t_final / dt must be integer for every dt.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Descending order: coarsest first, finest (reference) last.
    dt_sorted = sorted(dt_values, reverse=True)
    dt_ref = dt_sorted[-1]

    g = make_grid(p_base)  # spatial grid fixed

    # Initial diagnostics (same for all dt)
    u0_arr = _u0(g)
    p_tmp = replace(p_base, dt=dt_ref)
    m_tmp = precompute_multipliers(p_tmp, g)
    lw0 = recover_longwave(u0_arr, g, p_tmp, m_tmp, full=True)
    M0 = mass(u0_arr, g)
    E0 = energy_reduced(u0_arr, lw0["Q"], g, p_tmp)

    u_finals = {}
    rows = []

    for dt in dt_sorted:
        Nt = _check_divisibility(p_base.t_final, dt)
        p = replace(p_base, dt=dt)
        m = precompute_multipliers(p, g)
        u = _u0(g)

        t0 = time.perf_counter()
        for _ in range(Nt):
            u, _ = strang_step(u, g, p, m)
        cpu = time.perf_counter() - t0

        lw = recover_longwave(u, g, p, m, full=True)
        M_f = mass(u, g)
        E_f = energy_reduced(u, lw["Q"], g, p)

        u_finals[dt] = u.copy()
        rows.append({
            "dt": dt,
            "Nt": Nt,
            "cpu_time_seconds": cpu,
            "RE_M_max": abs(M_f - M0) / (abs(M0) + _eps_I(M0)),
            "RE_E_max": abs(E_f - E0) / (abs(E0) + _eps_I(E0)),
            "error_h": None,
            "observed_order": None,
        })

    # Errors against reference
    u_ref = u_finals[dt_ref]
    for row in rows:
        dt = row["dt"]
        if dt == dt_ref:
            row["error_h"] = 0.0
        else:
            diff = u_finals[dt] - u_ref
            row["error_h"] = float(
                np.sqrt(g.dx * g.dy * np.sum(np.abs(diff) ** 2))
            )

    # Observed orders: p_i stored in row i (comparing row i-1 and row i)
    # Row 0 stays None (no previous). Row -1 stays None (reference, e=0).
    for i in range(1, len(rows) - 1):
        e_prev = rows[i - 1]["error_h"]
        e_curr = rows[i]["error_h"]
        dt_prev = rows[i - 1]["dt"]
        dt_curr = rows[i]["dt"]
        if e_prev and e_curr and e_prev > 0.0 and e_curr > 0.0:
            rows[i]["observed_order"] = (
                math.log(e_prev / e_curr) / math.log(dt_prev / dt_curr)
            )

    with open(outdir / "results.json", "w") as f:
        json.dump(rows, f, indent=2)
    _write_csv(rows, outdir / "results.csv")
    _write_timestep_convergence_table(rows, outdir)

    print(f"[Exp 4] Time-step refinement -> {outdir}")
    return {"results": rows, "output_dir": str(outdir)}


def _write_timestep_convergence_table(rows: list, output_dir: str | Path) -> None:
    """Write CSV + LaTeX table for the temporal convergence study (Exp 4)."""
    outdir = Path(output_dir)
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_table_csv(rows, table_dir / "timestep_convergence.csv")

    tex_rows = []
    for r in rows:
        order_str = (
            _fmt_latex_sci(r["observed_order"])
            if r["observed_order"] is not None
            else "---"
        )
        err_str = (
            _fmt_latex_sci(r["error_h"])
            if r["error_h"] not in (None, 0.0)
            else "\\text{ref}"
        )
        tex_rows.append(
            f"{_fmt_latex_sci(r['dt'])} & {r['Nt']} & "
            f"{err_str} & {order_str} \\\\"
        )

    tex = r"""% chktex-file 8
% chktex-file 24
\begin{table}[!ht]
\centering
\caption{Temporal convergence study. $e_h(\Delta t)$ is the discrete $L^2$ error
against the finest-$\Delta t$ reference solution at $t = T$.}
\label{tab:timestep_convergence}
\begin{tabular}{cccc}
\toprule
$\Delta t$ & $N_t$ & $e_h(\Delta t)$ & Observed order \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (table_dir / "timestep_convergence.tex").write_text(tex, encoding="utf-8")


# ─────────────────────────── Experiment 5: Spatial resolution ────────────────

def run_spatial_resolution_study(
    p_base,
    nx_ny_pairs,
    dt_fixed=None,
    output_dir=_HERE / "outputs/exp5_spatial",
):
    """
    Runs for different (Nx, Ny) with a fixed dt.
    Reports spectral tail ratio and invariant drifts.
    Saves Fourier spectra for fig:spectral_resolution_plot.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for Nx, Ny in nx_ny_pairs:
        dt = dt_fixed if dt_fixed is not None else p_base.dt
        p = replace(p_base, Nx=Nx, Ny=Ny, dt=dt)
        Nt = _check_divisibility(p.t_final, p.dt)
        g = make_grid(p)
        m = precompute_multipliers(p, g)
        u = _u0(g)

        lw0 = recover_longwave(u, g, p, m, full=True)
        M0 = mass(u, g)
        E0 = energy_reduced(u, lw0["Q"], g, p)

        for _ in range(Nt):
            u, _ = strang_step(u, g, p, m)
        # Compute tail ratio once at the final state;
        # calling it every step would add one FFT per step unnecessarily.
        tail_max = spectral_tail_ratio(u, g)

        lw = recover_longwave(u, g, p, m, full=True)
        M_f = mass(u, g)
        E_f = energy_reduced(u, lw["Q"], g, p)

        label = f"{Nx}x{Ny}"
        np.save(outdir / f"uhat_{label}.npy", np.abs(np.fft.fft2(u)))

        results.append({
            "Nx": Nx, "Ny": Ny,
            "dx": g.dx, "dy": g.dy,
            "tail_ratio_max": tail_max,
            "RE_M_max": abs(M_f - M0) / (abs(M0) + _eps_I(M0)),
            "RE_E_max": abs(E_f - E0) / (abs(E0) + _eps_I(E0)),
        })

    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    _write_csv(results, outdir / "results.csv")

    print(f"[Exp 5] Spatial resolution study -> {outdir}")
    return {"results": results, "output_dir": str(outdir)}


# ──────────────────────────── Experiment 6: Boundary diagnostics ─────────────

def run_boundary_study(
    p_base,
    lx_ly_pairs,
    band=4,
    output_dir=_HERE / "outputs/exp6_boundary",
):
    """
    Runs for different (Lx, Ly) with same (Nx, Ny) and dt.
    Reports B_partial_max (eq. max_boundary_amplitude).
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for Lx, Ly in lx_ly_pairs:
        p = replace(p_base, Lx=Lx, Ly=Ly)
        Nt = _check_divisibility(p.t_final, p.dt)
        g = make_grid(p)
        m = precompute_multipliers(p, g)
        u = _u0(g)

        B_max = boundary_amplitude(u, band)
        for _ in range(Nt):
            u, _ = strang_step(u, g, p, m)
            B_max = max(B_max, boundary_amplitude(u, band))

        results.append({
            "Lx": Lx, "Ly": Ly, "band": band, "B_partial_max": B_max,
        })

    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    _write_boundary_table(results, outdir)

    print(f"[Exp 6] Boundary study -> {outdir}")
    return {"results": results, "output_dir": str(outdir)}


def _write_boundary_table(results: list, output_dir: str | Path) -> None:
    """Write CSV + LaTeX table for the periodic-boundary compatibility study (Exp 6)."""
    outdir = Path(output_dir)
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_table_csv(results, table_dir / "boundary_contamination.csv")

    tex_rows = []
    for r in results:
        Lx, Ly = r["Lx"], r["Ly"]
        tex_rows.append(
            f"        $[-{Lx/2:.0f},{Lx/2:.0f})\\times[-{Ly/2:.0f},{Ly/2:.0f})$ & "
            f"{r['band']} & "
            f"{_fmt_latex_sci(r['B_partial_max'])} \\\\"
        )

    tex = r"""% chktex-file 8
% chktex-file 9
% chktex-file 17
% chktex-file 3
% chktex-file 24
\begin{table}[!ht]
    \centering
    \caption{Periodic-boundary compatibility diagnostic. The quantity
    $B_{\partial}^{\max}$ is the maximum of $|u_h|$ over a four-cell
    boundary band and over all saved time levels. The small values indicate
    negligible interaction between the localized packet and the periodic
    boundary over $0\leq t\leq 2$.}
    \label{tab:boundary_contamination}
    \begin{tabular}{lcc}
        \toprule
        Domain & Band (cells) & $B_{\partial}^{\max}$ \\
        \midrule
""" + "\n".join(tex_rows) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    (table_dir / "boundary_contamination.tex").write_text(tex, encoding="utf-8")



# ────────────────────────────── Experiment 7: Computational cost ─────────────

def run_cost_study(
    p_base,
    nx_ny_pairs,
    n_repeats=3,
    output_dir=_HERE / "outputs/exp7_cost",
):
    """
    Measures CPU time for Q-only and full (w,v,Q) recovery at different grids.
    Takes the minimum over n_repeats runs to reduce OS jitter.
    Computes C_step and C_norm (eq. normalized_fft_cost).
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for Nx, Ny in nx_ny_pairs:
        for recover_full, mode_label in [(False, "Q_only"), (True, "full_wvQ")]:
            p = replace(p_base, Nx=Nx, Ny=Ny, recover_full_longwave=recover_full)
            Nt = _check_divisibility(p.t_final, p.dt)
            g = make_grid(p)
            m = precompute_multipliers(p, g)

            cpu_min = float("inf")
            for _ in range(n_repeats):
                u = _u0(g)
                t0 = time.perf_counter()
                for _ in range(Nt):
                    u, _ = strang_step(u, g, p, m)
                cpu_min = min(cpu_min, time.perf_counter() - t0)

            results.append({
                "Nx": Nx, "Ny": Ny,
                "recovery_mode": mode_label,
                "Nt": Nt,
                "ffts_per_step": 8 if recover_full else 6,
                "cpu_time_seconds": cpu_min,
                "C_step": cpu_min / Nt,
                "C_norm": _normalized_cost(cpu_min, Nt, Nx, Ny),
            })

    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    _write_csv(results, outdir / "results.csv")

    print(f"[Exp 7] Cost study -> {outdir}")
    return {"results": results, "output_dir": str(outdir)}



# ─────────────── Experiment 8: Babaoglu exact travelling-wave benchmark ──────

def _babaoglu_default_cases() -> list:
    """
    Periodic-boundary-compatible special cases of the sech--tanh--tanh
    travelling wave reported by Babaoglu and Erbay.

    The fully oblique case k1*k2 != 0 is supported by the formulas below, but
    it is not used as the default because a non-axis-aligned sech stripe crosses
    the edges of a rectangular periodic box.  The two default stripe cases keep
    the short-wave envelope localized with respect to the periodic direction and
    separately exercise the x- and y-long-wave recoveries.
    """
    return [
        {
            "label": "x_stripe",
            "k1": 1.0, "k2": 0.0,
            "l1": 0.20, "l2": 0.0,
            "Lambda": 1.0,
            "zeta0": 0.0,
            "phase0": 0.0,
        },
        {
            "label": "y_stripe",
            "k1": 0.0, "k2": 1.0,
            "l1": 0.0, "l2": -0.10,
            "Lambda": 1.0,
            "zeta0": 0.0,
            "phase0": 0.0,
        },
    ]


def _babaoglu_benchmark_params(p_base: GDSSParams) -> GDSSParams:
    """
    Parameter set used by the exact benchmark.

    It keeps the same grid and time settings as p_base but switches to the
    Babaoglu--Erbay normalized cubic coefficient gamma=2 and to a focusing
    long-wave coupling xi=-4.  The long-wave coefficients satisfy the structural
    condition theta^2=(phi-psi)(eta-chi).
    """
    return replace(
        p_base,
        alpha=1.0, beta=1.0,
        gamma=2.0, xi=-4.0,
        psi=1.0, eta=2.0,
        phi=2.0, chi=1.0,
        theta=1.0,
        dealias_density=False,
        recover_full_longwave=True,
    )


def _babaoglu_coefficients(p: GDSSParams, g, case: dict) -> dict:
    """
    Compute the coefficients of the sech travelling-wave solution.

    Ansatz:
        u = f(zeta) exp(i(l1*x + l2*y - Omega*t + phase0)),
        zeta = k1*x + k2*y - c*t + zeta0,
        f = A0 sech(sqrt(Lambda)*zeta).

    For the periodic FFT recovery, the zero Fourier mode of the density is
    removed.  In the axis-aligned default cases this changes Q from S*rho to
    S*(rho-rho_bar), which is a spatially uniform phase correction.  The
    frequency Omega below includes this discrete mean correction.
    """
    k1 = float(case.get("k1", 1.0))
    k2 = float(case.get("k2", 0.0))
    l1 = float(case.get("l1", 0.0))
    l2 = float(case.get("l2", 0.0))
    Lam = float(case.get("Lambda", 1.0))
    zeta0 = float(case.get("zeta0", 0.0))

    A_disp = p.alpha * k1**2 + p.beta * k2**2
    if A_disp <= 0.0:
        raise ValueError("Babaoglu benchmark requires alpha*k1^2 + beta*k2^2 > 0.")

    a_lw = p.psi * k1**2 + p.eta * k2**2
    b_lw = p.theta * k1 * k2
    c_lw = p.phi * k1**2 + p.chi * k2**2
    Delta = a_lw * c_lw - b_lw**2
    if abs(Delta) <= 1.0e-14:
        raise ValueError("Singular long-wave travelling-wave coefficient matrix.")

    G1 = (c_lw * k1 - b_lw * k2) / Delta
    G2 = (-b_lw * k1 + a_lw * k2) / Delta
    S = k1 * G1 + k2 * G2
    cubic_eff = p.gamma + p.xi * S
    if cubic_eff >= 0.0:
        raise ValueError(
            "The sech benchmark requires gamma + xi*S < 0. "
            f"Got gamma + xi*S = {cubic_eff}."
        )

    amp2 = -2.0 * A_disp * Lam / cubic_eff
    amp = math.sqrt(amp2)
    sqrt_Lam = math.sqrt(Lam)

    # Discrete mean correction for the zero-mode-free periodic recovery.
    zeta_init = k1 * g.X + k2 * g.Y + zeta0
    rho_init = amp2 / np.cosh(sqrt_Lam * zeta_init) ** 2
    rho_bar = float(np.mean(rho_init))

    speed = 2.0 * (p.alpha * k1 * l1 + p.beta * k2 * l2)
    linear_phase = p.alpha * l1**2 + p.beta * l2**2
    Omega = linear_phase - A_disp * Lam - p.xi * S * rho_bar

    Gamma = cubic_eff / (2.0 * A_disp)
    return {
        "k1": k1, "k2": k2, "l1": l1, "l2": l2,
        "Lambda": Lam, "sqrt_Lambda": sqrt_Lam,
        "zeta0": zeta0, "phase0": float(case.get("phase0", 0.0)),
        "A_disp": A_disp,
        "a_lw": a_lw, "b_lw": b_lw, "c_lw": c_lw, "Delta": Delta,
        "G1": G1, "G2": G2, "S": S,
        "Gamma": Gamma, "cubic_eff": cubic_eff,
        "amplitude": amp, "amplitude_squared": amp2,
        "rho_bar": rho_bar,
        "speed": speed,
        "Omega": Omega,
    }


def _babaoglu_exact_fields(g, coeffs: dict, t: float) -> dict:
    """Return exact u, rho, Q and auxiliary raw potentials at time t."""
    k1 = coeffs["k1"]
    k2 = coeffs["k2"]
    l1 = coeffs["l1"]
    l2 = coeffs["l2"]
    sqrt_Lam = coeffs["sqrt_Lambda"]
    amp = coeffs["amplitude"]

    zeta = k1 * g.X + k2 * g.Y - coeffs["speed"] * t + coeffs["zeta0"]
    phase = l1 * g.X + l2 * g.Y - coeffs["Omega"] * t + coeffs["phase0"]
    sech = 1.0 / np.cosh(sqrt_Lam * zeta)
    f = amp * sech
    rho = f**2
    u = f * np.exp(1j * phase)

    # Periodic zero-mode-free Q used by the FFT solver in the default cases.
    Q = coeffs["S"] * (rho - coeffs["rho_bar"])

    # Raw R^2 travelling-wave potentials, useful for visual reference only.
    tanh = np.tanh(sqrt_Lam * zeta)
    w_raw = coeffs["G1"] * coeffs["amplitude_squared"] / sqrt_Lam * tanh
    v_raw = coeffs["G2"] * coeffs["amplitude_squared"] / sqrt_Lam * tanh

    return {
        "u": u,
        "abs_u": np.abs(u),
        "rho": rho,
        "Q": Q,
        "w_raw": w_raw,
        "v_raw": v_raw,
        "zeta": zeta,
        "phase": phase,
    }


def _l2_norm(f: Array, g) -> float:
    return float(np.sqrt(g.dx * g.dy * np.sum(np.abs(f) ** 2)))


def _phase_aligned_error(u_num: Array, u_ex: Array, g) -> tuple[float, float]:
    """
    Return absolute and relative phase-aligned L2 errors.

    The alignment removes one global phase, which is useful because the
    zero-mode-free periodic recovery introduces a spatially uniform phase shift.
    """
    inner = g.dx * g.dy * np.vdot(u_ex, u_num)
    phase = np.angle(inner) if abs(inner) > 0.0 else 0.0
    u_aligned = np.exp(1j * phase) * u_ex
    err = _l2_norm(u_num - u_aligned, g)
    den = _l2_norm(u_ex, g)
    return err, err / max(den, 1.0e-30)


def _babaoglu_error_row(
    n: int,
    t: float,
    u_num: Array,
    lw_num: dict,
    exact: dict,
    g,
) -> dict:
    """Compute exact-solution errors at a time level."""
    u_ex = exact["u"]
    Q_ex = exact["Q"]
    amp_err = np.abs(u_num) - exact["abs_u"]
    raw_u_err = u_num - u_ex
    pal_abs, pal_rel = _phase_aligned_error(u_num, u_ex, g)
    Q_err = lw_num["Q"] - Q_ex
    den_u = max(_l2_norm(u_ex, g), 1.0e-30)
    den_amp = max(_l2_norm(exact["abs_u"], g), 1.0e-30)
    den_Q = max(_l2_norm(Q_ex, g), 1.0e-30)
    return {
        "n": int(n),
        "t": float(t),
        "u_L2_error": _l2_norm(raw_u_err, g),
        "u_L2_relative_error": _l2_norm(raw_u_err, g) / den_u,
        "u_phase_aligned_L2_error": pal_abs,
        "u_phase_aligned_L2_relative_error": pal_rel,
        "abs_u_L2_error": _l2_norm(amp_err, g),
        "abs_u_L2_relative_error": _l2_norm(amp_err, g) / den_amp,
        "abs_u_Linf_error": float(np.max(np.abs(amp_err))),
        "Q_L2_error": _l2_norm(Q_err, g),
        "Q_L2_relative_error": _l2_norm(Q_err, g) / den_Q,
        "Q_Linf_error": float(np.max(np.abs(Q_err))),
        "boundary_max": boundary_amplitude(u_num),
        "tail_ratio": spectral_tail_ratio(u_num, g),
    }


def _babaoglu_snapshot_dict(t: float, u_num: Array, lw_num: dict, exact: dict) -> dict:
    """Build one field snapshot for Exp. 8 time-series export."""
    Q_num = np.asarray(lw_num["Q"])
    w_num = np.asarray(lw_num.get("w", np.zeros_like(Q_num)))
    v_num = np.asarray(lw_num.get("v", np.zeros_like(Q_num)))
    abs_u_num = np.abs(u_num)
    abs_u_exact = np.asarray(exact["abs_u"])
    Q_exact = np.asarray(exact["Q"])

    return {
        "t": float(t),
        "abs_u_num": abs_u_num.copy(),
        "re_u_num": np.real(u_num).copy(),
        "im_u_num": np.imag(u_num).copy(),
        "Q_num": np.real(Q_num).copy(),
        "w_num": np.real(w_num).copy(),
        "v_num": np.real(v_num).copy(),
        "abs_u_exact": abs_u_exact.copy(),
        "Q_exact": Q_exact.copy(),
        "w_raw_exact": np.asarray(exact["w_raw"]).copy(),
        "v_raw_exact": np.asarray(exact["v_raw"]).copy(),
        "abs_u_error": (abs_u_num - abs_u_exact).copy(),
        "Q_error": (np.real(Q_num) - Q_exact).copy(),
    }


def _write_babaoglu_time_series_npz(
    snapshots: list,
    case_dir: str | Path,
    x: Array,
    y: Array,
) -> None:
    """Write Exp. 8 temporal field snapshots for ParaView animation export."""
    if not snapshots:
        return
    case_dir = Path(case_dir)
    keys = [k for k in snapshots[0].keys() if k != "t"]
    payload = {
        "x": np.asarray(x),
        "y": np.asarray(y),
        "times": np.asarray([s["t"] for s in snapshots], dtype=float),
    }
    for key in keys:
        payload[key] = np.stack([np.asarray(s[key]) for s in snapshots])
    np.savez_compressed(case_dir / "time_series.npz", **payload)


def _write_babaoglu_exact_table(rows: list, output_dir: str | Path) -> None:
    """Write a compact LaTeX table for the exact benchmark."""
    outdir = Path(output_dir)
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_table_csv(rows, table_dir / "babaoglu_exact_benchmark.csv")

    tex_rows = []
    for r in rows:
        tex_rows.append(
            f"{_fmt_latex_case(r['case'])} & {r['Nx']} & {_fmt_latex_sci(r['dt'])} & "
            f"{_fmt_latex_sci(r['u_phase_aligned_L2_relative_error'])} & "
            f"{_fmt_latex_sci(r['abs_u_L2_relative_error'])} & "
            f"{_fmt_latex_sci(r['Q_L2_relative_error'])} & "
            f"{_fmt_latex_sci(r.get('observed_order_phase_aligned'))} \\\\"
        )

    tex = r"""% chktex-file 8
% chktex-file 24
\begin{table}[!ht]
\centering
\caption{Exact Babaoglu--Erbay travelling-wave benchmark for the FFT--Strang GDSS solver.}
\label{tab:babaoglu_exact_benchmark}
\begin{tabular}{lccccc c}
\toprule
Case & $N_x=N_y$ & $\Delta t$ & $e_u^{\mathrm{rel}}$ & $e_{|u|}^{\mathrm{rel}}$ & $e_Q^{\mathrm{rel}}$ & Order \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (table_dir / "babaoglu_exact_benchmark.tex").write_text(tex, encoding="utf-8")


def run_babaoglu_exact_benchmark(
    p_base,
    dt_values=None,
    cases=None,
    output_dir=_HERE / "outputs/exp8_babaoglu_exact",
    save_history_for_base_dt: bool = True,
    save_time_series: bool = True,
    n_time_series_frames: int = 21,
):
    """
    Exact travelling-wave benchmark based on Babaoglu--Erbay sech solutions.

    The existing experiments are left unchanged.  This routine appends an
    additional benchmark that initializes the solver with an exact travelling
    wave and compares the numerical final state with the closed-form solution.

    Outputs:
        results.json/csv
        tables/babaoglu_exact_benchmark.tex
        <case>/history_dt_base.csv
        <case>/time_series.npz
        <case>/final_comparison.npz
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    p_exact_base = _babaoglu_benchmark_params(p_base)
    if cases is None:
        cases = _babaoglu_default_cases()
    if dt_values is None:
        # Use a compact, coarser exact-benchmark refinement sequence.  The
        # baseline dt may be very small for the invariant diagnostics, and using
        # it here would add considerable cost while the exact error is often
        # dominated by spatial and periodic-truncation effects.
        Nt_from_base = int(round(p_exact_base.t_final / p_exact_base.dt))
        Nt_coarse = min(max(4, Nt_from_base), 20)
        dt_values = [p_exact_base.t_final / (Nt_coarse * 2**k) for k in range(3)]
    dt_sorted = sorted(dt_values, reverse=True)
    dt_base = dt_sorted[0]

    all_rows = []
    all_histories = {}

    for case in cases:
        label = str(case.get("label", f"k{case.get('k1', 1.0)}_{case.get('k2', 0.0)}"))
        case_dir = outdir / label
        case_dir.mkdir(parents=True, exist_ok=True)

        case_rows = []
        for dt in dt_sorted:
            p = replace(p_exact_base, dt=float(dt))
            Nt = _check_divisibility(p.t_final, p.dt)
            g = make_grid(p)
            m = precompute_multipliers(p, g)
            coeffs = _babaoglu_coefficients(p, g, case)

            exact0 = _babaoglu_exact_fields(g, coeffs, 0.0)
            u = exact0["u"].copy()
            lw0 = recover_longwave(u, g, p, m, full=True)
            M0 = mass(u, g)
            E0 = energy_reduced(u, lw0["Q"], g, p)

            history = []
            time_series_snapshots = []
            is_base_dt = save_history_for_base_dt and abs(dt - dt_base) <= 1.0e-15
            if is_base_dt:
                history.append(_babaoglu_error_row(0, 0.0, u, lw0, exact0, g))
                if save_time_series:
                    time_series_snapshots.append(
                        _babaoglu_snapshot_dict(0.0, u, lw0, exact0)
                    )

            start = time.perf_counter()
            save_every = max(1, Nt // max(1, n_time_series_frames - 1))
            for n in range(1, Nt + 1):
                u, _ = strang_step(u, g, p, m)
                if is_base_dt:
                    if (n % save_every == 0) or (n == Nt):
                        t_n = n * p.dt
                        lw_n = recover_longwave(u, g, p, m, full=True)
                        exact_n = _babaoglu_exact_fields(g, coeffs, t_n)
                        history.append(_babaoglu_error_row(n, t_n, u, lw_n, exact_n, g))
                        if save_time_series:
                            time_series_snapshots.append(
                                _babaoglu_snapshot_dict(t_n, u, lw_n, exact_n)
                            )
            cpu = time.perf_counter() - start

            t_final = Nt * p.dt
            lw_f = recover_longwave(u, g, p, m, full=True)
            exact_f = _babaoglu_exact_fields(g, coeffs, t_final)
            err_f = _babaoglu_error_row(Nt, t_final, u, lw_f, exact_f, g)
            M_f = mass(u, g)
            E_f = energy_reduced(u, lw_f["Q"], g, p)

            row = {
                "case": label,
                "dt": float(dt),
                "Nt": int(Nt),
                "Nx": int(p.Nx),
                "Ny": int(p.Ny),
                "Lx": float(p.Lx),
                "Ly": float(p.Ly),
                "t_final": float(t_final),
                "cpu_time_seconds": float(cpu),
                "C_step": float(cpu / Nt),
                "RE_M_final": abs(M_f - M0) / (abs(M0) + _eps_I(M0)),
                "RE_E_final": abs(E_f - E0) / (abs(E0) + _eps_I(E0)),
                **{k: v for k, v in err_f.items() if k not in ("n", "t")},
                **{f"coeff_{k}": v for k, v in coeffs.items()
                   if isinstance(v, (int, float, np.floating))},
            }
            case_rows.append(row)
            all_rows.append(row)

            if is_base_dt:
                _write_csv(history, case_dir / "history_dt_base.csv")
                if save_time_series:
                    _write_babaoglu_time_series_npz(
                        time_series_snapshots, case_dir, g.x, g.y
                    )
                all_histories[label] = history
                np.savez_compressed(
                    case_dir / "final_comparison.npz",
                    x=g.x, y=g.y,
                    u_num=u,
                    abs_u_num=np.abs(u),
                    Q_num=lw_f["Q"],
                    w_num=lw_f.get("w", np.zeros_like(lw_f["Q"])),
                    v_num=lw_f.get("v", np.zeros_like(lw_f["Q"])),
                    u_exact=exact_f["u"],
                    abs_u_exact=exact_f["abs_u"],
                    Q_exact=exact_f["Q"],
                    w_raw_exact=exact_f["w_raw"],
                    v_raw_exact=exact_f["v_raw"],
                    abs_u_error=np.abs(u) - exact_f["abs_u"],
                    Q_error=lw_f["Q"] - exact_f["Q"],
                    t=np.array([t_final]),
                )

        # Observed orders by case, comparing consecutive dt values.
        for i in range(1, len(case_rows)):
            e_prev = case_rows[i - 1]["u_phase_aligned_L2_relative_error"]
            e_curr = case_rows[i]["u_phase_aligned_L2_relative_error"]
            dt_prev = case_rows[i - 1]["dt"]
            dt_curr = case_rows[i]["dt"]
            if e_prev > 0.0 and e_curr > 0.0:
                order = math.log(e_prev / e_curr) / math.log(dt_prev / dt_curr)
                case_rows[i]["observed_order_phase_aligned"] = order
                # all_rows contains the same dictionary object, so this updates it.
            else:
                case_rows[i]["observed_order_phase_aligned"] = None
        if case_rows:
            case_rows[0]["observed_order_phase_aligned"] = None

    with open(outdir / "results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    _write_csv(all_rows, outdir / "results.csv")
    _write_babaoglu_exact_table(all_rows, outdir)

    metadata = {
        "description": "Babaoglu--Erbay sech travelling-wave exact benchmark",
        "parameter_note": "gamma=2 and xi=-4 are used only for this exact benchmark.",
        "periodic_note": (
            "Default cases are axis-aligned stripes to keep |u| negligible at "
            "the periodic boundary.  Raw tanh potentials are saved only as visual "
            "references; Q is compared in its zero-mode-free periodic form."
        ),
        "params": p_exact_base.__dict__,
        "cases": cases,
        "dt_values": [float(dt) for dt in dt_sorted],
        "save_time_series": bool(save_time_series),
        "n_time_series_frames": int(n_time_series_frames),
        "time_series_note": (
            "Each case stores time_series.npz for ParaView animation export. "
            "The file contains numerical |u|, Re(u), Im(u), Q, w, v and matching exact/reference fields."
        ),
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"[Exp 8] Babaoglu exact benchmark -> {outdir}")
    return {"results": all_rows, "histories": all_histories, "output_dir": str(outdir)}

# ──────────────── Experiment 8b: Babaoglu spatial convergence study ──────────

def _write_babaoglu_spatial_table(rows: list, output_dir: str | Path) -> None:
    """Write CSV + LaTeX table for the spatial convergence study."""
    outdir = Path(output_dir)
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_table_csv(rows, table_dir / "babaoglu_spatial_convergence.csv")

    tex_rows = []
    for r in rows:
        tex_rows.append(
            f"{_fmt_latex_case(r['case'])} & {r['Nx']} & "
            f"{_fmt_latex_sci(r['u_phase_aligned_L2_relative_error'])} & "
            f"{_fmt_latex_sci(r['abs_u_L2_relative_error'])} & "
            f"{_fmt_latex_sci(r.get('observed_spatial_rate'))} \\\\"
        )

    tex = r"""% chktex-file 8
% chktex-file 24
\begin{table}[!ht]
\centering
\caption{Spatial convergence against the Babaoglu--Erbay exact solution (fixed $\Delta t$). The ``Approx.\ rate'' column is computed with the standard
log-ratio formula~\eqref{eq:observed_temporal_order} applied in the spatial
variable; for a spectrally (exponentially) convergent discretization this
quantity is not an algebraic convergence order and is reported only as a
qualitative indicator of the decay rate between consecutive grids. The large
value at $N_x=N_y=128$ reflects the transition from spectral decay to
saturation by the fixed temporal discretization error and round-off, not a
fourteenth-order method; see the discussion following this table.}
\label{tab:babaoglu_spatial_convergence}
\begin{tabular}{lcccc}
\toprule
Case & $N_x=N_y$ & $e_u^{\mathrm{rel}}$ & $e_{|u|}^{\mathrm{rel}}$ & Approx.\ rate \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (table_dir / "babaoglu_spatial_convergence.tex").write_text(tex, encoding="utf-8")


def run_babaoglu_spatial_convergence(
    p_base: GDSSParams,
    nx_ny_pairs=None,
    dt_fixed: Optional[float] = None,
    cases=None,
    output_dir=_HERE / "outputs/exp8b_babaoglu_spatial",
) -> Dict:
    """
    Spatial convergence study against the Babaoglu--Erbay exact travelling wave.

    Uses a fixed dt (4× finer than the base step so time error is negligible)
    and varies Nx = Ny.  The FFT solver achieves spectral (exponential) accuracy
    for smooth sech data; the L2 error should drop steeply on a log-linear plot.

    Observed rates are reported as log(e_prev/e_curr)/log(N_curr/N_prev) for
    reference; large values (≫ 2) confirm spectral convergence.

    Outputs per case directory:
        spatial_convergence.csv
    Outputs at experiment level:
        results.json, results.csv, tables/babaoglu_spatial_convergence.{csv,tex}
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    p_exact_base = _babaoglu_benchmark_params(p_base)
    if cases is None:
        cases = _babaoglu_default_cases()
    if nx_ny_pairs is None:
        N = p_base.Nx
        nx_ny_pairs = [
            (max(8, N // 4), max(8, N // 4)),
            (max(8, N // 2), max(8, N // 2)),
            (N, N),
            (2 * N, 2 * N),
        ]
    if dt_fixed is None:
        # 4× finer than the base dt → time error ≈16× smaller than at base dt.
        Nt_base = max(4, int(round(p_exact_base.t_final / p_exact_base.dt)))
        dt_fixed = p_exact_base.t_final / (Nt_base * 4)

    all_rows: List[Dict] = []

    for case in cases:
        label = str(case.get("label", f"k{case.get('k1', 1.0)}_{case.get('k2', 0.0)}"))
        case_dir = outdir / label
        case_dir.mkdir(parents=True, exist_ok=True)

        case_rows: List[Dict] = []
        for Nx, Ny in nx_ny_pairs:
            p = replace(p_exact_base, Nx=int(Nx), Ny=int(Ny), dt=float(dt_fixed))
            Nt = _check_divisibility(p.t_final, p.dt)
            g = make_grid(p)
            m = precompute_multipliers(p, g)
            coeffs = _babaoglu_coefficients(p, g, case)

            u = _babaoglu_exact_fields(g, coeffs, 0.0)["u"].copy()

            t0 = time.perf_counter()
            for _ in range(Nt):
                u, _ = strang_step(u, g, p, m)
            cpu = time.perf_counter() - t0

            lw_f = recover_longwave(u, g, p, m, full=True)
            exact_f = _babaoglu_exact_fields(g, coeffs, p.t_final)
            err_f = _babaoglu_error_row(Nt, p.t_final, u, lw_f, exact_f, g)

            row: Dict = {
                "case": label,
                "Nx": int(Nx), "Ny": int(Ny),
                "dt": float(dt_fixed),
                "Nt": int(Nt),
                "cpu_time_seconds": float(cpu),
                **{k: v for k, v in err_f.items() if k not in ("n", "t")},
                "observed_spatial_rate": None,
            }
            case_rows.append(row)
            all_rows.append(row)

        # Polynomial approximation of the spatial rate for consecutive grid pairs.
        # For a spectral method this number grows with N; values ≫ 2 confirm
        # spectral (super-algebraic) convergence.
        for i in range(1, len(case_rows)):
            e_prev = case_rows[i - 1]["u_phase_aligned_L2_relative_error"]
            e_curr = case_rows[i]["u_phase_aligned_L2_relative_error"]
            N_prev = case_rows[i - 1]["Nx"]
            N_curr = case_rows[i]["Nx"]
            if (e_prev is not None and e_curr is not None
                    and e_prev > 0.0 and e_curr > 0.0 and N_curr > N_prev):
                # case_rows[i] and all_rows share the same dict object.
                case_rows[i]["observed_spatial_rate"] = (
                    math.log(e_prev / e_curr) / math.log(N_curr / N_prev)
                )

        _write_csv(case_rows, case_dir / "spatial_convergence.csv")

    with open(outdir / "results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    _write_csv(all_rows, outdir / "results.csv")
    _write_babaoglu_spatial_table(all_rows, outdir)

    print(f"[Exp 8b] Babaoglu spatial convergence -> {outdir}")
    return {"results": all_rows, "output_dir": str(outdir)}


# ──────────────────────────────────── Main orchestrator ───────────────────────

def run_all_experiments(
    p_base,
    output_root=_HERE / "outputs",
    snapshot_times=None,
    dt_values_for_refinement=None,
    nx_ny_for_resolution=None,
    lx_ly_for_boundary=None,
    nx_ny_for_cost=None,
    run_exact_benchmark: bool = True,
    exact_benchmark_dt_values=None,
    exact_benchmark_cases=None,
    exact_benchmark_time_series_frames: int = 21,
    run_spatial_benchmark: bool = True,
    spatial_benchmark_nx_ny_pairs=None,
    spatial_benchmark_dt_fixed: Optional[float] = None,
    t_final_baseline: Optional[float] = None,
    t_final_benchmark: Optional[float] = None,
    t_final_convergence: float = 1.0,
    t_final_cost: float = 0.1,
    save_every_baseline: int = 20,
):
    """
    Convenience wrapper: runs all experiments in sequence.
    All parameters have sensible defaults; override any via keyword arguments.

    Time-horizon parameters (all controllable from run.py):
      t_final_baseline    — Exps 2/3, 6: main conservation/boundary run.
                            Defaults to p_base.t_final if not given.
      t_final_benchmark   — Exp 8: Babaoglu temporal benchmark.
                            Defaults to t_final_baseline if not given.
      t_final_convergence — Exps 4, 5, 8b: convergence studies.
                            Short (default 1.0) so the error is dominated by
                            the scheme, not by accumulated integration time.
      t_final_cost        — Exp 7: cost-scaling study.
                            Needs only enough steps for stable wall-clock
                            timing; default 0.1 gives Nt=100 at dt=1e-3.

    save_every_baseline controls how often Exp 2/3 saves diagnostic rows.
    With dt=1e-3 and t_final=2, a value of 20 gives 100 rows — enough for
    smooth drift curves.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    # Resolve time horizons: explicit arguments take priority over p_base.t_final.
    _t_baseline  = t_final_baseline  if t_final_baseline  is not None else p_base.t_final
    _t_benchmark = t_final_benchmark if t_final_benchmark is not None else _t_baseline

    # Build one parameter variant per distinct time horizon.
    p_baseline = replace(p_base, t_final=_t_baseline)
    p_bench    = replace(p_base, t_final=_t_benchmark)
    p_conv     = replace(p_base, t_final=t_final_convergence)
    p_cost     = replace(p_base, t_final=t_final_cost)

    if snapshot_times is None:
        T = _t_baseline
        snapshot_times = [0.0, T / 4, T / 2, 3 * T / 4, T]
    if dt_values_for_refinement is None:
        # Sequence guaranteed to divide t_final_convergence exactly.
        dt_values_for_refinement = _default_dt_sequence(
            t_final_convergence, p_base.dt, n_levels=4)
    if nx_ny_for_resolution is None:
        N = p_base.Nx
        nx_ny_for_resolution = [(N // 2, N // 2), (N, N), (2 * N, 2 * N)]
    if lx_ly_for_boundary is None:
        L = p_base.Lx
        lx_ly_for_boundary = [(L, L), (2 * L, 2 * L)]
    if nx_ny_for_cost is None:
        N = p_base.Nx
        nx_ny_for_cost = [(N // 2, N // 2), (N, N), (2 * N, 2 * N)]

    all_results = {}
    # Exp 1: static recovery — grid only, no time stepping.
    all_results["exp1"] = run_recovery_verification(
        p_base, output_dir=root / "exp1_recovery")
    # Exp 2/3: baseline simulation — t_final_baseline.
    all_results["exp2_3"] = run_simulation_with_snapshots(
        p_baseline, snapshot_times=snapshot_times,
        output_dir=root / "exp2_baseline", save_every=save_every_baseline)
    # Exp 4: temporal order — t_final_convergence, dt halved 4 times.
    all_results["exp4"] = run_time_step_study(
        p_conv, dt_values=dt_values_for_refinement,
        output_dir=root / "exp4_timestep")
    # Exp 5: spatial resolution — t_final_convergence, fixed dt.
    all_results["exp5"] = run_spatial_resolution_study(
        p_conv, nx_ny_pairs=nx_ny_for_resolution,
        output_dir=root / "exp5_spatial")
    # Exp 6: boundary contamination — t_final_baseline.
    all_results["exp6"] = run_boundary_study(
        p_baseline, lx_ly_pairs=lx_ly_for_boundary,
        output_dir=root / "exp6_boundary")
    # Exp 7: cost scaling — short t_final_cost (only needs stable wall-clock timing).
    all_results["exp7"] = run_cost_study(
        p_cost, nx_ny_pairs=nx_ny_for_cost,
        output_dir=root / "exp7_cost")
    # Exp 8: Babaoglu temporal benchmark — t_final_benchmark.
    if run_exact_benchmark:
        all_results["exp8"] = run_babaoglu_exact_benchmark(
            p_bench,
            dt_values=exact_benchmark_dt_values,
            cases=exact_benchmark_cases,
            output_dir=root / "exp8_babaoglu_exact",
            save_time_series=True,
            n_time_series_frames=exact_benchmark_time_series_frames)
    # Exp 8b: Babaoglu spatial convergence — t_final_convergence.
    if run_spatial_benchmark:
        all_results["exp8b"] = run_babaoglu_spatial_convergence(
            p_conv,
            nx_ny_pairs=spatial_benchmark_nx_ny_pairs,
            dt_fixed=spatial_benchmark_dt_fixed,
            cases=exact_benchmark_cases,
            output_dir=root / "exp8b_babaoglu_spatial")

    with open(root / "all_results_summary.json", "w") as f:
        json.dump(
            {k: v.get("results") or v.get("summary")
             for k, v in all_results.items()},
            f, indent=2, default=str,
        )

    return all_results


if __name__ == "__main__":
    from gdss_solver import GDSSParams
    params = GDSSParams(
        Nx=64, Ny=64, Lx=40.0, Ly=40.0,
        dt=5e-3, t_final=0.05,
        alpha=1.0, beta=1.0, gamma=1.0, xi=1.0,
        psi=1.0, eta=1.0, phi=2.0, chi=0.5, theta=None,
    )
    run_all_experiments(params, output_root=_HERE / "outputs")
