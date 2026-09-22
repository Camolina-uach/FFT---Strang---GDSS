"""
gdss_manufactured_plots.py -- figures and LaTeX table for the coupled
long-wave verification (revision experiment).

Reads the CSV files written by gdss_manufactured.py and produces
    manufactured_fields.{pdf,png}        exact rho, w, v, Q of the manufactured solution
    manufactured_convergence.{pdf,png}   recovery error vs N (manufactured and refined reference)
    manufactured_longwave_table.tex      LaTeX table for the manuscript

Usage:
    python gdss_manufactured_plots.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gdss_manufactured import PARAMETER_SETS, manufactured_fields
from gdss_plots import CMAP_SIGN, DPI, FIGW, _savefig
from gdss_solver import GDSSParams, make_grid

HERE = Path(__file__).resolve().parent


def _read(path: Path) -> List[Dict[str, float]]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({k: (v if k == "parameter_set" else float(v)) for k, v in r.items()})
    return out


def plot_fields(outdir: Path, N: int = 256) -> None:
    p = GDSSParams(Nx=N, Ny=N, Lx=40.0, Ly=40.0, **PARAMETER_SETS["baseline"])
    g = make_grid(p)
    ex = manufactured_fields(p, g.X, g.Y)
    names = [("rho", r"$\rho-\langle\rho\rangle$"), ("w", r"$w$"),
             ("v", r"$v$"), ("Q", r"$Q=w_x+v_y$")]
    fig, axes = plt.subplots(2, 2, figsize=(FIGW, FIGW * 0.86))
    for ax, (key, title) in zip(axes.ravel(), names):
        fld = ex[key] - (np.mean(ex[key]) if key == "rho" else 0.0)
        vmax = float(np.max(np.abs(fld)))
        cf = ax.contourf(g.X, g.Y, fld, levels=40, cmap=CMAP_SIGN, vmin=-vmax, vmax=vmax)
        cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.03)
        cb.locator = matplotlib.ticker.MaxNLocator(5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$", fontsize=10)
        ax.set_ylabel("$y$", fontsize=10)
        ax.set_aspect("equal")
    fig.tight_layout()
    _savefig(fig, outdir, "manufactured_fields")


def plot_convergence(outdir: Path, man: List[dict], ref: List[dict]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW * 1.6, FIGW * 0.62))
    colors = {"w": "C0", "v": "C1", "Q": "C2"}

    base = [r for r in man if r["parameter_set"] == "baseline"]
    for dealias, ls, tag in ((0, "-", "unfiltered"), (1, "--", r"$2/3$-filtered")):
        rows = sorted((r for r in base if int(r["dealias"]) == dealias), key=lambda r: r["N"])
        N = [r["N"] for r in rows]
        for f in ("w", "v", "Q"):
            a1.semilogy(N, [max(r[f"e_{f}_l2"], 1e-17) for r in rows], ls, marker="o",
                        ms=3.5, color=colors[f],
                        label=f"${f}$, {tag}")
    rows = sorted((r for r in base if int(r["dealias"]) == 0), key=lambda r: r["N"])
    a1.semilogy([r["N"] for r in rows], [r["e_v_l2_theta0"] for r in rows], ":",
                color="k", label=r"$v$, recovery with $\theta=0$")
    a1.set_xlabel(r"$N_x=N_y$", fontsize=11)
    a1.set_ylabel("relative $L^2$ error", fontsize=11)
    a1.set_title("(a) Manufactured long-wave solution", fontsize=11)
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(fontsize=7.5, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 0.93))

    ref = sorted(ref, key=lambda r: r["N"])
    N = [r["N"] for r in ref]
    for f, c in (("w", "C0"), ("v", "C1"), ("Q", "C2"), ("u", "C3")):
        a2.semilogy(N, [r[f"e_{f}_l2"] for r in ref], "o-", ms=3.5, color=c, label=f"${f}$")
    a2.set_xlabel(r"$N_x=N_y$", fontsize=11)
    a2.set_ylabel("relative $L^2$ difference", fontsize=11)
    a2.set_title(rf"(b) Gaussian run at $T={ref[0]['T']:g}$ vs. $N={int(ref[0]['N_ref'])}$",
                 fontsize=11)
    a2.grid(True, which="both", alpha=0.3)
    a2.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, outdir, "manufactured_convergence")


def _sci(x: float) -> str:
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def write_table(outdir: Path, man: List[dict], Ns=(32, 48, 64, 96, 128)) -> None:
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Manufactured-solution test of the coupled long-wave recovery "
        r"(unfiltered density) for the baseline ($\theta=1/\sqrt2$) and "
        r"benchmark ($\theta=1$) parameters. Relative $L^2$ errors of the recovered fields "
        r"against the exact fields~\eqref{eq:manufactured_fields}. The last column "
        r"repeats the recovery with the cross-coupling switched off ($\theta=0$), "
        r"which leaves an $O(10^{-1})$ error in $v$.}",
        r"\label{tab:manufactured_longwave}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Parameters & $N_x=N_y$ & $e_w$ & $e_v$ & $e_Q$ & $e_v$ ($\theta=0$) \\",
        r"\colrule",
    ]
    for name, label in (("baseline", "Baseline"), ("babaoglu", "Benchmark")):
        for N in Ns:
            r = next(r for r in man if r["parameter_set"] == name
                     and int(r["dealias"]) == 0 and int(r["N"]) == N)
            lines.append(rf"{label if N == Ns[0] else ''} & {N} & {_sci(r['e_w_l2'])} & "
                         rf"{_sci(r['e_v_l2'])} & {_sci(r['e_Q_l2'])} & "
                         rf"{_sci(r['e_v_l2_theta0'])} \\")
        if name == "baseline":
            lines.append(r"\colrule")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}", ""]
    (outdir / "manufactured_longwave_table.tex").write_text("\n".join(lines))
    print(f"wrote {outdir / 'manufactured_longwave_table.tex'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "revision")
    a = ap.parse_args()
    man = _read(a.out / "manufactured_longwave.csv")
    ref_path = a.out / "reference_longwave_convergence.csv"
    plot_fields(a.out)
    if ref_path.exists():
        plot_convergence(a.out, man, _read(ref_path))
    write_table(a.out, man)


if __name__ == "__main__":
    main()
