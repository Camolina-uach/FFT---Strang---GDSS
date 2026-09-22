"""
gdss_long_horizon_plots.py -- figure and LaTeX table for the long-horizon
drift runs written by gdss_long_horizon.py (revision experiment).

Produces
    long_horizon_drift.{pdf,png}   drift histories and dt-scaling of the energy drift
    long_horizon_table.tex         LaTeX table for the manuscript

Usage:
    python gdss_long_horizon_plots.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gdss_plots import FIGW, _savefig

HERE = Path(__file__).resolve().parent


def _read(path: Path) -> Dict[str, np.ndarray]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def load_runs(outdir: Path) -> Dict[float, Dict[str, np.ndarray]]:
    runs = {}
    for f in sorted(outdir.glob("long_horizon_dt*.csv")):
        dt = float(f.stem.replace("long_horizon_dt", ""))
        runs[dt] = _read(f)
    return dict(sorted(runs.items(), reverse=True))


def plot(outdir: Path, runs) -> None:
    dt_min = min(runs)
    fine = runs[dt_min]
    t = fine["t"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FIGW * 1.4, FIGW * 0.52))

    for key, lab, c in (("rel_M", r"$M_h$", "C0"), ("rel_Jx", r"$J_{x,h}$", "C1"),
                        ("rel_Jy", r"$J_{y,h}$", "C2"), ("rel_E_red", r"$E_h$", "C3")):
        a1.semilogy(t[1:], np.maximum(fine[key][1:], 1e-17), color=c, lw=1.2, label=lab)
    a1.set_xlabel("$t$", fontsize=11)
    a1.set_ylabel("relative drift", fontsize=11)
    a1.set_title(rf"(a) Invariant drift, $\Delta t={dt_min:g}$", fontsize=10.5)
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(fontsize=8.5, loc="center right")

    for i, (dt, r) in enumerate(runs.items()):
        scale = (dt_min / dt) ** 2
        a2.plot(r["t"], r["rel_E_red"] * scale, color=f"C{i}", lw=1.3,
                ls=("-", "--", ":")[i % 3],
                label=rf"$\Delta t={dt:g}$" + ("" if dt == dt_min else
                                                 rf", $\times({dt_min:g}/{dt:g})^2$"))
    a2.set_xlabel("$t$", fontsize=11)
    a2.set_ylabel(r"relative energy drift (rescaled)", fontsize=11)
    a2.set_title(r"(b) $\Delta t^2$ scaling of the energy drift", fontsize=10.5)
    a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a2.grid(True, alpha=0.3)
    a2.legend(fontsize=8)

    a3.semilogy(t[1:], fine["boundary"][1:], color="C4", lw=1.2, label=r"$B_\partial$ (boundary band)")
    a3.semilogy(t[1:], fine["tail"][1:], color="C5", lw=1.2, label=r"$T_u$ (spectral tail)")
    a3.semilogy(t, fine["max_abs_u"], color="k", lw=1.0, ls="--", label=r"$\max|u_h|$")
    a3.set_ylim(1e-26, 3)
    a3.set_xlabel("$t$", fontsize=11)
    a3.set_title("(c) Monitors", fontsize=10.5)
    a3.grid(True, which="both", alpha=0.3)
    a3.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, outdir, "long_horizon_drift")


def _sci(x: float) -> str:
    m, e = f"{x:.3e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def write_table(outdir: Path, runs) -> None:
    dts = sorted(runs, reverse=True)
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Long-horizon run ($T=20$, domain $[-80,80)^2$, $N_x=N_y=512$): "
        r"maximum relative drifts over $0\le t\le20$ and final-time energy drift for "
        r"three time steps. The last column is the observed order of the final "
        r"energy drift between consecutive time steps.}",
        r"\label{tab:long_horizon}",
        r"\footnotesize",
        r"\begin{tabular}{@{}cccccc@{}}",
        r"\toprule",
        r"$\Delta t$ & $\max\delta_M$ & $\max\delta_{J}$ & $\max\delta_E$ & "
        r"$\delta_E(T)$ & Order \\",
        r"\colrule",
    ]
    prev = None
    for dt in dts:
        r = runs[dt]
        dJ = max(r["rel_Jx"].max(), r["rel_Jy"].max())
        eT = r["rel_E_red"][-1]
        order = "--" if prev is None else f"{np.log(prev[1] / eT) / np.log(prev[0] / dt):.2f}"
        lines.append(rf"{_sci(dt)} & {_sci(r['rel_M'].max())} & {_sci(dJ)} & "
                     rf"{_sci(r['rel_E_red'].max())} & {_sci(eT)} & {order} \\")
        prev = (dt, eT)
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}", ""]
    (outdir / "long_horizon_table.tex").write_text("\n".join(lines))
    print(f"wrote {outdir / 'long_horizon_table.tex'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "revision")
    a = ap.parse_args()
    runs = load_runs(a.out)
    plot(a.out, runs)
    write_table(a.out, runs)


if __name__ == "__main__":
    main()
