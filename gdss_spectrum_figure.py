"""
gdss_spectrum_figure.py -- spectral-resolution figure with the tail region
marked (revision; replaces the manuscript figure produced by
gdss_plots.plot_spectrum_resolution, which is kept unchanged).

The spectra are regenerated with the archived experiment function
gdss_experiments.run_spatial_resolution_study (grids 64^2, 128^2, 256^2,
T = 1, dt = 1e-3), and the onset p = N_x/3 of the tail region K_tail used for
the tail ratio T_u (the complement of the 2/3 dealiasing band) is drawn as a
dashed line for each grid.

Usage:
    python gdss_spectrum_figure.py [--out DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gdss_experiments import run_spatial_resolution_study
from gdss_plots import FIGH, FIGW, _savefig
from gdss_solver import GDSSParams

HERE = Path(__file__).resolve().parent
GRIDS = [(64, 64), (128, 128), (256, 256)]


def regenerate_spectra(exp5_dir: Path) -> None:
    p = GDSSParams(Nx=128, Ny=128, Lx=40.0, Ly=40.0, dt=1.0e-3, t_final=1.0,
                   alpha=1.0, beta=1.0, gamma=1.0, xi=1.0,
                   psi=1.0, eta=1.0, phi=2.0, chi=0.5, theta=None,
                   dealias_density=True, recover_full_longwave=True)
    run_spatial_resolution_study(replace(p), GRIDS, output_dir=exp5_dir)


def plot(exp5_dir: Path, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIGW * 1.4, FIGH))
    for i, (Nx, Ny) in enumerate(GRIDS):
        uhat = np.load(exp5_dir / f"uhat_{Nx}x{Ny}.npy")
        mid = uhat[:, 0]
        norm = mid[0] if mid[0] > 0 else 1.0
        k = np.arange(Nx // 2)
        ax.semilogy(k, mid[: Nx // 2] / norm, color=f"C{i}", lw=1.5,
                    label=f"$N_x = N_y = {Nx}$")
        ax.axvline(Nx / 3, color=f"C{i}", ls="--", lw=1.0, alpha=0.9)
        ax.text(Nx / 3 + 1, 0.55, rf"$p={Nx}/3$", transform=ax.get_xaxis_transform(),
                color=f"C{i}", fontsize=9, va="center")
    ax.set_xlabel("Mode index $p$", fontsize=12)
    ax.set_ylabel(r"$|\hat{u}_{p,0}| / |\hat{u}_{0,0}|$", fontsize=12)
    ax.set_title(r"Fourier spectrum — $k_x$ midline; dashed: onset of $\mathcal{K}_{\mathrm{tail}}$",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, outdir, "spectrum_resolution")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "revision")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing spectra instead of recomputing them")
    a = ap.parse_args()
    exp5_dir = a.out / "exp5_spatial"
    if not (a.reuse and (exp5_dir / "uhat_256x256.npy").exists()):
        regenerate_spectra(exp5_dir)
    plot(exp5_dir, a.out)


if __name__ == "__main__":
    main()
