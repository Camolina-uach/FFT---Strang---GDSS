#!/usr/bin/env python3
"""Documented command-line entry point for GDSS reproducibility runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdss_experiments import run_all_experiments
from gdss_paraview_export import export_exp8_paraview
from gdss_plots import plot_all
from gdss_solver import GDSSParams


HERE = Path(__file__).resolve().parent


def manuscript_params() -> GDSSParams:
    """Return the parameter set used by the manuscript driver."""
    return GDSSParams(
        Nx=128,
        Ny=128,
        Lx=40.0,
        Ly=40.0,
        dt=1.0e-3,
        t_final=2.0,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        xi=1.0,
        psi=1.0,
        eta=1.0,
        phi=2.0,
        chi=0.5,
        theta=None,
        dealias_density=True,
        recover_full_longwave=True,
    )


def run_paper(output_root: Path, with_paraview: bool) -> None:
    """Run the manuscript experiments and figures."""
    run_all_experiments(
        manuscript_params(),
        output_root=output_root,
        t_final_baseline=2.0,
        t_final_benchmark=2.0,
        t_final_convergence=1.0,
        t_final_cost=0.1,
        snapshot_times=[0.0, 0.5, 1.0, 1.5, 2.0],
        save_every_baseline=20,
        run_exact_benchmark=True,
        exact_benchmark_time_series_frames=21,
        run_spatial_benchmark=True,
    )
    plot_all(output_root=output_root)
    if with_paraview:
        export_exp8_paraview(output_root=output_root, export_time_series=True)


def run_revision(output_root: Path) -> None:
    """Run the experiments added in the revision (v1.1.0) and their figures."""
    import gdss_long_horizon
    import gdss_long_horizon_plots
    import gdss_manufactured
    import gdss_manufactured_plots
    import gdss_spectrum_figure

    out = output_root / "revision"
    out.mkdir(parents=True, exist_ok=True)
    gdss_manufactured._write_environment(out)
    gdss_manufactured.run_manufactured(out)
    gdss_manufactured.run_reference(out)
    for dt in (4.0e-3, 2.0e-3, 1.0e-3):
        gdss_long_horizon.run(dt, T=20.0, every=0.1, out=out)
    gdss_spectrum_figure.regenerate_spectra(out / "exp5_spatial")
    gdss_spectrum_figure.plot(out / "exp5_spatial", out)
    man = gdss_manufactured_plots._read(out / "manufactured_longwave.csv")
    gdss_manufactured_plots.plot_fields(out)
    gdss_manufactured_plots.plot_convergence(
        out, man, gdss_manufactured_plots._read(out / "reference_longwave_convergence.csv"))
    gdss_manufactured_plots.write_table(out, man)
    runs = gdss_long_horizon_plots.load_runs(out)
    gdss_long_horizon_plots.plot(out, runs)
    gdss_long_horizon_plots.write_table(out, runs)


def run_smoke(output_root: Path) -> None:
    """Exercise the orchestration quickly; values are not paper references."""
    params = GDSSParams(
        Nx=32,
        Ny=32,
        Lx=20.0,
        Ly=20.0,
        dt=5.0e-3,
        t_final=2.0e-2,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        xi=1.0,
        psi=1.0,
        eta=1.0,
        phi=2.0,
        chi=0.5,
        theta=None,
        dealias_density=True,
        recover_full_longwave=True,
    )
    run_all_experiments(
        params,
        output_root=output_root,
        snapshot_times=[0.0, 0.01, 0.02],
        dt_values_for_refinement=[0.005, 0.0025, 0.00125],
        nx_ny_for_resolution=[(16, 16), (32, 32)],
        lx_ly_for_boundary=[(20.0, 20.0), (40.0, 40.0)],
        nx_ny_for_cost=[(16, 16), (32, 32)],
        run_exact_benchmark=False,
        run_spatial_benchmark=False,
        t_final_baseline=0.02,
        t_final_convergence=0.02,
        t_final_cost=0.01,
        save_every_baseline=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper = subparsers.add_parser("paper", help="run manuscript experiments and figures")
    paper.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "outputs",
        help="output directory (default: repository/outputs)",
    )
    paper.add_argument(
        "--with-paraview",
        action="store_true",
        help="also create the large ParaView time-series export",
    )

    revision = subparsers.add_parser(
        "revision",
        help="run the experiments added in the revision (manufactured solution, "
             "long-horizon drift, spectrum figure with tail markers)",
    )
    revision.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "outputs",
        help="output directory (default: repository/outputs)",
    )

    smoke = subparsers.add_parser("smoke", help="run a small, fast workflow check")
    smoke.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "smoke_outputs",
        help="output directory (default: repository/smoke_outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.command == "paper":
        run_paper(output_root, args.with_paraview)
    elif args.command == "revision":
        run_revision(output_root)
    else:
        run_smoke(output_root)
    print(f"Completed successfully. Outputs: {output_root}")


if __name__ == "__main__":
    main()
