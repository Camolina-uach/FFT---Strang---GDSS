"""
gdss_paraview_export.py — ParaView export utilities for GDSS experiment outputs.

This script converts the saved Exp. 8 Babaoglu--Erbay final_comparison.npz
and time_series.npz files into ParaView-readable files.

Typical usage, from the same folder as run.py:

    python gdss_paraview_export.py --output-root outputs

or, if you want to point directly to Exp. 8:

    python gdss_paraview_export.py --exp8-dir outputs/exp8_babaoglu_exact

Recommended ParaView workflow for the time evolution:
    1. Open the generated *_time_series.pvd file.
    2. ParaView will read the sequence as a time-dependent dataset.
    3. Color by abs_u_num, v_num, w_num, Q_num, etc.
    4. Use Warp By Scalar if you want a true height surface for |u|, v, or w.

CSV workflow:
    Open a *_paraview_points.csv file and apply Table To Points.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np


_FIELD_CANDIDATES = {
    "abs_u_num": ("abs_u_num", "abs_u", "rho"),
    "re_u_num": ("re_u_num",),
    "im_u_num": ("im_u_num",),
    "v_num": ("v_num", "v", "v_raw_exact"),
    "w_num": ("w_num", "w", "w_raw_exact"),
    "Q_num": ("Q_num", "Q", "Q_exact"),
    "abs_u_exact": ("abs_u_exact",),
    "v_raw_exact": ("v_raw_exact",),
    "w_raw_exact": ("w_raw_exact",),
    "Q_exact": ("Q_exact",),
    "abs_u_error": ("abs_u_error",),
    "Q_error": ("Q_error",),
}


def _field_from_npz(
    data: np.lib.npyio.NpzFile,
    candidates: Iterable[str],
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Return the first available 2D candidate field, converted to a real array."""
    keys = set(data.files)
    for key in candidates:
        if key not in keys:
            continue
        arr = np.asarray(data[key])
        if key == "rho":
            arr = np.sqrt(np.maximum(np.real(arr), 0.0))
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        arr = np.asarray(arr, dtype=float)
        if arr.shape == shape:
            return arr
    return None


def _field_series_from_npz(
    data: np.lib.npyio.NpzFile,
    candidates: Iterable[str],
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Return the first available 3D candidate series, shape (nt,nx,ny)."""
    keys = set(data.files)
    for key in candidates:
        if key not in keys:
            continue
        arr = np.asarray(data[key])
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        arr = np.asarray(arr, dtype=float)
        if arr.shape == shape:
            return arr
    return None


def _load_fields(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load x, y and all available 2D scalar fields from final_comparison.npz."""
    data = np.load(npz_path)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    shape = (x.size, y.size)

    fields: dict[str, np.ndarray] = {}
    for out_name, candidates in _FIELD_CANDIDATES.items():
        arr = _field_from_npz(data, candidates, shape)
        if arr is not None:
            fields[out_name] = arr

    if not fields:
        raise ValueError(f"No scalar fields could be extracted from {npz_path}.")
    return x, y, fields


def _load_time_series(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load x, y, times and available scalar field series from time_series.npz."""
    data = np.load(npz_path)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    shape = (times.size, x.size, y.size)

    fields: dict[str, np.ndarray] = {}
    for out_name, candidates in _FIELD_CANDIDATES.items():
        arr = _field_series_from_npz(data, candidates, shape)
        if arr is not None:
            fields[out_name] = arr

    if not fields:
        raise ValueError(f"No time-series scalar fields could be extracted from {npz_path}.")
    return x, y, times, fields


def _write_legacy_vtk_structured_grid(
    vtk_path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    fields: dict[str, np.ndarray],
    title: str,
) -> Path:
    """Write one 2D structured grid as an ASCII legacy VTK file."""
    vtk_path = Path(vtk_path)
    vtk_path.parent.mkdir(parents=True, exist_ok=True)
    nx = x.size
    ny = y.size
    npts = nx * ny
    field_names = list(fields.keys())

    with open(vtk_path, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"POINTS {npts} float\n")
        # VTK structured grid ordering: i changes fastest, then j.
        for j, yj in enumerate(y):
            for i, xi in enumerate(x):
                f.write(f"{xi:.16e} {yj:.16e} 0.0000000000000000e+00\n")

        f.write(f"\nPOINT_DATA {npts}\n")
        for name in field_names:
            safe_name = name.replace(" ", "_")
            f.write(f"SCALARS {safe_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            arr = np.asarray(fields[name], dtype=float)
            if arr.shape != (nx, ny):
                raise ValueError(f"Field {name} has shape {arr.shape}; expected {(nx, ny)}.")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{float(arr[i, j]):.16e}\n")
            f.write("\n")
    return vtk_path


def export_final_comparison_to_csv(
    npz_path: str | Path,
    csv_path: str | Path | None = None,
) -> Path:
    """Export one Exp. 8 final_comparison.npz file as a ParaView point CSV."""
    npz_path = Path(npz_path)
    x, y, fields = _load_fields(npz_path)
    if csv_path is None:
        csv_path = npz_path.with_name("paraview_points.csv")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = list(fields.keys())
    z_field_names = [f"z_{name}" for name in field_names]
    header = ["i", "j", "x", "y", "z"] + z_field_names + field_names

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for j, yj in enumerate(y):
            for i, xi in enumerate(x):
                values = [float(fields[name][i, j]) for name in field_names]
                row = [i, j, xi, yj, 0.0] + values + values
                writer.writerow(row)

    return csv_path


def export_final_comparison_to_legacy_vtk(
    npz_path: str | Path,
    vtk_path: str | Path | None = None,
) -> Path:
    """Export one Exp. 8 final_comparison.npz file as a legacy VTK structured grid."""
    npz_path = Path(npz_path)
    x, y, fields = _load_fields(npz_path)
    if vtk_path is None:
        vtk_path = npz_path.with_name("paraview_structured.vtk")
    return _write_legacy_vtk_structured_grid(
        vtk_path, x, y, fields, title="GDSS Babaoglu final-state fields"
    )


def _write_pvd_collection(pvd_path: str | Path, vtk_files: list[Path], times: np.ndarray) -> Path:
    """Write a ParaView .pvd collection file referencing a VTK time sequence."""
    pvd_path = Path(pvd_path)
    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for vtk_file, t in zip(vtk_files, times):
            rel = escape(vtk_file.name)
            f.write(f'    <DataSet timestep="{float(t):.16e}" group="" part="0" file="{rel}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    return pvd_path


def export_time_series_to_legacy_vtk(
    npz_path: str | Path,
    outdir: str | Path | None = None,
    case_label: str | None = None,
) -> tuple[Path, list[Path]]:
    """
    Export one Exp. 8 time_series.npz as a ParaView animation sequence.

    Returns:
        pvd_path, vtk_frame_paths

    Open the .pvd file in ParaView to load the whole time sequence at once.
    """
    npz_path = Path(npz_path)
    if case_label is None:
        case_label = npz_path.parent.name
    if outdir is None:
        outdir = npz_path.parent / "paraview_time_series"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x, y, times, field_series = _load_time_series(npz_path)
    nt = times.size
    vtk_files: list[Path] = []
    for n in range(nt):
        fields_n = {name: arr[n] for name, arr in field_series.items()}
        vtk_path = outdir / f"{case_label}_{n:04d}.vtk"
        _write_legacy_vtk_structured_grid(
            vtk_path,
            x,
            y,
            fields_n,
            title=f"GDSS Babaoglu time-series fields, case={case_label}, frame={n}",
        )
        vtk_files.append(vtk_path)

    pvd_path = outdir / f"{case_label}_time_series.pvd"
    _write_pvd_collection(pvd_path, vtk_files, times)
    return pvd_path, vtk_files


def export_time_series_to_csv_sequence(
    npz_path: str | Path,
    outdir: str | Path | None = None,
    case_label: str | None = None,
) -> list[Path]:
    """
    Export time_series.npz as one CSV file per time frame.

    VTK/PVD is recommended for ParaView quality and convenience; CSV is provided
    for users who specifically want point tables.
    """
    npz_path = Path(npz_path)
    if case_label is None:
        case_label = npz_path.parent.name
    if outdir is None:
        outdir = npz_path.parent / "paraview_time_series_csv"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x, y, times, field_series = _load_time_series(npz_path)
    field_names = list(field_series.keys())
    z_field_names = [f"z_{name}" for name in field_names]
    header = ["time_index", "time", "i", "j", "x", "y", "z"] + z_field_names + field_names

    csv_files: list[Path] = []
    for n, t in enumerate(times):
        csv_path = outdir / f"{case_label}_{n:04d}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for j, yj in enumerate(y):
                for i, xi in enumerate(x):
                    values = [float(field_series[name][n, i, j]) for name in field_names]
                    row = [n, float(t), i, j, xi, yj, 0.0] + values + values
                    writer.writerow(row)
        csv_files.append(csv_path)
    return csv_files


def export_case(case_dir: str | Path, export_time_series: bool = True, export_csv_time_series: bool = False) -> dict[str, Path | list[Path]] | None:
    """Export one case directory containing final_comparison.npz and/or time_series.npz."""
    case_dir = Path(case_dir)
    outputs: dict[str, Path | list[Path]] = {}

    final_npz = case_dir / "final_comparison.npz"
    if final_npz.exists():
        csv_path = case_dir / f"{case_dir.name}_paraview_points.csv"
        vtk_path = case_dir / f"{case_dir.name}_paraview_structured.vtk"
        outputs["final_csv"] = export_final_comparison_to_csv(final_npz, csv_path)
        outputs["final_vtk"] = export_final_comparison_to_legacy_vtk(final_npz, vtk_path)

    ts_npz = case_dir / "time_series.npz"
    if export_time_series and ts_npz.exists():
        pvd_path, vtk_files = export_time_series_to_legacy_vtk(
            ts_npz,
            outdir=case_dir / "paraview_time_series",
            case_label=case_dir.name,
        )
        outputs["time_pvd"] = pvd_path
        outputs["time_vtk_files"] = vtk_files
        if export_csv_time_series:
            outputs["time_csv_files"] = export_time_series_to_csv_sequence(
                ts_npz,
                outdir=case_dir / "paraview_time_series_csv",
                case_label=case_dir.name,
            )

    return outputs or None


def _resolve_exp8_dir(output_root: str | Path | None = None, exp8_dir: str | Path | None = None) -> Path:
    """Accept either output_root='outputs' or exp8_dir='outputs/exp8_babaoglu_exact'."""
    if exp8_dir is not None:
        return Path(exp8_dir)
    root = Path("outputs") if output_root is None else Path(output_root)
    candidate = root / "exp8_babaoglu_exact"
    if candidate.exists():
        return candidate
    return root


def export_exp8_paraview(
    output_root: str | Path | None = None,
    exp8_dir: str | Path | None = None,
    export_time_series: bool = True,
    export_csv_time_series: bool = False,
) -> list[dict[str, Path | list[Path]]]:
    """Export all Exp. 8 case directories to ParaView-readable files."""
    exp8 = _resolve_exp8_dir(output_root=output_root, exp8_dir=exp8_dir)
    if not exp8.exists():
        raise FileNotFoundError(f"Exp. 8 directory not found: {exp8}")

    outputs: list[dict[str, Path | list[Path]]] = []
    for case_dir in sorted(p for p in exp8.iterdir() if p.is_dir()):
        exported = export_case(
            case_dir,
            export_time_series=export_time_series,
            export_csv_time_series=export_csv_time_series,
        )
        if exported is not None:
            outputs.append(exported)
            print(f"[ParaView] {case_dir.name}:")
            for key, value in exported.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} files")
                else:
                    print(f"  {key}: {value}")

    if not outputs:
        raise FileNotFoundError(
            f"No final_comparison.npz or time_series.npz files found inside case directories of {exp8}."
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GDSS Exp. 8 outputs for ParaView.")
    parser.add_argument("--output-root", default=None, help="Root outputs folder, e.g. outputs")
    parser.add_argument("--exp8-dir", default=None, help="Direct Exp. 8 folder, e.g. outputs/exp8_babaoglu_exact")
    parser.add_argument("--no-time-series", action="store_true", help="Only export final-state files, not temporal sequences.")
    parser.add_argument("--csv-time-series", action="store_true", help="Also export one CSV point table per time frame.")
    args = parser.parse_args()
    export_exp8_paraview(
        output_root=args.output_root,
        exp8_dir=args.exp8_dir,
        export_time_series=not args.no_time_series,
        export_csv_time_series=args.csv_time_series,
    )


if __name__ == "__main__":
    main()
