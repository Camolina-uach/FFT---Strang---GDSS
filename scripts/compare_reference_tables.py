#!/usr/bin/env python3
"""Compare regenerated manuscript CSV tables with archived references."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "reference_results" / "paper_tables"
TABLE_PATHS = {
    "recovery_diagnostics.csv": "exp1_recovery/tables/recovery_diagnostics.csv",
    "invariant_balance_summary.csv": "exp2_baseline/tables/invariant_balance_summary.csv",
    "invariant_balance_times.csv": "exp2_baseline/tables/invariant_balance_times.csv",
    "longwave_balance_summary.csv": "exp2_baseline/tables/longwave_balance_summary.csv",
    "timestep_convergence.csv": "exp4_timestep/tables/timestep_convergence.csv",
    "boundary_contamination.csv": "exp6_boundary/tables/boundary_contamination.csv",
    "babaoglu_exact_benchmark.csv": "exp8_babaoglu_exact/tables/babaoglu_exact_benchmark.csv",
    "babaoglu_spatial_convergence.csv": "exp8b_babaoglu_spatial/tables/babaoglu_spatial_convergence.csv",
}
IGNORED_COLUMNS = {"cpu_time_seconds", "C_step", "C_norm"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compare_table(reference: Path, candidate: Path, rtol: float, atol: float) -> list[str]:
    errors: list[str] = []
    ref_columns, ref_rows = read_csv(reference)
    got_columns, got_rows = read_csv(candidate)
    if ref_columns != got_columns:
        errors.append(f"columns differ: expected {ref_columns}, got {got_columns}")
        return errors
    if len(ref_rows) != len(got_rows):
        errors.append(f"row count differs: expected {len(ref_rows)}, got {len(got_rows)}")
        return errors

    for row_number, (expected, got) in enumerate(zip(ref_rows, got_rows), start=2):
        for column in ref_columns:
            if column in IGNORED_COLUMNS:
                continue
            ref_value = expected[column]
            got_value = got[column]
            ref_number = as_float(ref_value)
            got_number = as_float(got_value)
            if ref_number is not None and got_number is not None:
                if not math.isclose(got_number, ref_number, rel_tol=rtol, abs_tol=atol):
                    errors.append(
                        f"row {row_number}, {column}: expected {ref_number:.17g}, "
                        f"got {got_number:.17g}"
                    )
            elif ref_value != got_value:
                errors.append(
                    f"row {row_number}, {column}: expected {ref_value!r}, got {got_value!r}"
                )
            if len(errors) >= 20:
                errors.append("comparison stopped after 20 differences")
                return errors
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--rtol", type=float, default=1.0e-7)
    parser.add_argument("--atol", type=float, default=5.0e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failed = False
    for name, relative_path in TABLE_PATHS.items():
        reference = REFERENCE_DIR / name
        candidate = args.outputs / relative_path
        if not candidate.exists():
            print(f"MISSING {name}: {candidate}")
            failed = True
            continue
        errors = compare_table(reference, candidate, args.rtol, args.atol)
        if errors:
            failed = True
            print(f"FAIL    {name}")
            for error in errors:
                print(f"        {error}")
        else:
            print(f"PASS    {name}")
    if failed:
        print("Reference comparison failed.")
        return 1
    print("All eight manuscript tables agree within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
