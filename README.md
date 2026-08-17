# FFT Strang solver for the generalized Davey--Stewartson system

Research code and reproducibility material for the manuscript **“A reproducible
FFT-based Strang-splitting baseline for the generalized Davey--Stewartson
system”**, prepared for the *International Journal of Modern Physics C*.

Repository: <https://github.com/Camolina-uach/FFT---Strang---GDSS>

The five original Python files used to generate the manuscript results are kept
unchanged at the repository root. Their SHA-256 hashes and the version-selection
audit are recorded in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib

The validated environment was Python 3.13.2, NumPy 2.3.2, and Matplotlib
3.10.6 on macOS; it is recorded in `requirements-validated.txt`. Other recent
versions should work, but small floating-point and timing differences are expected.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce.py smoke
```

The smoke run uses a small grid and a short time interval. It verifies the full
experiment orchestration without attempting to reproduce the paper tables.

## Reproduce the paper results

```bash
python reproduce.py paper
python scripts/compare_reference_tables.py
```

This runs Experiments 1--8b with the manuscript parameters, generates the paper
figures, and writes all products below `outputs/`. The ParaView time-series
export is optional because it is large:

```bash
python reproduce.py paper --with-paraview
```

The archival driver used for the original results remains available as:

```bash
python run.py
```

It performs the full run, figures, and ParaView export in one command. Existing
`outputs/` files may be overwritten by either full command, so archive important
runs before re-executing them.

## Repository contents

- `gdss_solver.py`: Fourier pseudospectral solver, Strang step, invariants, and diagnostics.
- `gdss_experiments.py`: numerical experiments and table generation.
- `gdss_plots.py`: manuscript figures (PDF and PNG).
- `gdss_paraview_export.py`: VTK/CSV exports for ParaView.
- `run.py`: unchanged archival full-run driver.
- `reproduce.py`: documented command-line driver for smoke and paper runs.
- `reference_results/paper_tables/`: the eight CSV tables used in the manuscript.
- `scripts/compare_reference_tables.py`: tolerant comparison against the reference tables.
- `tests/`: fast numerical and integrity tests.
- `docs/`: experiment map, reproducibility notes, and provenance audit.

Runtime measurements (`cpu_time_seconds`, `C_step`, and `C_norm`) depend on the
machine and are intentionally excluded from reference-value comparisons.

## Citation and licensing

Use [`CITATION.cff`](CITATION.cff) to cite this software. Add the article DOI and
the archived software DOI after acceptance/publication. The code is distributed
under the permissive [`BSD-3-Clause`](LICENSE) license, with authorization from
the copyright holders.

Spanish instructions are available in [`README_ES.md`](README_ES.md).
