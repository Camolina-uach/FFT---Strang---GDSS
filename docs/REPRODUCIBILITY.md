# Reproducibility notes

## Levels of verification

1. `python -m unittest discover -s tests -v` checks source integrity and fast
   numerical properties (long-wave recovery and mass preservation).
2. `python reproduce.py smoke` executes all non-Babaoglu experiment pathways
   on a small grid.
3. `python reproduce.py paper` performs the manuscript computation and figure
   generation.
4. `python scripts/compare_reference_tables.py` compares the regenerated paper
   tables with the archived CSV files.

## Expected variation

FFT roundoff may vary slightly with NumPy, BLAS/FFT implementation, CPU, and
operating system. The comparison script therefore uses `rtol=1e-7` and
`atol=5e-12` by default. It excludes `cpu_time_seconds`, `C_step`, and `C_norm`,
which are performance measurements rather than invariant scientific values.

The cost-scaling experiment should be interpreted by trend and FFT count, not
by equality of wall-clock values. PDF metadata and font embedding may also make
figure-file hashes differ even when the plotted data agree.

## Storage

The full run can create hundreds of megabytes, especially when ParaView time
series are requested. Generated `outputs/` are excluded from Git. Keep the small
reference tables in version control and deposit a full immutable release in an
archival service such as Zenodo if journal policy or peer review requires it.

## Recommended release procedure

1. Confirm the author list and software version. Authorization for the
   BSD-3-Clause release was confirmed on 2026-08-17.
2. Run the unit tests, full paper workflow, and table comparison in a clean environment.
3. Commit and tag the release (for example `v1.0.0`).
4. Archive that tag with Zenodo to obtain a DOI.
5. Add the DOI to `CITATION.cff`, the README, and the manuscript data/code-availability statement.

See `PRE_PUBLICATION_CHECKLIST.md` for the GitHub-specific final review.
