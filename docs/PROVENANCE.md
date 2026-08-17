# Provenance and version-selection audit

Audit date: 2026-08-17.

The selected source was the local candidate directory named
`FFT Strang GDSS Article v2`.

An identically named copy under `Article FFT Strang GDSS/` contains byte-for-byte
identical versions of all five Python files. The candidates below `Old/` have
different hashes for the driver, experiment, and plotting files and were treated
as superseded versions.

The decisive cross-check was the scientific output: all eight CSV tables stored
with the selected source match byte-for-byte the tables incorporated into the
v4/v5 manuscript workflow.

## Original scientific source hashes

| File | SHA-256 |
|---|---|
| `gdss_solver.py` | `998e54ba7bf8fb5fef2d82a6cbfeaf45596f51528b4504aa20eac691719ea3bc` |
| `gdss_experiments.py` | `5ee86448830b2d5eeb0f8af607139a0ee55c82dc9116f63a4adb76668e9fcd30` |
| `gdss_plots.py` | `8331178d04c042dfd064619810265a1adb08f06e111e858b472cfe08b220c8ec` |
| `gdss_paraview_export.py` | `075acb3109b2f5c76281d35998ec819538c0fa62ad0f2acea2c4a664a20d6a22` |
| `run.py` | `5b124289e64d57dcd2f44018c7a0871e18c7af4af40b56a16a2b3879733a5ca3` |

`tests/test_source_integrity.py` enforces these hashes so future edits cannot be
mistaken for the code that generated the manuscript results. Enhancements should
be added in new files or accompanied by a documented release and new references.
