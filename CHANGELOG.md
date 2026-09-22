# Changelog

## 1.1.0 — 2026-09-21 (manuscript revision)

- Added a manufactured-solution test of the coupled long-wave recovery, built
  from the adjugate of the long-wave operator, and a refined-reference
  convergence study of the recovered `w`, `v`, `Q` along the baseline run
  (`gdss_manufactured.py`, `gdss_manufactured_plots.py`,
  `tests/test_manufactured.py`).
- Added a long-horizon drift run to `T = 20` on an enlarged domain for three
  time steps (`gdss_long_horizon.py`, `gdss_long_horizon_plots.py`).
- Added the spectral-resolution figure with the tail region `K_tail` marked
  (`gdss_spectrum_figure.py`).
- Added the `revision` command to `reproduce.py`.
- The five archived scientific sources are unchanged (hash test still passes).

## 1.0.0 — 2026-08-17

- Selected the final `FFT Strang GDSS Article v2` implementation after a hash
  audit of the available versions.
- Preserved the five scientific Python files without modification.
- Added reproducibility drivers, tests, dependency files, citation metadata,
  bilingual instructions, provenance documentation, and the eight manuscript
  CSV reference tables.
- Prepared the package for a public GitHub repository: adopted BSD-3-Clause,
  removed the private local source path, and added automated GitHub Actions tests.
- Recorded author authorization for BSD-3-Clause, normalized public file
  permissions, and enforced LF line endings for cross-platform hash stability.
- Added the public GitHub repository URL to the citation and availability metadata.
