# Validation record

Validation date: 2026-08-17

Platform: macOS, Python 3.13.2, NumPy 2.3.2, Matplotlib 3.10.6

## Completed checks

- Source integrity: all five archived scientific-source SHA-256 hashes passed.
- Unit tests: 3/3 passed.
- Smoke workflow: Experiments 1--7 completed on a reduced grid.
- Full manuscript workflow: Experiments 1--8b completed with manuscript parameters.
- Figure generation: 36 PDF/PNG files were created successfully.
- Table regression: all eight manuscript CSV tables passed with `rtol=1e-7`
  and `atol=5e-12`; machine-dependent timing columns were excluded.
- Public-release review: BSD-3-Clause authorization recorded, privacy/secret
  scan clean, LF line endings enforced, and GitHub Actions configuration parsed.

The optional ParaView export was not repeated during packaging because the
archived exporter itself is unchanged and the export is a large derived product.
It can be generated with `python reproduce.py paper --with-paraview` or the
unchanged archival command `python run.py`.

Generated validation outputs were removed after the checks. They are fully
regenerable and excluded by `.gitignore`; the eight small manuscript reference
tables remain in the repository.
