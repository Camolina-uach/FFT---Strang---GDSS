# Contributing

Please open an issue describing the numerical or documentation change before a
substantial modification. Scientific changes should include:

1. the mathematical rationale;
2. a focused test that fails before the change and passes afterward;
3. a new full reproduction run when reference values may change; and
4. an update to `CHANGELOG.md` and `docs/PROVENANCE.md`.

Do not replace the reference tables without documenting why the manuscript
values changed. Keep machine-dependent timings separate from scientific
regression checks.
