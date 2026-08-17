# Reference results

`paper_tables/` contains the eight CSV tables used by the manuscript. They are
small, platform-independent reference artifacts and are intentionally versioned.

Use:

```bash
python scripts/compare_reference_tables.py
```

after a full run. Performance columns are ignored. Scientific numerical values
are compared using the tolerances described in `docs/REPRODUCIBILITY.md`.
