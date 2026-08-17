from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "gdss_solver.py": "998e54ba7bf8fb5fef2d82a6cbfeaf45596f51528b4504aa20eac691719ea3bc",
    "gdss_experiments.py": "5ee86448830b2d5eeb0f8af607139a0ee55c82dc9116f63a4adb76668e9fcd30",
    "gdss_plots.py": "8331178d04c042dfd064619810265a1adb08f06e111e858b472cfe08b220c8ec",
    "gdss_paraview_export.py": "075acb3109b2f5c76281d35998ec819538c0fa62ad0f2acea2c4a664a20d6a22",
    "run.py": "5b124289e64d57dcd2f44018c7a0871e18c7af4af40b56a16a2b3879733a5ca3",
}


class SourceIntegrityTests(unittest.TestCase):
    def test_original_scientific_sources_are_unchanged(self) -> None:
        for filename, expected in EXPECTED_SHA256.items():
            with self.subTest(filename=filename):
                digest = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
                self.assertEqual(digest, expected)


if __name__ == "__main__":
    unittest.main()
