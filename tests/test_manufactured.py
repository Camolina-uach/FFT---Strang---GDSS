from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gdss_manufactured import PARAMETER_SETS, manufactured_fields, rel_l2  # noqa: E402
from gdss_solver import (  # noqa: E402
    GDSSParams,
    make_grid,
    precompute_multipliers,
    recover_longwave,
)


def _recover(set_name: str, N: int, dealias: bool, theta_override=None):
    lw = dict(PARAMETER_SETS[set_name])
    if theta_override is not None:
        lw["theta"] = theta_override
    p = GDSSParams(Nx=N, Ny=N, Lx=40.0, Ly=40.0, dealias_density=dealias,
                   recover_full_longwave=True, **lw)
    g = make_grid(p)
    ref = GDSSParams(Nx=N, Ny=N, Lx=40.0, Ly=40.0, **PARAMETER_SETS[set_name])
    ex = manufactured_fields(ref, g.X, g.Y)
    u = np.sqrt(ex["rho"] + 1.0 - ex["rho"].min()).astype(np.complex128)
    rec = recover_longwave(u, g, p, precompute_multipliers(p, g), full=True)
    return rec, ex


class ManufacturedLongwaveTests(unittest.TestCase):
    def test_coupled_recovery_reaches_round_off(self) -> None:
        for set_name in PARAMETER_SETS:
            for N, dealias in ((96, False), (128, True)):
                with self.subTest(set=set_name, N=N, dealias=dealias):
                    rec, ex = _recover(set_name, N, dealias)
                    for f in ("w", "v", "Q"):
                        self.assertLess(rel_l2(rec[f], ex[f]), 1.0e-12)

    def test_recovery_without_coupling_fails(self) -> None:
        rec, ex = _recover("baseline", 96, False, theta_override=0.0)
        self.assertGreater(rel_l2(rec["v"], ex["v"]), 1.0e-2)


if __name__ == "__main__":
    unittest.main()
