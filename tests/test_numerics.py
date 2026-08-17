from __future__ import annotations

import unittest

from gdss_solver import (
    GDSSParams,
    gaussian_initial_data,
    hermitian_defect,
    longwave_interaction_residual,
    longwave_residuals,
    make_grid,
    mass,
    precompute_multipliers,
    recover_longwave,
    strang_step,
    zero_mode_residual,
)


class SolverSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = GDSSParams(
            Nx=32,
            Ny=32,
            Lx=20.0,
            Ly=20.0,
            dt=1.0e-3,
            t_final=1.0e-3,
            recover_full_longwave=True,
        )
        self.grid = make_grid(self.params)
        self.multipliers = precompute_multipliers(self.params, self.grid)
        self.u0 = gaussian_initial_data(
            self.grid, amplitude=1.0, width=3.0, kx0=0.2, ky0=-0.1
        )

    def test_longwave_recovery_residuals_are_small(self) -> None:
        recovered = recover_longwave(
            self.u0, self.grid, self.params, self.multipliers, full=True
        )
        self.assertLess(
            longwave_interaction_residual(recovered, self.grid, self.params), 1.0e-12
        )
        self.assertLess(zero_mode_residual(recovered, self.grid), 1.0e-12)
        self.assertLess(hermitian_defect(recovered), 1.0e-10)
        residual_w, residual_v = longwave_residuals(
            recovered, self.grid, self.params
        )
        self.assertLess(residual_w, 1.0e-12)
        self.assertLess(residual_v, 1.0e-12)

    def test_one_strang_step_preserves_mass(self) -> None:
        initial_mass = mass(self.u0, self.grid)
        u1, _ = strang_step(
            self.u0, self.grid, self.params, self.multipliers
        )
        relative_drift = abs(mass(u1, self.grid) - initial_mass) / initial_mass
        self.assertLess(relative_drift, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
