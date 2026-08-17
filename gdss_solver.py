"""
gdss_solver.py — Fourier pseudo-spectral Strang splitting baseline for the GDSS.

Model
-----
    i u_t + alpha u_xx + beta u_yy = gamma |u|^2 u + xi Q u,   Q = w_x + v_y,
    psi w_xx + eta w_yy + theta v_xy = d_x(|u|^2),
    phi v_xx + chi v_yy + theta w_xy = d_y(|u|^2).

Design
------
- Periodic box, NumPy native FFT ordering.
- 2/3 dealiasing applied only to the density used in the nonlocal recovery;
  u itself is never projected.
- All diagnostics defined here match the manuscript (sec:balance_diagnostics).

Corrections relative to earlier drafts
---------------------------------------
1. momentum():               sign of jx/jy_density corrected.
2. energy_longwave_quadratic(): cross-term corrected to +2*theta*wx*vy.
3. longwave_interaction_residual(): new — implements R_lw from the manuscript.
4. zero_mode_residual():     new — implements Z_lw.
5. hermitian_defect():       new — implements H_f.
6. summarize_history():      relative drift uses eps_I = 1e-14*max(1,|I_0|).
7. diagnostics_row():        R_lw, Z_lw, H_defect included when available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from typing import Dict, Optional, Tuple

import numpy as np

Array = np.ndarray


# ═══════════════════════════════════════════════════════════════════ dataclasses


@dataclass
class GDSSParams:
    """
    All model and numerical parameters in one place.
    theta=None triggers the structural condition theta = sqrt((phi-psi)(eta-chi)).
    """
    # Short-wave PDE coefficients
    alpha: float = 1.0
    beta:  float = 1.0
    gamma: float = 1.0
    xi:    float = 1.0

    # Long-wave PDE coefficients
    psi:   float = 1.0
    eta:   float = 1.0
    phi:   float = 2.0
    chi:   float = 0.5
    theta: Optional[float] = None   # set to structural value if None

    # Domain
    Lx: float = 40.0
    Ly: float = 40.0

    # Spatial grid
    Nx: int = 128
    Ny: int = 128

    # Time integration
    dt:      float = 1.0e-3
    t_final: float = 0.05

    # Numerical options
    dealias_density:      bool  = True
    recover_full_longwave: bool  = True
    zero_tol_factor:      float = 100.0  # multiplier for Delta near-zero check

    def __post_init__(self) -> None:
        if self.theta is None:
            val = (self.phi - self.psi) * (self.eta - self.chi)
            if val < 0:
                raise ValueError(
                    "Structural theta is not real: (phi-psi)(eta-chi) < 0."
                )
            self.theta = float(np.sqrt(val))
        if self.Nx % 2 or self.Ny % 2:
            raise ValueError("Nx and Ny must be even for the 2/3 dealiasing mask.")


@dataclass
class Grid:
    x:            Array   # physical x-coordinates, shape (Nx,)
    y:            Array   # physical y-coordinates, shape (Ny,)
    X:            Array   # meshgrid X, shape (Nx, Ny)
    Y:            Array   # meshgrid Y, shape (Nx, Ny)
    dx:           float
    dy:           float
    KX:           Array   # wavenumber meshgrid, shape (Nx, Ny)
    KY:           Array
    mode_x:       Array   # integer mode indices, shape (Nx, Ny)
    mode_y:       Array
    dealias_mask: Array   # bool, True inside the 2/3 band
    zero_mode:    Array   # bool, True only at (KX, KY) = (0, 0)


@dataclass
class Multipliers:
    Ehalf:                 Array   # linear half-step exponential, shape (Nx, Ny)
    Mw:                    Array   # long-wave multiplier for w
    Mv:                    Array   # long-wave multiplier for v
    MQ:                    Array   # long-wave multiplier for Q = w_x + v_y
    Delta:                 Array   # denominator of the Fourier system
    valid:                 Array   # bool: nonzero and non-singular modes
    min_abs_delta_nonzero: float   # smallest |Delta| over nonzero modes


# ════════════════════════════════════════════════════════════ grid and precomputation


def make_grid(p: GDSSParams) -> Grid:
    """Build the uniform periodic grid and precompute wavenumber arrays."""
    dx = p.Lx / p.Nx
    dy = p.Ly / p.Ny

    x = np.linspace(-p.Lx / 2.0, p.Lx / 2.0, p.Nx, endpoint=False)
    y = np.linspace(-p.Ly / 2.0, p.Ly / 2.0, p.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    kx = 2.0 * np.pi * np.fft.fftfreq(p.Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(p.Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    mode_x_1d = np.fft.fftfreq(p.Nx) * p.Nx   # integer mode indices
    mode_y_1d = np.fft.fftfreq(p.Ny) * p.Ny
    mode_x, mode_y = np.meshgrid(mode_x_1d, mode_y_1d, indexing="ij")

    dealias_mask = (
        (np.abs(mode_x) <= p.Nx / 3.0) & (np.abs(mode_y) <= p.Ny / 3.0)
    )
    zero_mode = (KX == 0.0) & (KY == 0.0)

    return Grid(
        x=x, y=y, X=X, Y=Y, dx=dx, dy=dy,
        KX=KX, KY=KY, mode_x=mode_x, mode_y=mode_y,
        dealias_mask=dealias_mask, zero_mode=zero_mode,
    )


def precompute_multipliers(p: GDSSParams, g: Grid) -> Multipliers:
    """
    Precompute all Fourier multipliers for the long-wave recovery and
    the linear dispersive half-step.

    Long-wave system in Fourier space:
        (-psi k^2 - eta l^2) w_hat - theta*k*l * v_hat = i*k * rho_hat
        -theta*k*l * w_hat + (-phi k^2 - chi l^2) v_hat = i*l * rho_hat

    With  a = psi*k^2 + eta*l^2,  b = theta*k*l,  c = phi*k^2 + chi*l^2,
          Delta = a*c - b^2,
    Cramer's rule gives:
        Mw = -i(c*k - b*l) / Delta
        Mv = -i(-b*k + a*l) / Delta
        MQ = (c*k^2 - 2*b*k*l + a*l^2) / Delta
    """
    KX, KY = g.KX, g.KY

    a = p.psi * KX**2 + p.eta * KY**2
    b = p.theta * KX * KY
    c = p.phi * KX**2 + p.chi * KY**2
    Delta = a * c - b**2

    nonzero = ~g.zero_mode
    min_abs_delta = (
        float(np.min(np.abs(Delta[nonzero]))) if np.any(nonzero) else 0.0
    )

    eps = np.finfo(float).eps
    tol = p.zero_tol_factor * eps * max(1.0, float(np.max(np.abs(Delta))))
    valid = nonzero & (np.abs(Delta) > tol)

    Mw = np.zeros_like(Delta, dtype=np.complex128)
    Mv = np.zeros_like(Delta, dtype=np.complex128)
    MQ = np.zeros_like(Delta, dtype=np.complex128)

    Mw[valid] = -1j * (c[valid] * KX[valid] - b[valid] * KY[valid]) / Delta[valid]
    Mv[valid] = -1j * (-b[valid] * KX[valid] + a[valid] * KY[valid]) / Delta[valid]
    MQ[valid] = (
        c[valid] * KX[valid]**2
        - 2.0 * b[valid] * KX[valid] * KY[valid]
        + a[valid] * KY[valid]**2
    ) / Delta[valid]

    omega = p.alpha * KX**2 + p.beta * KY**2
    Ehalf = np.exp(-1j * omega * p.dt / 2.0)

    return Multipliers(
        Ehalf=Ehalf, Mw=Mw, Mv=Mv, MQ=MQ,
        Delta=Delta, valid=valid,
        min_abs_delta_nonzero=min_abs_delta,
    )


# ═══════════════════════════════════════════════════════════ spectral operations


def fft2(f: Array) -> Array:
    return np.fft.fft2(f)


def ifft2(fhat: Array) -> Array:
    return np.fft.ifft2(fhat)


def spectral_dx(f: Array, g: Grid) -> Array:
    """Spectral derivative d/dx via Fourier multiplication by i*kx."""
    return ifft2(1j * g.KX * fft2(f))


def spectral_dy(f: Array, g: Grid) -> Array:
    """Spectral derivative d/dy via Fourier multiplication by i*ky."""
    return ifft2(1j * g.KY * fft2(f))


def filter_density_hat(rho: Array, g: Grid, p: GDSSParams) -> Array:
    """
    FFT of rho, optionally dealiased with the 2/3 mask, zero mode zeroed.
    This filtered density is used exclusively for the nonlocal recovery.
    """
    rho_hat = fft2(rho)
    if p.dealias_density:
        rho_hat = g.dealias_mask * rho_hat
    rho_hat[g.zero_mode] = 0.0
    return rho_hat


# ═══════════════════════════════════════════════════════════ long-wave recovery


def recover_longwave(
    u: Array,
    g: Grid,
    p: GDSSParams,
    m: Multipliers,
    full: Optional[bool] = None,
) -> Dict[str, Array]:
    """
    Recover Q (and optionally w, v) from rho = |u|^2.

    The *local* density used in the nonlinear phase rotation is the unfiltered
    rho = |u|^2.  Only the density fed into the Fourier multipliers is filtered.

    Returns a dict with keys:
        rho, rho_hat_f, Q_hat, Q
        (and w_hat, v_hat, w, v if full=True)
    """
    if full is None:
        full = p.recover_full_longwave

    rho = np.abs(u) ** 2
    rho_hat_f = filter_density_hat(rho, g, p)

    Q_hat = m.MQ * rho_hat_f
    Q_hat[g.zero_mode] = 0.0
    Q = np.real(ifft2(Q_hat))

    out: Dict[str, Array] = {
        "rho": rho, "rho_hat_f": rho_hat_f, "Q_hat": Q_hat, "Q": Q,
    }

    if full:
        w_hat = m.Mw * rho_hat_f
        v_hat = m.Mv * rho_hat_f
        w_hat[g.zero_mode] = 0.0
        v_hat[g.zero_mode] = 0.0
        out.update({
            "w_hat": w_hat,
            "v_hat": v_hat,
            "w": np.real(ifft2(w_hat)),
            "v": np.real(ifft2(v_hat)),
        })

    return out


# ═══════════════════════════════════════════════════════════ Strang splitting


def linear_half_step(u: Array, m: Multipliers) -> Array:
    """Apply the linear dispersive half-step: u <- ifft(Ehalf * fft(u))."""
    return ifft2(m.Ehalf * fft2(u))


def nonlinear_step(u: Array, Q: Array, p: GDSSParams) -> Array:
    """
    Apply the exact nonlinear nonlocal phase rotation over dt:
        u <- exp(-i * dt * (gamma*|u|^2 + xi*Q)) * u.
    |u|^2 and Q are frozen during this subflow (intensity invariance).
    """
    rho = np.abs(u) ** 2
    return np.exp(-1j * p.dt * (p.gamma * rho + p.xi * Q)) * u


def strang_step(
    u: Array, g: Grid, p: GDSSParams, m: Multipliers
) -> Tuple[Array, Dict[str, Array]]:
    """
    One Strang step:  u^{n+1} = Phi_L^{dt/2} o Phi_N^{dt} o Phi_L^{dt/2} (u^n).

    Returns (u_new, lw) where lw is the long-wave dict computed at the
    intermediate state u^{(1)} (used for the nonlinear phase; not the
    diagnostic lw, which should be recomputed from u_new if needed).
    """
    u1 = linear_half_step(u, m)
    lw = recover_longwave(u1, g, p, m, full=p.recover_full_longwave)
    u2 = nonlinear_step(u1, lw["Q"], p)
    return linear_half_step(u2, m), lw


# ═══════════════════════════════════════════════════════════ conservation quantities


def mass(u: Array, g: Grid) -> float:
    """Discrete mass M_h = ||u||_h^2 = dx*dy * sum |u_{j,l}|^2."""
    return float(g.dx * g.dy * np.sum(np.abs(u) ** 2))


def momentum(u: Array, g: Grid) -> Tuple[float, float]:
    """
    Discrete momenta consistent with the manuscript definition:
        J_x = -2 Im <D_x u, u>_h
            = Re[ i (conj(u) * ux - u * conj(ux)) ] * dx * dy
        J_y = -2 Im <D_y u, u>_h  (analogous)

    Note: the operand order matters.  The incorrect version
    1j*(u*conj(ux) - conj(u)*ux) gives -J_x (wrong sign).
    """
    ux = spectral_dx(u, g)
    uy = spectral_dy(u, g)

    jx_density = 1j * (np.conj(u) * ux - u * np.conj(ux))
    jy_density = 1j * (np.conj(u) * uy - u * np.conj(uy))

    Jx = float(np.real(g.dx * g.dy * np.sum(jx_density)))
    Jy = float(np.real(g.dx * g.dy * np.sum(jy_density)))
    return Jx, Jy


def energy_reduced(u: Array, Q: Array, g: Grid, p: GDSSParams) -> float:
    """
    Reduced Hamiltonian diagnostic (slaved long waves via interaction identity):
        E = alpha||u_x||^2 + beta||u_y||^2 + gamma/2 * int|u|^4 + xi/2 * int|u|^2*Q.
    This is the primary energy diagnostic for the FFT-Strang baseline.
    Not expected to be exactly conserved by the splitting method.
    """
    ux  = spectral_dx(u, g)
    uy  = spectral_dy(u, g)
    rho = np.abs(u) ** 2

    e = (
        p.alpha * np.sum(np.abs(ux) ** 2)
        + p.beta  * np.sum(np.abs(uy) ** 2)
        + 0.5 * p.gamma * np.sum(rho ** 2)
        + 0.5 * p.xi    * np.sum(rho * Q)
    )
    return float(np.real(g.dx * g.dy * e))


def energy_longwave_quadratic(
    u: Array,
    lw: Dict[str, Array],
    g: Grid,
    p: GDSSParams,
) -> Optional[float]:
    """
    Full-field Hamiltonian diagnostic using w and v explicitly.

    From the interaction identity (integrating the long-wave PDEs by parts):
        int rho Q = int (psi*wx^2 + eta*wy^2 + phi*vx^2 + chi*vy^2
                         + 2*theta*wx*vy)

    The cross term is +2*theta*wx*vy (NOT -2*theta*wy*vx).
    Returns None if w and v are not in lw.
    """
    if "w" not in lw or "v" not in lw:
        return None

    w, v = lw["w"], lw["v"]
    ux = spectral_dx(u, g)
    uy = spectral_dy(u, g)
    wx = np.real(spectral_dx(w, g))
    wy = np.real(spectral_dy(w, g))
    vx = np.real(spectral_dx(v, g))
    vy = np.real(spectral_dy(v, g))
    rho = np.abs(u) ** 2

    long_part = (
        p.psi * wx**2
        + p.eta * wy**2
        + p.phi * vx**2
        + p.chi * vy**2
        + 2.0 * p.theta * wx * vy   # +2θ <w_x, v_y> from interaction identity
    )
    e = (
        p.alpha * np.sum(np.abs(ux) ** 2)
        + p.beta  * np.sum(np.abs(uy) ** 2)
        + 0.5 * p.gamma * np.sum(rho ** 2)
        + 0.5 * p.xi    * np.sum(long_part)
    )
    return float(np.real(g.dx * g.dy * e))


# ═══════════════════════════════════════════════════════ manuscript diagnostics


def longwave_interaction_residual(
    lw: Dict[str, Array], g: Grid, p: GDSSParams
) -> Optional[float]:
    """
    R_lw^n from eq. longwave_consistency_residual in the manuscript:
        R_lw = |<Q, rho_f>_h - B_lw| / (|B_lw| + eps_B)

    <Q, rho_f>_h is a physical-space quadrature.
    B_lw is computed from the separately reconstructed w, v via spectral
    derivatives — a different numerical route from Q, providing a genuine check.

    Because Q, w, v all come from the same rho_hat_f, the identity holds to
    round-off in a correct implementation.  Large values signal sign errors,
    normalization inconsistencies, or loss of Hermitian symmetry.

    Returns None if w, v are not in lw (full recovery not requested).
    """
    if "w" not in lw or "v" not in lw:
        return None

    rho_f = np.real(ifft2(lw["rho_hat_f"]))
    inner_Q_rho = float(g.dx * g.dy * np.sum(lw["Q"] * rho_f))

    w, v = lw["w"], lw["v"]
    wx = np.real(spectral_dx(w, g))
    wy = np.real(spectral_dy(w, g))
    vx = np.real(spectral_dx(v, g))
    vy = np.real(spectral_dy(v, g))

    B_lw = float(g.dx * g.dy * np.sum(
        p.psi * wx**2
        + p.eta * wy**2
        + p.phi * vx**2
        + p.chi * vy**2
        + 2.0 * p.theta * wx * vy
    ))

    eps_B = 1e-14 * max(1.0, abs(B_lw))
    return float(abs(inner_Q_rho - B_lw) / (abs(B_lw) + eps_B))


def zero_mode_residual(lw: Dict[str, Array], g: Grid) -> float:
    """
    Z_lw^n from eq. zero_mode_residual in the manuscript:
        Z_lw = |w_hat_{0,0}| + |v_hat_{0,0}| + |Q_hat_{0,0}|.
    Should remain at round-off level when mean-free normalization is enforced.
    Always available (Q_hat is always in lw).
    """
    z = float(abs(lw["Q_hat"][g.zero_mode].sum()))
    if "w_hat" in lw:
        z += float(abs(lw["w_hat"][g.zero_mode].sum()))
    if "v_hat" in lw:
        z += float(abs(lw["v_hat"][g.zero_mode].sum()))
    return z


def hermitian_defect(lw: Dict[str, Array]) -> float:
    """
    H_f^n from eq. hermitian_symmetry_defect in the manuscript:
        H_f = max_{p,q} |f_hat_{-p,-q} - conj(f_hat_{p,q})|.

    Applied to rho_hat_f, Q_hat, and (if available) w_hat, v_hat.
    The negative-frequency index (-p,-q) maps to array index
    (Nx-p)%Nx, (Ny-q)%Ny — implemented via reverse + roll.
    """
    fields = {"rho_hat_f": lw["rho_hat_f"], "Q_hat": lw["Q_hat"]}
    for key in ("w_hat", "v_hat"):
        if key in lw:
            fields[key] = lw[key]

    max_def = 0.0
    for fhat in fields.values():
        # fhat_neg[i,j] = fhat[(Nx-i)%Nx, (Ny-j)%Ny]  (negative frequency)
        fhat_neg = np.roll(np.roll(fhat[::-1, ::-1], 1, axis=0), 1, axis=1)
        max_def = max(max_def, float(np.max(np.abs(fhat_neg - np.conj(fhat)))))
    return max_def


def longwave_residuals(
    lw: Dict[str, Array], g: Grid, p: GDSSParams
) -> Optional[Tuple[float, float]]:
    """
    Relative spectral residuals of the two long-wave PDEs in Fourier space:
        res_w = ||(-a*w_hat - b*v_hat) - i*kx*rho_hat_f|| / ||i*kx*rho_hat_f||
        res_v = ||(-b*w_hat - c*v_hat) - i*ky*rho_hat_f|| / ||i*ky*rho_hat_f||

    This verifies the accuracy of the Fourier solve itself — distinct from
    longwave_interaction_residual, which checks the interaction identity.
    Returns None if w_hat, v_hat are not in lw.
    """
    if "w_hat" not in lw or "v_hat" not in lw:
        return None

    wh, vh, rhoh = lw["w_hat"], lw["v_hat"], lw["rho_hat_f"]

    lhs1 = -(p.psi * g.KX**2 + p.eta * g.KY**2) * wh - p.theta * g.KX * g.KY * vh
    rhs1 = 1j * g.KX * rhoh
    lhs2 = -(p.phi * g.KX**2 + p.chi * g.KY**2) * vh - p.theta * g.KX * g.KY * wh
    rhs2 = 1j * g.KY * rhoh

    num1 = np.linalg.norm((lhs1 - rhs1).ravel())
    num2 = np.linalg.norm((lhs2 - rhs2).ravel())
    den1 = max(np.linalg.norm(rhs1.ravel()), 1.0e-30)
    den2 = max(np.linalg.norm(rhs2.ravel()), 1.0e-30)

    return float(num1 / den1), float(num2 / den2)


# ═══════════════════════════════════════════════════════ auxiliary diagnostics


def boundary_amplitude(u: Array, band: int = 4) -> float:
    """
    Max |u| in a boundary band of width `band` grid points.
    Checks whether the solution remains negligible near the periodic boundary.
    """
    if band <= 0:
        return 0.0
    edges = np.r_[
        np.abs(u[:band,  :]).ravel(),
        np.abs(u[-band:, :]).ravel(),
        np.abs(u[:,  :band]).ravel(),
        np.abs(u[:, -band:]).ravel(),
    ]
    return float(np.max(edges))


def spectral_tail_ratio(u: Array, g: Grid, alpha_tail: float = 2.0 / 3.0) -> float:
    """
    Relative energy in high-frequency modes:
        T_u = sum_{|p|>alpha_tail*Nx/2 or |q|>alpha_tail*Ny/2} |u_hat|^2
              / sum_{p,q} |u_hat|^2.
    alpha_tail=2/3 (default) corresponds to the 2/3 dealiasing boundary.
    A small value indicates the solution is well resolved spectrally.
    """
    uhat = fft2(u)
    tail = (
        (np.abs(g.mode_x) > alpha_tail * u.shape[0] / 2.0)
        | (np.abs(g.mode_y) > alpha_tail * u.shape[1] / 2.0)
    )
    total = np.sum(np.abs(uhat) ** 2)
    if total == 0.0:
        return 0.0
    return float(np.sum(np.abs(uhat[tail]) ** 2) / total)


# ════════════════════════════════════════════════════════════ initial data


def gaussian_initial_data(
    g: Grid,
    amplitude: float = 1.0,
    width:     float = 4.0,
    kx0:       float = 0.0,
    ky0:       float = 0.0,
) -> Array:
    """
    Gaussian wave packet:  u_0 = amplitude * exp(-r^2/width^2) * exp(i(kx0*x+ky0*y)).
    """
    r2 = g.X**2 + g.Y**2
    return (amplitude * np.exp(-r2 / width**2)
            * np.exp(1j * (kx0 * g.X + ky0 * g.Y)))


# ════════════════════════════════════════════════════════ history and summary


def _eps_I(v0: float) -> float:
    """Regularisation: eps_I = 1e-14 * max(1, |I_0|) (from the manuscript)."""
    return 1e-14 * max(1.0, abs(v0))


def diagnostics_row(
    n: int,
    t: float,
    u: Array,
    lw: Dict[str, Array],
    g: Grid,
    p: GDSSParams,
) -> Dict:
    """
    Compute all diagnostics at a single time level.
    Includes all quantities needed for tabs and figs in sec:numerical_experiments.
    """
    M         = mass(u, g)
    Jx, Jy    = momentum(u, g)
    E_red     = energy_reduced(u, lw["Q"], g, p)
    E_lw      = energy_longwave_quadratic(u, lw, g, p)
    R_lw      = longwave_interaction_residual(lw, g, p)
    res       = longwave_residuals(lw, g, p)

    row: Dict = {
        "n":            int(n),
        "t":            float(t),
        "M":            M,
        "Jx":           Jx,
        "Jy":           Jy,
        "E_reduced":    E_red,
        "Z_lw":         zero_mode_residual(lw, g),
        "H_defect":     hermitian_defect(lw),
        "boundary_max": boundary_amplitude(u),
        "tail_ratio":   spectral_tail_ratio(u, g),
        "Q_L2":         float(np.sqrt(g.dx * g.dy * np.sum(lw["Q"] ** 2))),
        "rho_L2":       float(np.sqrt(g.dx * g.dy * np.sum(np.abs(u) ** 4))),
    }

    if E_lw is not None:
        row["E_longwave_quadratic"] = E_lw
    if R_lw is not None:
        row["R_lw"] = R_lw
    if res is not None:
        row["res_w"] = res[0]
        row["res_v"] = res[1]

    return row


def summarize_history(rows: list) -> Dict:
    """
    Compute maximum absolute and relative drifts over all rows.
    Relative drift: RE_I = |I_n - I_0| / (|I_0| + eps_I)
    with eps_I = 1e-14 * max(1, |I_0|), matching the manuscript definition.
    """
    first = rows[0]
    summary: Dict = {}

    for key in ["M", "Jx", "Jy", "E_reduced"]:
        vals = np.array([r[key] for r in rows], dtype=float)
        v0 = float(first[key])
        eps = _eps_I(v0)
        summary[f"{key}_0"]      = v0
        summary[f"AD_{key}_max"] = float(np.max(np.abs(vals - v0)))
        summary[f"RE_{key}_max"] = float(np.max(np.abs(vals - v0)) / (abs(v0) + eps))

    if "E_longwave_quadratic" in first:
        vals = np.array([r["E_longwave_quadratic"] for r in rows], dtype=float)
        v0 = float(first["E_longwave_quadratic"])
        eps = _eps_I(v0)
        summary["E_longwave_quadratic_0"]      = v0
        summary["AD_E_longwave_quadratic_max"] = float(np.max(np.abs(vals - v0)))
        summary["RE_E_longwave_quadratic_max"] = float(
            np.max(np.abs(vals - v0)) / (abs(v0) + eps)
        )

    for key in ["boundary_max", "tail_ratio", "Q_L2", "Z_lw", "H_defect"]:
        if key in first:
            vals = np.array([r[key] for r in rows if key in r], dtype=float)
            summary[f"{key}_max"] = float(np.max(vals))

    for key in ["R_lw", "res_w", "res_v"]:
        if key in first:
            summary[f"{key}_max"] = float(
                np.max([r[key] for r in rows if key in r])
            )

    return summary


# ════════════════════════════════════════════════════════ baseline runner


def run_baseline(
    p: GDSSParams,
    output_dir: str | Path = "outputs_gdss",
    save_every: int = 10,
) -> Dict:
    """
    Run a baseline simulation and save diagnostics and final fields.

    Output files:
        params.json       — copy of GDSSParams
        summary.json      — max drifts and cost metrics
        history.csv       — per-step diagnostics at every save_every steps
        final_fields.npz  — u, abs_u, rho, Q, w, v at t_final
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    g = make_grid(p)
    m = precompute_multipliers(p, g)
    u = gaussian_initial_data(g, amplitude=1.0, width=4.0, kx0=0.2, ky0=-0.1)
    Nt = int(round(p.t_final / p.dt))

    rows = []
    lw = recover_longwave(u, g, p, m, full=p.recover_full_longwave)
    rows.append(diagnostics_row(0, 0.0, u, lw, g, p))

    start = time.perf_counter()
    for n in range(1, Nt + 1):
        u, _ = strang_step(u, g, p, m)
        if n % save_every == 0 or n == Nt:
            lw = recover_longwave(u, g, p, m, full=p.recover_full_longwave)
            rows.append(diagnostics_row(n, n * p.dt, u, lw, g, p))
    cpu = time.perf_counter() - start

    summary = summarize_history(rows)
    import math as _math
    summary.update({
        "cpu_time_seconds":          cpu,
        "C_step":                    cpu / Nt,
        "C_norm":                    cpu / (Nt * p.Nx * p.Ny * _math.log(p.Nx * p.Ny)),
        "Nt":                        Nt,
        "actual_t_final":            Nt * p.dt,
        "dt":                        p.dt,
        "Nx":                        p.Nx,
        "Ny":                        p.Ny,
        "min_abs_delta_nonzero":     m.min_abs_delta_nonzero,
        "num_invalid_nonzero_modes": int(np.sum((~m.valid) & (~g.zero_mode))),
    })

    with open(outdir / "params.json", "w") as f:
        json.dump(asdict(p), f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    keys = list(rows[0].keys())
    with open(outdir / "history.csv", "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    final_lw = recover_longwave(u, g, p, m, full=True)
    np.savez_compressed(
        outdir / "final_fields.npz",
        x=g.x, y=g.y, u=u, abs_u=np.abs(u),
        rho=np.abs(u) ** 2,
        Q=final_lw["Q"], w=final_lw["w"], v=final_lw["v"],
    )

    return {"summary": summary, "rows": rows, "output_dir": str(outdir)}


# ════════════════════════════════════════════════════════════════════ entry point

if __name__ == "__main__":
    params = GDSSParams(
        Nx=128, Ny=128, Lx=40.0, Ly=40.0,
        dt=1.0e-3, t_final=0.05,
        alpha=1.0, beta=1.0, gamma=1.0, xi=1.0,
        psi=1.0, eta=1.0, phi=2.0, chi=0.5,
        theta=None,
        dealias_density=True,
        recover_full_longwave=True,
    )
    result = run_baseline(params, output_dir="outputs_gdss", save_every=10)
    print(json.dumps(result["summary"], indent=2))
