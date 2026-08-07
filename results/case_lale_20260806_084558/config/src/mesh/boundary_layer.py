#!/usr/bin/env python3
"""Physics-based first-cell-height sizing for wall-resolved RANS meshes.

Computes the first cell height Delta_y1 for a target y+ using
flat-plate turbulent skin-friction correlations:

    Re_c   = rho * U_inf * c / mu
    C_f    = 0.026 / Re_c^(1/7)          (Schlichting power-law fit)
    tau_w  = 0.5 * C_f * rho * U_inf^2
    u_tau  = sqrt(tau_w / rho)
    dy1    = y_plus * mu / (rho * u_tau)

For the Spalart-Allmaras turbulence model resolved to the wall we target
y+ approximately 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

R_AIR = 287.058   # J/(kg K)
GAMMA = 1.4


@dataclass
class Freestream:
    """Freestream state. Give (M, T, p) or (U, T, p)."""
    T: float                 # K
    p: float                 # Pa
    U: float | None = None   # m/s
    M: float | None = None   # -
    mu: float = 1.8e-5       # Pa s (matches constant/thermophysicalProperties)
    chord: float = 1.0       # m

    def __post_init__(self):
        self.a = math.sqrt(GAMMA * R_AIR * self.T)
        if self.U is None:
            self.U = self.M * self.a
        if self.M is None:
            self.M = self.U / self.a
        self.rho = self.p / (R_AIR * self.T)
        self.Re = self.rho * self.U * self.chord / self.mu


def first_cell_height(fs: Freestream, y_plus: float = 1.0) -> dict:
    """Return dy1 and the intermediate quantities (for the README table)."""
    cf = 0.026 / fs.Re ** (1.0 / 7.0)
    tau_w = 0.5 * cf * fs.rho * fs.U ** 2
    u_tau = math.sqrt(tau_w / fs.rho)
    dy1 = y_plus * fs.mu / (fs.rho * u_tau)
    return {
        "Re_c": fs.Re,
        "Cf": cf,
        "tau_w": tau_w,
        "u_tau": u_tau,
        "dy1": dy1,
        "y_plus_target": y_plus,
    }


def bl_layer_count(dy1: float, total_height: float, ratio: float = 1.2) -> int:
    """Number of geometrically-growing layers needed to fill `total_height`.

    Solves  dy1 * (r^n - 1)/(r - 1) >= H  for n.
    """
    return math.ceil(
        math.log(1.0 + total_height * (ratio - 1.0) / dy1) / math.log(ratio)
    )


# --- Operating points used in this project ---------------------------------
# A) DAFoam official tutorial baseline (validated numbers).
TUTORIAL = Freestream(T=300.0, p=101325.0, U=248.0)

# B) He et al. (2019) / AIAA ADODG Case 2 benchmark: M = 0.729, Re = 6.5e6.
#    p is chosen so that Re_c = 6.5e6 at chord 1 m:
#    rho = Re * mu / (U * c);  p = rho * R * T
_M, _T = 0.729, 288.15
_U = _M * math.sqrt(GAMMA * R_AIR * _T)
_rho = 6.5e6 * 1.8e-5 / _U
BENCHMARK = Freestream(T=_T, p=_rho * R_AIR * _T, M=_M)

# C) LALE point from Oettershagen et al. (2017), AtlantikSolar UAV.
#    Physical setup: c = 0.305 m, U = 8.30 m/s, altitude 500 m ISA.
#    ISA at 500 m: T = 284.90 K, p = 95461 Pa, mu = 1.7737e-5 Pa s.
#    Result: Re = 1.6663e5, M = 0.0245.
LALE_500M = Freestream(T=284.90, p=95461.0, U=8.30, mu=1.7737e-5, chord=0.305)


if __name__ == "__main__":
    cases = [
        ("Tutorial (U=248, T=300, p=101325, c=1 m)", TUTORIAL),
        ("Benchmark (M=0.729, Re=6.5e6, c=1 m)", BENCHMARK),
        ("LALE 500 m (U=8.30, c=0.305 m)", LALE_500M),
    ]
    for name, fs in cases:
        r = first_cell_height(fs, y_plus=1.0)
        print(f"\n{name}")
        print(f"  M = {fs.M:.4f}, U = {fs.U:.3f} m/s, rho = {fs.rho:.4f} kg/m3, "
              f"p = {fs.p:.0f} Pa, chord = {fs.chord} m")
        print(f"  Re_c   = {r['Re_c']:.4g}")
        print(f"  Cf     = {r['Cf']:.5g}")
        print(f"  u_tau  = {r['u_tau']:.4g} m/s")
        print(f"  dy1    = {r['dy1']:.4g} m  (for y+ = 1)")
        print(f"  layers to fill 0.02c at r=1.2: "
              f"{bl_layer_count(r['dy1'], 0.02 * fs.chord)}")
