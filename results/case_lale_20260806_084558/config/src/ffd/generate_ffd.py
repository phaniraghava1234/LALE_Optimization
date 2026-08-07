#!/usr/bin/env python3
"""Free-Form Deformation (FFD) control-lattice generator.

Writes the Plot3D ASCII ``.xyz`` control lattice that pyGeo wraps around
the airfoil. The optimizer's shape design variables are the
y-displacements of these control points.

FFD is the trivariate Bernstein/B-spline mapping::

    X(u, v, w) = sum_i sum_j sum_k  B_i(u) B_j(v) B_k(w) * P_ijk

Every mesh surface node is embedded once at its parametric coordinates
(u, v, w). When a control point P_ijk moves, the embedded geometry
morphs smoothly and with analytic derivatives dX/dP - exactly what the
adjoint chain rule needs.

Lattice conventions:

    * NX chordwise stations x NY = 2 (below/above) x NZ = 2 (spanwise pair)
    * The bounding box fits the airfoil with a small margin.
    * It must fully contain every 'wing' surface point or pyGeo aborts.

Writes ``../case/FFD/wingFFD.xyz``, the same file the DAFoam model reads.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "mesh"))
from airfoil_io import read_profile_pair  # noqa: E402

NX = 30          # chordwise control points (=> 2*(NX-2)+2 shape DOFs, see model)
NY = 2           # one row below, one above the airfoil
NZ = 2           # z = 0 and z = Z_SPAN planes (2D case)
Z_SPAN = 0.01
MARGIN_X = 0.005  # chordwise margin beyond LE/TE
MARGIN_Y = 0.02   # vertical margin beyond min/max thickness


def build_lattice(xy_ss: np.ndarray, xy_ps: np.ndarray) -> np.ndarray:
    x_min = min(xy_ss[:, 0].min(), xy_ps[:, 0].min()) - MARGIN_X
    x_max = max(xy_ss[:, 0].max(), xy_ps[:, 0].max()) + MARGIN_X
    y_min = xy_ps[:, 1].min() - MARGIN_Y
    y_max = xy_ss[:, 1].max() + MARGIN_Y

    x = np.linspace(x_min, x_max, NX)
    y = np.array([y_min, y_max])
    z = np.array([0.0, Z_SPAN])

    P = np.zeros((NX, NY, NZ, 3))
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                P[i, j, k] = [x[i], y[j], z[k]]
    return P


def write_plot3d(path: str, P: np.ndarray) -> None:
    nx, ny, nz, _ = P.shape
    with open(path, "w") as f:
        f.write("1\n")
        f.write(f"{nx} {ny} {nz}\n")
        for dim in range(3):
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{P[i, j, k, dim]:.10f} ")
            f.write("\n")


STEMS = {"tutorial": "RAE2822", "benchmark": "RAE2822", "lale": "MH139F"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="../case")
    ap.add_argument("--preset", default="tutorial", choices=list(STEMS),
                    help="Chooses which airfoil profile files to wrap.")
    args = ap.parse_args()

    stem = STEMS[args.preset]
    ss, ps = read_profile_pair(
        os.path.join(args.case, "profiles", f"{stem}_PS.profile"),
        os.path.join(args.case, "profiles", f"{stem}_SS.profile"),
    )
    P = build_lattice(ss, ps)
    out = os.path.join(args.case, "FFD", "wingFFD.xyz")
    write_plot3d(out, P)

    # containment check: every airfoil point strictly inside the box
    pts = np.vstack([ss, ps])
    ok = (
        (pts[:, 0] > P[..., 0].min()).all()
        and (pts[:, 0] < P[..., 0].max()).all()
        and (pts[:, 1] > P[..., 1].min()).all()
        and (pts[:, 1] < P[..., 1].max()).all()
    )
    print(f"[OK] wrote {out}  ({NX}x{NY}x{NZ} lattice)")
    print(f"[QC] airfoil fully inside FFD box: {'YES' if ok else 'NO -- FIX MARGINS'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
