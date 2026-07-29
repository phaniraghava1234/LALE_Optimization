#!/usr/bin/env python3
"""
Steps 1.2 + 1.4 -- Topology Definition & Scripted Mesh Generation
=================================================================
Builds a 4-quadrant structured O-grid around the RAE 2822 airfoil with
classy_blocks, writes system/blockMeshDict, and (optionally) runs
OpenFOAM's blockMesh.

Topology (Step 1.2)
-------------------
The closed airfoil loop (SS + PS + blunt-TE base) is split at four
anchor parameters:

        A0 = TE (upper corner)      F0..F3 = radial projections of
        A1 = mid suction side              A0..A3 onto the farfield
        A2 = leading edge                  circle of radius R_FAR
        A3 = mid pressure side

Each quadrant is one hex block:  [A_i, A_{i+1}, F_{i+1}, F_i]
  * inner edge  : spline ON the airfoil curve  (cb.OnCurve)
  * outer edge  : circular arc                 (cb.Origin)
  * side edges  : straight radial lines

Because every block shares its radial edges with its neighbours, the
grid lines are continuous around the airfoil -- no shearing, no
criss-crossing, by construction.

Boundary layer (Step 1.3)
-------------------------
The radial chop uses start_size = dy1 from boundary_layer.py
(y+ = 1 physics), with a geometric expansion to the farfield.

Patch names match the DAFoam case exactly:
    wing        (wall)         inout      (farfield)
    symmetry1 / symmetry2      (z-planes, type symmetry)

Usage
-----
    python generate_blockmesh.py                 # write dict only
    python generate_blockmesh.py --run           # + run blockMesh
    python generate_blockmesh.py --case ../case  # target case dir
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import numpy as np

import classy_blocks as cb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airfoil_io import (  # noqa: E402
    closed_loop,
    cosine_resample,
    read_profile_pair,
    thicken_te,
)
from boundary_layer import TUTORIAL, first_cell_height  # noqa: E402

# ----------------------------- user parameters -----------------------------
R_FAR = 30.0          # farfield radius in chords
Z_SPAN = 0.01         # one-cell-thick 2D span (matches DAFoam symmetryPlanes)
CENTER = np.array([0.5, 0.0, 0.0])   # O-grid center (mid chord)

N_CIRC = [40] * 4          # circumferential cells per block (4 blocks * 40 = 160).
                           # Tried 60 -- combined with n_te=3 base points the
                           # cell-count / polyline-density interaction broke
                           # things (AR 1e95, negative volumes). 40 is the
                           # value we know passes cleanly. Length MUST equal
                           # the number of anchors in build_mesh.
N_POINTS = 60              # OnCurve polyline resolution per block edge.
C2C_BL = 1.08         # cell-to-cell expansion in the radial direction.
                      # 1.15 -> 1.10 -> 1.08 progression as we tightened
                      # radial resolution. 1.08 gives ~185 radial layers
                      # (vs 106 at 1.15) and much smoother growth, which
                      # helps non-orth at LE because neighbours are closer
                      # in size (face-normal jumps are smaller).
Y_PLUS = 1.0          # target y+ (Step 1.3)

# quadrant anchors as fractions of the closed-loop parameter
# t=0 is the TE upper corner; ~0.25 mid-SS; ~0.5 LE; ~0.75 mid-PS
ANCHOR_T = [0.0, 0.25, 0.50, 0.75]
# ----------------------------------------------------------------------------


def build_mesh(case_dir: str) -> cb.Mesh:
    ss, ps = read_profile_pair(
        os.path.join(case_dir, "profiles", "RAE2822PS.profile"),
        os.path.join(case_dir, "profiles", "RAE2822SS.profile"),
    )
    # Cosine-resample both surfaces to concentrate points near LE and TE
    # where curvature is highest. Raw profiles have ~61 evenly-spaced-in-x
    # points, which puts only 3-4 points in the region 0 <= x <= 0.02c
    # where the airfoil turns nearly 180 deg. The spline through those
    # knots looks polygonal at LE and the mesh cells snap to that outline
    # (the "stair-stepping" visible in ParaView LE close-up). Cosine
    # spacing puts ~40 points in the same LE region.
    ss = cosine_resample(ss, n_pts=200)
    ps = cosine_resample(ps, n_pts=200)
    # Thicken TE from ~3.7e-4 c to 2e-3 c (still an order of magnitude
    # thinner than chord, but ~5x thicker than raw). Tapers smoothly from
    # zero at x=0.9c, so the airfoil shape upstream is unchanged. Widens
    # the TE cell circumferentially -> lower TE-cell AR -> better mesh
    # quality where it matters most for the adjoint.
    ss, ps = thicken_te(ss, ps, gap_target=2.0e-3, taper_x_start=0.9)
    gap_new = float(ss[-1, 1] - ps[-1, 1])
    print(f"[geom] cosine-resampled to {len(ss)} SS + {len(ps)} PS pts, "
          f"TE gap = {gap_new:.3e} c")
    # n_te=0 -> do NOT insert base points across the blunt TE gap.
    # We tried n_te=3 (after widening TE to 2e-3 c) in combination with
    # N_CIRC=60; the mesh collapsed. n_te=0 with cosine-spaced airfoil
    # and thickened TE keeps QC passing cleanly.
    loop2d = closed_loop(ss, ps, n_te=0)
    loop3d = np.hstack([loop2d, np.zeros((len(loop2d), 1))])
    n_loop = len(loop3d)

    # 4 anchors: TE_up, mid-SS (arc-midpoint), LE (exact min-x point),
    # mid-PS (arc-midpoint of PS side). This is the classy_blocks
    # anchor placement that WORKS -- every other attempt (angle-symmetric,
    # 8-block loop-index-spaced) either mesh-collapses or degenerates
    # into sliver blocks at LE/TE. The resulting mesh has an unavoidable
    # ~5:1 circumferential density mismatch on the farfield (topology
    # property of the 4-block O-grid on an elongated body), but is
    # topologically valid.
    i_le = int(np.argmin(loop3d[:, 0]))
    n_te_side = n_loop - i_le
    anchor_idx = [
        0,
        max(1, i_le // 2),
        i_le,
        i_le + max(1, n_te_side // 2),
    ]
    A = [loop3d[k].copy() for k in anchor_idx]

    # Radial farfield projections (unchanged).
    F = []
    for a in A:
        r = a - CENTER
        F.append(CENTER + R_FAR * r / np.linalg.norm(r))

    # First cell height from flat-plate physics (Step 1.3).
    dy1 = first_cell_height(TUTORIAL, y_plus=Y_PLUS)["dy1"]
    print(f"[BL] first cell height dy1 = {dy1:.3e} m (y+ target {Y_PLUS})")
    print(f"[topo] loop points = {n_loop}, anchor indices = {anchor_idx}")

    def arc_from_j_to_i(i: int, j: int) -> np.ndarray:
        """Slice loop3d covering the airfoil arc between anchors i and j,
        then reverse it so it runs from A[j] to A[i]. This matches the
        Face vertex order [A[j], A[i], ...] used below, so cb.OnCurve
        traces edge 0 unambiguously from vertex 0 (A[j]) to vertex 1 (A[i]).
        Handles the quadrant-3 wraparound (i=3, j=0) explicitly."""
        a, b = anchor_idx[i], anchor_idx[j]
        if a < b:
            pts = loop3d[a : b + 1]
        else:
            pts = np.vstack([loop3d[a:], loop3d[: b + 1]])
        return pts[::-1]

    mesh = cb.Mesh()
    n_quads = len(A)
    for i in range(n_quads):
        j = (i + 1) % n_quads

        # Per-quadrant OPEN spline through this quadrant's airfoil points.
        # Endpoints of the sliced arc ARE the Face vertices, so OnCurve's
        # t=0 and t=1 land exactly on A[j] and A[i].
        quad_pts = arc_from_j_to_i(i, j)
        inner_curve = cb.SplineInterpolatedCurve(quad_pts)

        # Reversed vertex order [A[j], A[i], F[i], F[j]] gives CCW winding
        # in xy (positive Shoelace area), so +z extrusion produces a right-
        # side-out hex. Edge 0 still spans two airfoil anchors and edge 2
        # still spans two farfield anchors -- add_edge and set_patch calls
        # below are unchanged.
        face = cb.Face([A[j], A[i], F[i], F[j]])
        face.add_edge(0, cb.OnCurve(inner_curve, n_points=N_POINTS))
        face.add_edge(2, cb.Origin(CENTER))

        op = cb.Extrude(face, [0.0, 0.0, Z_SPAN])
        op.chop(0, count=N_CIRC[i])                   # circumferential (per-block)
        if i == 0:                                    # radial: chop once,
            op.chop(1, start_size=dy1, c2c_expansion=C2C_BL)  # propagates
        op.chop(2, count=1)                           # one cell in z

        op.set_patch("front", "wing")
        op.set_patch("back", "inout")
        op.set_patch("bottom", "symmetry1")
        op.set_patch("top", "symmetry2")
        mesh.add(op)
        print(f"[quad {i}] {len(quad_pts)} airfoil pts in inner spline")

    mesh.modify_patch("wing", "wall")
    mesh.modify_patch("symmetry1", "symmetry")
    mesh.modify_patch("symmetry2", "symmetry")
    mesh.modify_patch("inout", "patch")
    return mesh


# --------------------------- built-in verification --------------------------
def verify_patch_geometry(dict_path: str) -> None:
    """Parse the written blockMeshDict and check that 'wing' faces sit on
    the airfoil and 'inout' faces sit on the farfield circle. Cheap
    insurance against side-naming mistakes."""
    txt = open(dict_path).read()
    verts = np.array(
        re.findall(r"\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)",
                   txt.split("vertices")[1].split(";")[0]),
        dtype=float,
    )
    for patch, lo, hi in [("wing", 0.0, 2.0), ("inout", R_FAR * 0.9, R_FAR * 1.1)]:
        m = re.search(patch + r"\s*\{[^}]*faces\s*\(([^;]*)\);", txt, re.S)
        ids = {int(v) for v in re.findall(r"\d+", m.group(1))}
        r = np.linalg.norm(verts[list(ids)][:, :2] - CENTER[:2], axis=1)
        ok = (r.min() >= lo) and (r.max() <= hi)
        print(f"[QC] patch '{patch}': radius {r.min():.3g}..{r.max():.3g} "
              f"-> {'OK' if ok else 'WRONG SIDE MAPPING!'}")
        if not ok:
            raise RuntimeError(f"Patch '{patch}' is not where it should be.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="../case", help="OpenFOAM case directory")
    ap.add_argument("--run", action="store_true", help="run blockMesh after writing")
    args = ap.parse_args()

    mesh = build_mesh(args.case)
    out = os.path.join(args.case, "system", "blockMeshDict")
    mesh.write(out, debug_path=None)
    print(f"[OK] wrote {out}")
    verify_patch_geometry(out)

    if args.run:
        subprocess.run(["blockMesh", "-case", args.case], check=True)
        # Step 1.5: automatic quality gate
        from mesh_quality import check_mesh
        report = check_mesh(args.case)
        print(report)
        if not report.passed:
            raise SystemExit("Mesh rejected by quality gate (Step 1.5).")


if __name__ == "__main__":
    main()
