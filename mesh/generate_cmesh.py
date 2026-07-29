#!/usr/bin/env python3
"""C-topology mesh generator for the RAE 2822 airfoil (seven blocks).

Follows the ``classy_blocks`` C-mesh recipe from
``classy_examples/examples/operation/airfoil_2d.py``, adapted for the
RAE 2822 blunt trailing edge by adding a dedicated TE base block.

Topology
--------
7 blocks arranged around the airfoil:

    P_front----P_top----P_top_TE----P_top_wake       y=+H_top
      |          |         |            |
      |  BLK 1t  | BLK 2t  |  BLK 3t    |            (top row)
      |          |         |            |
      |   (arc) split_up  TE_up-------wake_up_TE     y=+gap/2
      |         (SS arc)   | TE BASE(7) |
    P_front----LE          |            |
      |         (PS arc)   | TE BASE(7) |
      |   (arc) split_low TE_low-------wake_low_TE   y=-gap/2
      |          |         |            |
      |  BLK 1b  | BLK 2b  |  BLK 3b    |            (bottom row)
      |          |         |            |
    P_front----P_bot----P_bot_TE----P_bot_wake       y=-H_top

- Blocks 1t / 1b: front region with a half-circle outer boundary of
  radius R_FRONT centered at the split point (x_split, 0). Airfoil edge
  is the LE-to-split spline for each surface.
- Blocks 2t / 2b: mid region above/below the airfoil from split to TE.
  Airfoil edge is the split-to-TE spline. Outer edge is a straight line
  at y = +/- H_top.
- Blocks 3t / 3b: rear/wake region from TE column to L_wake downstream.
  All straight edges. The inner edge (y = +/- gap/2) is the TE base line.
- Block 7 (TE base): small strip between blocks 3t and 3b, spanning the
  blunt-TE gap in the wake. Vertical extent is the TE gap (~2e-3 c).

Every block-to-block interface is a matching pair of shared corners so
cell counts propagate automatically via classy_blocks' chop system.

Usage
-----
    python generate_cmesh.py --case ../case --run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

import classy_blocks as cb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from airfoil_io import (  # noqa: E402
    cosine_resample,
    read_profile_pair,
    thicken_te,
)
from boundary_layer import TUTORIAL, first_cell_height  # noqa: E402

# ----------------------------- user parameters -----------------------------
X_SPLIT   = 0.30      # airfoil split point (fraction of chord); LE-side vs mid-side
CHORD_TE  = 0.998     # airfoil x at TE (per the RAE 2822 profile files)
R_FRONT   = 30.0      # front half-circle radius, in chords
L_WAKE    = 30.0      # wake extension length from TE, in chords
H_TOP     = 30.0      # rectangular top/bot boundary y, in chords
Z_SPAN    = 0.01      # 2D slab thickness (matches DAFoam symmetryPlanes)

N_CIRC_FRONT = 120    # cells along the direction 1 axis of blocks 1t/1b.
                      # Was 60 -- doubled to reduce the cell-size discontinuity
                      # at the split column (x=0.3) where front blocks (arc
                      # length ~47c) meet mid blocks (arc length ~0.7c).
                      # Perfect matching would need thousands of cells; 120
                      # halves the visible disparity.
N_CIRC_MID   = 80     # cells along airfoil arc in blocks 2t/2b (split to TE)
WAKE_START   = 0.01   # start_size at TE for wake x grading. Matches the
                      # last mid-block cell size (~0.009c) so the transition
                      # from mid blocks to wake blocks is smooth.
WAKE_C2C     = 1.10   # cell-to-cell expansion in wake x. Cells grow gradually.
N_TE_BASE    = 10     # cells across the TE gap inside block 7 (y direction)
C2C_BL       = 1.15   # radial expansion ratio near the wall
Y_PLUS       = 1.0    # target y+ for wall spacing
# ---------------------------------------------------------------------------


def _find_split_indices(xy: np.ndarray, x_split: float) -> int:
    """Return the index i such that xy[i, 0] is closest to x_split."""
    return int(np.argmin(np.abs(xy[:, 0] - x_split)))


def _to_3d(pt2d: np.ndarray) -> np.ndarray:
    """Append z=0 to a 2D point/array."""
    if pt2d.ndim == 1:
        return np.array([pt2d[0], pt2d[1], 0.0])
    return np.hstack([pt2d, np.zeros((len(pt2d), 1))])


def build_mesh(case_dir: str) -> cb.Mesh:
    # ---------- 1. Load and prep the airfoil ------------------------------
    ss, ps = read_profile_pair(
        os.path.join(case_dir, "profiles", "RAE2822PS.profile"),
        os.path.join(case_dir, "profiles", "RAE2822SS.profile"),
    )
    ss = cosine_resample(ss, n_pts=200)
    ps = cosine_resample(ps, n_pts=200)
    ss, ps = thicken_te(ss, ps, gap_target=2.0e-3, taper_x_start=0.9)

    # Split each surface at X_SPLIT
    i_ss = _find_split_indices(ss, X_SPLIT)
    i_ps = _find_split_indices(ps, X_SPLIT)
    ss_front = ss[: i_ss + 1]                     # LE -> split_up (incl. endpoints)
    ss_mid   = ss[i_ss:]                          # split_up -> TE_up
    ps_front = ps[: i_ps + 1]                     # LE -> split_low
    ps_mid   = ps[i_ps:]                          # split_low -> TE_low

    gap_now = float(ss[-1, 1] - ps[-1, 1])
    print(f"[geom] {len(ss)} SS + {len(ps)} PS pts (cosine-resampled), "
          f"TE gap = {gap_now:.3e} c")
    print(f"[topo] split at x={X_SPLIT}: i_ss={i_ss} (y_up={ss[i_ss,1]:.4f}), "
          f"i_ps={i_ps} (y_low={ps[i_ps,1]:.4f})")

    # ---------- 2. Corner points -----------------------------------------
    LE          = np.array([0.0, 0.0, 0.0])
    split_up    = _to_3d(ss[i_ss])
    split_low   = _to_3d(ps[i_ps])
    TE_up       = _to_3d(ss[-1])
    TE_low      = _to_3d(ps[-1])

    P_front     = np.array([X_SPLIT - R_FRONT, 0.0, 0.0])
    P_top       = np.array([X_SPLIT,           +H_TOP, 0.0])
    P_bot       = np.array([X_SPLIT,           -H_TOP, 0.0])
    P_top_TE    = np.array([CHORD_TE,          +H_TOP, 0.0])
    P_bot_TE    = np.array([CHORD_TE,          -H_TOP, 0.0])
    P_top_wake  = np.array([CHORD_TE + L_WAKE, +H_TOP, 0.0])
    P_bot_wake  = np.array([CHORD_TE + L_WAKE, -H_TOP, 0.0])
    wake_up_TE  = np.array([CHORD_TE + L_WAKE, TE_up[1],  0.0])
    wake_low_TE = np.array([CHORD_TE + L_WAKE, TE_low[1], 0.0])

    # Wall spacing from y+=1 physics
    dy1 = first_cell_height(TUTORIAL, y_plus=Y_PLUS)["dy1"]
    print(f"[BL] first cell height dy1 = {dy1:.3e} m (y+ target {Y_PLUS})")

    # Half-circle center: the split column at y=0. cb.Origin() places an
    # arc through the two endpoint vertices centered on this point.
    circle_center = np.array([X_SPLIT, 0.0, 0.0])

    # ---------- 3. Build the airfoil-arc splines -------------------------
    ss_front_curve = cb.SplineInterpolatedCurve(_to_3d(ss_front))
    ss_mid_curve   = cb.SplineInterpolatedCurve(_to_3d(ss_mid))
    ps_front_curve = cb.SplineInterpolatedCurve(_to_3d(ps_front))
    ps_mid_curve   = cb.SplineInterpolatedCurve(_to_3d(ps_mid))

    # ---------- 4. Assemble the 7 blocks ---------------------------------
    mesh = cb.Mesh()

    # Chop plan:
    #   Blocks 1t / 1b / 2t / 2b (surface-adjacent) set RADIAL chops with
    #   BL grading (start_size=dy1, c2c=C2C_BL) so cells near the airfoil
    #   are at y+=1 spacing. The `invert` flag depends on whether the block-
    #   local axis's "start" vertex is at the wall or the farfield.
    #
    #   Blocks 3t / 3b / 7 (wake) don't touch the airfoil directly, so
    #   their radial chops just inherit via propagation on shared edges
    #   (we set counts only on their non-radial direction).

    # --- BLK 1 top: front upper (half-circle to LE-to-split) ------------
    # [P_front, LE, split_up, P_top]. dir 0 = RADIAL, axis "start" at v0/v3
    # (P_front, P_top -- both farfield). Wall is at v1/v2 (LE, split_up).
    # invert=True puts dy1 at the wall side.
    face_1t = cb.Face([P_front, LE, split_up, P_top])
    face_1t.add_edge(1, cb.OnCurve(ss_front_curve, n_points=100))
    face_1t.add_edge(3, cb.Origin(circle_center))
    op_1t = cb.Extrude(face_1t, [0.0, 0.0, Z_SPAN])
    op_1t.chop(0, end_size=dy1, c2c_expansion=1.0 / C2C_BL)  # RADIAL BL
    op_1t.chop(1, count=N_CIRC_FRONT)                                 # CIRCUMF
    op_1t.chop(2, count=1)
    op_1t.set_patch("right", "wing")     # edge 1 = airfoil (LE -> split_up)
    op_1t.set_patch("left", "inout")     # edge 3 = half-circle arc
    op_1t.set_patch("bottom", "symmetry1")
    op_1t.set_patch("top", "symmetry2")
    mesh.add(op_1t)

    # --- BLK 1 bot: front lower (mirror) --------------------------------
    # [P_front, P_bot, split_low, LE]. dir 1 = RADIAL, axis "start" at
    # v0/v1 (P_front, P_bot -- both farfield). Wall at v2/v3. invert=True.
    face_1b = cb.Face([P_front, P_bot, split_low, LE])
    face_1b.add_edge(0, cb.Origin(circle_center))
    face_1b.add_edge(2, cb.OnCurve(
        cb.SplineInterpolatedCurve(_to_3d(ps_front[::-1])),
        n_points=100,
    ))
    op_1b = cb.Extrude(face_1b, [0.0, 0.0, Z_SPAN])
    op_1b.chop(0, count=N_CIRC_FRONT)                                 # CIRCUMF
    op_1b.chop(1, end_size=dy1, c2c_expansion=1.0 / C2C_BL)  # RADIAL BL
    op_1b.chop(2, count=1)
    op_1b.set_patch("front", "inout")    # edge 0 = half-circle arc
    op_1b.set_patch("back", "wing")      # edge 2 = airfoil (split_low -> LE)
    op_1b.set_patch("bottom", "symmetry1")
    op_1b.set_patch("top", "symmetry2")
    mesh.add(op_1b)

    # --- BLK 2 top: mid upper -------------------------------------------
    # [split_up, TE_up, P_top_TE, P_top]. dir 1 = RADIAL, "start" at v0/v1
    # (split_up, TE_up -- both AIRFOIL/wall). Wall at start, no invert.
    face_2t = cb.Face([split_up, TE_up, P_top_TE, P_top])
    face_2t.add_edge(0, cb.OnCurve(ss_mid_curve, n_points=100))
    op_2t = cb.Extrude(face_2t, [0.0, 0.0, Z_SPAN])
    # Circumferential grading: start small at split_up (~0.005c, close to
    # block 1t's airfoil cell size at that corner) and grow at 1.01 per
    # cell toward TE_up (ending ~0.012c, close to wake first cell 0.01c).
    # Smooths the split->mid transition AND the mid->wake transition.
    op_2t.chop(0, start_size=0.005, c2c_expansion=1.01)               # CIRCUMF graded
    op_2t.chop(1, start_size=dy1, c2c_expansion=C2C_BL)               # RADIAL BL (no invert)
    op_2t.chop(2, count=1)
    op_2t.set_patch("front", "wing")     # edge 0 = airfoil (split_up -> TE_up)
    op_2t.set_patch("back", "inout")     # edge 2 = top farfield
    op_2t.set_patch("bottom", "symmetry1")
    op_2t.set_patch("top", "symmetry2")
    mesh.add(op_2t)

    # --- BLK 2 bot: mid lower (mirror) ----------------------------------
    # [split_low, P_bot, P_bot_TE, TE_low]. dir 0 = RADIAL, "start" at
    # v0/v3 (split_low, TE_low -- both AIRFOIL/wall). No invert.
    face_2b = cb.Face([split_low, P_bot, P_bot_TE, TE_low])
    face_2b.add_edge(3, cb.OnCurve(
        cb.SplineInterpolatedCurve(_to_3d(ps_mid[::-1])),
        n_points=100,
    ))
    op_2b = cb.Extrude(face_2b, [0.0, 0.0, Z_SPAN])
    op_2b.chop(0, start_size=dy1, c2c_expansion=C2C_BL)               # RADIAL BL (no invert)
    # Circumferential grading (mirror of block 2t; see 2t comment for rationale)
    op_2b.chop(1, start_size=0.005, c2c_expansion=1.01)               # CIRCUMF graded
    op_2b.chop(2, count=1)
    op_2b.set_patch("right", "inout")    # edge 1 = bottom farfield
    op_2b.set_patch("left", "wing")      # edge 3 = airfoil (TE_low -> split_low)
    op_2b.set_patch("bottom", "symmetry1")
    op_2b.set_patch("top", "symmetry2")
    mesh.add(op_2b)

    # --- BLK 3 top: wake upper (radial count inherits from 2t) ----------
    face_3t = cb.Face([TE_up, wake_up_TE, P_top_wake, P_top_TE])
    op_3t = cb.Extrude(face_3t, [0.0, 0.0, Z_SPAN])
    op_3t.chop(0, start_size=WAKE_START, c2c_expansion=WAKE_C2C)      # WAKE-X: fine at TE, growing
    op_3t.chop(2, count=1)
    op_3t.set_patch("right", "inout")    # edge 1 = right farfield
    op_3t.set_patch("back", "inout")     # edge 2 = top farfield
    op_3t.set_patch("bottom", "symmetry1")
    op_3t.set_patch("top", "symmetry2")
    mesh.add(op_3t)

    # --- BLK 3 bot: wake lower (radial count inherits from 2b) ----------
    face_3b = cb.Face([TE_low, P_bot_TE, P_bot_wake, wake_low_TE])
    op_3b = cb.Extrude(face_3b, [0.0, 0.0, Z_SPAN])
    op_3b.chop(1, start_size=WAKE_START, c2c_expansion=WAKE_C2C)      # WAKE-X: fine at TE, growing
    op_3b.chop(2, count=1)
    op_3b.set_patch("right", "inout")    # edge 1 = bottom farfield
    op_3b.set_patch("back", "inout")     # edge 2 = right farfield
    op_3b.set_patch("bottom", "symmetry1")
    op_3b.set_patch("top", "symmetry2")
    mesh.add(op_3b)

    # --- BLK 7: TE base ---------------------------------------------------
    # First-try [TE_up, wake_up_TE, wake_low_TE, TE_low] was CW (Shoelace
    # -0.12); blockMesh rejected it as inside-out. Reversed to CCW below.
    # Vertex order [TE_up, TE_low, wake_low_TE, wake_up_TE]:
    #   edge 0 (TE_up -> TE_low, vertical)      : ACROSS-GAP (airfoil back!)
    #   edge 1 (TE_low -> wake_low_TE, horiz)   : WAKE-X
    #   edge 2 (wake_low_TE -> wake_up_TE, vert): ACROSS-GAP
    #   edge 3 (wake_up_TE -> TE_up, horiz)     : WAKE-X
    #   direction 0 : ACROSS-GAP -> chop(0) = N_TE_BASE
    #   direction 1 : WAKE-X     -> chop(1) = N_WAKE
    face_7 = cb.Face([TE_up, TE_low, wake_low_TE, wake_up_TE])
    op_7 = cb.Extrude(face_7, [0.0, 0.0, Z_SPAN])
    op_7.chop(0, count=N_TE_BASE)                                     # ACROSS-GAP
    op_7.chop(1, start_size=WAKE_START, c2c_expansion=WAKE_C2C)       # WAKE-X: fine at TE, growing
    op_7.chop(2, count=1)
    op_7.set_patch("front", "wing")      # edge 0 = airfoil back (TE_up -> TE_low)
    op_7.set_patch("back", "inout")      # edge 2 = right farfield
    op_7.set_patch("bottom", "symmetry1")
    op_7.set_patch("top", "symmetry2")
    mesh.add(op_7)

    # Modify patch TYPES (default is "patch"; symmetry needs its own type)
    mesh.modify_patch("wing", "wall")
    mesh.modify_patch("symmetry1", "symmetry")
    mesh.modify_patch("symmetry2", "symmetry")
    mesh.modify_patch("inout", "patch")

    return mesh


# --------------------------- built-in verification --------------------------
def verify_patch_geometry(dict_path: str) -> None:
    """Cheap sanity check: parse the written blockMeshDict and confirm the
    'wing' patch has vertices at radii close to the airfoil (< 2c from
    origin) and 'inout' faces are far (> 20c). Doesn't verify winding, only
    that patches map to the intended physical regions."""
    import re
    txt = open(dict_path).read()
    verts = np.array(
        re.findall(r"\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)",
                   txt.split("vertices")[1].split(";")[0]),
        dtype=float,
    )
    for patch, lo, hi in [("wing", 0.0, 2.0), ("inout", 20.0, 45.0)]:
        m = re.search(patch + r"\s*\{[^}]*faces\s*\(([^;]*)\);", txt, re.S)
        if not m:
            print(f"[QC] patch '{patch}' not defined in dict "
                  f"(no set_patch calls; using default patches)")
            continue
        ids = {int(v) for v in re.findall(r"\d+", m.group(1))}
        if not ids:
            print(f"[QC] patch '{patch}' defined but has no faces (empty)")
            continue
        r = np.linalg.norm(verts[list(ids)][:, :2], axis=1)
        ok = (r.min() >= lo) and (r.max() <= hi)
        print(f"[QC] patch '{patch}': r = {r.min():.3g} .. {r.max():.3g} "
              f"-> {'OK' if ok else 'CHECK MANUALLY'}")
    # Also list what patches WERE defined, since we removed set_patch calls
    patch_names = re.findall(r"^\s*(\w+)\s*\{\s*type\s+", txt, re.M)
    if patch_names:
        print(f"[QC] blockMeshDict patches: {patch_names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="../case", help="OpenFOAM case directory")
    ap.add_argument("--run", action="store_true", help="run blockMesh + checkMesh after writing")
    args = ap.parse_args()

    mesh = build_mesh(args.case)
    out = os.path.join(args.case, "system", "blockMeshDict")
    mesh.write(out, debug_path=None)
    print(f"[OK] wrote {out}")
    verify_patch_geometry(out)

    if args.run:
        subprocess.run(["blockMesh", "-case", args.case], check=True)
        from mesh_quality import check_mesh
        report = check_mesh(args.case)
        print(report)
        if not report.passed:
            raise SystemExit("Mesh rejected by quality gate (Step 1.5).")


if __name__ == "__main__":
    main()
