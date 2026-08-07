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
from boundary_layer import (  # noqa: E402
    TUTORIAL, BENCHMARK, LALE_500M, first_cell_height,
)

# Preset registry — pairs an operating point with a set of profile files
# and per-preset mesh-density tuning.
#
# n_circ_front: cells along the front C-block outer arc (~47 chord). The
#   inner (airfoil-side) cell size at the split column x=0.30 is
#   0.30 / n_circ_front. Match this to N_CIRC_MID (=0.70/N_CIRC_MID) for
#   a smooth transition at the split. Transonic cases historically used
#   120 for LE resolution; the LALE case at low Re can use 40 without
#   losing physics.
# Per-preset configuration. Every mesh-shape and grading knob is here so
# regenerating with `--preset tutorial|benchmark` reproduces the historical
# transonic mesh, and `--preset lale` produces the tuned LALE mesh.
#
# Keys (all required per preset):
#   stem            : profile file basename (case_dir/profiles/<stem>_{PS,SS}.profile)
#   flow            : Freestream from boundary_layer.py, drives y+ -> dy1
#   n_circ_front    : cells along the airfoil arc in blocks 1t/1b (LE region)
#   y_plus          : target y+ at the wall (drives first-cell height)
#   n_profile_pts   : cosine-resample count for the airfoil profile spline
#   R_FRONT         : front half-circle radius, in chords
#   L_WAKE          : wake extension length aft of TE, in chords
#   H_TOP           : rectangular top/bot boundary y, in chords
#                     (MUST equal R_FRONT: the semicircle at (X_SPLIT, 0)
#                      with radius R_FRONT must reach P_top=(X_SPLIT, +H_TOP))
#   WAKE_START      : first wake cell size at TE (start_size in wake blocks)
#   WAKE_C2C        : cell-to-cell expansion in wake x
#   N_TE_BASE       : cells across the TE gap in block 7 (y direction)
#   C2C_BL          : radial expansion ratio in the BL grading
#   mid_chord_mode  : block 2 (mid, airfoil-to-TE) chord chop mode:
#                     "c2c_101"    -> start=0.005, c2c=1.01     (transonic default)
#                     "match_wake" -> start=0.005, end=WAKE_START (LALE smoothing)
_TRANSONIC_COMMON = dict(
    R_FRONT=30.0, L_WAKE=30.0, H_TOP=30.0,
    WAKE_START=0.01, WAKE_C2C=1.10, N_TE_BASE=10,
    C2C_BL=1.15, mid_chord_mode="c2c_101",
    n_circ_front=120, y_plus=1.0, n_profile_pts=200,
)
PRESETS = {
    "tutorial":  {"stem": "RAE2822", "flow": TUTORIAL,  **_TRANSONIC_COMMON},
    "benchmark": {"stem": "RAE2822", "flow": BENCHMARK, **_TRANSONIC_COMMON},
    # LALE at Re=1.66e5, incompressible DASimpleFoam with wall functions.
    # Smaller domain (10c) since low-Re incompressible needs less far-field.
    # Coarser LE cluster (n_profile_pts=100, n_circ_front=60) prevents the
    # 5e-10 min-volume LE cells that caused SA production runaway.
    # y+=30 target (dy1 ~ 1.1e-3) matches nutUSpaldingWallFunction on the wall.
    # Wake tuning (start=0.02, c2c=1.05) keeps GAMG well-conditioned.
    # Block 2 chord in "match_wake" mode -> smooth cell transition at TE.
    "lale": {
        "stem": "MH139F", "flow": LALE_500M,
        "R_FRONT": 10.0, "L_WAKE": 10.0, "H_TOP": 10.0,
        "WAKE_START": 0.02, "WAKE_C2C": 1.05, "N_TE_BASE": 3,
        "C2C_BL": 1.12, "mid_chord_mode": "match_wake",
        "n_circ_front": 60, "y_plus": 30.0, "n_profile_pts": 100,
    },
}

# ----------------------------- topology constants --------------------------
# These are geometric invariants shared by all presets. If you need to change
# them, do so per-preset (they'd then move into PRESETS above).
X_SPLIT   = 0.30      # airfoil split point (fraction of chord); LE-side vs mid-side
CHORD_TE  = 0.998     # airfoil x at TE (per the RAE 2822 profile files)
Z_SPAN    = 0.01      # 2D slab thickness (matches DAFoam symmetryPlanes)
N_CIRC_MID = 80       # cells along airfoil arc in blocks 2t/2b (split to TE)
                      # (retained for compatibility; not currently referenced)
# ---------------------------------------------------------------------------


def _find_split_indices(xy: np.ndarray, x_split: float) -> int:
    """Return the index i such that xy[i, 0] is closest to x_split."""
    return int(np.argmin(np.abs(xy[:, 0] - x_split)))


def _to_3d(pt2d: np.ndarray) -> np.ndarray:
    """Append z=0 to a 2D point/array."""
    if pt2d.ndim == 1:
        return np.array([pt2d[0], pt2d[1], 0.0])
    return np.hstack([pt2d, np.zeros((len(pt2d), 1))])


def build_mesh(case_dir: str, preset: str = "tutorial") -> cb.Mesh:
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset!r}; valid: {list(PRESETS)}")
    cfg = PRESETS[preset]
    # Unpack per-preset mesh parameters into local names so the block-build
    # code below reads naturally. All shape/grading knobs come from `cfg`
    # -- there are no more mesh-shape globals.
    stem            = cfg["stem"]
    n_circ_front    = cfg["n_circ_front"]
    n_profile_pts   = cfg["n_profile_pts"]
    R_FRONT         = cfg["R_FRONT"]
    L_WAKE          = cfg["L_WAKE"]
    H_TOP           = cfg["H_TOP"]
    WAKE_START      = cfg["WAKE_START"]
    WAKE_C2C        = cfg["WAKE_C2C"]
    N_TE_BASE       = cfg["N_TE_BASE"]
    C2C_BL          = cfg["C2C_BL"]
    mid_chord_mode  = cfg["mid_chord_mode"]

    # Geometric consistency: the front semicircle at (X_SPLIT, 0) with
    # radius R_FRONT must pass through the corner points P_top and P_bot
    # sitting at (X_SPLIT, +/-H_TOP), so R_FRONT must equal H_TOP.
    assert abs(R_FRONT - H_TOP) < 1e-9, (
        f"[preset {preset!r}] R_FRONT ({R_FRONT}) must equal H_TOP "
        f"({H_TOP}) for the front semicircle to close on the "
        "rectangular block corners."
    )
    # ---------- 1. Load and prep the airfoil ------------------------------
    ss, ps = read_profile_pair(
        os.path.join(case_dir, "profiles", f"{stem}_PS.profile"),
        os.path.join(case_dir, "profiles", f"{stem}_SS.profile"),
    )
    ss = cosine_resample(ss, n_pts=n_profile_pts)
    ps = cosine_resample(ps, n_pts=n_profile_pts)
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

    # Wall spacing from y+ target physics for the selected preset
    fs = PRESETS[preset]["flow"]
    y_plus_target = PRESETS[preset]["y_plus"]
    dy1 = first_cell_height(fs, y_plus=y_plus_target)["dy1"]
    print(f"[BL] preset {preset!r}: Re = {fs.Re:.3e}, "
          f"first cell height dy1 = {dy1:.3e} m (y+ target {y_plus_target})")

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
    op_1t.chop(1, count=n_circ_front)                                 # CIRCUMF
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
    op_1b.chop(0, count=n_circ_front)                                 # CIRCUMF
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
    # Circumferential grading, chosen by preset's mid_chord_mode:
    #  c2c_101    : historical transonic default, start=0.005, c2c=1.01
    #               (block 2 last cell ends up ~0.012c, close to but not
    #                exactly matching WAKE_START)
    #  match_wake : LALE smoothing, start=0.005, end=WAKE_START
    #               (guarantees smooth transition at TE column into block 3)
    if mid_chord_mode == "c2c_101":
        op_2t.chop(0, start_size=0.005, c2c_expansion=1.01)
    elif mid_chord_mode == "match_wake":
        op_2t.chop(0, start_size=0.005, end_size=WAKE_START)
    else:
        raise ValueError(f"Unknown mid_chord_mode {mid_chord_mode!r}")
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
    if mid_chord_mode == "c2c_101":
        op_2b.chop(1, start_size=0.005, c2c_expansion=1.01)
    else:  # "match_wake" (only other valid option; already validated above)
        op_2b.chop(1, start_size=0.005, end_size=WAKE_START)
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
    ap.add_argument("--preset", default="tutorial", choices=list(PRESETS),
                    help="Operating-point preset (chooses profile stem and Re "
                         "for wall spacing).")
    ap.add_argument("--run", action="store_true",
                    help="run blockMesh + checkMesh after writing")
    args = ap.parse_args()

    mesh = build_mesh(args.case, preset=args.preset)
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
            raise SystemExit("Mesh rejected by quality gate.")


if __name__ == "__main__":
    main()
