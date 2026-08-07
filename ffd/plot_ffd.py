#!/usr/bin/env python3
"""
Visualize an FFD lattice (Plot3D .xyz) overlaid on the airfoil surface.

Reads any Plot3D-format FFD file and produces a PNG showing:
  * airfoil surface (SS = blue, PS = red)
  * lattice grid lines (thin black)
  * control points colored by row (bottom = red circles, top = blue)
  * LE / TE columns highlighted (the coupled DVs)

Usage:
    python ffd/plot_ffd.py                            # defaults
    python ffd/plot_ffd.py --ffd case/FFD/other.xyz   # different file
    python ffd/plot_ffd.py --show                     # also open the plot
    python ffd/plot_ffd.py --out my_plot.png          # custom output name
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def read_plot3d(path: str) -> np.ndarray:
    """Read a single-block Plot3D ASCII .xyz file.

    Returns an (nx, ny, nz, 3) array of control point positions.
    Assumes the writer wrote:
        line 1: "1"                (n_blocks)
        line 2: "nx ny nz"
        line 3: all x values (nx*ny*nz numbers, whitespace separated)
        line 4: all y values (nx*ny*nz numbers)
        line 5: all z values (nx*ny*nz numbers)
    with per-dimension loop order  k (outer) -> j -> i (inner).
    """
    with open(path) as f:
        n_blocks = int(f.readline().strip())
        if n_blocks != 1:
            raise ValueError(
                f"expected single-block Plot3D, got n_blocks={n_blocks}"
            )
        nx, ny, nz = (int(v) for v in f.readline().split())
        rows = []
        for _ in range(3):
            rows.append([float(v) for v in f.readline().split()])

    P = np.zeros((nx, ny, nz, 3))
    for dim, row in enumerate(rows):
        arr = np.asarray(row).reshape(nz, ny, nx)  # loop order k,j,i
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    P[i, j, k, dim] = arr[k, j, i]
    return P


def plot_ffd(
    P: np.ndarray,
    xy_ss: np.ndarray,
    xy_ps: np.ndarray,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """Draw the airfoil + the FFD box (z=0 slice) with control points."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny, nz, _ = P.shape
    k = 0  # z=0 slice; z=Z_SPAN copy is identical in 2D FFDs

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Airfoil surfaces
    ax.plot(xy_ss[:, 0], xy_ss[:, 1], "b-", lw=1.4, label="SS (upper)")
    ax.plot(xy_ps[:, 0], xy_ps[:, 1], "r-", lw=1.4, label="PS (lower)")

    # Lattice grid lines: vertical (each chordwise station) and horizontal (each y-row)
    for i in range(nx):
        x_col = [P[i, j, k, 0] for j in range(ny)]
        y_col = [P[i, j, k, 1] for j in range(ny)]
        ax.plot(x_col, y_col, "-", color="0.4", lw=0.6, alpha=0.7, zorder=1)
    for j in range(ny):
        x_row = [P[i, j, k, 0] for i in range(nx)]
        y_row = [P[i, j, k, 1] for i in range(nx)]
        ax.plot(x_row, y_row, "-", color="0.4", lw=0.6, alpha=0.7, zorder=1)

    # Control points (open circles), colored by row
    lower = np.array([P[i, 0, k] for i in range(nx)])
    upper = np.array([P[i, ny - 1, k] for i in range(nx)])
    ax.plot(
        lower[:, 0], lower[:, 1], "o",
        mec="crimson", mfc="white", mew=1.6, ms=9,
        label=f"FFD lower row (j=0), {nx} pts", zorder=3,
    )
    ax.plot(
        upper[:, 0], upper[:, 1], "o",
        mec="royalblue", mfc="white", mew=1.6, ms=9,
        label=f"FFD upper row (j={ny-1}), {nx} pts", zorder=3,
    )

    # LE/TE column dashed connectors (the two "coupled" DVs -- upper and lower
    # move oppositely to preserve the LE / TE camber line).
    ax.plot(
        [P[0, 0, k, 0], P[0, ny - 1, k, 0]],
        [P[0, 0, k, 1], P[0, ny - 1, k, 1]],
        "--", color="green", lw=2.0, zorder=2, label="LE column (coupled)",
    )
    ax.plot(
        [P[nx - 1, 0, k, 0], P[nx - 1, ny - 1, k, 0]],
        [P[nx - 1, 0, k, 1], P[nx - 1, ny - 1, k, 1]],
        "--", color="darkorange", lw=2.0, zorder=2, label="TE column (coupled)",
    )

    # Cosmetics
    ax.set_aspect("equal")
    box_x_pad = 0.05
    box_y_pad = 0.02
    ax.set_xlim(P[..., 0].min() - box_x_pad, P[..., 0].max() + box_x_pad)
    ax.set_ylim(P[..., 1].min() - box_y_pad, P[..., 1].max() + box_y_pad)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    total = nx * ny * nz
    ax.set_title(
        f"FFD lattice: {nx}×{ny}×{nz} = {total} control points  "
        f"(z=0 slice; z={P[0,0,-1,2]:.3g} slice is identical in 2D)"
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] wrote {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an FFD lattice")
    parser.add_argument(
        "--ffd", default="case/FFD/wingFFD.xyz",
        help="path to Plot3D .xyz FFD file (default: %(default)s)",
    )
    parser.add_argument(
        "--profile-dir", default="case/profiles",
        help="directory containing the two .profile files",
    )
    parser.add_argument(
        "--out", default="ffd_plot.png",
        help="output PNG filename, saved next to this script (default: %(default)s)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="open the plot interactively (implies save)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root so the script works from any CWD
    project_root = Path(__file__).resolve().parent.parent
    ffd_path = Path(args.ffd)
    if not ffd_path.is_absolute():
        ffd_path = project_root / ffd_path
    if not ffd_path.is_file():
        raise SystemExit(f"[plot_ffd] FFD file not found: {ffd_path}")

    profile_dir = Path(args.profile_dir)
    if not profile_dir.is_absolute():
        profile_dir = project_root / profile_dir
    # Auto-detect the airfoil stem by glob (e.g., RAE2822 or MH139F).
    ss_candidates = sorted(profile_dir.glob("*_SS.profile"))
    if not ss_candidates:
        raise SystemExit(f"[plot_ffd] no *_SS.profile found in {profile_dir}")
    stem = ss_candidates[0].name[:-len("_SS.profile")]
    print(f"[plot_ffd] using profile stem: {stem}")
    ps_path = profile_dir / f"{stem}_PS.profile"
    ss_path = profile_dir / f"{stem}_SS.profile"
    for p in (ps_path, ss_path):
        if not p.is_file():
            raise SystemExit(f"[plot_ffd] profile file not found: {p}")

    # Load
    P = read_plot3d(str(ffd_path))

    sys.path.insert(0, str(project_root / "mesh"))
    from airfoil_io import read_profile_pair  # noqa: E402
    ss, ps = read_profile_pair(str(ps_path), str(ss_path))

    out_path = Path(__file__).resolve().parent / args.out
    plot_ffd(P, ss, ps, save_path=str(out_path), show=args.show)


if __name__ == "__main__":
    main()
