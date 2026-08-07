#!/usr/bin/env python3
"""Mesh figures for any archived case -- overview panels and a deformed
baseline-vs-optimized comparison.

Works on a plain workstation: reads the ASCII polyMesh directly, so neither
OpenFOAM nor pyvista is required.

    # Overview only
    python postprocessing/plot_mesh.py \
        --mesh results/benchmark_baseline_coldstart_20260729/constant/polyMesh \
        --title "RAE 2822" --out rae2822_mesh.png

    # Overview plus baseline-vs-deformed comparison
    python postprocessing/plot_mesh.py \
        --mesh   results/benchmark_baseline_coldstart_20260729/constant/polyMesh \
        --points results/benchmark_baseline_coldstart_20260729/8001/polyMesh/points \
        --deformed results/benchmark_eval9_coldstart_20260729/10392/polyMesh/points \
        --title "RAE 2822" --labels "Baseline" "Optimized (eval 9)" \
        --out rae2822_mesh.png --compare-out rae2822_mesh_deformation.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foam_mesh import (ordered_slice_ids, patch_faces,  # noqa: E402
                       read_points)

C_A = "#4a6fa5"
C_B = "#b03a2e"


def _as_points(src, mesh_dir: str | None = None) -> np.ndarray:
    """Accept either an already-loaded point array or a path to read."""
    if isinstance(src, np.ndarray):
        return src
    if src is None:
        src = os.path.join(mesh_dir, "points")
    return read_points(src)


def load(mesh_dir: str, points=None):
    """Cell outlines, wing contour ids and the point array for a mesh."""
    pts = _as_points(points, mesh_dir)
    cells = patch_faces(mesh_dir, "symmetry1")
    wing = ordered_slice_ids(mesh_dir, "wing", pts)
    return pts, cells, wing


def _closed(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1])


def overview(mesh_dir: str, out: str, title: str, points=None,
             subtitle: str = "") -> None:
    pts, cells, wing = load(mesh_dir, points)
    polys = [pts[f][:, :2] for f in cells]
    w = pts[wing]
    far = max(abs(pts[:, 0]).max(), abs(pts[:, 1]).max())

    views = [
        (f"Full domain ({far:.0f} chords)",
         (-far * 1.02, far * 1.05), (-far * 1.02, far * 1.02), 0.06),
        ("Airfoil and wake", (-0.55, 2.1), (-0.85, 0.85), 0.30),
        ("Leading edge", (-0.035, 0.13), (-0.075, 0.075), 0.45),
        ("Trailing edge", (0.93, 1.05), (-0.05, 0.07), 0.45),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
    for ax, (t, xl, yl, lw) in zip(axes.ravel(), views):
        ax.add_collection(PolyCollection(polys, facecolors="none",
                                         edgecolors=C_A, linewidths=lw))
        ax.plot(*_closed(w), "-", color=C_B, lw=0.9)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")

    head = f"{title} C-mesh: {len(cells):,} hex cells"
    fig.suptitle(head + (f" — {subtitle}" if subtitle else ""), y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[OK] {out}  ({len(cells):,} cells)")


def compare(mesh_dir: str, base_points, defo_points, out: str,
            title: str, labels: tuple[str, str],
            xlim=(0.30, 0.80), ylim=(-0.12, 0.14)) -> None:
    """Baseline vs deformed mesh, zoomed on the airfoil.

    Each panel also carries the *other* case's surface as a dashed line, so a
    sub-1 % chord change stays visible where the mesh alone would not show it.
    """
    pa, cells, wing = load(mesh_dir, base_points)
    pb = _as_points(defo_points, mesh_dir)
    if pa.shape != pb.shape:
        raise ValueError(f"point counts differ: {pa.shape} vs {pb.shape}")

    a, b = pa[wing], pb[wing]
    dy = (b[:, 1] - a[:, 1]) * 1e3
    peak, xpk = np.abs(dy).max(), a[int(np.argmax(np.abs(dy))), 0]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, (p, other, lab, c, oc) in zip(axes, (
            (pa, b, labels[0], C_A, C_B),
            (pb, a, labels[1], C_B, C_A))):
        ax.add_collection(PolyCollection([p[f][:, :2] for f in cells],
                                         facecolors="none", edgecolors=c,
                                         linewidths=0.35))
        ax.plot(*_closed(p[wing]), "-", color=c, lw=1.4)
        ax.plot(*_closed(other), "--", color=oc, lw=1.0, alpha=0.85)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(lab, fontsize=11)
        ax.set_xlabel("x/c")
    axes[0].set_ylabel("y/c")

    fig.suptitle(f"{title}: mesh deformed by IDWarp, not regenerated — "
                 f"peak surface |dy| = {peak:.2f} mm/c at x/c = {xpk:.2f} "
                 f"({peak / 10:.2f} % chord); dashed line is the other case",
                 y=0.98, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[OK] {out}  (peak |dy| = {peak:.3f} mm/c at x/c = {xpk:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="polyMesh directory (topology)")
    ap.add_argument("--points", default=None,
                    help="override points file for the baseline state")
    ap.add_argument("--deformed", default=None,
                    help="points file of the deformed/optimized state")
    ap.add_argument("--title", default="Airfoil")
    ap.add_argument("--labels", nargs=2, default=("Baseline", "Optimized"))
    ap.add_argument("--zoom", nargs=4, type=float, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-out", default=None)
    args = ap.parse_args()

    overview(args.mesh, args.out, args.title, args.points)

    if args.deformed:
        if not args.compare_out:
            raise SystemExit("--deformed requires --compare-out")
        kw = {}
        if args.zoom:
            kw = {"xlim": tuple(args.zoom[:2]), "ylim": tuple(args.zoom[2:])}
        compare(args.mesh, args.points or os.path.join(args.mesh, "points"),
                args.deformed, args.compare_out, args.title,
                tuple(args.labels), **kw)


if __name__ == "__main__":
    main()
