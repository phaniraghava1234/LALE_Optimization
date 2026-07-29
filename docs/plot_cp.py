#!/usr/bin/env python3
"""
Plot Cp along the airfoil chord from ParaView-exported wing-surface CSV(s).

Usage:
    # Single run (baseline OR optimized only, produces one PNG in its folder):
    python docs/plot_cp.py <csv>

    # Overlay mode (baseline + optimized, produces four PNGs):
    python docs/plot_cp.py <baseline_csv> --optimized <optimized_csv>

Overlay mode outputs
--------------------
  <baseline_folder>/cp.png
        Single-run Cp + airfoil for the baseline.
  <optimized_folder>/cp.png
        Single-run Cp + airfoil for the optimized case.
  <baseline_folder>/cp_overlay.png
        Cp overlay only (no airfoil) with baseline vs optimized.
  <baseline_folder>/shape_comparison.png
        Full-chord airfoil shape overlay + zoomed insets on near-LE and
        near-shock regions. Transparent markers on the optimized shape
        so the underlying baseline curve can be tracked.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("baseline_csv", help="First (baseline) CSV")
parser.add_argument("--optimized", default=None,
                    help="Second CSV (optimized). Enables overlay mode.")
parser.add_argument("--x-te-cut", type=float, default=0.98,
                    help="Trim x/c > this (blunt TE ambiguity zone)")
args = parser.parse_args()


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {candidates} in {list(df.columns)}")


def load_and_split(csv_path, x_te_cut):
    df = pd.read_csv(csv_path)
    cp_col = find_col(df, ["Cp", "Cp0", "Cp1"])
    x_col = find_col(df, ["Points:0", "Points_0"])
    y_col = find_col(df, ["Points:1", "Points_1"])
    z_col = find_col(df, ["Points:2", "Points_2"])

    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    Cp = df[cp_col].values

    # Filter to one spanwise station.
    unique_z = np.unique(np.round(z, 6))
    mask_z = np.isclose(z, unique_z[0], atol=1e-5)
    x, y, Cp = x[mask_z], y[mask_z], Cp[mask_z]

    # Trim near-TE ambiguity.
    m = x <= x_te_cut
    x, y, Cp = x[m], y[m], Cp[m]

    # Chord-line split.
    i_le5 = np.argsort(x)[:5]
    i_te5 = np.argsort(x)[-5:]
    x_le, y_le = x[i_le5].mean(), y[i_le5].mean()
    x_te, y_te = x[i_te5].mean(), y[i_te5].mean()
    y_chord = y_le + (y_te - y_le) * (x - x_le) / (x_te - x_le)
    upper = y > y_chord
    lower = y < y_chord

    xu, yu, Cu = x[upper], y[upper], Cp[upper]
    xl, yl, Cl = x[lower], y[lower], Cp[lower]

    ou = np.argsort(xu); xu, yu, Cu = xu[ou], yu[ou], Cu[ou]
    ol = np.argsort(xl); xl, yl, Cl = xl[ol], yl[ol], Cl[ol]
    print(f"[{csv_path}] u={len(xu)} l={len(xl)} "
          f"Cp_min={min(Cu.min(), Cl.min()):.3f} "
          f"Cp_max={max(Cu.max(), Cl.max()):.3f}")
    return xu, yu, Cu, xl, yl, Cl


# ----------------------------------------------------------------------- #
def plot_single(data, out_path, title):
    """Panel Cp + airfoil geometry."""
    xu, yu, Cu, xl, yl, Cl = data
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(xu, Cu, "-", color="#c0392b", lw=2, label="Upper")
    ax.plot(xl, Cl, "-", color="#1f4e79", lw=2, label="Lower")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("$C_p$")
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    ax.legend()

    ax = axes[1]
    ax.plot(xu, yu, "-", color="#c0392b", lw=1.5)
    ax.plot(xl, yl, "-", color="#1f4e79", lw=1.5)
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_cp_overlay(base, opt, out_path):
    """Only Cp curves — baseline vs optimized. No airfoil."""
    xu, _, Cu, xl, _, Cl = base
    xuo, _, Cuo, xlo, _, Clo = opt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xu, Cu, "-", color="#7f7f7f", lw=1.6, label="Baseline, upper")
    ax.plot(xl, Cl, "--", color="#7f7f7f", lw=1.6, label="Baseline, lower")
    ax.plot(xuo, Cuo, "-", color="#c0392b", lw=2.2, label="Optimized, upper")
    ax.plot(xlo, Clo, "--", color="#1f4e79", lw=2.2, label="Optimized, lower")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x / c")
    ax.set_ylabel("$C_p$")
    ax.set_title(
        "Pressure coefficient overlay — benchmark preset\n"
        "M=0.729, Re=6.5e6, target $C_L^*$=0.824")
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def _draw_shape(ax, base, opt, marker_alpha=0.6, marker_size=6, lw=1.8,
                equal_aspect=True):
    """Helper: draw baseline + optimized shape on ax."""
    xu, yu, _, xl, yl, _ = base
    xuo, yuo, _, xlo, ylo, _ = opt
    ax.plot(xu,  yu,  "-", color="#7f7f7f", lw=lw, label="Baseline upper")
    ax.plot(xl,  yl,  "-", color="#7f7f7f", lw=lw, label="Baseline lower")
    ax.plot(xuo, yuo, "-", color="#c0392b", lw=lw, label="Optimized upper")
    ax.plot(xlo, ylo, "-", color="#1f4e79", lw=lw, label="Optimized lower")
    ax.plot(xuo, yuo, "o", color="#c0392b", ms=marker_size, alpha=marker_alpha)
    ax.plot(xlo, ylo, "o", color="#1f4e79", ms=marker_size, alpha=marker_alpha)
    if equal_aspect:
        ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)


def _draw_shape_hollow(ax, base, opt, ms=5, lw=1.6, stride=8):
    """Alternative style with HOLLOW optimized markers, thinned by `stride`.

    stride > 1 keeps only every stride-th marker, so densely-spaced points
    at LE / TE don't visually pile up.
    """
    xu, yu, _, xl, yl, _ = base
    xuo, yuo, _, xlo, ylo, _ = opt
    ax.plot(xu,  yu,  "-", color="#7f7f7f", lw=lw, label="Baseline upper")
    ax.plot(xl,  yl,  "-", color="#7f7f7f", lw=lw, label="Baseline lower")
    ax.plot(xuo, yuo, "-",  color="#c0392b", lw=lw, label="Optimized upper")
    ax.plot(xlo, ylo, "-",  color="#1f4e79", lw=lw, label="Optimized lower")
    ax.plot(xuo[::stride], yuo[::stride], "o",
            mfc="none", mec="#c0392b", ms=ms, mew=1.2)
    ax.plot(xlo[::stride], ylo[::stride], "o",
            mfc="none", mec="#1f4e79", ms=ms, mew=1.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)


def plot_shape_full(base, opt, out_path):
    """Plot 1 of 3 — full airfoil overview with thinned hollow markers."""
    fig, ax = plt.subplots(figsize=(14, 6))
    # Subsample every 4th point to avoid LE cosine-cluster crowding.
    _draw_shape_hollow(ax, base, opt, ms=5, lw=1.6, stride=4)
    ax.set_title(
        "Airfoil shape — Baseline (grey solid) vs Optimized Eval 9 (colored + hollow markers)\n"
        "Markers thinned (every 4th point) so LE cosine-spacing does not visually crowd")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_shape_zooms(base, opt, out_path):
    """Plot 2 of 3 — near-LE and near-shock zooms side by side.

    Both panels use equal aspect ratio (true geometric proportions).
    """
    fig, (ax_le, ax_sh) = plt.subplots(1, 2, figsize=(18, 6))

    _draw_shape(ax_le, base, opt, marker_alpha=0.7, marker_size=6, lw=1.8,
                equal_aspect=True)
    ax_le.set_xlim(-0.005, 0.10)
    ax_le.set_ylim(-0.04, 0.05)
    ax_le.set_title("Zoom — near leading edge (x/c ≤ 0.10)")
    ax_le.set_xlabel("x / c")
    ax_le.set_ylabel("y / c")
    ax_le.legend(loc="lower right", fontsize=9)

    _draw_shape(ax_sh, base, opt, marker_alpha=0.7, marker_size=6, lw=1.8,
                equal_aspect=True)
    ax_sh.set_xlim(0.30, 0.75)
    ax_sh.set_ylim(0.02, 0.07)
    ax_sh.set_title("Zoom — near-shock region on upper surface (0.30 ≤ x/c ≤ 0.75)")
    ax_sh.set_xlabel("x / c")
    ax_sh.set_ylabel("y / c")
    ax_sh.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[wrote] {out_path}")


def plot_shape_delta(base, opt, out_path):
    """Plot 3 of 3 — Δy = y_optimized − y_baseline per surface."""
    xu,  yu,  _, xl,  yl,  _ = base
    xuo, yuo, _, xlo, ylo, _ = opt

    # Interpolate optimized onto baseline x so we can subtract.
    yuo_on_base = np.interp(xu, xuo, yuo)
    ylo_on_base = np.interp(xl, xlo, ylo)
    dy_upper = yuo_on_base - yu
    dy_lower = ylo_on_base - yl

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xu, dy_upper * 1e3, "-o", color="#c0392b", ms=4, lw=1.6,
            label="Δy upper = y_opt − y_base")
    ax.plot(xl, dy_lower * 1e3, "-o", color="#1f4e79", ms=4, lw=1.6,
            label="Δy lower = y_opt − y_base")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x / c")
    ax.set_ylabel("Δy  (mm on 1 m chord = ‰ of chord)")
    ax.set_title(
        "Shape delta:  y_optimized − y_baseline  along each surface\n"
        "Δy_upper > 0 = upper moved OUT (thickening).  "
        "Δy_lower < 0 = lower moved DOWN (thickening).")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[wrote] {out_path}")


# ----------------------------------------------------------------------- #
base = load_and_split(args.baseline_csv, args.x_te_cut)
base_dir = os.path.dirname(os.path.abspath(args.baseline_csv))

if args.optimized:
    opt = load_and_split(args.optimized, args.x_te_cut)
    opt_dir = os.path.dirname(os.path.abspath(args.optimized))

    # Individual plots (Cp + airfoil) in each case's own folder.
    plot_single(base, os.path.join(base_dir, "cp.png"),
                "Pressure coefficient — Baseline (benchmark)")
    plot_single(opt, os.path.join(opt_dir, "cp.png"),
                "Pressure coefficient — Optimized Eval 9 cold-start (benchmark)")
    # Cp overlay (no airfoil) in baseline folder.
    plot_cp_overlay(base, opt, os.path.join(base_dir, "cp_overlay.png"))
    # Shape comparison — three separate PNGs in baseline folder.
    plot_shape_full  (base, opt, os.path.join(base_dir, "shape_full.png"))
    plot_shape_zooms (base, opt, os.path.join(base_dir, "shape_zooms.png"))
    plot_shape_delta (base, opt, os.path.join(base_dir, "shape_delta.png"))
else:
    # Just one CSV — one plot in its folder.
    plot_single(base, os.path.join(base_dir, "cp.png"),
                "Pressure coefficient — single run")
