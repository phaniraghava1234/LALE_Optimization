#!/usr/bin/env python3
"""Generate the Case 2 (MH139-F low-Reynolds) figures from an archived run.

Runs on a plain workstation -- no DAFoam container, no pyvista. Reads the
archive produced by `mdo/archive_run.py` and writes PNGs alongside the
existing optimization history plots.

    python postprocessing/plot_lale_figures.py \
        --archive results/case_lale_20260806_084558
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foam_mesh import (ordered_slice_ids, patch_face_owners,  # noqa: E402
                       patch_faces, patch_point_ids, read_labels, read_points,
                       reconstruct_points, reconstruct_saved_points,
                       reconstruct_saved_scalar)
from plot_mesh import compare as mesh_compare  # noqa: E402
from plot_mesh import overview as mesh_overview  # noqa: E402

C_BASE = "#3b3b3b"
C_OPT = "#c0392b"
C_FEAS = "#2471a3"


def parse_summary(path: str) -> dict[str, np.ndarray]:
    """Pull the per-evaluation table out of opt_summary.md."""
    rows = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\|\s*(\d+)\s*\|\s*(base|eval)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        raise ValueError(f"no evaluation rows found in {path}")

    col = lambda i: np.array([float(r[i]) for r in rows])  # noqa: E731
    return {
        "eval": np.array([int(r[0]) for r in rows]),
        "shape_norm": col(2),
        "cd": col(4),
        "counts": col(5),
        "cl": col(7),
        "cl_err": col(8),
        "ld": col(9),
        "thick": col(10),
        "vol": col(11),
        "rcon": col(12),
    }


def fig_shape(archive: str, time_name: str, out: str) -> None:
    mesh_dir = os.path.join(archive, "mesh", "polyMesh")
    base = read_points(os.path.join(mesh_dir, "points"))
    defo = reconstruct_points(os.path.join(archive, "flow"), time_name,
                              len(base))

    # Contour order comes from face connectivity, so it is valid for a
    # cambered section where an angular sort would interleave the surfaces.
    ids = ordered_slice_ids(mesh_dir, "wing", base)
    b, o = base[ids], defo[ids]

    # The walk starts at the leading edge; the far point is the trailing
    # edge, so the two halves of the loop are the two surfaces.
    te = int(np.argmax(b[:, 0]))
    halves = ((slice(0, te + 1), C_OPT), (slice(te, None), C_FEAS))
    # Label whichever half sits higher at mid-chord as the upper surface.
    first_mean = b[:te + 1, 1].mean()
    names = (("upper surface", "lower surface") if first_mean >= 0
             else ("lower surface", "upper surface"))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.8), height_ratios=[2.0, 1.0])

    ax1.plot(np.append(b[:, 0], b[0, 0]), np.append(b[:, 1], b[0, 1]),
             "-", color=C_BASE, lw=1.4, label="Baseline MH139-F")
    ax1.plot(np.append(o[:, 0], o[0, 0]), np.append(o[:, 1], o[0, 1]),
             "--", color=C_OPT, lw=1.6, label="Optimized (eval 50)")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_ylabel("y/c")
    ax1.set_title(
        "MH139-F shape change, Re = 1.67e5, $C_L$ = 0.975 at fixed "
        r"$\alpha$ = 4.5$\degree$")

    dy = (o[:, 1] - b[:, 1]) * 1e3
    for (sl, c), nm in zip(halves, names):
        ax2.plot(b[sl, 0], dy[sl], "-", color=c, lw=1.4, label=nm)

    ax2.axhline(0, color="k", lw=0.6)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    ax2.set_xlabel("x/c")
    ax2.set_ylabel(r"$\Delta y$  [mm/c]")
    peak = np.abs(dy).max()
    ax2.set_title(f"Displacement from baseline, peak |dy| = {peak:.1f} mm/c "
                  f"({peak / 10:.2f}% chord)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[OK] {out}  (peak |dy| = {peak:.2f} mm/c, {len(ids)} contour pts)")


def fig_mesh(archive: str, out: str) -> None:
    """Overview panels, delegated to the shared mesh renderer."""
    mesh_overview(os.path.join(archive, "mesh", "polyMesh"), out, "MH139-F",
                  subtitle=r"$y^+ pprox 30$ wall functions, "
                           "10-chord far field")


def fig_mesh_deformation(archive: str, out: str, eval_no: int) -> None:
    """Baseline vs deformed near-wall mesh.

    The deformed state lives in the decomposed per-evaluation snapshot, so it
    is reconstructed into a global point array before rendering.
    """
    mesh_dir = os.path.join(archive, "mesh", "polyMesh")
    base = read_points(os.path.join(mesh_dir, "points"))
    defo = reconstruct_saved_points(
        os.path.join(archive, "optimization", "saved", f"eval{eval_no:04d}"),
        os.path.join(archive, "flow"), len(base))
    mesh_compare(mesh_dir, base, defo, out, "MH139-F",
                 ("Baseline", f"Optimized (eval {eval_no})"),
                 xlim=(0.55, 1.06), ylim=(-0.10, 0.14))


def surface_cp(archive: str, eval_no: int, u_cfd: float):
    """Wall Cp along the wing contour for one saved evaluation.

    Returns (x, y, cp, i_te) with the faces ordered from the leading edge
    around the section; `i_te` is the index of the trailing-edge face, so
    [:i_te+1] and [i_te:] are the two surfaces.
    """
    mesh_dir = os.path.join(archive, "mesh", "polyMesh")
    flow = os.path.join(archive, "flow")
    ev_dir = os.path.join(archive, "optimization", "saved",
                          f"eval{eval_no:04d}")

    owners = patch_face_owners(mesh_dir, "wing")
    n_cells = int(read_labels(os.path.join(mesh_dir, "owner")).max()) + 1
    p = reconstruct_saved_scalar(ev_dir, flow, "p", n_cells)
    pts = reconstruct_saved_points(ev_dir, flow,
                                   len(read_points(
                                       os.path.join(mesh_dir, "points"))))

    faces = patch_faces(mesh_dir, "wing")
    order = ordered_slice_ids(mesh_dir, "wing", pts)
    pos = {int(pid): i for i, pid in enumerate(order)}
    n = len(order)

    # Rank each face by where its in-plane edge sits along the contour.
    rank = np.empty(len(faces))
    for k, f in enumerate(faces):
        ps = [pos[int(q)] for q in f if int(q) in pos]
        if len(ps) == 2 and abs(ps[0] - ps[1]) > n / 2:
            rank[k] = 0.0                      # face straddling the LE seam
        else:
            rank[k] = float(np.mean(ps)) if ps else np.nan

    keep = ~np.isnan(rank)
    idx = np.argsort(rank[keep])
    cent = np.array([pts[f].mean(axis=0) for f in faces])[keep][idx]
    cp = (p[owners][keep][idx]) / (0.5 * u_cfd ** 2)
    return cent[:, 0], cent[:, 1], cp, int(np.argmax(cent[:, 0]))


def fig_cp(archive: str, out: str, u_cfd: float,
           evals: tuple[int, int]) -> None:
    base_no, opt_no = evals
    xb, _, cpb, teb = surface_cp(archive, base_no, u_cfd)
    xo, _, cpo, teo = surface_cp(archive, opt_no, u_cfd)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for (x, cp, te), c, lbl in (
            ((xb, cpb, teb), C_BASE,
             f"baseline (eval {base_no}), $C_L$ = 0.696"),
            ((xo, cpo, teo), C_OPT,
             f"optimized (eval {opt_no}), $C_L$ = 0.975")):
        ax.plot(x[:te + 1], cp[:te + 1], "-", color=c, lw=1.5, label=lbl)
        ax.plot(x[te:], cp[te:], "-", color=c, lw=1.5)

    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("$C_p$")
    ax.set_title("Surface pressure, MH139-F at Re = 1.67e5, "
                 r"$\alpha$ = 4.5$\degree$ (fixed)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[OK] {out}  (baseline Cp in [{cpb.min():.3f}, {cpb.max():.3f}], "
          f"optimized [{cpo.min():.3f}, {cpo.max():.3f}])")


def fig_convergence(archive: str, out: str) -> None:
    d = parse_summary(
        os.path.join(archive, "optimization", "opt_summary.md"))
    ev = d["eval"]

    feas = np.where(np.abs(d["cl_err"]) < 1e-3)[0]
    first = int(ev[feas[0]]) if len(feas) else int(ev[0])
    best = int(ev[-1])

    fig, (ax1, axz, ax2) = plt.subplots(
        3, 1, figsize=(10, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.8]})

    i_first = int(np.where(ev == first)[0][0])
    cd_first, cd_last = d["counts"][i_first], d["counts"][-1]

    # --- panel 1: full range, phases shaded --------------------------------
    lo, hi = d["counts"].min(), d["counts"].max()
    ax1.set_ylim(lo - 0.08 * (hi - lo), hi + 0.10 * (hi - lo))
    ax1.axvspan(ev[0], first, color="#f2b1b1", alpha=0.35, zorder=0)
    ax1.axvspan(first, ev[-1], color="#b8d8ef", alpha=0.30, zorder=0)
    ax1.plot(ev, d["counts"], "-o", color=C_BASE, ms=3, lw=1.2, zorder=3)
    ax1.plot(first, cd_first, "o", color=C_FEAS, ms=8, zorder=5)
    ax1.plot(best, cd_last, "o", color=C_OPT, ms=8, zorder=5)
    ax1.text(0.055, 0.10, "seeking\nfeasibility", transform=ax1.transAxes,
             ha="center", va="bottom", fontsize=9, color="#8c2f2f")
    ax1.text(0.60, 0.10, "minimizing $C_D$ at constant lift",
             transform=ax1.transAxes, ha="center", va="bottom",
             fontsize=9, color="#1b4f72")
    ax1.annotate(f"undeformed MH139-F: {d['counts'][0]:.1f} ct at "
                 f"$C_L$ = {d['cl'][0]:.3f}\n"
                 f"infeasible — lift constraint not met",
                 (ev[0], d["counts"][0]), xytext=(11, 292),
                 textcoords="data", fontsize=9, color=C_BASE,
                 va="center",
                 arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.0,
                                 shrinkA=0, shrinkB=6))
    ax1.set_ylabel("$C_D$  [drag counts]")
    ax1.grid(alpha=0.3)
    ax1.set_title("MH139-F low-Reynolds optimization, 50 SLSQP evaluations",
                  pad=10)

    # --- panel 2: zoom on the feasible region ------------------------------
    m = ev >= first
    axz.plot(ev[m], d["counts"][m], "-o", color=C_BASE, ms=3, lw=1.2)
    axz.plot(first, cd_first, "o", color=C_FEAS, ms=8, zorder=5)
    axz.plot(best, cd_last, "o", color=C_OPT, ms=8, zorder=5)

    zlo, zhi = d["counts"][m].min(), d["counts"][m].max()
    axz.set_ylim(zlo - 0.22 * (zhi - zlo), zhi + 0.08 * (zhi - zlo))

    axz.annotate(f"first feasible (eval {first})\n{cd_first:.1f} ct",
                 (first, cd_first), textcoords="offset points",
                 xytext=(14, 10), va="bottom", fontsize=9, color=C_FEAS)
    axz.annotate(f"final (eval {best})\n{cd_last:.1f} ct",
                 (best, cd_last), textcoords="offset points",
                 xytext=(-12, 6), ha="right", fontsize=9, color=C_OPT)

    # GEMSEO reports eval 8 as the optimum, but it is only feasible under the
    # loose default eq_tolerance -- worth showing rather than hiding.
    arte = np.where(np.abs(d["cl_err"]) > 5e-3)[0]
    arte = arte[(ev[arte] > first)]
    if len(arte):
        j = int(arte[np.argmin(d["counts"][arte])])
        axz.plot(ev[j], d["counts"][j], "v", color="#7d3c98", ms=8, zorder=5)
        axz.annotate(f"eval {ev[j]}: GEMSEO's reported optimum — "
                     f"{d['counts'][j]:.1f} ct,\nbut $C_L$ err = "
                     f"{d['cl_err'][j]:+.4f}, a tolerance artifact",
                     (ev[j], d["counts"][j]), textcoords="offset points",
                     xytext=(30, -14), fontsize=8, color="#7d3c98",
                     va="top", ha="left",
                     arrowprops=dict(arrowstyle="->", color="#7d3c98",
                                     lw=0.9, shrinkA=0, shrinkB=4))
    axz.set_ylabel("$C_D$  [drag counts]")
    axz.grid(alpha=0.3)
    axz.set_title(f"Feasible region only: {cd_first:.1f} $\\rightarrow$ "
                  f"{cd_last:.1f} ct  ({cd_last - cd_first:+.1f} ct, "
                  f"{100 * (cd_last - cd_first) / cd_first:+.1f} %)",
                  fontsize=10, pad=6)

    # --- panel 3: lift ------------------------------------------------------
    ax2.axhline(0.975, color=C_OPT, lw=1.0, ls="--",
                label="target $C_L$ = 0.975")
    ax2.plot(ev, d["cl"], "-o", color=C_BASE, ms=3, lw=1.2, label="$C_L$")
    ax2.set_ylabel("$C_L$")
    ax2.set_xlabel("Primal evaluation")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    i0 = int(np.where(ev == first)[0][0])
    print(f"[OK] {out}  (eval {first}: {d['counts'][i0]:.1f} ct -> "
          f"eval {best}: {d['counts'][-1]:.1f} ct, "
          f"{d['counts'][-1] - d['counts'][i0]:+.1f} ct)")


def fig_polar(archive: str, out: str) -> None:
    """Trajectory in the (C_L, C_D) plane -- shows the two phases as a path."""
    d = parse_summary(
        os.path.join(archive, "optimization", "opt_summary.md"))
    feas = np.where(np.abs(d["cl_err"]) < 1e-3)[0]
    k = int(feas[0]) if len(feas) else 0

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    for ax in (axa, axb):
        ax.plot(d["cl"][:k + 1], d["counts"][:k + 1], "-o", color="#8c2f2f",
                ms=4, lw=1.2, label="phase 1: seeking feasibility")
        ax.plot(d["cl"][k:], d["counts"][k:], "-o", color=C_FEAS, ms=4,
                lw=1.2, label="phase 2: minimizing $C_D$")
        ax.axvline(0.975, color=C_OPT, ls="--", lw=1.0,
                   label="lift constraint $C_L$ = 0.975")
        ax.plot(d["cl"][-1], d["counts"][-1], "*", color=C_OPT, ms=16,
                zorder=5)
        ax.set_xlabel("$C_L$")
        ax.grid(alpha=0.3)

    axa.plot(d["cl"][0], d["counts"][0], "s", color=C_BASE, ms=9, zorder=5)
    axa.annotate("undeformed\n(infeasible)", (d["cl"][0], d["counts"][0]),
                 textcoords="offset points", xytext=(10, 2), fontsize=9,
                 color=C_BASE)
    axa.set_ylabel("$C_D$  [drag counts]")
    axa.set_title("Full trajectory", fontsize=11)
    axa.legend(loc="upper left", frameon=False, fontsize=8.5)

    # Zoom on the feasible cluster, which the full view collapses to a sliver.
    cl_f, cd_f = d["cl"][k:], d["counts"][k:]
    padx = 0.35 * (cl_f.max() - cl_f.min() + 1e-9)
    pady = 0.10 * (cd_f.max() - cd_f.min())
    axb.set_xlim(cl_f.min() - padx, cl_f.max() + padx)
    axb.set_ylim(cd_f.min() - pady, cd_f.max() + 2.2 * pady)
    axb.annotate(f"optimized\n{d['counts'][-1]:.1f} ct",
                 (d["cl"][-1], d["counts"][-1]),
                 textcoords="offset points", xytext=(10, 8), fontsize=9,
                 color=C_OPT)
    axb.set_title("Zoom: the optimizer slides down the constraint",
                  fontsize=11)

    fig.suptitle("Design trajectory in the ($C_L$, $C_D$) plane", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[OK] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True,
                    help="archive directory under results/")
    ap.add_argument("--time", default=None,
                    help="flow time directory holding the deformed mesh "
                         "(default: the latest one found)")
    ap.add_argument("--u-cfd", type=float, default=2.531,
                    help="CFD freestream speed used to non-dimensionalize Cp")
    ap.add_argument("--base-eval", type=int, default=1)
    ap.add_argument("--opt-eval", type=int, default=50)
    ap.add_argument("--outdir", default=None,
                    help="where to write PNGs (default: <archive>/optimization)")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.archive, "optimization")
    os.makedirs(outdir, exist_ok=True)

    time_name = args.time
    if time_name is None:
        p0 = os.path.join(args.archive, "flow", "processor0")
        times = [t for t in os.listdir(p0)
                 if re.fullmatch(r"\d+(\.\d+)?", t)
                 and os.path.isdir(os.path.join(p0, t, "polyMesh"))]
        if not times:
            raise SystemExit("no time directory with a polyMesh found")
        time_name = max(times, key=float)
        print(f"[..] using flow time {time_name}")

    fig_shape(args.archive, time_name,
              os.path.join(outdir, "lale_shape_comparison.png"))
    fig_cp(args.archive, os.path.join(outdir, "lale_cp_overlay.png"),
           args.u_cfd, (args.base_eval, args.opt_eval))
    fig_mesh(args.archive, os.path.join(outdir, "lale_mesh.png"))
    fig_mesh_deformation(args.archive,
                         os.path.join(outdir, "lale_mesh_deformation.png"),
                         args.opt_eval)
    fig_convergence(args.archive,
                    os.path.join(outdir, "lale_convergence.png"))
    fig_polar(args.archive, os.path.join(outdir, "lale_trajectory.png"))


if __name__ == "__main__":
    main()
