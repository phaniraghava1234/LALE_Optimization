#!/usr/bin/env python3
"""
Step 1.1 -- Data Ingestion
==========================
Reads airfoil coordinate files and splits them logically into
suction side (SS) and pressure side (PS), both running LE -> TE.

Supported formats:
  * Selig .dat  : single loop, TE -> upper -> LE -> lower -> TE
  * DAFoam .profile pair : separate PS/SS files, LE -> TE (as in the
    official DAFoam RAE2822 tutorial, ../case/profiles/)

The output convention used by every downstream tool in this project is:
    xy_ss : (N,2) array, LE (x=0) -> TE, upper surface
    xy_ps : (N,2) array, LE (x=0) -> TE, lower surface
"""
from __future__ import annotations

import numpy as np


def read_selig(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a Selig-format .dat file and split at the leading edge.

    Returns (xy_ss, xy_ps), both ordered LE -> TE.
    """
    # Skip a header line if the first line is not numeric
    with open(path) as f:
        first = f.readline()
    try:
        [float(v) for v in first.split()]
        skip = 0
    except ValueError:
        skip = 1

    pts = np.loadtxt(path, skiprows=skip)
    if pts.shape[1] > 2:
        pts = pts[:, :2]

    # LE = point of minimum x (robust for cambered airfoils)
    i_le = int(np.argmin(pts[:, 0]))

    # Selig runs TE -> LE along the upper surface, then LE -> TE lower
    xy_ss = pts[: i_le + 1][::-1]   # reverse so it runs LE -> TE
    xy_ps = pts[i_le:]

    return _dedup(xy_ss), _dedup(xy_ps)


def read_profile_pair(path_ps: str, path_ss: str) -> tuple[np.ndarray, np.ndarray]:
    """Read DAFoam-style PS/SS .profile files (already LE -> TE)."""
    xy_ps = np.loadtxt(path_ps)[:, :2]
    xy_ss = np.loadtxt(path_ss)[:, :2]
    return _dedup(xy_ss), _dedup(xy_ps)


def closed_loop(xy_ss: np.ndarray, xy_ps: np.ndarray, n_te: int = 5) -> np.ndarray:
    """Assemble a single closed loop for O-grid meshing.

    Order: TE (upper) -> SS -> LE -> PS -> TE (lower) -> blunt-TE base
    closing back to the start. `n_te` points span the blunt trailing edge.
    A sharp TE (identical last points) collapses n_te automatically.
    """
    loop = np.vstack([xy_ss[::-1], xy_ps[1:]])  # TE_up ... LE ... TE_low
    te_up, te_low = xy_ss[-1], xy_ps[-1]
    gap = np.linalg.norm(te_up - te_low)
    if gap > 1e-10:
        base = np.linspace(te_low, te_up, n_te)[1:-1]  # exclude endpoints
        loop = np.vstack([loop, base])
    return loop


def geometry_report(xy_ss: np.ndarray, xy_ps: np.ndarray) -> dict:
    """Basic sanity metrics used by the meshing QC step."""
    chord = max(xy_ss[:, 0].max(), xy_ps[:, 0].max()) - min(
        xy_ss[:, 0].min(), xy_ps[:, 0].min()
    )
    # max thickness by interpolating PS onto SS x-stations
    x = xy_ss[:, 0]
    t = xy_ss[:, 1] - np.interp(x, xy_ps[:, 0], xy_ps[:, 1])
    return {
        "chord": float(chord),
        "n_ss": len(xy_ss),
        "n_ps": len(xy_ps),
        "max_thickness": float(t.max()),
        "x_max_thickness": float(x[np.argmax(t)]),
        "te_gap": float(np.linalg.norm(xy_ss[-1] - xy_ps[-1])),
    }


def _dedup(xy: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Remove consecutive duplicate points (they break spline fits)."""
    keep = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) > tol:
            keep.append(i)
    return xy[keep]


def thicken_te(
    xy_ss: np.ndarray,
    xy_ps: np.ndarray,
    gap_target: float = 2.0e-3,
    taper_x_start: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen the trailing-edge gap by symmetric, tapered y-offset.

    The raw RAE 2822 has a TE gap of ~3.7e-4 c. The paper's mesh uses a
    somewhat thicker TE (visible in Figure 10, ~4-5e-3 c), which improves
    mesh cell aspect ratio in the TE region. This helper offsets SS points
    upward and PS points downward, with an offset that TAPERS to zero at
    `taper_x_start` (so the airfoil shape upstream of that x is unchanged).

    Weight uses a cosine (smoothstep) taper: w(t) = 0.5 * (1 - cos(pi*t))
    for t in [0, 1] where t is the normalized position from taper_x_start
    to the airfoil's max x. That gives C^1 smoothness at the taper start.

    Only fires when gap_target > current_gap.
    """
    te_up = xy_ss[-1]
    te_low = xy_ps[-1]
    current_gap = float(te_up[1] - te_low[1])
    if gap_target <= current_gap:
        return xy_ss.copy(), xy_ps.copy()

    half_offset = (gap_target - current_gap) / 2.0

    def _offset(xy: np.ndarray, direction: int) -> np.ndarray:
        out = xy.copy()
        mask = out[:, 0] >= taper_x_start
        if not mask.any():
            return out
        x_max = float(out[:, 0].max())
        t = (out[mask, 0] - taper_x_start) / (x_max - taper_x_start)
        weight = 0.5 * (1.0 - np.cos(np.pi * t))
        out[mask, 1] += direction * half_offset * weight
        return out

    return _offset(xy_ss, +1), _offset(xy_ps, -1)


def cosine_resample(xy: np.ndarray, n_pts: int = 200) -> np.ndarray:
    """Resample one airfoil surface at cosine-spaced arc-length positions.

    Cosine spacing puts points where the geometry needs them: dense near
    LE and TE (high curvature), sparse mid-chord.

    IMPORTANT: this uses **parametric arc-length spline fitting**, NOT
    y = f(x). Fitting y = f(x) fails at LE, where dy/dx -> infinity (the
    surface tangent is nearly vertical); a CubicSpline in x then over-
    shoots wildly in the first few x-stations near LE, producing invalid
    y-values, invalid mesh cells, and skewness ~10000 downstream. A
    parametric spline (x(s), y(s)) fitted along accumulated chord length
    s doesn't have that singularity.

    Steps:
      1. splprep fits a periodic-free parametric cubic spline (x(u), y(u))
         through the raw points with u in [0, 1] proportional to
         accumulated chord length.
      2. Cosine-space u: u_i = 0.5 * (1 - cos(pi * i / (n_pts - 1))).
         This clusters u near 0 and 1, so the sampled points cluster near
         the LE (u=0 for LE->TE input) and near the TE.
      3. splev evaluates x(u) and y(u) at the cosine-spaced u's.

    The input should be already LE->TE (project convention).
    """
    from scipy.interpolate import splev, splprep

    # Deduplicate before spline fit -- splprep chokes on repeated points.
    xy = _dedup(xy)
    # k=3 cubic, s=0 interpolating (no smoothing), u parameterized by
    # accumulated chord length (splprep default).
    tck, _u = splprep([xy[:, 0], xy[:, 1]], s=0.0, k=3)
    theta = np.linspace(0.0, np.pi, n_pts)
    u_new = 0.5 * (1.0 - np.cos(theta))  # 0 to 1, cosine-spaced
    x_new, y_new = splev(u_new, tck)
    return np.column_stack([np.asarray(x_new), np.asarray(y_new)])


def plot_airfoil(
    xy_ss: np.ndarray,
    xy_ps: np.ndarray,
    save_path: str | None = None,
    show: bool = False,
    n_te: int = 5,
) -> None:
    """Diagnostic plot of the ingested airfoil.

    Draws:
      * SS (blue) and PS (red), both LE -> TE
      * LE marker (green circle at the min-x point)
      * Max-thickness stations (black squares, connected by a dashed line)
      * Closed-loop overlay (thin gray) with the blunt-TE base points
        (magenta triangles) inserted by `closed_loop`
      * A zoomed inset around the trailing edge

    Matplotlib is imported lazily so the meshing pipeline never pays for it.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loop = closed_loop(xy_ss, xy_ps, n_te=n_te)
    gap = float(np.linalg.norm(xy_ss[-1] - xy_ps[-1]))
    n_base = (n_te - 2) if gap > 1e-10 else 0
    base_pts = loop[-n_base:] if n_base > 0 else None

    rep = geometry_report(xy_ss, xy_ps)
    x_t = rep["x_max_thickness"]
    y_top = float(np.interp(x_t, xy_ss[:, 0], xy_ss[:, 1]))
    y_bot = float(np.interp(x_t, xy_ps[:, 0], xy_ps[:, 1]))

    fig, ax = plt.subplots(figsize=(10, 3.6))

    loop_closed = np.vstack([loop, loop[:1]])
    ax.plot(loop_closed[:, 0], loop_closed[:, 1],
            color="lightgray", lw=0.8, zorder=1, label="closed loop")

    ax.plot(xy_ss[:, 0], xy_ss[:, 1], "b-", lw=1.5,
            label=f"SS (upper), N={len(xy_ss)}")
    ax.plot(xy_ps[:, 0], xy_ps[:, 1], "r-", lw=1.5,
            label=f"PS (lower), N={len(xy_ps)}")

    x_le = float(xy_ss[np.argmin(xy_ss[:, 0]), 0])
    y_le = float(xy_ss[np.argmin(xy_ss[:, 0]), 1])
    ax.plot(x_le, y_le, "go", ms=8, label=f"LE ({x_le:.4f}, {y_le:.4f})")

    ax.plot([x_t, x_t], [y_bot, y_top], "k--", lw=0.8)
    ax.plot([x_t, x_t], [y_top, y_bot], "ks", ms=6,
            label=f"max t/c = {rep['max_thickness']*100:.2f}% @ x/c = {x_t:.3f}")

    if base_pts is not None:
        ax.plot(base_pts[:, 0], base_pts[:, 1], "m^", ms=7,
                label=f"blunt-TE base ({n_base} pts, gap = {gap:.2e} c)")

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title(f"Airfoil ingestion QC  |  chord = {rep['chord']:.4f}  "
                 f"|  TE gap = {rep['te_gap']:.3e} c")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # TE zoom inset --------------------------------------------------------
    x_lo, x_hi = 0.985, 1.005
    m_ss = xy_ss[:, 0] >= x_lo
    m_ps = xy_ps[:, 0] >= x_lo
    ax_te = ax.inset_axes([0.60, 0.60, 0.35, 0.35])
    ax_te.plot(xy_ss[m_ss, 0], xy_ss[m_ss, 1], "b-o", ms=3, lw=1)
    ax_te.plot(xy_ps[m_ps, 0], xy_ps[m_ps, 1], "r-o", ms=3, lw=1)
    if base_pts is not None:
        ax_te.plot(base_pts[:, 0], base_pts[:, 1], "m^", ms=7)
    ax_te.set_xlim(x_lo, x_hi)
    ax_te.set_aspect("equal")
    ax_te.set_title("TE zoom", fontsize=8)
    ax_te.tick_params(labelsize=7)
    ax_te.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"[plot] wrote {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="RAE 2822 ingestion QC")
    parser.add_argument("--plot", action="store_true",
                        help="save a diagnostic PNG of the ingested airfoil")
    parser.add_argument("--show", action="store_true",
                        help="also open the plot interactively (implies --plot)")
    parser.add_argument("--out", default="airfoil_qc.png",
                        help="output file name for --plot (default: %(default)s)")
    args = parser.parse_args()

    profiles = Path(__file__).resolve().parent.parent / "case" / "profiles"
    ps_path = profiles / "RAE2822PS.profile"
    ss_path = profiles / "RAE2822SS.profile"
    for p in (ps_path, ss_path):
        if not p.is_file():
            raise SystemExit(f"[airfoil_io] profile not found: {p}")

    ss, ps = read_profile_pair(str(ps_path), str(ss_path))
    print(json.dumps(geometry_report(ss, ps), indent=2))

    if args.plot or args.show:
        out_path = Path(__file__).resolve().parent / args.out
        plot_airfoil(ss, ps, save_path=str(out_path), show=args.show)
