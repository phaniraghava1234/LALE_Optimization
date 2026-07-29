#!/usr/bin/env python3
"""
Cp distribution on the wing patch (baseline validation, Phase 3.3).

    Cp = (p - p_inf) / (0.5 * rho_inf * U_inf^2)

Reads the latest OpenFOAM time directory with pyvista's OpenFOAM reader
(pip install pyvista inside the DAFoam environment).

    python plot_cp.py --case ../case --preset tutorial
Overlay a reference: put two columns (x/c, Cp) in a text file and pass
--ref cook_case6.dat (Cook et al. 1979, AGARD AR-138, Case 6 is the
classic experimental comparison for M=0.729).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "case"))
from dafoam_model import PRESETS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="../case")
parser.add_argument("--preset", default="tutorial")
parser.add_argument("--ref", default=None, help="optional x/c,Cp reference file")
parser.add_argument("--out", default="cp_distribution.png")
args = parser.parse_args()

p = PRESETS[args.preset]
rho_inf = p["p0"] / (287.0 * p["T0"])
q_inf = 0.5 * rho_inf * p["U0"] ** 2

# --- load latest time, wing patch ---
foam = os.path.join(args.case, "case.foam")
open(foam, "a").close()
reader = pv.POpenFOAMReader(foam)
reader.set_active_time_value(reader.time_values[-1])
reader.cell_to_point_creation = True
mesh = reader.read()
wing = mesh["boundary"]["wing"]

pts = wing.points
pfield = wing.point_data["p"]
mask = np.abs(pts[:, 2]) < 1e-6          # z = 0 plane only
x, y = pts[mask, 0], pts[mask, 1]
cp = (pfield[mask] - p["p0"]) / q_inf

# split upper/lower by camber-free heuristic (y sign relative to chord line)
upper = y >= np.interp(x, [x.min(), x.max()], [0, 0])
order_u, order_l = np.argsort(x[upper]), np.argsort(x[~upper])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x[upper][order_u], cp[upper][order_u], "b-", lw=1.5, label="upper (SS)")
ax.plot(x[~upper][order_l], cp[~upper][order_l], "r-", lw=1.5, label="lower (PS)")
if args.ref:
    ref = np.loadtxt(args.ref)
    ax.plot(ref[:, 0], ref[:, 1], "ko", ms=3, mfc="none", label="reference")
ax.invert_yaxis()
ax.axhline(0, color="k", lw=0.5)
# sonic Cp*: Cp* = (2/(g M^2)) [ ((2 + (g-1)M^2)/(g+1))^(g/(g-1)) - 1 ]
g, M = 1.4, p["U0"] / np.sqrt(1.4 * 287.0 * p["T0"])
cp_star = 2 / (g * M**2) * (((2 + (g - 1) * M**2) / (g + 1)) ** (g / (g - 1)) - 1)
ax.axhline(cp_star, color="g", ls="--", lw=1, label=f"Cp* (sonic) = {cp_star:.2f}")
ax.set_xlabel("x/c"); ax.set_ylabel("$C_p$")
ax.set_title(f"RAE 2822  M={M:.3f}  --  latest time")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=200)
print(f"[OK] wrote {args.out}")
