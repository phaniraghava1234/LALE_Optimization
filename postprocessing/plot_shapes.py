#!/usr/bin/env python3
"""
Phase 5.3 -- compare the baseline and optimized airfoil shapes.

The optimized surface is read from the deformed OpenFOAM mesh
(the wing patch points after the final IDWarp), the baseline from
the original .profile files.

    python plot_shapes.py --case ../case
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
                                "..", "mesh"))
from airfoil_io import read_profile_pair  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="../case")
parser.add_argument("--out", default="shape_comparison.png")
args = parser.parse_args()

ss, ps = read_profile_pair(
    os.path.join(args.case, "profiles", "RAE2822PS.profile"),
    os.path.join(args.case, "profiles", "RAE2822SS.profile"),
)

foam = os.path.join(args.case, "case.foam")
open(foam, "a").close()
reader = pv.POpenFOAMReader(foam)
reader.set_active_time_value(reader.time_values[-1])
wing = reader.read()["boundary"]["wing"]
pts = wing.points
mask = np.abs(pts[:, 2]) < 1e-6
xo, yo = pts[mask, 0], pts[mask, 1]
order = np.argsort(np.arctan2(yo - 0.0, xo - 0.5))  # angular sort around mid-chord

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(ss[:, 0], ss[:, 1], "k-", lw=1, label="baseline RAE 2822")
ax.plot(ps[:, 0], ps[:, 1], "k-", lw=1)
ax.plot(xo[order], yo[order], "r--", lw=1.2, label="optimized")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
ax.set_title("Shape change: shock-weakening recamber expected near x/c ~ 0.5-0.7")
fig.tight_layout(); fig.savefig(args.out, dpi=200)
print(f"[OK] wrote {args.out}")
