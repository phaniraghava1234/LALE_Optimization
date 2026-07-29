#!/usr/bin/env python3
"""
Phase 5.3 -- optimization history from GEMSEO's HDF5 database.

    python plot_history.py --db ../case/opt_history.h5
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gemseo.algos.opt_problem import OptimizationProblem

parser = argparse.ArgumentParser()
parser.add_argument("--db", default="../case/opt_history.h5")
parser.add_argument("--out", default="opt_convergence.png")
args = parser.parse_args()

problem = OptimizationProblem.import_hdf(args.db)
db = problem.database

cd, cl = [], []
for x in db.get_x_history():
    values = db.get(x)
    if "CD" in values:
        cd.append(float(np.atleast_1d(values["CD"])[0]))
        cl.append(float(np.atleast_1d(values.get("CL", np.nan))[0]))

it = np.arange(len(cd))
fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
ax[0].plot(it, np.array(cd) * 1e4, "o-", ms=3)
ax[0].set_ylabel("$C_D$ [counts]")
ax[0].set_title("Adjoint-based optimization convergence")
ax[1].plot(it, cl, "s-", ms=3, color="tab:red")
ax[1].set_ylabel("$C_L$"); ax[1].set_xlabel("function evaluation")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=200)
print(f"[OK] wrote {args.out}  (CD: {cd[0]*1e4:.1f} -> {cd[-1]*1e4:.1f} counts)")
