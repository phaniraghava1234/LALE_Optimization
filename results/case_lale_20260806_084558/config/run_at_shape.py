#!/usr/bin/env python3
"""Post-optimization primal-only run at a specific design point (LALE).

Loads a shape.npy (58-dim design vector) file, pushes it into the DAFoam
model, runs ONE primal solve, and lets DAFoam write the resulting flow
field to ``processor*/latestTime/``. No adjoint, no SLSQP, no optimization.

Purpose: reconstruct a specific evaluation's flow field for visualisation
after the fact (DAFoam overwrites ``processor*/latestTime`` with each
primal during the optimization loop, so only the LAST evaluation's
state survives to disk).

Alpha is fixed at 4.5 deg via the case BC files; there is no separate
--aoa argument as in the transonic case_lale.

    mpirun -np 4 python -u run_at_shape.py --shape eval_best_shape.npy

Prints CD, CL, and constraint values at the end. Compare to opt_summary.md
row to confirm the shape was reapplied correctly.
"""
import argparse

import numpy as np
from mpi4py import MPI

from dafoam_model import OF, build_problem

parser = argparse.ArgumentParser()
parser.add_argument("--shape", required=True, help=".npy file, 58-dim")
parser.add_argument("--preset", default="lale", choices=["lale"])
args = parser.parse_args()

shape = np.load(args.shape)
assert shape.shape == (58,), f"shape must be (58,), got {shape.shape}"

prob = build_problem(args.preset)
prob.set_val("shape", shape)
prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print("\n================ REPLAY RESULTS ================")
    print(f"  preset: {args.preset}")
    print(f"  |shape|:      {np.linalg.norm(shape):.4e}")
    print(f"  alpha:        4.5 deg (fixed via 0/U BC)")
    for name, path in OF.items():
        val = prob.get_val(path)
        val_str = f"{float(val):.6f}" if val.size == 1 else \
                  f"min={float(np.min(val)):.4f}, max={float(np.max(val)):.4f}"
        print(f"  {name:10s}   {val_str}")
    print("================================================")
    print("Flow field saved to processor*/latestTime/")
    print("Run `reconstructPar -latestTime` next to merge for ParaView.")
