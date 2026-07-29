#!/usr/bin/env python3
"""
Post-optimization primal-only run at a specific design point.

Loads a `shape.npy` (58-dim) and `aoa.npy` (1-dim) file, pushes them into
the DAFoam model, runs ONE primal solve, and lets DAFoam write the resulting
flow field to `processor*/latestTime/`. No adjoint, no SLSQP, no optimization.

Purpose: reconstruct a specific evaluation's flow field for visualization
after the fact (since DAFoam overwrites processor*/latestTime with each
primal during the optimization loop, only the LAST evaluation's state
survives to disk).

    mpirun -np 4 python -u run_at_shape.py \
        --shape eval9_shape.npy --aoa eval9_aoa.npy --preset benchmark

Prints CD, CL, and constraint values at the end. Compare to opt_summary.md
row to confirm the shape was reapplied correctly.
"""
import argparse

import numpy as np
from mpi4py import MPI

from dafoam_model import OF, build_problem

parser = argparse.ArgumentParser()
parser.add_argument("--shape", required=True, help=".npy file, 58-dim")
parser.add_argument("--aoa",   required=True, help=".npy file, 1-dim")
parser.add_argument("--preset", default="benchmark",
                    choices=["tutorial", "benchmark"])
args = parser.parse_args()

shape = np.load(args.shape)
aoa   = np.load(args.aoa)
assert shape.shape == (58,), f"shape must be (58,), got {shape.shape}"
assert aoa.shape == (1,),    f"aoa must be (1,), got {aoa.shape}"

prob = build_problem(args.preset)

# Push shape and reconstruct the full patchV = [U0, aoa]
U0 = prob.model.params["U0"]
prob.set_val("shape", shape)
prob.set_val("patchV", np.array([U0, float(aoa[0])]))

prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print("\n================ REPLAY RESULTS ================")
    print(f"  preset: {args.preset}")
    print(f"  |shape|:      {np.linalg.norm(shape):.4e}")
    print(f"  aoa:          {float(aoa[0]):.4f} deg")
    for name, path in OF.items():
        val = prob.get_val(path)
        val_str = f"{float(val):.6f}" if val.size == 1 else \
                  f"min={float(np.min(val)):.4f}, max={float(np.max(val)):.4f}"
        print(f"  {name:10s}   {val_str}")
    print("================================================")
    print("Flow field saved to processor*/latestTime/")
    print("Run `reconstructPar -latestTime` next to merge for ParaView.")
