#!/usr/bin/env python3
"""Standalone baseline primal RANS solve for the RAE 2822 case.

Runs the forward compressible RANS solver once on the baseline (undeformed)
mesh at the operating-point preset selected on the command line and
reports the resulting force coefficients (CD, CL) and the values of the
geometric constraints (thickness, volume, LE radius).

    mpirun -np 4 python run_primal.py                 # tutorial preset
    mpirun -np 4 python run_primal.py --preset benchmark
    mpirun -np 4 python run_primal.py --trim          # secant on alpha until CL = CL*

Baseline sanity ranges:

    tutorial preset:
        CL is trimmed to 0.700 (with --trim) or read at aoa=2.525 deg.
    benchmark preset (He et al. 2019, ADODG Case 2):
        M=0.729, Re=6.5e6, CL* = 0.824. Literature baseline CD is
        approximately 199 drag counts. A distinct suction-side shock at
        x/c ~ 0.55 must appear in the Cp / Mach field.
"""
import argparse

from mpi4py import MPI

from dafoam_model import OF, build_problem, make_options
from dafoam.mphys import OptFuncs

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="tutorial",
                    choices=["tutorial", "benchmark"])
parser.add_argument("--trim", action="store_true",
                    help="adjust alpha (secant iterations) until CL = CL_target")
args = parser.parse_args()

prob = build_problem(args.preset)
daOptions, _, params = make_options(args.preset)

if args.trim:
    optFuncs = OptFuncs(daOptions, prob)
    optFuncs.findFeasibleDesign(
        [OF["CL"]], ["patchV"], targets=[params["CL_target"]],
        designVarsComp=[1], epsFD=[1e-1],
    )
else:
    prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print("\n================ BASELINE RESULTS ================")
    for name, path in OF.items():
        print(f"  {name:10s} = {prob.get_val(path)}")
    print(f"  patchV [U, alpha_deg] = {prob.get_val('patchV')}")
    print("==================================================")
