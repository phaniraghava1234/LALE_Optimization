#!/usr/bin/env python3
"""
Phase 3.3 -- Baseline Primal Run
================================
Runs the forward CFD once on the baseline RAE 2822 mesh and reports
CD / CL plus the geometric constraint values.

    mpirun -np 8 python run_primal.py            # tutorial preset
    mpirun -np 8 python run_primal.py --preset benchmark
    mpirun -np 8 python run_primal.py --trim     # drive alpha to hit CL*

Sanity targets (see README, "Validating the baseline"):
  * tutorial preset : CL trimmed to 0.700
  * benchmark preset: at CL = 0.824, literature CD is ~0.0180-0.0205
    for RANS-SA on this class of mesh (He et al. 2019 report
    CD ~ 199 counts baseline). A strong suction-side shock near
    x/c ~ 0.55 must be visible in the Cp/Mach fields.
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
