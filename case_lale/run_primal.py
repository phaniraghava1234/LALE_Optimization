#!/usr/bin/env python3
"""Standalone baseline primal RANS solve for the MH139-F LALE case.

Runs the forward incompressible RANS solver once on the baseline
(undeformed) MH139-F mesh at alpha = 4.5 deg (fixed) and reports the
resulting force coefficients (CD, CL) plus the geometric constraint
values (thickness, volume, LE radius).

    mpirun -np 4 python run_primal.py

Baseline sanity ranges (SA fully turbulent, unit-chord representation
of the physical MH139-F chord = 0.305 m airfoil):

    CD approximately 100-250 counts. SA overpredicts CD compared to
    XFOIL because it cannot represent the laminar separation bubble on
    MH139-F at Re = 1.7e5.

    CL approximately 0.6-0.9 at alpha = 4.5 deg. The target CL* = 0.975
    (from level-flight equilibrium of the AtlantikSolar UAV) is
    intentionally higher than what the baseline shape produces; the
    optimizer must add camber to reach it.
"""
import argparse

from mpi4py import MPI

from dafoam_model import OF, build_problem

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="lale", choices=["lale"])
args = parser.parse_args()

prob = build_problem(args.preset)
prob.run_model()

if MPI.COMM_WORLD.rank == 0:
    print("\n================ BASELINE RESULTS ================")
    for name, path in OF.items():
        print(f"  {name:10s} = {prob.get_val(path)}")
    print("==================================================")
