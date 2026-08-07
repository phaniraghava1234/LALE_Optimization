#!/usr/bin/env python3
"""Discrete-adjoint gradient verification against central finite differences.

Two modes are exposed:

    --task check_totals
        OpenMDAO's ``check_totals`` runs central finite differences on
        every design variable and reports the relative error against the
        adjoint totals. Exhaustive but costs 2 x n_DV extra primal
        solves; run on a coarse mesh when practical.

    --task directional  (default)
        A single random-direction dot-product test:

            FD:      (F(x + h d) - F(x - h d)) / (2 h)
            Adjoint: (dF/dx) . d

        Costs only two additional primal solves. Catches sign or
        scaling bugs immediately.

Acceptance guidance: with a tightly converged primal (primalMinResTol,
gmresRelTol both driven low) the relative error typically settles near
1e-4 or below. Looser primal convergence inflates the FD/adjoint
disagreement without indicating a code bug; see the discussion in
``docs/PROJECT_REPORT.md``.

Usage:

    mpirun -np 4 python verify_gradients.py --task directional
    mpirun -np 4 python verify_gradients.py --task check_totals
"""
import argparse

import numpy as np
from mpi4py import MPI

from dafoam_model import OF, WRT, build_problem

comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="directional",
                    choices=["directional", "check_totals"])
parser.add_argument("--preset", default="lale", choices=["lale"])
parser.add_argument("--step", type=float, default=1e-3,
                    help="FD step (also try 1e-2 and 1e-4: the plateau "
                         "between truncation and cancellation error is "
                         "where the comparison is meaningful)")
args = parser.parse_args()

prob = build_problem(args.preset)
prob.run_model()

of_list = [OF["CD"], OF["CL"]]

if args.task == "check_totals":
    # exhaustive check on all DVs -- expensive, coarse mesh recommended
    prob.check_totals(of=of_list, wrt=WRT, compact_print=True,
                      step=args.step, form="central", step_calc="abs")

else:
    # ---- directional derivative test ----
    rng = np.random.default_rng(42)
    shape0 = prob.get_val("shape").copy()
    d = rng.standard_normal(shape0.size)
    d /= np.linalg.norm(d)
    h = args.step * 1e-1  # shape moves are O(1e-2); keep the probe small

    # adjoint totals: one primal (done) + one adjoint per function
    totals = prob.compute_totals(of=of_list, wrt=["shape"])
    adj = {f: float(totals[(f, "shape")] @ d) for f in of_list}

    if comm.rank == 0:
        print("\n---- SANITY CHECK: shape/scaling ----")
        print(f"  shape0 magnitude    = {np.linalg.norm(shape0):.6e}")
        print(f"  h * d magnitude     = {h * np.linalg.norm(d):.6e}")
        print(f"  n_shape             = {shape0.size}")
        for f in of_list:
            row = totals[(f, 'shape')].ravel()
            print(f"  d{f}/dshape norm    = {np.linalg.norm(row):.6e}")
            print(f"  d{f}/dshape max|.|  = {np.max(np.abs(row)):.6e}")

    # central FD: two more primals
    def eval_at(x):
        prob.set_val("shape", x)
        prob.run_model()
        return {f: float(prob.get_val(f)) for f in of_list}

    fp = eval_at(shape0 + h * d)
    fm = eval_at(shape0 - h * d)
    prob.set_val("shape", shape0)

    if comm.rank == 0:
        print("\n========== DIRECTIONAL GRADIENT VERIFICATION ==========")
        print(f"  probe step h = {h:.1e}, random unit direction, seed 42")
        for f in of_list:
            fd = (fp[f] - fm[f]) / (2 * h)
            rel = abs(adj[f] - fd) / max(abs(fd), 1e-16)
            verdict = "PASS" if rel < 5e-3 else "INVESTIGATE"
            print(f"  {f}")
            print(f"     adjoint dF.d = {adj[f]: .8e}")
            print(f"     central  FD  = {fd: .8e}")
            print(f"     rel. error   = {rel: .3e}   [{verdict}]")
        print("=======================================================")
        print("If errors stall around 1e-2: tighten the primal "
              "(primalFuncStdTol) and adjoint (gmresRelTol) tolerances "
              "before blaming the math.")
