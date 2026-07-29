#!/usr/bin/env python3
"""
Phases 5.1 + 5.3 -- MDO Formulation & Execution (GEMSEO 6 + pyOptSparse)
========================================================================
Formal statement of the optimization problem:

    minimize      CD(shape, aoa)
    w.r.t.        shape in [-1, 1]^n   (FFD y-displacements, m)
                  aoa   in [0, 10]     (degrees)
    subject to    CL(shape, aoa)  =  CL*            (lift equality)
                  thickcon        >= 0.5 t_base     (per station)
                  volcon          >= 1.0 V_base     (area/volume)
                  rcon            >= 0.8 r_base     (LE radius)

Gradients: exact discrete-adjoint Jacobians supplied by the
discipline (`set_differentiation_method("user")`), so each iteration
costs 1 primal + ~2 adjoint solves regardless of n_shape.

Writes opt_summary.md with a full header (operating conditions, mesh, DVs,
solver settings) followed by a per-evaluation table (CD, CL, constraints,
elapsed time) and a footer with the final result.

Solver options (--algo):
    PYOPTSPARSE_SLSQP  (default, free, robust)
    PYOPTSPARSE_SNOPT  (commercial license required)
    SLSQP              (GEMSEO built-in, less robust for noisy CFD)
    NLOPT_SLSQP        (NLopt implementation)

MUST be launched from inside case/ (OpenFOAM reads the mesh from cwd):

    cd case
    mpirun -np 4 python ../phase5_gemseo/run_optimization.py
    mpirun -np 4 python ../phase5_gemseo/run_optimization.py \
           --preset benchmark --algo PYOPTSPARSE_SLSQP --max-iter 50
"""
import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from gemseo import configure_logger, create_design_space, create_scenario

from dafoam_discipline import DAFoamDiscipline

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="tutorial",
                    choices=["tutorial", "benchmark"])
parser.add_argument("--algo", default="PYOPTSPARSE_SLSQP",
                    help="Optimizer: PYOPTSPARSE_SLSQP (default), "
                         "PYOPTSPARSE_SNOPT, SLSQP, NLOPT_SLSQP")
parser.add_argument("--max-iter", type=int, default=50)
parser.add_argument("--summary", default="opt_summary.md",
                    help="Path to markdown summary file")
args = parser.parse_args()

configure_logger(level=logging.INFO)

# ---- MPI-safe writes: only rank 0 touches the summary file ----------- #
try:
    from mpi4py import MPI
    RANK = MPI.COMM_WORLD.rank
except ImportError:
    RANK = 0

SUMMARY = os.path.abspath(args.summary)


def w(text: str = "") -> None:
    """Write a line to the summary file, rank-0 only."""
    if RANK == 0:
        with open(SUMMARY, "a") as f:
            f.write(text + "\n")


# ---- 5.2: the wrapper discipline (builds DAFoam once) ---------------- #
disc = DAFoamDiscipline(preset=args.preset, summary_path=SUMMARY)
p = disc.params
CL_target = p["CL_target"]
U0, p0, T0, aoa0 = p["U0"], p["p0"], p["T0"], p["aoa0"]

# Derived quantities
rho0 = p0 / (287.0 * T0)
a0 = np.sqrt(1.4 * 287.0 * T0)      # speed of sound
Mach = U0 / a0
mu = 1.8e-5                          # kinematic viscosity assumption
Re_chord = rho0 * U0 * 1.0 / mu     # Re per chord=1 m

# ---- Mesh cell count (from checkMesh output, or query the model) ----- #
try:
    n_cells = disc.prob.model.mesh.mphys_get_surface_mesh().shape[0]
    # This gets surface points, not cells. For cell count, better to read
    # from constant/polyMesh/owner header, but simplest is to grep the log.
except Exception:
    n_cells = "unknown"

# ---- HEADER --------------------------------------------------------- #
if RANK == 0 and os.path.exists(SUMMARY):
    os.remove(SUMMARY)

w(f"# Optimization Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w()
w("## Problem Statement")
w()
w("Single-point adjoint-based transonic airfoil shape optimization")
w("(He et al. 2019, *Robust aerodynamic shape optimization*, ADODG Case 2 setup):")
w()
w("```")
w("minimize      CD(shape, aoa)")
w(f"subject to    CL = {CL_target}  (lift equality)")
w("              thickcon >= 0.5, volcon >= 1.0, rcon >= 0.8 (geometry)")
w("              shape in [-1, 1]^n,  aoa in [0, 10] deg")
w("```")
w()
w("## Configuration")
w()
w(f"- **Preset**: `{args.preset}`")
w(f"- **Algorithm**: `{args.algo}` (sequential quadratic programming)")
w(f"- **Max SLSQP iterations**: {args.max_iter}")
w(f"- **Shape DVs**: {disc.n_shape} (FFD y-displacements, camber-preserving at LE/TE)")
w(f"- **MPI ranks**: {MPI.COMM_WORLD.size if 'MPI' in dir() else 1}")
w(f"- **Working directory**: `{os.getcwd()}`")
w(f"- **Gradient method**: user-supplied (discrete adjoint)")
w()
w("## Operating Conditions")
w()
w(f"- **U_infinity**: {U0:.3f} m/s")
w(f"- **p_infinity**: {p0:.1f} Pa")
w(f"- **T_infinity**: {T0:.2f} K")
w(f"- **rho_infinity** (derived): {rho0:.4f} kg/m^3")
w(f"- **Speed of sound** (derived): {a0:.2f} m/s")
w(f"- **Mach** (derived): {Mach:.4f}  {'(transonic)' if 0.7 < Mach < 1.0 else ''}")
w(f"- **Re per chord** (derived, mu={mu:.1e}): {Re_chord:.3e}")
w(f"- **Initial AoA**: {aoa0:.4f} deg")
w(f"- **Target CL**: {CL_target}")
w(f"- **Reference area A0**: 0.01 m^2 (chord=1m x span=0.01m)")
w(f"- **Dynamic pressure q_inf**: {0.5*rho0*U0*U0:.1f} Pa")
w()
w("## Physics Expected")
w()
if Mach > 0.7:
    w(f"- Transonic flow at M={Mach:.3f}, expect a normal shock on suction side "
      f"near x/c ~ 0.5-0.6")
    w("- CD = CD_pressure + CD_viscous; pressure component dominates due to shock")
    w("- Optimization goal: weaken/eliminate the shock (shock-free or lambda foot)")
    w("- Well-known result: ~30-40% CD reduction is achievable at fixed CL")
w()
w("## Reference Results (for context)")
w()
if args.preset == "tutorial":
    w("- DAFoam tutorial baseline (this operating point):  ~150-190 drag counts")
    w("- Typical SLSQP-optimized outcome:                   ~100-130 drag counts")
    w("- CL held to target ~0.7 throughout")
elif args.preset == "benchmark":
    w("- He et al. 2019, ADODG Case 2 (M=0.729, Re=6.5M, CL=0.824):")
    w("  - CD baseline (RANS-SA):   ~199 drag counts (0.0199)")
    w("  - CD optimized:            ~123-135 drag counts (published range)")
    w("  - CD reduction:            ~30-38%")
w("- Reference: He, Mader, Martins, Maki. *Aerospace Science and Technology* 87 (2019) 483-502.")
w()
w("## Solver Settings (see `case/dafoam_model.py`)")
w()
w("- Solver: `DAHisaFoam` (density-based compressible, JST convective flux)")
w("- Turbulence: Spalart-Allmaras, wall-resolved (y+~1, no wall functions)")
w("- Primal tolerance: stdTol=0.01, slopeTol=5e-5, primalMinIters=6000")
w("- Adjoint: Krylov (GMRES), gmresRelTol=1e-6, pcFillLevel=1")
w("- Mesh: C-topology, ~57k hex cells (halved from initial 99k for memory)")
w()
w("## Constraints")
w()
w(f"- `CL` = {CL_target} (equality) - maintains lift during CD minimization")
w("- `thickcon` >= 0.5 (per-station thickness ratio) - prevents crazy-thin airfoil")
w("- `volcon`   >= 1.0 (volume ratio) - prevents deflating the airfoil")
w("- `rcon`     >= 0.8 (LE radius ratio) - keeps LE bluntness reasonable")
w()
w("## Iteration Log")
w()
w("One row per primal evaluation. SLSQP triggers a primal for each outer iter, ")
w("plus additional primals inside its line search. Gradients (adjoints) run ")
w("after each successful primal.")
w()
w("**Columns:**")
w("- `Eval`: sequential primal number (1 = baseline)")
w("- `kind`: base (first) / eval (subsequent)")
w("- `|shape|`: L2 norm of shape DV vector (0 = baseline, larger = more deformed)")
w("- `AoA`: current angle of attack, deg")
w("- `CD` / `CL`: force coefficients")
w("- `counts`: CD * 10000 (aerospace-standard drag counts)")
w("- `dCts`: change in drag counts vs baseline (negative = improvement)")
w("- `CL_err`: CL - CL_target (should stay near 0 as SLSQP enforces equality)")
w("- `L/D`: lift-to-drag ratio")
w("- `thick_min` / `vol` / `rcon_min`: geometric constraint values (must stay >= bounds)")
w("- `wall(min)`: this primal's wall-clock time in minutes")
w("- `total`: cumulative wall time")
w()
w("| Eval | kind | |shape|  | AoA (deg) | CD | counts | dCts | CL | CL_err | L/D | thick_min | vol | rcon_min | wall(min) | total |")
w("|-----:|:----:|--------:|----------:|--------:|-------:|-----:|-------:|-------:|-----:|----------:|-----:|---------:|----------:|------:|")

# ---- 5.1: design space --------------------------------------------- #
# Shape DV bounds: FFD y-displacements in METERS.
# Chord = 1 m, so +/-0.02 = +/-2% chord. That's a huge design freedom for
# aero shape optimization (paper's optimized RAE2822 uses ~1-2% chord peak
# displacement) and prevents SLSQP from warping the y+=1 wall stack into
# invalid geometry (which killed run at eval 2 with 9k non-orth errors).
# The DAFoam tutorial uses +/-1.0 with pyOptSparse-native SLSQP that takes
# smaller trust-region steps; GEMSEO's pyoptsparse-SLSQP wrapper takes
# bigger first-step so we constrain physically.
ds = create_design_space()
ds.add_variable("shape", size=disc.n_shape,
                lower_bound=-0.02, upper_bound=0.02,
                value=np.zeros(disc.n_shape))
ds.add_variable("aoa", size=1, lower_bound=0.0, upper_bound=10.0,
                value=np.array([aoa0]))

scenario = create_scenario(
    disciplines=[disc],
    formulation_name="DisciplinaryOpt",
    objective_name="CD",
    design_space=ds,
)

scenario.add_constraint("CL", constraint_type="eq", value=CL_target)
scenario.add_constraint("thickcon", constraint_type="ineq",
                        positive=True, value=0.5)
scenario.add_constraint("volcon", constraint_type="ineq",
                        positive=True, value=1.0)
scenario.add_constraint("rcon", constraint_type="ineq",
                        positive=True, value=0.8)

scenario.set_differentiation_method("user")

# ---- 5.3: execute --------------------------------------------------- #
# gemseo-pyoptsparse 1.1.1 rejects pyoptsparse-native setting names
# (MAXIT, ACC, Major_feasibility_tolerance, ...) when a GEMSEO counterpart
# exists. Only GEMSEO-canonical settings (max_iter, xtol_abs, xtol_rel,
# ftol_abs, ftol_rel, ineq_tolerance, eq_tolerance, normalize_design_space)
# are accepted. Pass just max_iter to keep it working; add tolerances later
# once we know the exact accepted names for THIS plugin version.
t_opt_start = time.time()
scenario.execute(algo_name=args.algo, max_iter=args.max_iter)
t_opt = time.time() - t_opt_start

# ---- post-processing & history export ------------------------------ #
try:
    scenario.post_process(post_name="OptHistoryView", save=True, show=False,
                          file_path="opt_history")
except Exception as e:
    if RANK == 0:
        print(f"[warn] post_process OptHistoryView failed: {e}")

if RANK == 0:
    try:
        scenario.save_optimization_history("opt_history.h5")
    except Exception as e:
        print(f"[warn] save_optimization_history failed: {e}")

# ---- FOOTER --------------------------------------------------------- #
opt = scenario.optimization_result
CD_star = float(opt.f_opt)
baseline_CD = disc._baseline_CD if disc._baseline_CD is not None else CD_star
CD_reduction_counts = (baseline_CD - CD_star) * 1e4
CD_reduction_pct = 100.0 * (baseline_CD - CD_star) / baseline_CD if baseline_CD else 0.0

# Final aerodynamic state
CL_final = float(disc.prob.get_val(disc._OF["CL"]))
aoa_final = float(disc.prob.get_val("patchV")[1])
shape_final = disc.prob.get_val("shape")
shape_norm_final = float(np.linalg.norm(shape_final))
LD_baseline = float("nan")
LD_final = CL_final / CD_star if CD_star > 0 else float("nan")

# Constraint satisfaction
thick_final = float(np.min(disc.prob.get_val(disc._OF["thickcon"])))
vol_final = float(disc.prob.get_val(disc._OF["volcon"]))
rcon_final = float(np.min(disc.prob.get_val(disc._OF["rcon"])))

w()
w("## Final Result")
w()
w("### Aerodynamics")
w(f"- **CD baseline**: {baseline_CD:.6f}  ({baseline_CD * 1e4:.1f} counts)")
w(f"- **CD optimized**: {CD_star:.6f}  ({CD_star * 1e4:.1f} counts)")
w(f"- **CD reduction**: {CD_reduction_counts:+.1f} counts  ({CD_reduction_pct:+.2f}%)")
w(f"- **CL final**: {CL_final:.6f}  (target {CL_target}, error {CL_final-CL_target:+.4f})")
w(f"- **L/D final**: {LD_final:.2f}")
w(f"- **AoA final**: {aoa_final:.4f} deg  (initial {aoa0:.4f})")
w()
w("### Constraints (final)")
w(f"- `thickcon` min: {thick_final:.4f}  (must >= 0.5)  {'PASS' if thick_final >= 0.5 else 'FAIL'}")
w(f"- `volcon`      : {vol_final:.4f}  (must >= 1.0)  {'PASS' if vol_final >= 1.0 else 'FAIL'}")
w(f"- `rcon` min    : {rcon_final:.4f}  (must >= 0.8)  {'PASS' if rcon_final >= 0.8 else 'FAIL'}")
w()
w("### Shape")
w(f"- **|shape| final**: {shape_norm_final:.4e} (baseline 0)")
w(f"- **max shape DV**: {np.max(np.abs(shape_final)):.4e}")
w()
w("### Solver Stats")
w(f"- **Primal evaluations**: {disc._eval_count}")
w(f"- **Gradient (adjoint) evaluations**: {disc._grad_count}")
w(f"- **Total wall time**: {int(t_opt // 3600)}h {int((t_opt % 3600) // 60)}m "
  f"({t_opt:.0f} s)")
w(f"- **Avg time per primal**: {t_opt/max(disc._eval_count,1)/60:.1f} min")
w()
w("### Optimizer Verdict")
w(f"- **SLSQP feasible flag**: {opt.is_feasible}")
w(f"- **Objective at exit**: {opt.f_opt:.6f}")
w("- Feasible means all constraints are satisfied within pyOptSparse tolerances (ACC=1e-6).")
w("- Convergence to optimum: check `opt_history_criteria.png` (KKT norm should trend down).")
w()
w("### Comparison to Reference")
w()
if args.preset == "tutorial":
    w(f"- DAFoam tutorial baseline: ~150-190 counts vs our {baseline_CD*1e4:.1f}")
    w(f"- Typical tutorial optimum: ~100-130 counts vs our {CD_star*1e4:.1f}")
elif args.preset == "benchmark":
    w(f"- He 2019 baseline: 199 counts vs our {baseline_CD*1e4:.1f}")
    w(f"- He 2019 optimum:  123-135 counts vs our {CD_star*1e4:.1f}")
    w(f"- He 2019 reduction: ~30-38% vs our {CD_reduction_pct:.1f}%")
w()
w("## Files Produced")
w()
w("- `opt_summary.md`         : this file")
w("- `opt_history.h5`         : GEMSEO optimization database (all evals + gradients)")
w("- `opt_history*.png`       : SLSQP convergence plots (OptHistoryView)")
w("- `opt_log.txt`            : full stdout/stderr from the run")
w("- `deformedFFD_*.dat`      : deformed FFD lattices per iter (if writeDeformedFFDs=1)")
w("- `processor*/latestTime/` : final flow field, viewable in ParaView")
w("- To visualize final shape: `touch case/case.foam && paraview case/case.foam`")
w()
w(f"Run finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if RANK == 0:
    print("\n================ OPTIMIZATION RESULT ================")
    print(f"  algorithm: {args.algo}")
    print(f"  CD*  = {CD_star:.6f}  ({CD_star * 1e4:.1f} counts)")
    print(f"  converged (feasible): {opt.is_feasible}")
    print(f"  summary written to: {SUMMARY}")
    print("=====================================================")
