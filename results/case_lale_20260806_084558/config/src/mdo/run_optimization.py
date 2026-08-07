#!/usr/bin/env python3
"""MDO scenario driver for the adjoint-based airfoil optimization.

Formal problem statement (variables depend on preset — see below):

    minimize      C_D(shape, aoa)
    w.r.t.        shape in [-0.02, 0.02]^n   (FFD y-displacements, m)
                  aoa   in [0, 10]           (degrees, transonic only)
    subject to    C_L(shape, aoa)  =  C_L*   (lift equality)
                  thickcon         >= 0.5    (per-station thickness ratio)
                  volcon           >= 1.0    (airfoil-area ratio)
                  rcon             >= 0.8    (LE radius ratio)

Presets:

    tutorial   compressible RAE 2822 at sea level (DAFoam tutorial)
    benchmark  compressible RAE 2822 at He 2019 ADODG Case 2 conditions
    lale       incompressible MH139-F at LALE cruise (alpha fixed 4.5 deg,
               no aoa design variable, uses case_lale/)

Gradients are supplied by the discrete adjoint of the RANS equations
(via DAFoam) and delivered to GEMSEO through the user-supplied Jacobian
pathway (``set_differentiation_method("user")``). Each SLSQP outer
iteration costs one primal solve plus roughly one adjoint solve per
active functional, independent of the design-variable count.

Supported optimizers (``--algo``):

    PYOPTSPARSE_SLSQP   default; free; well-tested for aero shape opt
    PYOPTSPARSE_SNOPT   requires a commercial SNOPT license
    SLSQP               GEMSEO built-in wrapping scipy SLSQP
    NLOPT_SLSQP         NLopt implementation

Must be launched from inside the case directory because OpenFOAM reads
the mesh from the current working directory:

    cd case          (or case_lale)
    mpirun -np 4 python ../mdo/run_optimization.py --preset benchmark
    mpirun -np 4 python ../mdo/run_optimization.py --preset lale
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
                    choices=["tutorial", "benchmark", "lale"])
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
    NRANKS = MPI.COMM_WORLD.size
except ImportError:
    RANK = 0
    NRANKS = 1

SUMMARY = os.path.abspath(args.summary)

# Per-evaluation design vectors (small .npy files).
SHAPE_DIR = os.path.abspath("shapes")

# Per-evaluation full state: deformed mesh points, flow fields and FFD
# control points, one subdirectory per evaluation. DAFoam only preserves a
# solution when it runs an adjoint, so line-search evaluations would
# otherwise be overwritten; writing from the discipline's own per-primal
# hook captures all of them. Saving the mesh (rather than only the shape)
# also means a later replay loads the exact geometry instead of recreating
# it by re-warping.
SAVE_DIR = os.path.abspath("saved")


def w(text: str = "") -> None:
    """Write a line to the summary file, rank-0 only."""
    if RANK == 0:
        with open(SUMMARY, "a") as f:
            f.write(text + "\n")


# ---- Preset-specific flags -------------------------------------------- #
IS_LALE = (args.preset == "lale")

# ---- Build the wrapper discipline (builds DAFoam once) --------------- #
disc = DAFoamDiscipline(preset=args.preset, summary_path=SUMMARY,
                        fixed_aoa=IS_LALE, shape_dir=SHAPE_DIR,
                        save_dir=SAVE_DIR)
p = disc.params
CL_target = p["CL_target"]
U0, aoa0 = p["U0"], p["aoa0"]

# Derived operating-point quantities.
# Compressible presets store p0, T0; incompressible stores rho0, nu directly.
if IS_LALE:
    rho0 = p["rho0"]
    nu = p["nu"]
    mu = rho0 * nu
    T0 = float("nan")            # no thermodynamics in incompressible
    p0 = float("nan")
    a0 = float("nan")
    Mach = U0 / 340.3            # sea-level speed of sound for a rough M ref
else:
    p0, T0 = p["p0"], p["T0"]
    rho0 = p0 / (287.0 * T0)
    a0 = np.sqrt(1.4 * 287.0 * T0)
    Mach = U0 / a0
    mu = 1.8e-5                  # compressible presets assume dry-air value
    nu = mu / rho0

Re_chord = rho0 * U0 * 1.0 / mu  # Re per unit chord in the CFD

# ---- HEADER --------------------------------------------------------- #
if RANK == 0 and os.path.exists(SUMMARY):
    os.remove(SUMMARY)

w(f"# Optimization Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
w()
w("## Problem Statement")
w()
if IS_LALE:
    w("Single-point adjoint-based low-Reynolds airfoil shape optimization")
    w("(Oettershagen et al. 2017, AtlantikSolar UAV cruise condition):")
    w()
    w("```")
    w("minimize      C_D(shape)                     (alpha fixed at 4.5 deg)")
    w(f"subject to    C_L = {CL_target}  (lift equality)")
    w("              thickcon >= 0.5, volcon >= 1.0, rcon >= 0.8")
    w("              shape in [-0.02, 0.02]^n   (m; ~2% chord)")
    w("```")
else:
    w("Single-point adjoint-based transonic airfoil shape optimization")
    w("(He et al. 2019, *Robust aerodynamic shape optimization*, ADODG Case 2):")
    w()
    w("```")
    w("minimize      C_D(shape, aoa)")
    w(f"subject to    C_L = {CL_target}  (lift equality)")
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
w(f"- **Alpha**: {'fixed at ' + str(aoa0) + ' deg (NOT a DV)' if IS_LALE else 'design variable (initial ' + str(aoa0) + ' deg)'}")
w(f"- **MPI ranks**: {NRANKS}")
w(f"- **Working directory**: `{os.getcwd()}`")
w(f"- **Gradient method**: user-supplied (discrete adjoint)")
w()
w("## Operating Conditions")
w()
w(f"- **U_infinity**: {U0:.3f} m/s (CFD representation)")
if IS_LALE:
    w(f"- **rho_infinity**: {rho0:.4f} kg/m^3 (500 m ISA)")
    w(f"- **nu**: {nu:.4e} m^2/s")
    w(f"- **Mach**: {Mach:.4f} (incompressible regime, fully incompressible solver)")
else:
    w(f"- **p_infinity**: {p0:.1f} Pa")
    w(f"- **T_infinity**: {T0:.2f} K")
    w(f"- **rho_infinity** (derived): {rho0:.4f} kg/m^3")
    w(f"- **Speed of sound** (derived): {a0:.2f} m/s")
    w(f"- **Mach** (derived): {Mach:.4f}  {'(transonic)' if 0.7 < Mach < 1.0 else ''}")
w(f"- **Re per chord** (unit-chord CFD): {Re_chord:.3e}")
w(f"- **Initial AoA**: {aoa0:.4f} deg")
w(f"- **Target CL**: {CL_target}")
w(f"- **Reference area A0**: 0.01 m^2 (chord=1m x span=0.01m)")
w(f"- **Dynamic pressure q_inf**: {0.5*rho0*U0*U0:.4f} Pa")
w()
w("## Physics Expected")
w()
if IS_LALE:
    w(f"- Low-Reynolds incompressible flow, Re = {Re_chord:.2e}, M = {Mach:.3f}")
    w("- Fully turbulent SA (no laminar-transition model); absolute CD is over-predicted")
    w("  compared to XFOIL because the SA closure cannot represent the laminar")
    w("  separation bubble known to exist on the MH-series at these conditions.")
    w("- Optimization signal (gradient) remains valid; the optimizer will find shape")
    w("  moves that reduce CD at the fixed CL target.")
elif Mach > 0.7:
    w(f"- Transonic flow at M = {Mach:.3f}, expect a normal shock on suction side "
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
elif args.preset == "lale":
    w("- Reference aircraft: AtlantikSolar LALE UAV (Oettershagen et al. 2017).")
    w("- Section-CL target 0.975 is derived from level-flight equilibrium of")
    w("  the aircraft at 500 m ISA, U = 8.30 m/s (measured cruise from the")
    w("  81-hour endurance flight), c = 0.305 m, S = 1.735 m^2, W = 67.98 N.")
    w("- No published CD target for the SECTION (paper reports whole-aircraft")
    w("  power); this study is a demonstrator of the pipeline at LALE Re, not")
    w("  a paper-reproduction case.")
    w("- Reference: Oettershagen et al. *Journal of Field Robotics* 34(7) (2017) 1352-1385.")
w()
w("## Solver Settings")
w()
if IS_LALE:
    w("- Solver: `DASimpleFoam` (incompressible SIMPLE)")
    w("- Turbulence: Spalart-Allmaras, wall-resolved (y+~1)")
    w("- Primal tolerance: stdTol=1e-3, slopeTol=5e-6, primalMinIters=8000")
    w("- Adjoint: Krylov (GMRES), gmresRelTol=1e-6, pcFillLevel=1")
    w("- Mesh: C-topology, ~32-45k hex cells")
else:
    w("- Solver: `DAHisaFoam` (density-based compressible, JST convective flux)")
    w("- Turbulence: Spalart-Allmaras, wall-resolved (y+~1, no wall functions)")
    w("- Primal tolerance: stdTol=0.01, slopeTol=5e-5, primalMinIters=8000")
    w("- Adjoint: Krylov (GMRES), gmresRelTol=1e-6, pcFillLevel=1")
    w("- Mesh: C-topology, ~57k hex cells")
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
w("- `AoA`: current angle of attack, deg (fixed for LALE, DV for transonic)")
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

# ---- Design space ---------------------------------------------------- #
# Shape DV bounds: FFD y-displacements in METERS.
# Chord = 1 m in the CFD, so +/-0.02 = +/-2% chord.
ds = create_design_space()
ds.add_variable("shape", size=disc.n_shape,
                lower_bound=-0.02, upper_bound=0.02,
                value=np.zeros(disc.n_shape))
if not IS_LALE:
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

# ---- execute --------------------------------------------------------- #
# gemseo-pyoptsparse 1.1.1 rejects pyoptsparse-native setting names
# (MAXIT, ACC, ...) when a GEMSEO counterpart exists. Only GEMSEO-canonical
# settings (max_iter, xtol_rel, ftol_rel, eq_tolerance, ineq_tolerance,
# normalize_design_space) are accepted. We pass only max_iter here.
# ---- incremental history backup ------------------------------------- #
# By default GEMSEO only writes opt_history.h5 when the scenario finishes,
# so a crash at iteration 7 loses every evaluation and gradient. Backing the
# database up as it goes keeps the file current throughout the run.
# The keyword names for this API differ across GEMSEO versions, so
# introspect the signature rather than guessing.
HISTORY = os.path.abspath("opt_history.h5")
if RANK == 0:
    try:
        import inspect

        sig = inspect.signature(scenario.set_optimization_history_backup)
        kw = {}
        if "erase" in sig.parameters:
            kw["erase"] = True
        for flag in ("at_each_function_call", "at_each_iteration"):
            if flag in sig.parameters:
                kw[flag] = True
                break
        if "file_path" in sig.parameters:
            scenario.set_optimization_history_backup(file_path=HISTORY, **kw)
        else:
            scenario.set_optimization_history_backup(HISTORY, **kw)
        print(f"[info] opt_history.h5 backed up incrementally -> {HISTORY}")
    except Exception as e:
        print(f"[warn] incremental history backup unavailable ({e}); "
              "opt_history.h5 will only be written at the end of the run")

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
if IS_LALE:
    aoa_final = aoa0  # fixed at 4.5 deg throughout
else:
    aoa_final = float(disc.prob.get_val("patchV")[1])
shape_final = disc.prob.get_val("shape")
shape_norm_final = float(np.linalg.norm(shape_final))
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
if IS_LALE:
    w(f"- **AoA**: {aoa_final:.4f} deg  (fixed hard constraint throughout)")
else:
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
w("- Feasible means all constraints are satisfied within GEMSEO tolerances")
w("  (default eq_tolerance=0.01 for the CL equality).")
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
elif args.preset == "lale":
    w("- No paper-reported SECTION CD to compare against.")
    w("- Cross-check with XFOIL polar of MH139-F at Re = 1.66e5 (offline).")
w()
w("## Files Produced")
w()
w("### Written once per run")
w()
w("| File | Contents |")
w("|---|---|")
w("| `opt_summary.md` | this file — one row per evaluation |")
w("| `opt_history.h5` | GEMSEO database: every design vector, function and gradient. Backed up incrementally, so it survives a crash |")
w("| `opt_history*.png` | SLSQP convergence plots (OptHistoryView) |")
w("| `opt_log.txt` | full stdout/stderr |")
w("| `constant/polyMesh/` | mesh **topology** — invariant, since FFD moves vertices but never re-connects cells |")
w("| `system/`, `constant/*Properties`, `0_orig/` | numerics, fluid properties, boundary conditions — invariant |")
w()
w("### Written once per evaluation")
w()
w("| Path | Contents | Size |")
w("|---|---|---|")
w("| `shapes/evalNNNN_shape.npy` | design vector | ~0.5 kB |")
w("| `saved/evalNNNN/processorP/polyMesh/points.gz` | **deformed mesh** for this shape | ~200 kB × ranks |")
w("| `saved/evalNNNN/processorP/{U,p,phi,nut,nuTilda}.gz` | flow field | ~450 kB × ranks |")
w("| `saved/evalNNNN/ffd_coef.npy` | deformed FFD control points | ~3 kB |")
w()
w("These are written by the discipline's per-primal hook, so **every**")
w("evaluation is captured. DAFoam's own solution-preservation happens inside")
w("its adjoint routine, which means line-search evaluations (objective only,")
w("no gradient) would otherwise be overwritten by the next evaluation.")
w()
w("Saving the mesh rather than only the shape matters for reproducing a")
w("design point: a replay can load the exact geometry instead of recreating")
w("it by re-warping, so any disagreement is attributable to the flow solve")
w("alone rather than to geometry and flow together.")
w()
w("### Reproducing evaluation NNNN")
w()
w("```bash")
w("# from the saved mesh and flow field (no re-solve)")
w("cp -r saved/evalNNNN/processor* .")
w("reconstructPar -latestTime && paraview case_lale.foam")
w()
w("# or re-solve from the design vector")
w("mpirun -np 4 python run_at_shape.py --shape shapes/evalNNNN_shape.npy")
w("```")
w()
w(f"Run finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---- archive the run ------------------------------------------------ #
# Every run overwrites the case in place, so freeze mesh, numerics,
# geometry, source, optimizer database and per-evaluation shapes into
# results/<run_name>/ before anything else touches this directory.
ARCHIVE_DIR = None
if RANK == 0:
    try:
        from archive_run import archive_run

        try:
            resolved_da_options = disc.prob.model.daOptions
        except AttributeError:
            resolved_da_options = None

        ARCHIVE_DIR = archive_run(
            case_dir=os.getcwd(),
            # Archive processor*/ too, so every evaluation's flow field can
            # be reconstructPar'd from the archive later without re-solving.
            include_flow=True,
            metadata={
                "preset": args.preset,
                "algorithm": args.algo,
                "max_iter": args.max_iter,
                "n_ranks": NRANKS,
                "wall_time_s": round(t_opt, 1),
                "n_evaluations": disc._eval_count,
                "n_gradients": disc._grad_count,
                "daOptions": resolved_da_options,
                "results": {
                    "CD_baseline": baseline_CD,
                    "CD_optimum": CD_star,
                    "CD_reduction_counts": round(CD_reduction_counts, 2),
                    "CD_reduction_pct": round(CD_reduction_pct, 2),
                    "CL_final": CL_final,
                    "CL_target": CL_target,
                    "aoa_final_deg": aoa_final,
                    "LD_final": LD_final,
                    "shape_norm_final": shape_norm_final,
                    "feasible": bool(opt.is_feasible),
                },
            },
        )
    except Exception as e:
        print(f"[warn] run archive failed: {e}")

if RANK == 0:
    print("\n================ OPTIMIZATION RESULT ================")
    print(f"  preset: {args.preset}, algorithm: {args.algo}")
    print(f"  CD*  = {CD_star:.6f}  ({CD_star * 1e4:.1f} counts)")
    print(f"  converged (feasible): {opt.is_feasible}")
    print(f"  summary written to: {SUMMARY}")
    if ARCHIVE_DIR:
        print(f"  run archived to:     {ARCHIVE_DIR}")
    print("=====================================================")
