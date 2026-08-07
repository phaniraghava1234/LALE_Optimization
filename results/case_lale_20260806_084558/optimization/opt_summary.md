# Optimization Run: 2026-08-06 00:08:20

## Problem Statement

Single-point adjoint-based low-Reynolds airfoil shape optimization
(Oettershagen et al. 2017, AtlantikSolar UAV cruise condition):

```
minimize      C_D(shape)                     (alpha fixed at 4.5 deg)
subject to    C_L = 0.975  (lift equality)
              thickcon >= 0.5, volcon >= 1.0, rcon >= 0.8
              shape in [-0.02, 0.02]^n   (m; ~2% chord)
```

## Configuration

- **Preset**: `lale`
- **Algorithm**: `PYOPTSPARSE_SLSQP` (sequential quadratic programming)
- **Max SLSQP iterations**: 50
- **Shape DVs**: 58 (FFD y-displacements, camber-preserving at LE/TE)
- **Alpha**: fixed at 4.5 deg (NOT a DV)
- **MPI ranks**: 6
- **Working directory**: `/home/dafoamuser/mount/case_lale`
- **Gradient method**: user-supplied (discrete adjoint)

## Operating Conditions

- **U_infinity**: 2.531 m/s (CFD representation)
- **rho_infinity**: 1.1673 kg/m^3 (500 m ISA)
- **nu**: 1.5196e-05 m^2/s
- **Mach**: 0.0074 (incompressible regime, fully incompressible solver)
- **Re per chord** (unit-chord CFD): 1.666e+05
- **Initial AoA**: 4.5000 deg
- **Target CL**: 0.975
- **Reference area A0**: 0.01 m^2 (chord=1m x span=0.01m)
- **Dynamic pressure q_inf**: 3.7403 Pa

## Physics Expected

- Low-Reynolds incompressible flow, Re = 1.67e+05, M = 0.007
- Fully turbulent SA (no laminar-transition model); absolute CD is over-predicted
  compared to XFOIL because the SA closure cannot represent the laminar
  separation bubble known to exist on the MH-series at these conditions.
- Optimization signal (gradient) remains valid; the optimizer will find shape
  moves that reduce CD at the fixed CL target.

## Reference Results (for context)

- Reference aircraft: AtlantikSolar LALE UAV (Oettershagen et al. 2017).
- Section-CL target 0.975 is derived from level-flight equilibrium of
  the aircraft at 500 m ISA, U = 8.30 m/s (measured cruise from the
  81-hour endurance flight), c = 0.305 m, S = 1.735 m^2, W = 67.98 N.
- No published CD target for the SECTION (paper reports whole-aircraft
  power); this study is a demonstrator of the pipeline at LALE Re, not
  a paper-reproduction case.
- Reference: Oettershagen et al. *Journal of Field Robotics* 34(7) (2017) 1352-1385.

## Solver Settings

- Solver: `DASimpleFoam` (incompressible SIMPLE)
- Turbulence: Spalart-Allmaras, wall-resolved (y+~1)
- Primal tolerance: stdTol=1e-3, slopeTol=5e-6, primalMinIters=8000
- Adjoint: Krylov (GMRES), gmresRelTol=1e-6, pcFillLevel=1
- Mesh: C-topology, ~32-45k hex cells

## Constraints

- `CL` = 0.975 (equality) - maintains lift during CD minimization
- `thickcon` >= 0.5 (per-station thickness ratio) - prevents crazy-thin airfoil
- `volcon`   >= 1.0 (volume ratio) - prevents deflating the airfoil
- `rcon`     >= 0.8 (LE radius ratio) - keeps LE bluntness reasonable

## Iteration Log

One row per primal evaluation. SLSQP triggers a primal for each outer iter, 
plus additional primals inside its line search. Gradients (adjoints) run 
after each successful primal.

**Columns:**
- `Eval`: sequential primal number (1 = baseline)
- `kind`: base (first) / eval (subsequent)
- `|shape|`: L2 norm of shape DV vector (0 = baseline, larger = more deformed)
- `AoA`: current angle of attack, deg (fixed for LALE, DV for transonic)
- `CD` / `CL`: force coefficients
- `counts`: CD * 10000 (aerospace-standard drag counts)
- `dCts`: change in drag counts vs baseline (negative = improvement)
- `CL_err`: CL - CL_target (should stay near 0 as SLSQP enforces equality)
- `L/D`: lift-to-drag ratio
- `thick_min` / `vol` / `rcon_min`: geometric constraint values (must stay >= bounds)
- `wall(min)`: this primal's wall-clock time in minutes
- `total`: cumulative wall time

| Eval | kind | |shape|  | AoA (deg) | CD | counts | dCts | CL | CL_err | L/D | thick_min | vol | rcon_min | wall(min) | total |
|-----:|:----:|--------:|----------:|--------:|-------:|-----:|-------:|-------:|-----:|----------:|-----:|---------:|----------:|------:|
|    1 | base | 0.0000e+00 |  4.5000 | 0.026374 |  263.7 |    +0.0 | 0.696469 | -0.2785 |  26.41 | 1.0000 | 1.0000 | 1.0000 |   4.7 | 00h04m |
|    2 | eval | 2.4922e-02 |  4.5000 | 0.035604 |  356.0 |   +92.3 | 0.922293 | -0.0527 |  25.90 | 0.9930 | 1.0000 | 0.9984 |   4.7 | 00h16m |
|    3 | eval | 3.3869e-02 |  4.5000 | 0.037842 |  378.4 |  +114.7 | 0.961017 | -0.0140 |  25.40 | 0.9743 | 1.0000 | 1.0023 |   4.0 | 00h30m |
|    4 | eval | 3.8098e-02 |  4.5000 | 0.038519 |  385.2 |  +121.4 | 0.974143 | -0.0009 |  25.29 | 0.9663 | 1.0000 | 1.0047 |   4.3 | 00h45m |
|    5 | eval | 3.8292e-02 |  4.5000 | 0.038657 |  386.6 |  +122.8 | 0.975001 | +0.0000 |  25.22 | 0.9661 | 1.0000 | 1.0052 |   4.1 | 01h00m |
|    6 | eval | 3.8063e-02 |  4.5000 | 0.038444 |  384.4 |  +120.7 | 0.974923 | -0.0001 |  25.36 | 0.9665 | 1.0000 | 1.0062 |   4.2 | 01h15m |
|    7 | eval | 3.7218e-02 |  4.5000 | 0.038067 |  380.7 |  +116.9 | 0.973766 | -0.0012 |  25.58 | 0.9688 | 1.0000 | 1.0113 |   4.2 | 01h29m |
|    8 | eval | 3.8135e-02 |  4.5000 | 0.037011 |  370.1 |  +106.4 | 0.966240 | -0.0088 |  26.11 | 0.9651 | 1.0000 | 1.0268 |   4.2 | 01h44m |
|    9 | eval | 3.7319e-02 |  4.5000 | 0.037701 |  377.0 |  +113.3 | 0.972739 | -0.0023 |  25.80 | 0.9688 | 1.0000 | 1.0180 |   4.5 | 01h48m |
|   10 | eval | 3.8822e-02 |  4.5000 | 0.037416 |  374.2 |  +110.4 | 0.973308 | -0.0017 |  26.01 | 0.9649 | 1.0000 | 1.0258 |   4.2 | 02h03m |
|   11 | eval | 3.9940e-02 |  4.5000 | 0.037586 |  375.9 |  +112.1 | 0.975058 | +0.0001 |  25.94 | 0.9629 | 1.0000 | 1.0280 |   4.2 | 02h17m |
|   12 | eval | 3.9271e-02 |  4.5000 | 0.037450 |  374.5 |  +110.8 | 0.974006 | -0.0010 |  26.01 | 0.9641 | 1.0000 | 1.0267 |   4.5 | 02h21m |
|   13 | eval | 4.0839e-02 |  4.5000 | 0.037546 |  375.5 |  +111.7 | 0.974856 | -0.0001 |  25.96 | 0.9611 | 1.0000 | 1.0311 |   4.1 | 02h36m |
|   14 | eval | 3.9795e-02 |  4.5000 | 0.037441 |  374.4 |  +110.7 | 0.974396 | -0.0006 |  26.02 | 0.9631 | 1.0000 | 1.0282 |   4.5 | 02h40m |
|   15 | eval | 4.2311e-02 |  4.5000 | 0.037517 |  375.2 |  +111.4 | 0.974238 | -0.0008 |  25.97 | 0.9583 | 1.0000 | 1.0364 |   4.1 | 02h54m |
|   16 | eval | 4.0422e-02 |  4.5000 | 0.037435 |  374.3 |  +110.6 | 0.974382 | -0.0006 |  26.03 | 0.9618 | 1.0000 | 1.0303 |   4.6 | 02h59m |
|   17 | eval | 4.2823e-02 |  4.5000 | 0.037498 |  375.0 |  +111.2 | 0.974515 | -0.0005 |  25.99 | 0.9574 | 1.0000 | 1.0383 |   4.3 | 03h13m |
|   18 | eval | 4.1118e-02 |  4.5000 | 0.037422 |  374.2 |  +110.5 | 0.974540 | -0.0005 |  26.04 | 0.9605 | 1.0000 | 1.0327 |   5.3 | 03h19m |
|   19 | eval | 4.3573e-02 |  4.5000 | 0.037485 |  374.8 |  +111.1 | 0.974421 | -0.0006 |  26.00 | 0.9560 | 1.0000 | 1.0415 |   4.3 | 03h33m |
|   20 | eval | 4.1767e-02 |  4.5000 | 0.037418 |  374.2 |  +110.4 | 0.974604 | -0.0004 |  26.05 | 0.9593 | 1.0000 | 1.0352 |   4.6 | 03h38m |
|   21 | eval | 4.3944e-02 |  4.5000 | 0.037457 |  374.6 |  +110.8 | 0.974653 | -0.0003 |  26.02 | 0.9553 | 1.0000 | 1.0437 |   4.3 | 03h52m |
|   22 | eval | 4.2503e-02 |  4.5000 | 0.037386 |  373.9 |  +110.1 | 0.974710 | -0.0003 |  26.07 | 0.9579 | 1.0000 | 1.0381 |   4.6 | 03h57m |
|   23 | eval | 4.4974e-02 |  4.5000 | 0.037402 |  374.0 |  +110.3 | 0.974437 | -0.0006 |  26.05 | 0.9534 | 1.0000 | 1.0489 |   4.3 | 04h12m |
|   24 | eval | 4.3412e-02 |  4.5000 | 0.037359 |  373.6 |  +109.9 | 0.974714 | -0.0003 |  26.09 | 0.9562 | 1.0000 | 1.0423 |   4.6 | 04h16m |
|   25 | eval | 4.5898e-02 |  4.5000 | 0.037364 |  373.6 |  +109.9 | 0.974263 | -0.0007 |  26.07 | 0.9518 | 1.0000 | 1.0542 |   4.3 | 04h31m |
|   26 | eval | 4.4346e-02 |  4.5000 | 0.037320 |  373.2 |  +109.5 | 0.974691 | -0.0003 |  26.12 | 0.9544 | 1.0000 | 1.0470 |   4.5 | 04h35m |
|   27 | eval | 4.6276e-02 |  4.5000 | 0.037314 |  373.1 |  +109.4 | 0.974473 | -0.0005 |  26.12 | 0.9511 | 1.0000 | 1.0576 |   4.4 | 04h50m |
|   28 | eval | 4.5222e-02 |  4.5000 | 0.037267 |  372.7 |  +108.9 | 0.974767 | -0.0002 |  26.16 | 0.9529 | 1.0000 | 1.0520 |   4.5 | 04h54m |
|   29 | eval | 4.7300e-02 |  4.5000 | 0.037214 |  372.1 |  +108.4 | 0.973905 | -0.0011 |  26.17 | 0.9495 | 1.0000 | 1.0661 |   4.3 | 05h08m |
|   30 | eval | 4.6212e-02 |  4.5000 | 0.037187 |  371.9 |  +108.1 | 0.974642 | -0.0004 |  26.21 | 0.9511 | 1.0000 | 1.0591 |   4.6 | 05h13m |
|   31 | eval | 4.6957e-02 |  4.5000 | 0.037194 |  371.9 |  +108.2 | 0.974421 | -0.0006 |  26.20 | 0.9500 | 1.0000 | 1.0683 |   5.5 | 05h29m |
|   32 | eval | 4.6499e-02 |  4.5000 | 0.037147 |  371.5 |  +107.7 | 0.974680 | -0.0003 |  26.24 | 0.9506 | 1.0000 | 1.0631 |   6.1 | 05h35m |
|   33 | eval | 4.6183e-02 |  4.5000 | 0.037088 |  370.9 |  +107.1 | 0.973298 | -0.0017 |  26.24 | 0.9514 | 1.0000 | 1.0732 |   5.5 | 05h51m |
|   34 | eval | 4.6286e-02 |  4.5000 | 0.037077 |  370.8 |  +107.0 | 0.974462 | -0.0005 |  26.28 | 0.9510 | 1.0000 | 1.0677 |   5.5 | 05h57m |
|   35 | eval | 4.5522e-02 |  4.5000 | 0.037143 |  371.4 |  +107.7 | 0.974549 | -0.0005 |  26.24 | 0.9522 | 1.0000 | 1.0701 |   4.4 | 06h12m |
|   36 | eval | 4.6087e-02 |  4.5000 | 0.037072 |  370.7 |  +107.0 | 0.974604 | -0.0004 |  26.29 | 0.9513 | 1.0000 | 1.0683 |   4.5 | 06h17m |
|   37 | eval | 4.4542e-02 |  4.5000 | 0.037139 |  371.4 |  +107.7 | 0.974377 | -0.0006 |  26.24 | 0.9537 | 1.0000 | 1.0687 |   4.9 | 06h32m |
|   38 | eval | 4.5730e-02 |  4.5000 | 0.037074 |  370.7 |  +107.0 | 0.974605 | -0.0004 |  26.29 | 0.9518 | 1.0000 | 1.0684 |   4.8 | 06h37m |
|   39 | eval | 4.5929e-02 |  4.5000 | 0.037157 |  371.6 |  +107.8 | 0.974595 | -0.0004 |  26.23 | 0.9515 | 1.0000 | 1.0683 |   4.6 | 06h41m |
|   40 | eval | 4.6071e-02 |  4.5000 | 0.037071 |  370.7 |  +107.0 | 0.974605 | -0.0004 |  26.29 | 0.9513 | 1.0000 | 1.0683 |   4.4 | 06h46m |
|   41 | eval | 4.3852e-02 |  4.5000 | 0.037108 |  371.1 |  +107.3 | 0.974203 | -0.0008 |  26.25 | 0.9547 | 1.0000 | 1.0675 |   4.6 | 07h00m |
|   42 | eval | 4.5411e-02 |  4.5000 | 0.037068 |  370.7 |  +106.9 | 0.974654 | -0.0003 |  26.29 | 0.9523 | 1.0000 | 1.0681 |   4.5 | 07h05m |
|   43 | eval | 4.3694e-02 |  4.5000 | 0.037121 |  371.2 |  +107.5 | 0.974540 | -0.0005 |  26.25 | 0.9549 | 1.0000 | 1.0668 |   6.1 | 07h21m |
|   44 | eval | 4.5019e-02 |  4.5000 | 0.037065 |  370.6 |  +106.9 | 0.974668 | -0.0003 |  26.30 | 0.9529 | 1.0000 | 1.0678 |   5.6 | 07h27m |
|   45 | eval | 4.3693e-02 |  4.5000 | 0.037136 |  371.4 |  +107.6 | 0.974659 | -0.0003 |  26.25 | 0.9549 | 1.0000 | 1.0670 |   5.9 | 07h44m |
|   46 | eval | 4.4793e-02 |  4.5000 | 0.037055 |  370.6 |  +106.8 | 0.974667 | -0.0003 |  26.30 | 0.9532 | 1.0000 | 1.0676 |   7.5 | 07h51m |
|   47 | eval | 4.3634e-02 |  4.5000 | 0.037131 |  371.3 |  +107.6 | 0.974708 | -0.0003 |  26.25 | 0.9549 | 1.0000 | 1.0677 |   6.9 | 08h09m |
|   48 | eval | 4.4594e-02 |  4.5000 | 0.037048 |  370.5 |  +106.7 | 0.974669 | -0.0003 |  26.31 | 0.9535 | 1.0000 | 1.0676 |   5.5 | 08h15m |
|   49 | eval | 4.3341e-02 |  4.5000 | 0.037108 |  371.1 |  +107.3 | 0.974617 | -0.0004 |  26.26 | 0.9554 | 1.0000 | 1.0690 |   6.5 | 08h32m |
|   50 | eval | 4.4309e-02 |  4.5000 | 0.037049 |  370.5 |  +106.8 | 0.974705 | -0.0003 |  26.31 | 0.9539 | 1.0000 | 1.0679 |   4.7 | 08h37m |

## Final Result

### Aerodynamics
- **CD baseline**: 0.026374  (263.7 counts)
- **CD optimized**: 0.037011  (370.1 counts)
- **CD reduction**: -106.4 counts  (-40.33%)
- **CL final**: 0.974705  (target 0.975, error -0.0003)
- **L/D final**: 26.34
- **AoA**: 4.5000 deg  (fixed hard constraint throughout)

### Constraints (final)
- `thickcon` min: 0.9539  (must >= 0.5)  PASS
- `volcon`      : 1.0000  (must >= 1.0)  PASS
- `rcon` min    : 1.0679  (must >= 0.8)  PASS

### Shape
- **|shape| final**: 4.4309e-02 (baseline 0)
- **max shape DV**: 2.0000e-02

### Solver Stats
- **Primal evaluations**: 50
- **Gradient (adjoint) evaluations**: 27
- **Total wall time**: 8h 37m (31039 s)
- **Avg time per primal**: 10.3 min

### Optimizer Verdict
- **SLSQP feasible flag**: True
- **Objective at exit**: 0.037011
- Feasible means all constraints are satisfied within GEMSEO tolerances
  (default eq_tolerance=0.01 for the CL equality).
- Convergence to optimum: check `opt_history_criteria.png` (KKT norm should trend down).

### Comparison to Reference

- No paper-reported SECTION CD to compare against.
- Cross-check with XFOIL polar of MH139-F at Re = 1.66e5 (offline).

## Files Produced

### Written once per run

| File | Contents |
|---|---|
| `opt_summary.md` | this file — one row per evaluation |
| `opt_history.h5` | GEMSEO database: every design vector, function and gradient. Backed up incrementally, so it survives a crash |
| `opt_history*.png` | SLSQP convergence plots (OptHistoryView) |
| `opt_log.txt` | full stdout/stderr |
| `constant/polyMesh/` | mesh **topology** — invariant, since FFD moves vertices but never re-connects cells |
| `system/`, `constant/*Properties`, `0_orig/` | numerics, fluid properties, boundary conditions — invariant |

### Written once per evaluation

| Path | Contents | Size |
|---|---|---|
| `shapes/evalNNNN_shape.npy` | design vector | ~0.5 kB |
| `saved/evalNNNN/processorP/polyMesh/points.gz` | **deformed mesh** for this shape | ~200 kB × ranks |
| `saved/evalNNNN/processorP/{U,p,phi,nut,nuTilda}.gz` | flow field | ~450 kB × ranks |
| `saved/evalNNNN/ffd_coef.npy` | deformed FFD control points | ~3 kB |

These are written by the discipline's per-primal hook, so **every**
evaluation is captured. DAFoam's own solution-preservation happens inside
its adjoint routine, which means line-search evaluations (objective only,
no gradient) would otherwise be overwritten by the next evaluation.

Saving the mesh rather than only the shape matters for reproducing a
design point: a replay can load the exact geometry instead of recreating
it by re-warping, so any disagreement is attributable to the flow solve
alone rather than to geometry and flow together.

### Reproducing evaluation NNNN

```bash
# from the saved mesh and flow field (no re-solve)
cp -r saved/evalNNNN/processor* .
reconstructPar -latestTime && paraview case_lale.foam

# or re-solve from the design vector
mpirun -np 4 python run_at_shape.py --shape shapes/evalNNNN_shape.npy
```

Run finished: 2026-08-06 08:45:58
