# Optimization Run: 2026-07-26 13:43:46

## Problem Statement

Single-point adjoint-based transonic airfoil shape optimization
(He et al. 2019, *Robust aerodynamic shape optimization*, ADODG Case 2 setup):

```
minimize      CD(shape, aoa)
subject to    CL = 0.7  (lift equality)
              thickcon >= 0.5, volcon >= 1.0, rcon >= 0.8 (geometry)
              shape in [-1, 1]^n,  aoa in [0, 10] deg
```

## Configuration

- **Preset**: `tutorial`
- **Algorithm**: `PYOPTSPARSE_SLSQP` (sequential quadratic programming)
- **Max SLSQP iterations**: 5
- **Shape DVs**: 58 (FFD y-displacements, camber-preserving at LE/TE)
- **MPI ranks**: 4
- **Working directory**: `/home/dafoamuser/mount/case`
- **Gradient method**: user-supplied (discrete adjoint)

## Operating Conditions

- **U_infinity**: 248.000 m/s
- **p_infinity**: 101325.0 Pa
- **T_infinity**: 300.00 K
- **rho_infinity** (derived): 1.1768 kg/m^3
- **Speed of sound** (derived): 347.19 m/s
- **Mach** (derived): 0.7143  (transonic)
- **Re per chord** (derived, mu=1.8e-05): 1.621e+07
- **Initial AoA**: 2.5252 deg
- **Target CL**: 0.7
- **Reference area A0**: 0.01 m^2 (chord=1m x span=0.01m)
- **Dynamic pressure q_inf**: 36189.9 Pa

## Physics Expected

- Transonic flow at M=0.714, expect a normal shock on suction side near x/c ~ 0.5-0.6
- CD = CD_pressure + CD_viscous; pressure component dominates due to shock
- Optimization goal: weaken/eliminate the shock (shock-free or lambda foot)
- Well-known result: ~30-40% CD reduction is achievable at fixed CL

## Reference Results (for context)

- DAFoam tutorial baseline (this operating point):  ~150-190 drag counts
- Typical SLSQP-optimized outcome:                   ~100-130 drag counts
- CL held to target ~0.7 throughout
- Reference: He, Mader, Martins, Maki. *Aerospace Science and Technology* 87 (2019) 483-502.

## Solver Settings (see `case/dafoam_model.py`)

- Solver: `DAHisaFoam` (density-based compressible, JST convective flux)
- Turbulence: Spalart-Allmaras, wall-resolved (y+~1, no wall functions)
- Primal tolerance: stdTol=0.01, slopeTol=5e-5, primalMinIters=6000
- Adjoint: Krylov (GMRES), gmresRelTol=1e-6, pcFillLevel=1
- Mesh: C-topology, ~57k hex cells (halved from initial 99k for memory)

## Constraints

- `CL` = 0.7 (equality) - maintains lift during CD minimization
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
- `AoA`: current angle of attack, deg
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
|    1 | base | 0.0000e+00 |  2.5252 | 0.011900 |  119.0 |    +0.0 | 0.747029 | +0.0470 |  62.78 | 1.0000 | 1.0000 | 1.0000 |  85.8 | 01h25m |
|    2 | eval | 7.8350e-04 |  2.2644 | 0.010948 |  109.5 |    -9.5 | 0.697999 | -0.0020 |  63.76 | 0.9991 | 1.0000 | 0.9999 | 134.3 | 04h00m |
|    3 | eval | 7.9109e-04 |  2.2547 | 0.010749 |  107.5 |   -11.5 | 0.701436 | +0.0014 |  65.26 | 0.9988 | 1.0000 | 0.9999 | 132.2 | 06h34m |
|    4 | eval | 7.0949e-04 |  2.2414 | 0.010662 |  106.6 |   -12.4 | 0.700278 | +0.0003 |  65.68 | 0.9991 | 1.0000 | 0.9994 | 124.7 | 09h01m |
|    5 | eval | 1.0253e-03 |  2.2195 | 0.010758 |  107.6 |   -11.4 | 0.698639 | -0.0014 |  64.94 | 0.9987 | 1.0001 | 0.9977 | 143.9 | 11h46m |

## Final Result

### Aerodynamics
- **CD baseline**: 0.011900  (119.0 counts)
- **CD optimized**: 0.010662  (106.6 counts)
- **CD reduction**: +12.4 counts  (+10.40%)
- **CL final**: 0.698639  (target 0.7, error -0.0014)
- **L/D final**: 65.53
- **AoA final**: 2.2195 deg  (initial 2.5252)

### Constraints (final)
- `thickcon` min: 0.9987  (must >= 0.5)  PASS
- `volcon`      : 1.0001  (must >= 1.0)  PASS
- `rcon` min    : 0.9977  (must >= 0.8)  PASS

### Shape
- **|shape| final**: 1.0253e-03 (baseline 0)
- **max shape DV**: 5.4540e-04

### Solver Stats
- **Primal evaluations**: 5
- **Gradient (adjoint) evaluations**: 4
- **Total wall time**: 11h 46m (42383 s)
- **Avg time per primal**: 141.3 min

### Optimizer Verdict
- **SLSQP feasible flag**: True
- **Objective at exit**: 0.010662
- Feasible means all constraints are satisfied within pyOptSparse tolerances (ACC=1e-6).
- Convergence to optimum: check `opt_history_criteria.png` (KKT norm should trend down).

### Comparison to Reference

- DAFoam tutorial baseline: ~150-190 counts vs our 119.0
- Typical tutorial optimum: ~100-130 counts vs our 106.6

## Files Produced

- `opt_summary.md`         : this file
- `opt_history.h5`         : GEMSEO optimization database (all evals + gradients)
- `opt_history*.png`       : SLSQP convergence plots (OptHistoryView)
- `opt_log.txt`            : full stdout/stderr from the run
- `deformedFFD_*.dat`      : deformed FFD lattices per iter (if writeDeformedFFDs=1)
- `processor*/latestTime/` : final flow field, viewable in ParaView
- To visualize final shape: `touch case/case.foam && paraview case/case.foam`

Run finished: 2026-07-27 01:30:23
