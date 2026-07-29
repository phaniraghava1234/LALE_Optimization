# Optimization Run: 2026-07-27 20:14:48

## Problem Statement

Single-point adjoint-based transonic airfoil shape optimization
(He et al. 2019, *Robust aerodynamic shape optimization*, ADODG Case 2 setup):

```
minimize      CD(shape, aoa)
subject to    CL = 0.824  (lift equality)
              thickcon >= 0.5, volcon >= 1.0, rcon >= 0.8 (geometry)
              shape in [-1, 1]^n,  aoa in [0, 10] deg
```

## Configuration

- **Preset**: `benchmark`
- **Algorithm**: `PYOPTSPARSE_SLSQP` (sequential quadratic programming)
- **Max SLSQP iterations**: 10
- **Shape DVs**: 58 (FFD y-displacements, camber-preserving at LE/TE)
- **MPI ranks**: 4
- **Working directory**: `/home/dafoamuser/mount/case`
- **Gradient method**: user-supplied (discrete adjoint)

## Operating Conditions

- **U_infinity**: 248.080 m/s
- **p_infinity**: 39011.0 Pa
- **T_infinity**: 288.15 K
- **rho_infinity** (derived): 0.4717 kg/m^3
- **Speed of sound** (derived): 340.26 m/s
- **Mach** (derived): 0.7291  (transonic)
- **Re per chord** (derived, mu=1.8e-05): 6.501e+06
- **Initial AoA**: 2.8000 deg
- **Target CL**: 0.824
- **Reference area A0**: 0.01 m^2 (chord=1m x span=0.01m)
- **Dynamic pressure q_inf**: 14515.8 Pa

## Physics Expected

- Transonic flow at M=0.729, expect a normal shock on suction side near x/c ~ 0.5-0.6
- CD = CD_pressure + CD_viscous; pressure component dominates due to shock
- Optimization goal: weaken/eliminate the shock (shock-free or lambda foot)
- Well-known result: ~30-40% CD reduction is achievable at fixed CL

## Reference Results (for context)

- He et al. 2019, ADODG Case 2 (M=0.729, Re=6.5M, CL=0.824):
  - CD baseline (RANS-SA):   ~199 drag counts (0.0199)
  - CD optimized:            ~123-135 drag counts (published range)
  - CD reduction:            ~30-38%
- Reference: He, Mader, Martins, Maki. *Aerospace Science and Technology* 87 (2019) 483-502.

## Solver Settings (see `case/dafoam_model.py`)

- Solver: `DAHisaFoam` (density-based compressible, JST convective flux)
- Turbulence: Spalart-Allmaras, wall-resolved (y+~1, no wall functions)
- Primal tolerance: stdTol=0.01, slopeTol=5e-5, primalMinIters=6000
- Adjoint: Krylov (GMRES), gmresRelTol=1e-6, pcFillLevel=1
- Mesh: C-topology, ~57k hex cells (halved from initial 99k for memory)

## Constraints

- `CL` = 0.824 (equality) - maintains lift during CD minimization
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
|    1 | base | 0.0000e+00 |  2.8000 | 0.018385 |  183.8 |    +0.0 | 0.788461 | -0.0355 |  42.89 | 1.0000 | 1.0000 | 1.0000 |  95.6 | 01h35m |
|    2 | eval | 1.7745e-03 |  2.9339 | 0.019400 |  194.0 |   +10.2 | 0.826844 | +0.0028 |  42.62 | 0.9988 | 1.0000 | 1.0010 | 124.4 | 03h59m |
|    3 | eval | 6.7276e-03 |  2.6375 | 0.016417 |  164.2 |   -19.7 | 0.798586 | -0.0254 |  48.64 | 0.9953 | 1.0000 | 1.0051 | 133.1 | 06h29m |
|    4 | eval | 5.3223e-03 |  2.6868 | 0.016149 |  161.5 |   -22.4 | 0.814745 | -0.0093 |  50.45 | 0.9960 | 1.0000 | 1.0040 | 139.3 | 09h10m |
|    5 | eval | 6.7483e-03 |  2.6752 | 0.015942 |  159.4 |   -24.4 | 0.822368 | -0.0016 |  51.59 | 0.9941 | 1.0000 | 1.0050 | 154.9 | 12h07m |
|    6 | eval | 1.2368e-02 |  2.4607 | 0.018022 |  180.2 |    -3.6 | 0.784408 | -0.0396 |  43.53 | 0.9871 | 1.0001 | 1.0083 | 176.6 | 15h29m |
|    7 | eval | 7.8952e-03 |  2.6294 | 0.015251 |  152.5 |   -31.3 | 0.825190 | +0.0012 |  54.11 | 0.9926 | 1.0000 | 1.0057 | 152.8 | 18h02m |
|    8 | eval | 2.0332e-02 |  2.1363 | 0.025661 |  256.6 |   +72.8 | 0.675661 | -0.1483 |  26.33 | 0.9790 | 1.0004 | 1.0125 | 156.4 | 21h09m |
|    9 | eval | 9.3854e-03 |  2.5680 | 0.014713 |  147.1 |   -36.7 | 0.826245 | +0.0022 |  56.16 | 0.9909 | 1.0001 | 1.0066 | 161.9 | 23h50m |
|   10 | eval | 1.6616e-02 |  2.2563 | 0.022346 |  223.5 |   +39.6 | 0.721383 | -0.1026 |  32.28 | 0.9824 | 1.0000 | 1.0101 | 165.9 | 27h18m |

## Final Result

### Aerodynamics
- **CD baseline**: 0.018385  (183.8 counts)
- **CD optimized**: 0.014713  (147.1 counts)
- **CD reduction**: +36.7 counts  (+19.97%)
- **CL final**: 0.721383  (target 0.824, error -0.1026)
- **L/D final**: 49.03
- **AoA final**: 2.2563 deg  (initial 2.8000)

### Constraints (final)
- `thickcon` min: 0.9824  (must >= 0.5)  PASS
- `volcon`      : 1.0000  (must >= 1.0)  PASS
- `rcon` min    : 1.0101  (must >= 0.8)  PASS

### Shape
- **|shape| final**: 1.6616e-02 (baseline 0)
- **max shape DV**: 6.2315e-03

### Solver Stats
- **Primal evaluations**: 10
- **Gradient (adjoint) evaluations**: 7
- **Total wall time**: 27h 18m (98283 s)
- **Avg time per primal**: 163.8 min

### Optimizer Verdict
- **SLSQP feasible flag**: True
- **Objective at exit**: 0.014713
- Feasible means all constraints are satisfied within pyOptSparse tolerances (ACC=1e-6).
- Convergence to optimum: check `opt_history_criteria.png` (KKT norm should trend down).

### Comparison to Reference

- He 2019 baseline: 199 counts vs our 183.8
- He 2019 optimum:  123-135 counts vs our 147.1
- He 2019 reduction: ~30-38% vs our 20.0%

## Files Produced

- `opt_summary.md`         : this file
- `opt_history.h5`         : GEMSEO optimization database (all evals + gradients)
- `opt_history*.png`       : SLSQP convergence plots (OptHistoryView)
- `opt_log.txt`            : full stdout/stderr from the run
- `deformedFFD_*.dat`      : deformed FFD lattices per iter (if writeDeformedFFDs=1)
- `processor*/latestTime/` : final flow field, viewable in ParaView
- To visualize final shape: `touch case/case.foam && paraview case/case.foam`

Run finished: 2026-07-28 23:33:08
