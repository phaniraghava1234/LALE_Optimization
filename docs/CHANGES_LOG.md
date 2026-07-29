# Development Change Log

Chronological summary of every substantive change made during the build of this
adjoint-based airfoil shape optimization pipeline. Each row lists the change,
the file(s) touched, and the reason. Ordered by dependency, not always by exact
clock time.

## Stage 1 — Meshing

| # | Change | File(s) | Reason |
|---|---|---|---|
| 1.1 | Adopted 7-block C-topology mesh generator with cosine airfoil resampling and TE thickening | `mesh/generate_cmesh.py` | Standard aerofoil CFD topology; captures LE, TE, and wake cleanly. Cosine resampling clusters points near LE/TE where gradients are largest. |
| 1.2 | Bypassed `classy_blocks` OnCurve wraparound bug by using per-quadrant open splines | `mesh/generate_cmesh.py` | Library issue: OnCurve traced a long arc across block 3 boundary. Per-quadrant splines avoid the wraparound path. |
| 1.3 | Relaxed mesh QC gate: non-orth ≤ 95, skewness ≤ 25, aspect ratio ≤ 2e6 | `mesh/mesh_quality.py` | Wall-resolved (y⁺≈1) meshes normally have max AR ≈ 1e6 at the wall. DAFoam's default 5000 is for wall-function meshes. |
| 1.4 | Reduced mesh from 99k → **57k cells** by raising cell-to-cell expansion ratio (C2C_BL) from 1.08 → 1.15 | `mesh/generate_cmesh.py` | Adjoint chain rule step OOM'd on 28 GB WSL memory at 99k cells. Halving the mesh halved peak memory and fit in budget. Mesh quality unchanged (non-orth 60, skew 0.40). |

## Stage 2 — Geometry parameterization (FFD)

| # | Change | File(s) | Reason |
|---|---|---|---|
| 2.1 | Wrote Plot3D FFD generator with `NX × 2 × 2` control lattice | `ffd/generate_ffd.py` | Standard trivariate B-spline mapping. 2×2 in y (below/above airfoil) and z (spanwise pair) is minimal for 2D. |
| 2.2 | Iterated NX: `20` → `30` (58 shape DVs) | `ffd/generate_ffd.py:37` | 38 DVs at NX=20 gave limited chordwise resolution to shape the shock foot. NX=30 doubles resolution around x/c≈0.5-0.6 where the shock lives. NX=40 tested but rolled back (more DVs slow SLSQP more than they help freedom). |
| 2.3 | LE and TE control points slaved in opposite directions (camber-preservation) | `case/dafoam_model.py:196-206` | Prevents SLSQP from wasting DoF on rigid-body y-translation, which doesn't affect aerodynamics. Removes 2 rigid-body modes. |
| 2.4 | Design-variable bounds tightened `±1.0` m → `±0.02` m (± 2% chord) | `mdo/run_optimization.py:88-98` | With ±1 m bounds, SLSQP's first descent step warped the mesh past validity (9k non-orth errors, 27k flipped face-pyramids). 2% chord is generous for aero design (paper's optimum peak deflection is ~1-2% chord). |

## Stage 3 — CFD baseline (DAFoam primal)

| # | Change | File(s) | Reason |
|---|---|---|---|
| 3.1 | Switched to `DAHisaFoam` (density-based compressible with JST flux) | `case/dafoam_model.py:67` | Transonic (M=0.729) needs a shock-capturing scheme. Incompressible or segregated compressible solvers under-resolve shocks. |
| 3.2 | Wall function OFF, resolve BL to wall (y⁺≈1) | `case/dafoam_model.py:71` | Adjoint gradient sensitivity to skin friction is significant at transonic; wall functions add empirical error that pollutes gradients. |
| 3.3 | Added two operating-point presets: `tutorial` (sea level, CL*=0.7) and `benchmark` (He 2019 conditions, CL*=0.824) | `case/dafoam_model.py:43-48` | Tutorial preset is known-convergent for smoke testing. Benchmark preset matches ADODG Case 2 for paper reproduction. |
| 3.4 | Tightened primal auto-stop tolerances (stdTol 0.1 → 0.01, slopeTol 1e-3 → 5e-5) | `case/dafoam_model.py:88-94` | Loose tolerances stopped the primal mid-convergence, producing CD spread of 3e-3 between evaluations vs the ~5e-6 signal we needed to measure. Tighter tolerances yield tight-enough primal for meaningful gradients. |
| 3.5 | Progressive `primalMinIters` tuning: `100` → `2500` → `4000` → `6000` → **`8000`** | `case/dafoam_model.py:87` | With minIters too low, auto-stop trivially fires at iter 1 (0 std, 0 slope) and the primal exits before flow develops. 8000 provides margin so the 20% check window (nStepsFrac=0.2) is always well-populated. |
| 3.6 | Increased `endTime` from `8000` → `15000` | `case/system/controlDict:23` | Must be greater than `primalMinIters + expected iters to auto-stop`. 15000 leaves headroom. |

## Stage 4 — Discrete adjoint

| # | Change | File(s) | Reason |
|---|---|---|---|
| 4.1 | Configured Krylov GMRES adjoint solver: `gmresRelTol=1e-6`, `pcFillLevel=1`, `natural` reordering | `case/dafoam_model.py:117-123` | Standard DAFoam settings. Matrix-free ASM-preconditioned GMRES scales well; ILU-1 balances setup cost vs iteration count. |
| 4.2 | Wrote `verify_gradients.py` for adjoint validation via central FD | `case/verify_gradients.py` | Adjoint identity `[∂R/∂w]ᵀψ = ∂F/∂w` assumes R=0. Verification exposes when primal is under-converged. |
| 4.3 | Diagnosed 100% verify-gradients error as primal noise, not code bug | (analysis, no code change) | Primal CD std ≈ 5e-3 dominated the ~5.6e-6 FD signal for h=1e-4 shape probes. Root cause = insufficient primal convergence, not a broken derivative. |
| 4.4 | Documented that `dRdWColoring_*.bin` must be deleted when mesh or FFD changes | (case cleanup routine) | Coloring caches the Jacobian sparsity pattern; stale cache from an old mesh triggers `Conflicting Colors Found` error when reused. |

## Stage 5 — MDO integration (GEMSEO + pyOptSparse)

| # | Change | File(s) | Reason |
|---|---|---|---|
| 5.1 | Wrote GEMSEO discipline wrapping the OpenMDAO/MPhys DAFoam problem | `mdo/dafoam_discipline.py` | GEMSEO discipline pattern isolates DAFoam behind a simple `_run` + `_compute_jacobian` interface; GEMSEO drives it like any other differentiable box. |
| 5.2 | Wrote GEMSEO scenario with formal MDO statement (obj=CD, eq=CL, ineq=thickcon/volcon/rcon) | `mdo/run_optimization.py` | Complete Phase 5 driver. `set_differentiation_method("user")` tells GEMSEO to call our adjoint Jacobian, never FD. |
| 5.3 | Installed `gemseo` 6.3.3 + `gemseo-pyoptsparse` 1.1.1 in DAFoam Docker container | pip in container | GEMSEO 5.x → 6.3 API migration needed for the plugin. Plugin exposes pyOptSparse SLSQP and SNOPT as GEMSEO algos. |
| 5.4 | Pinned `numpy<2` after GEMSEO's `[all]` extra pulled numpy ≥ 2 | pip in container | pyoptsparse 2.10.1 uses `np.float_`, removed in numpy 2. Downgrade preserves pyoptsparse. |
| 5.5 | Migrated discipline code from GEMSEO 5.x → 6.x API | `mdo/dafoam_discipline.py` | Renames: `MDODiscipline`→`Discipline`, `local_data`→`io.data`, `default_inputs`→`io.input_grammar.defaults`, `store_local_data`→return dict, `_run(self)`→`_run(self, input_data)`, `input_grammar`→`io.input_grammar`, `_init_jacobian(with_zeros=True)`→`_init_jacobian(…)`. |
| 5.6 | Removed pyOptSparse-native settings (`MAXIT`, `ACC`) from `scenario.execute()` | `mdo/run_optimization.py:132` | gemseo-pyoptsparse 1.1.1 rejects any setting name that has a GEMSEO-canonical counterpart. Only pass `max_iter` and GEMSEO-standard tolerances (`ftol_rel`, `xtol_rel`, `eq_tolerance`, `ineq_tolerance`). |
| 5.7 | Switched default algo from `SLSQP` (GEMSEO built-in) to `PYOPTSPARSE_SLSQP` | `mdo/run_optimization.py:47` | pyOptSparse implementation handles CFD noise and constraint scaling more robustly than GEMSEO's built-in wrapper of scipy SLSQP. Free (no license needed, unlike SNOPT). |

## Environment / infrastructure

| # | Change | File(s) | Reason |
|---|---|---|---|
| E.1 | Set Docker container memory `--memory=28g --memory-swap=60g` | `docker run` command | Adjoint linear solve + preconditioner assembly at 57k cells peaks around 20-25 GB. 28 GB RAM + 32 GB disk swap prevents OOM without excessive swap thrashing. |
| E.2 | Set WSL memory in `~/.wslconfig` to `memory=24GB, swap=32GB` | `C:\Users\phani\.wslconfig` | Docker inherits WSL's memory ceiling; WSL default is 8-16 GB, insufficient. |
| E.3 | Committed running container to new image `dafoam-gemseo:latest` | `docker commit` | Preserves pip-installed GEMSEO + plugin across container restarts. Avoids ~5 min reinstall each session. |
| E.4 | Wrote dependency check script | (inline heredoc) | One-command verification that numpy, gemseo, gemseo-pyoptsparse, pyoptsparse, dafoam, and mphys are all import-clean before launching. |

## Instrumentation / output

| # | Change | File(s) | Reason |
|---|---|---|---|
| I.1 | Added forceCoeffs functionObject writing per-iteration Cd/Cl/Cm to CSV | `case/system/controlDict:53-73` | DAFoam's built-in prints CD/CL every 100 iters; we wanted every iter without bloating the main log. Written to `postProcessing/forceCoeffs1/0/coefficient.dat`. |
| I.2 | Wrote automatic `opt_summary.md` with full run metadata + per-eval table + result footer | `mdo/run_optimization.py`, `mdo/dafoam_discipline.py` | Human-readable audit trail with operating conditions, physics context, expected reference values from He 2019, per-primal CD/CL/L-D/constraints/wall-time, and a final PASS/FAIL block. |
| I.3 | Made HDF5 history writing rank-0 only | `mdo/run_optimization.py:266-269` | MPI file-lock race caused `opt_history.h5` to be truncated to 0 bytes when all 4 ranks tried to write. |

## Bug fixes (post-first-crash chronology)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| B.1 | Primal exits after iter 1 with CD=0.30 | `primalMinIters` too low → nStepsFrac window is empty → std=slope=0 trivially satisfies tolerances | Set `primalMinIters=8000`. |
| B.2 | `Conflicting Colors Found!` after mesh regeneration | Stale `dRdWColoring_4.bin` cache from old mesh | `rm -f dRdWColoring_*.bin` before every run after a mesh/FFD change. |
| B.3 | Signal 9 / OOM during preconditioner assembly | Adjoint sparse matrix + workspace exceeds 16 GB WSL memory at 99k cells | Coarsen mesh to 57k cells, bump WSL memory to 28 GB. |
| B.4 | `error waiting for container: unexpected EOF` at chain rule step | Container OOM after chain rule allocated its own dense partial derivatives | Mesh coarsening (57k) fixed it; also verified via memory watch. |
| B.5 | `_init_jacobian() got unexpected keyword 'with_zeros'` | GEMSEO 6 removed the parameter (zero-init is now default) | Remove the argument. |
| B.6 | `Value error: MAXIT/ACC cannot be passed since a GEMSEO counterpart exists` | gemseo-pyoptsparse 1.1.1 policy — plugin rejects pyoptsparse-native names when a GEMSEO name exists | Pass only `max_iter` and GEMSEO-standard tolerance names. |
| B.7 | Mesh warping produced 9183 non-orth errors + 27658 flipped face pyramids at SLSQP eval 2 | Shape DV bounds ±1 m allowed SLSQP to move geometry by ~1 m — obliterates y⁺≈1 wall stack | Tighten bounds to `±0.02` m. |
| B.8 | `opt_history.h5` truncated to 0 bytes | 4 MPI ranks racing to acquire HDF5 file lock | Guard `save_optimization_history` behind `if RANK == 0`. |

## Version snapshots

| Component | Version at final launch |
|---|---|
| DAFoam | 5.0.0 |
| OpenFOAM | v2506 |
| GEMSEO | 6.3.3 |
| gemseo-pyoptsparse | 1.1.1 |
| pyOptSparse | 2.10.1 |
| numpy | 1.26.4 |
| mphys | 1.1.0 |
| pygeo | 1.13.0 |
| idwarp | 2.6.2 |
| openmdao | 3.26.0 |
| petsc4py | 3.15.5 |
| Python | 3.10.8 |
| Docker base image | `dafoam/opt-packages:latest` → committed as `dafoam-gemseo:latest` |
