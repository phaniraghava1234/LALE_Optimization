# Case Study 1 — Transonic Shape Optimization of the RAE 2822

Single-point adjoint-based shape optimization of the **RAE 2822** airfoil at
transonic cruise, driven by the discrete adjoint of the compressible RANS
equations.

Operating point matches the ADODG Case 2 setup used by He, Mader, Martins and
Maki (2019): **M = 0.729, Re = 6.5 × 10⁶, C_L\* = 0.824**.

For the shared pipeline description, see [`REPORT.md`](REPORT.md). For the
low-Reynolds companion study, see [`CASE_LOWRE_MH139F.md`](CASE_LOWRE_MH139F.md).

---

## Contents

1. [Scope of the claim](#1-scope-of-the-claim)
2. [Operating point](#2-operating-point)
3. [Optimization problem](#3-optimization-problem)
4. [Results](#4-results)
5. [Cold-start verification](#5-cold-start-verification)
6. [Physics](#6-physics)
7. [Comparison to the paper](#7-comparison-to-the-paper)
8. [Limitations](#8-limitations)
9. [Reproducing this run](#9-reproducing-this-run)
10. [References](#10-references)

---

## 1. Scope of the claim

**This is not a bit-for-bit reproduction of any published paper.**

The study draws on He et al. (2019), *Robust aerodynamic shape optimization —
from a circle to an airfoil* (Aerospace Science and Technology 87, 483–502),
specifically their ADODG Case 2 setup for the RAE 2822. The intent was to
build an end-to-end open-source adjoint MDO pipeline on DAFoam, exercise it at
the same operating point, and honestly assess where a practical
DAFoam + GEMSEO + SLSQP stack lands relative to a rigorous ADflow + SNOPT
setup.

Several parts are deliberately different:

| Aspect | Paper (He 2019) | This project |
|---|---|---|
| CFD solver | ADflow (structured multi-block, ANK/NK Newton–Krylov) | DAFoam v5 / `DAHisaFoam` (density-based FVM on polyMesh) |
| Mesh | O-mesh, knife-edge TE | C-mesh, 1 % chord blunt TE (`classy_blocks` needs non-zero TE) |
| Optimizer | SNOPT (commercial) | pyOptSparse SLSQP (free), via GEMSEO 6.3 |
| Outer framework | OpenMDAO + pyOptSparse directly | GEMSEO wraps OpenMDAO |
| Primal convergence | Density residual 10⁻¹⁵ | C_D std ≤ 0.01, slope ≤ 5e-5, `primalMinIters` = 8000 |
| Optimizer tolerances | opt 10⁻⁶, step 10⁻³, Hessian 50 | GEMSEO defaults (`ftol_rel` 10⁻⁶, `eq_tolerance` 10⁻²) |
| Iterations to termination | ≈ 500 (paper Fig. 11) | 10 (`max_iter` cap) |

Same in both: operating point, Spalart–Allmaras low-Re turbulence closure, JST
convective flux, FFD shape parameterization with camber-preserving LE/TE, and
the thickness/volume/LE-radius constraint set.

Every claim below marked "verified from the paper" cites the page of
`He2019c.pdf` where it appears. Everything else is an implementation choice
with its reasoning stated.

---

## 2. Operating point

`benchmark` preset in `case/dafoam_model.py`:

| Parameter | Value |
|---|---:|
| U∞ | 248.08 m/s |
| p∞ | 39 011 Pa |
| T∞ | 288.15 K |
| ρ∞ | 0.472 kg/m³ |
| Mach | 0.729 |
| Re per chord | 6.5 × 10⁶ |
| C_L\* | 0.824 |
| Initial angle of attack | 2.8° |

Mesh: C-topology, 57 522 hex cells, wall-resolved to y⁺ ≈ 1, far field ≈ 30
chords. Non-orthogonality 60, skewness 0.40.

![Mesh](../results/benchmark_baseline_coldstart_20260729/rae2822_mesh.png)

The wall stack is much finer than in the low-Reynolds case — the leading-edge
panel shows the y⁺ ≈ 1 resolution that transonic skin-friction sensitivity
requires, against the y⁺ ≈ 30 wall-function mesh used for
[Case 2](CASE_LOWRE_MH139F.md#57-mesh).

**Cell count history.** The first mesh had 99 k cells with a mild radial
expansion ratio. The adjoint chain-rule step exceeded 28 GB of WSL memory and
the container was killed. Rebuilding with a slightly more aggressive expansion
gave 57 k cells at unchanged quality metrics.

---

## 3. Optimization problem

```
minimize      C_D(s, α)
w.r.t.        s ∈ [-0.02, 0.02]^58 m,   α ∈ [0°, 10°]
subject to    C_L(s, α) = 0.824
              thickcon(s) ≥ 0.5, volcon(s) ≥ 1.0, rcon(s) ≥ 0.8
```

59 design variables total: 58 FFD y-displacements (30 × 2 × 2 lattice, minus
two rigid-body modes removed by LE/TE camber-preservation) **plus the angle of
attack**.

> Contrast with [Case 2](CASE_LOWRE_MH139F.md), where α is fixed and the shape
> alone must meet the lift target.

Configuration: `PYOPTSPARSE_SLSQP`, `max_iter = 10`, 4 MPI ranks.

---

## 4. Results

### 4.1 SLSQP trajectory (10 evaluations)

| Eval | Kind | \|shape\| (m) | α (°) | C_D | Counts | Δcts | C_L | L/D | Wall (min) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | base | 0.000e+00 | 2.800 | 0.018385 | 183.8 | 0.0 | 0.788 | 42.9 | 96 |
| 2 | eval | 1.775e-03 | 2.934 | 0.019400 | 194.0 | +10.2 | 0.827 | 42.6 | 124 |
| 3 | eval | 6.728e-03 | 2.638 | 0.016417 | 164.2 | −19.7 | 0.799 | 48.6 | 133 |
| 4 | eval | 5.322e-03 | 2.687 | 0.016149 | 161.5 | −22.4 | 0.815 | 50.5 | 139 |
| 5 | eval | 6.748e-03 | 2.675 | 0.015942 | 159.4 | −24.4 | 0.822 | 51.6 | 155 |
| 6 | eval | 1.237e-02 | 2.461 | 0.018022 | 180.2 | −3.6 | 0.784 | 43.5 | 177 |
| 7 | eval | 7.895e-03 | 2.629 | 0.015251 | 152.5 | −31.3 | 0.825 | 54.1 | 153 |
| 8 | eval | 2.033e-02 | 2.136 | 0.025661 | 256.6 | +72.8 | 0.676 | 26.3 | 156 |
| **9** | **eval** | **9.385e-03** | **2.568** | **0.014713** | **147.1** | **−36.7** | **0.826** | **56.2** | 162 |
| 10 | eval | 1.662e-02 | 2.256 | 0.022346 | 223.5 | +39.6 | 0.721 | 32.3 | 166 |

SLSQP's exploration–exploitation pattern is visible: descent steps (3–5, 7, 9)
punctuated by aggressive probes that violate the C_L equality (6, 8, 10) and
get rejected. Eval 9 is the deepest C_D that also satisfies all constraints.
Eval 10 is the final line-search probe.

### 4.2 Reported optimum (Eval 9)

| Metric | Baseline (Eval 1) | Optimized (Eval 9) | Change |
|---|---:|---:|---:|
| C_D | 0.018385 (183.8 ct) | **0.014713 (147.1 ct)** | **−36.7 ct (−20.0 %)** |
| C_L | 0.788 | 0.826 (target 0.824, err +0.002) | at target |
| L/D | 42.9 | 56.2 | +31 % |
| Angle of attack | 2.800° | 2.568° | −0.232° |
| thickcon min | 1.000 | 0.991 | PASS |
| volcon | 1.000 | 1.000 | PASS |
| rcon min | 1.000 | 1.007 | PASS |
| Wall time | — | 27 h 18 m | 4 MPI ranks |

**These are the warm-started numbers as SLSQP reported them. See § 5 for the
corrected figure.**

---

## 5. Cold-start verification

DAFoam's optimization loop resets the U, p and T fields between primals via
`primalInitCondition`, but **not** the turbulence variable ν̃. In strong-shock
transonic RANS this lets each primal inherit some field state from the
previous iteration and land on a slightly different attractor than a cold
start would.

The baseline and the Eval 9 shape were therefore re-run as **independent
cold-start primals** — freshly from freestream, no inherited state:

| Case | C_D (counts) | C_L |
|---|---:|---:|
| Cold-start baseline (α = 2.8°, shape = 0) | 183.8 | 0.788 |
| Cold-start Eval 9 shape (α = 2.568°) | **157.2** | 0.803 |
| Warm-started Eval 9 (as SLSQP reported) | 147.1 | 0.826 |

The cold-start Eval 9 C_D is about 10 counts higher than the warm-started
value. The warm start puts the flow on a *steady* attractor with a slightly
weaker shock; the cold start lands on a *limit-cycle* attractor with a
marginally stronger shock at the same geometry. Both are valid RANS solutions.

> **Honest cold-start result: 183.8 → 157.2 counts = −26.6 counts (−14.5 %).**
> This is the number quoted in the README and on the project page, not the
> warm-started −20.0 %.

---

## 6. Physics

Cp distribution overlay (baseline grey, optimized coloured):

![Cp overlay](../results/benchmark_baseline_coldstart_20260729/cp_overlay.png)

- Upper-surface **suction peak deepens** (Cp ≈ −1.0 → −1.4) — the optimizer
  added camber near the leading edge.
- **Shock jump shrinks** from ΔCp ≈ 0.75 to ΔCp ≈ 0.5 — weaker compression,
  less wave drag.
- Lower surface barely changes. The drag reduction is almost entirely
  upper-surface shock weakening.

Shape delta between baseline and optimized surfaces:

![Shape delta](../results/benchmark_baseline_coldstart_20260729/shape_delta.png)

Peak |Δy| ≈ 3 mm on a 1 m chord — about **0.3 % chord**. A very subtle
geometric change producing a large flow response, which is characteristic of
transonic shock-drag optimization.

Measured directly from the two archived meshes, the peak is **3.02 mm at
x/c = 0.546** — inside the shock region, which is what the Cp overlay above
would predict.

### 6.2 Mesh deformation

![Mesh deformation](../results/benchmark_baseline_coldstart_20260729/rae2822_mesh_deformation.png)

Baseline and optimized meshes over x/c = 0.3–0.8, each panel carrying the
other case's surface as a dashed line. At 0.3 % chord the change is invisible
in the mesh alone; the dashed overlay is what makes it legible. The optimized
upper surface sits above the baseline through x/c ≈ 0.5–0.7 — the crown
reshaping that weakens the shock.

Note that the cells are *deformed*, never regenerated: IDWarp carries the
wall stack with the surface, which is what keeps y⁺ ≈ 1 intact across every
design iteration and keeps analytic mesh sensitivities available for the
chain rule.

### 6.3 Archived outputs

- [`results/benchmark_run_20260729/`](../results/benchmark_run_20260729/) — full SLSQP loop outputs
- [`results/benchmark_baseline_coldstart_20260729/`](../results/benchmark_baseline_coldstart_20260729/) — cold-start baseline flow field, Cp CSV, plots
- [`results/benchmark_eval9_coldstart_20260729/`](../results/benchmark_eval9_coldstart_20260729/) — cold-start Eval 9 flow field, Cp CSV, design vectors

Cp-extraction algorithm: [`CP_PLOT_NOTES.md`](CP_PLOT_NOTES.md).

---

## 7. Comparison to the paper

Offered as engineering context, not as a replication claim.

### 7.1 What lines up

- **Operating point** — M = 0.729, Re = 6.5 × 10⁶, C_L\* = 0.824 [paper § 4.2]
- **Turbulence model** — Spalart–Allmaras, low-Re [§ 2.3]
- **Numerical flux family** — JST [§ 2.3]
- **Adjoint framework** — discrete adjoint, PETSc GMRES linear solve [§ 2.3]
- **Optimizer family** — SQP [§ 2.4]
- **Design-variable scheme** — FFD y-displacements with per-station and volume
  constraints [§ 4.2]

### 7.2 What differs (verified against the paper text)

| Quantity | Paper (verified page) | This project |
|---|---|---|
| CFD solver | **ADflow**, ANK/NK Newton–Krylov [p.7 § 2.3] | **DAFoam v5** `DAHisaFoam`, pseudo-timestepping |
| Mesh topology | **O-mesh**, knife-edge TE [p.20 Fig. 10] | **C-mesh**, 1 % chord blunt TE |
| Primal convergence | **Density residual 10⁻¹⁵** [p.20] | C_D std ≤ 0.01, `primalMinIters` = 8000 |
| Optimizer | **SNOPT** [p.8 § 2.4] | pyOptSparse SLSQP via GEMSEO |
| Tolerances | Optimality **10⁻⁶**, step **10⁻³**, Hessian **50** [p.20] | GEMSEO defaults |
| Iterations at termination | ≈ **500** (Fig. 11) | 10 (`max_iter` cap) |
| Design-variable count | 8–52 [p.20 § 4.2.1] | 58 shape + α |

### 7.3 Where the numbers do not line up

| Quantity | Paper (N_DV = 30 column, Fig. 11) | This project |
|---|---:|---:|
| Baseline C_D | ≈ 199 counts | 183.8 counts (cold-start) |
| Optimized C_D | 110.09 counts | 147.1 (warm) / 157.2 (cold) |
| C_D reduction | ≈ 45 % | 20 % (warm) / 14.5 % (cold) |

**None of these are physics differences.** In decreasing order of likely
contribution:

1. **Termination.** The paper runs ≈ 500 iterations at SNOPT tolerances of
   10⁻⁶ / 10⁻³. Ours stopped at `max_iter = 10` with a reduced-gradient
   residual still around 0.78 (post-hoc estimate). SLSQP was still descending
   when it was cut off. More outer iterations would recover a large fraction
   of the gap.
2. **Primal convergence rigour.** Density residual 10⁻¹⁵ versus C_D
   statistical convergence at std ≤ 0.01 is a very different standard of
   gradient accuracy.
3. **Solver difference.** ADflow's Newton–Krylov stack drives each primal to a
   true steady state. `DAHisaFoam`'s pseudo-timestepping can land on
   limit-cycle attractors for strong shocks — exactly what § 5 observed.
4. **Mesh topology.** C-mesh with blunt TE versus O-mesh with knife-edge TE:
   an estimated 3–8 counts, mostly base region.
5. **Bound choice.** ± 2 % chord is more conservative than typical.

### 7.4 What is deliberately not reproduced

**SNOPT internal convergence metrics** (merit function, feasibility and
optimality as plotted in the paper's Fig. 11) are SNOPT-proprietary
quantities. They are documented precisely only in Gill, Murray and Saunders
(2005), not in the He 2019 text. Reconstructing them from `opt_history.h5`
without that source would produce invented formulas. That approach was
attempted, rejected on principle, and dropped. Only unambiguous raw quantities
— C_D history, C_L versus target, per-constraint values — are plotted.

---

## 8. Limitations

- **Reported C_D depends on optimization history.** `primalInitCondition`
  resets U, p, T but not ν̃, so warm and cold starts can reach different
  steady attractors at the same shape. See § 5.
- **Blunt trailing edge.** `mesh/airfoil_io.py:thicken_te()` adds ≈ 1 % chord
  of blunt base because `classy_blocks` requires it. Raises C_D by 3–8 counts
  and produces a near-TE Cp artifact — see [`CP_PLOT_NOTES.md`](CP_PLOT_NOTES.md).
- **Adjoint accuracy is limited by primal convergence, not by code.**
  `verify_gradients.py` at optimization-grade tolerances shows meaningful
  FD/adjoint disagreement. Publication-quality verification needs
  `stdTol = 1e-3` and ~15 000 primal iterations, which was out of budget.
- **Terminated at `max_iter = 10`,** with the optimizer still descending.
- **Two-dimensional emulation** via a single-cell spanwise slab. Fine for the
  RAE 2822; a 3D wing would need fresh mesh topology.

---

## 9. Reproducing this run

Environment setup (Docker image, GEMSEO install) is in
[`REPORT.md`](REPORT.md) § 8.

Outside the container:

```bash
python mesh/generate_cmesh.py --case case --run
checkMesh -case case
cd ffd && python generate_ffd.py && cd ..
head -3 case/FFD/wingFFD.xyz    # expect "30 2 2"
```

Inside the DAFoam container:

```bash
cd case
rm -f dRdWColoring_*.bin opt_log.txt opt_summary.md
rm -rf processor* 0 postProcessing
cp -r 0_orig 0

# Baseline sanity primal (~2 hours):
mpirun -np 4 python -u run_primal.py --preset benchmark > baseline.log 2>&1

# Full optimization (~27 hours at max-iter 10):
mpirun -np 4 python -u ../mdo/run_optimization.py \
    --preset benchmark --algo PYOPTSPARSE_SLSQP --max-iter 10 \
    > opt_log.txt 2>&1 &
tail -f opt_log.txt
```

Regenerate the mesh figures from the archives (no container needed):

```bash
python postprocessing/plot_mesh.py \
    --mesh     results/benchmark_baseline_coldstart_20260729/constant/polyMesh \
    --points   results/benchmark_baseline_coldstart_20260729/8001/polyMesh/points \
    --deformed results/benchmark_eval9_coldstart_20260729/10392/polyMesh/points \
    --title "RAE 2822" \
    --labels "Baseline (α = 2.8°)" "Optimized (eval 9, α = 2.568°)" \
    --out         results/benchmark_baseline_coldstart_20260729/rae2822_mesh.png \
    --compare-out results/benchmark_baseline_coldstart_20260729/rae2822_mesh_deformation.png
```

Replay a specific design at cold start:

```bash
mpirun -np 4 python -u run_at_shape.py --preset benchmark \
    --shape results/benchmark_run_20260729/shapes/eval0009_shape.npy
```

---

## 10. References

- He, X., Mader, C.A., Martins, J.R.R.A., Maki, K.J. (2019). *Robust
  aerodynamic shape optimization — from a circle to an airfoil*. Aerospace
  Science and Technology 87, 483–502.
- Gill, P.E., Murray, W., Saunders, M.A. (2005). *SNOPT: An SQP Algorithm for
  Large-Scale Constrained Optimization*. SIAM Review 47(1), 99–131.
- Jameson, A., Schmidt, W., Turkel, E. (1981). *Numerical solutions of the
  Euler equations by finite volume methods using Runge–Kutta time-stepping
  schemes*. AIAA Paper 81-1259.
- Spalart, P.R., Allmaras, S.R. (1992). *A One-Equation Turbulence Model for
  Aerodynamic Flows*. AIAA Paper 92-0439.
- Shared software-stack references are listed in [`REPORT.md`](REPORT.md) § 10.

### Companion documents

- [`REPORT.md`](REPORT.md) — shared pipeline methodology.
- [`CASE_LOWRE_MH139F.md`](CASE_LOWRE_MH139F.md) — Case Study 2, low-Reynolds.
- [`CP_PLOT_NOTES.md`](CP_PLOT_NOTES.md) — Cp extraction and blunt-TE handling.
- [`WORKFLOW.md`](WORKFLOW.md) — workflow diagram.
</content>
