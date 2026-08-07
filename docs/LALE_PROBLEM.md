# LALE Problem — Shape Optimization of the MH139-F Airfoil

Second case study for the same pipeline. Low-altitude long-endurance (LALE)
operating point taken from the AtlantikSolar solar-powered UAV design of
Oettershagen et al. (2017).

The pipeline (mesh → FFD → CFD → adjoint → GEMSEO + pyOptSparse SLSQP) is
unchanged. The physics regime is entirely different from the transonic RAE
2822 case — low Reynolds, incompressible, fixed angle of attack.

---

## 1. Locked-in physical setup

| Symbol | Value | Source |
|---|---|---|
| Airfoil | **MH139-F** (custom low-Re, 11.6% thick) | Oettershagen 2017 §3.1 |
| Aircraft mass m | 6.93 kg | Table 1 |
| Weight W = m·g | 67.98 N | derived |
| Wing span b | 5.69 m | Table 1 |
| Wing chord c | **0.305 m** | Table 1 |
| Wing area S = b·c | 1.735 m² | derived (rectangular wing per §3.1) |
| Cruise airspeed U∞ | **8.30 m/s** | measured average during 81-hour endurance flight, §4 |
| Altitude | **500 m ISA** | our design choice (round number near actual test location Rafz, ~430 m) |
| Air density ρ | **1.1673 kg/m³** | ISA at 500 m |
| Dynamic viscosity μ | **1.7737 × 10⁻⁵ Pa·s** | Sutherland at T = 284.90 K |
| Kinematic viscosity ν | **1.5196 × 10⁻⁵ m²/s** | μ/ρ |
| Reynolds number Re | ρUc/μ = **166 626** ≈ **1.6 × 10⁵** | derived |
| Mach number M | U/a = 8.30/338.4 = **0.024** | derived — fully incompressible regime |
| Angle of attack α | **4.5° — HARD CONSTRAINT (fixed constant, NOT a design variable)** | user choice |
| Target lift coefficient C_L\* | **0.975** | W / (½ρU²S), assuming uniform spanwise loading |

Notes:
- The paper does not state an explicit design altitude. 500 m is our choice, close to Rafz elevation.
- C_L is set by level-flight equilibrium and does not depend on α. Fixing α at 4.5° instead of 4.0° does not change the C_L target; it only changes the geometric angle the section is asked to produce that C_L at.
- The C_L target uses the simplest assumption (uniform loading, 2D section CL = 3D wing CL). Finite-wing corrections would shift the section CL up by 10-27 %; these are neglected as is standard practice in 2D airfoil MDO papers.

## 2. Formal optimization problem

```
minimize      C_D(shape)
w.r.t.        shape ∈ [-0.02, 0.02]^n   (FFD y-displacements, m)
subject to    C_L(shape) = 0.975         (lift equality at α = 4.5°)
              thickcon(shape) ≥ 0.5      (per-station thickness ratio)
              volcon(shape)   ≥ 1.0      (airfoil-area ratio)
              rcon(shape)     ≥ 0.8      (LE radius ratio)
              α is CONSTANT at 4.5° — not a design variable
```

Compared to the transonic RAE 2822 case:
- α is removed from the design vector (fixed hard constraint)
- Design space is only 58 shape DVs (was 59 = 58 + aoa)
- All other constraints identical in form; only reference values change

---

## 3. Software / solver changes vs the transonic case

| Component | Transonic case | LALE case |
|---|---|---|
| CFD solver | `DAHisaFoam` (density-based compressible, JST) | **`DASimpleFoam`** (incompressible SIMPLE) |
| Fluid model | Compressible, includes T, ρ | Incompressible, ρ constant, no T |
| Turbulence | Spalart-Allmaras low-Re | Same (SA low-Re) — **note this is a simplification; SA does not resolve the laminar separation bubble known to exist for MH-series at Re ~ 10⁵** |
| BC fields | U, p, T, nut, nuTilda, alphat | U, p, nut, nuTilda (no T, no alphat) |
| Reference pressure | Absolute p (Pa) | Relative p / ρ (m²/s²) |
| Case directory | `case/` | `case_lale/` (fresh copy) |

The switch to incompressible is because M = 0.024 is well outside DAHisaFoam's design range. Compressible density-based solvers with JST become poorly conditioned as M → 0.

The SA-fully-turbulent choice is a known simplification. Absolute CD will be overpredicted (SA cannot represent the laminar separation bubble that MH airfoils naturally exhibit at these conditions). The optimization signal is still valid — the gradient points in a defensible design direction — but comparing absolute CD numbers to XFOIL or wind-tunnel data will show a mismatch.

---

## 4. Step-by-step implementation plan

### Stage 1 — Prepare the MH139-F airfoil

#### 1.1 Clean the raw `.dat` file

The file `airfoils/MH139-F.dat` was saved from a webpage and carries HTML wrapper
lines at the top and bottom (`<pre><div class="text_to_html">MH 139F-0`).

- Read the file, strip anything that isn't `x y` coordinate pairs
- Write a cleaned version to `airfoils/mh139f_clean.dat`

#### 1.2 Determine and split the format

Selig format (starts near TE upper, walks CCW → LE → TE lower). Split into
two files matching the format that `mesh/airfoil_io.py:read_profile_pair`
expects:

- `case_lale/profiles/MH139F_SS.profile` — suction (upper) surface, LE → TE
- `case_lale/profiles/MH139F_PS.profile` — pressure (lower) surface, LE → TE

#### 1.3 Smooth + resample

Reuse `mesh/airfoil_io.py:cosine_resample()` (splprep spline fit with
cosine redistribution). Target 200 points per side, matching the RAE 2822
setup.

#### 1.4 Trailing edge treatment

MH139-F has a sharp TE. `classy_blocks` requires a non-zero TE thickness.
Apply the existing `thicken_te()` helper with `gap_target = 2e-3` (in
normalized chord units — becomes ~0.6 mm after scaling to physical chord).

#### 1.5 Chord scaling decision

Two equivalent options:

- **Option A** (recommended): keep the profile files at unit chord
  (`x ∈ [0, 1]`) and let the mesh generator apply the 0.305 m scaling
  when building blocks. Cleaner separation.
- Option B: scale the profile coordinates by 0.305 at split time so
  they are already physical.

Going with Option A.

### Stage 2 — Create the `case_lale/` folder

```bash
cp -r case case_lale
```

Then wipe the RAE 2822 artefacts:

```bash
cd case_lale
rm -rf constant/polyMesh/          # will be regenerated for MH139-F
rm -f FFD/wingFFD.xyz              # will be regenerated
rm -f profiles/RAE2822*.profile    # source-of-truth is a different airfoil now
rm -rf processor* 0 postProcessing # any live run state
rm -f opt_log.txt opt_summary.md dRdWColoring_*.bin
rm -f *.foam
```

Then drop in the new profile files at `case_lale/profiles/MH139F_*.profile`.

### Stage 3 — Regenerate the mesh for MH139-F at physical scale

#### 3.1 Add a LALE preset in `mesh/boundary_layer.py`

Add a new preset alongside the existing `TUTORIAL` and `BENCHMARK`:

```python
LALE_500M = FlowConditions(
    U=8.30, rho=1.1673, mu=1.7737e-5,
    chord=0.305, y_plus=1.0,
)
```

#### 3.2 Update `mesh/generate_cmesh.py` for the new preset

Parameters that change:
- Profile file paths → `case_lale/profiles/MH139F_*.profile`
- `CHORD_TE = 0.998` (nondimensional; scaled at build time)
- `H_TOP`, `R_FRONT`, `L_WAKE` — retain as multiples of chord (30 × 0.305 = 9.15 m physical); no code change needed
- `C2C_BL` (radial expansion) — try 1.10 first; low-Re wall stack is thicker relative to chord so shallower expansion is fine
- First-cell height — computed automatically from the LALE_500M preset via `first_cell_height(FlowConditions)`

Expected mesh: **30 000 - 45 000 hex cells** (fewer than the transonic 57 k because the y+ = 1 wall stack is coarser at Re = 1.6 × 10⁵).

#### 3.3 Regenerate

```bash
python mesh/generate_cmesh.py --case case_lale --preset lale --run
checkMesh -case case_lale
```

Expected: non-orth < 65, skewness < 0.5, aspect ratio ≤ 2e5 (much smaller than the 1.65e6 at transonic because wall stack is coarser).

### Stage 4 — Regenerate the FFD lattice

#### 4.1 Update `ffd/generate_ffd.py`

- Point the profile reader at `case_lale/profiles/MH139F_*.profile`
- Same lattice: 30 × 2 × 2
- Bounding-box margins scale automatically because the airfoil bounding box shrinks (chord 0.305 vs 1.0)
- Same LE / TE camber-preservation

#### 4.2 Generate

```bash
cd ffd
python generate_ffd.py --case ../case_lale
cd ..
head -3 case_lale/FFD/wingFFD.xyz    # should show "30 2 2"
```

### Stage 5 — Switch the solver from compressible to incompressible

#### 5.1 New `case_lale/system/controlDict`

- `application simpleFoam;`
- `startFrom latestTime;`
- `endTime 15000;` (same iteration budget as transonic)
- `deltaT 1;` (pseudo-time in incompressible SIMPLE)
- `writeControl timeStep;` `writeInterval 8000;`
- Keep the `forceCoeffs1` functionObject; update constants:
  ```
  rho          rhoInf;
  rhoInf       1.1673;
  magUInf      8.30;
  lRef         0.305;
  Aref         0.00305;                  # = chord × 0.01 (2D slab span)
  liftDir      (-0.07846 0.99692 0);     # (-sin 4.5°, cos 4.5°, 0)
  dragDir      ( 0.99692 0.07846 0);     # ( cos 4.5°, sin 4.5°, 0)
  pitchAxis    (0 0 1);
  ```

#### 5.2 New `case_lale/system/fvSchemes`

Standard incompressible-SIMPLE schemes:

```
ddtSchemes       { default steadyState; }
gradSchemes      { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes {
    default                          none;
    div(phi,U)                       bounded Gauss linearUpwindV grad(U);
    div(phi,nuTilda)                 bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))   Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes    { default corrected; }
wallDist         { method meshWaveFrozen; }
```

#### 5.3 New `case_lale/system/fvSolution`

```
solvers {
    p    { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.01; }
    U    { solver smoothSolver; smoother GaussSeidel; nSweeps 1; tolerance 1e-8; relTol 0.1; }
    nuTilda { solver smoothSolver; smoother GaussSeidel; nSweeps 1; tolerance 1e-8; relTol 0.1; }
}
SIMPLE {
    nNonOrthogonalCorrectors 1;
    consistent               true;
    residualControl { p 1e-6; U 1e-7; nuTilda 1e-7; }
}
relaxationFactors {
    fields { p 0.3; }
    equations { U 0.7; nuTilda 0.7; }
}
```

#### 5.4 New `case_lale/0_orig/` initial and boundary conditions

Files to create / modify:

- `case_lale/0_orig/U`
  - Internal field: `uniform (8.2745 0.6512 0)` — that's (U cos 4.5°, U sin 4.5°, 0)
  - `wing`: `noSlip;`
  - `inout`: `freestreamVelocity;` with `freestreamValue uniform (8.2745 0.6512 0);`
  - `symmetry1`, `symmetry2`: `symmetry;`

- `case_lale/0_orig/p`
  - dimensions `[0 2 -2 0 0 0 0]` (incompressible: kinematic pressure = p/ρ)
  - Internal field: `uniform 0;`
  - `wing`: `zeroGradient;`
  - `inout`: `freestreamPressure;` with `freestreamValue uniform 0;`
  - symmetries: `symmetry;`

- `case_lale/0_orig/nuTilda`
  - Internal field: `uniform 4.5e-5;` (same seed as before; equivalent to turbulence intensity ~2 % at these conditions)
  - `wing`: `fixedValue uniform 0;`
  - `inout`: `freestreamValue uniform 4.5e-5;`

- `case_lale/0_orig/nut`
  - `wing`: `nutLowReWallFunction;` with `value uniform 0;`
  - `inout`: `zeroGradient;`

- **Delete** `case_lale/0_orig/T` and `case_lale/0_orig/alphat` (incompressible does not use them).
- **Delete** `case_lale/0_orig/include/freestreamConditions` if it references thermodynamic values.

#### 5.5 New `case_lale/constant/transportProperties`

```
transportModel  Newtonian;
nu              nu [0 2 -1 0 0 0 0] 1.5196e-5;
```

#### 5.6 Update `case_lale/constant/turbulenceProperties`

Keep RAS with Spalart-Allmaras (same as before). No change needed.

### Stage 6 — Update `case_lale/dafoam_model.py`

#### 6.1 Add the LALE preset

```python
PRESETS = {
    "tutorial":  dict(U0=248.0,  p0=101325.0, T0=300.0,   CL_target=0.7,   aoa0=2.52517169),
    "benchmark": dict(U0=248.08, p0=39011.0,  T0=288.15,  CL_target=0.824, aoa0=2.8),
    "lale":      dict(U0=8.30,   rho0=1.1673, nu=1.5196e-5,
                      CL_target=0.975, aoa0=4.5, incompressible=True),
}
```

#### 6.2 Solver branch inside `make_options`

- If `preset == "lale"`: `solverName = "DASimpleFoam"`; drop `T` from `primalInitCondition`; drop temperature-related normalization; keep `useWallFunction = False`.
- Otherwise: current DAHisaFoam pathway.

#### 6.3 Fix α — remove `patchV` from the design vector

Two implementation options:

- **Option A**: leave `patchV` as an input but set both `patchV.upper == patchV.lower == fixed value` in the design space. Ugly but preserves existing wiring.
- **Option B**: remove `patchV` from `inputInfo` entirely and set the freestream direction via the BC files in `0_orig/U`. The scenario has only `shape` as a design variable.

Going with **Option B** (cleaner).

Concretely in `make_options`:
- Delete the `"patchV"` entry from `inputInfo` when `preset == "lale"`
- Delete the `patchVelocityInputName` field from CD and CL function definitions

In `Top.setup` / `Top.configure`:
- Only `self.dvs.add_output("shape", ...)` — no `patchV` DV output
- Only `self.add_design_var("shape", ...)` — no aoa DV
- Freestream velocity direction comes entirely from `0_orig/U`

#### 6.4 Update reference area A0

```python
A0 = 0.305 * 0.01   # chord × 2D-slab span
```

for the LALE preset only; leave the transonic value alone.

### Stage 7 — Update `mdo/run_optimization.py`

#### 7.1 Design-space branch

```python
if args.preset == "lale":
    ds.add_variable("shape", size=disc.n_shape,
                    lower_bound=-0.02, upper_bound=0.02,
                    value=np.zeros(disc.n_shape))
    # aoa is NOT a DV; it is fixed at 4.5° via the BC files
else:
    ds.add_variable("shape", ...)                          # as before
    ds.add_variable("aoa", size=1, lower_bound=0.0,
                    upper_bound=10.0, value=np.array([aoa0]))
```

#### 7.2 Summary header

The auto-writer inside `run_optimization.py` computes derived quantities
(M, Re, q∞) from the preset. Nothing to change there — the formulas work
with the LALE preset once we've added it.

### Stage 8 — Sanity baseline primal

```bash
docker run -it --rm -u dafoamuser --memory=28g --memory-swap=60g \
    --mount "type=bind,src=$(pwd),target=/home/dafoamuser/mount" \
    -w /home/dafoamuser/mount \
    dafoam-gemseo:latest bash

cd case_lale
rm -f dRdWColoring_*.bin
rm -rf processor* 0
cp -r 0_orig 0

mpirun -np 4 python -u run_primal.py --preset lale > baseline_lale.log 2>&1
tail -f baseline_lale.log
```

Expected:
- CD in the range 100-250 drag counts (SA fully turbulent will overpredict; XFOIL would give ~90-130)
- CL somewhere between 0.5 and 0.9 (unoptimized MH139-F at α = 4.5° in 2D fully turbulent)
- Convergence in about 5000-10000 pseudo-time steps (incompressible SIMPLE with SA)

### Stage 9 — Verify adjoint (optional but recommended once)

```bash
mpirun -np 4 python -u verify_gradients.py --task directional --preset lale
```

Some FD/adjoint disagreement at optimization-grade primal tolerances is
expected, exactly as in the transonic case.

### Stage 10 — Launch the optimization

```bash
cd case_lale
rm -f dRdWColoring_*.bin opt_log.txt opt_summary.md
rm -rf processor* 0
cp -r 0_orig 0

mpirun -np 4 python -u ../mdo/run_optimization.py \
    --preset lale --algo PYOPTSPARSE_SLSQP --max-iter 10 \
    > opt_log.txt 2>&1 &

tail -f opt_log.txt
# In a second WSL terminal:
tail -f case_lale/opt_summary.md
```

### Stage 11 — Post-process

Reuse `docs/plot_cp.py` after exporting Cp CSV from ParaView (same
procedure as the transonic case; described in `docs/CP_PLOT_NOTES.md`).

Freestream constants for the Cp calculation in the LALE case:
- `rhoInf = 1.1673`
- `magUInf = 8.30`
- `p_ref = 0` (kinematic pressure)

Recall that in incompressible OpenFOAM, `p` from the solution is
kinematic pressure (p/ρ, dimensions m²/s²). To compute Cp:

```
Cp = (p_solution - p_inf_kinematic) / (0.5 * U∞²)
   = p_solution / (0.5 * 8.30²)     # p_inf_kinematic = 0
```

So the `Cp` Calculator expression in ParaView becomes:
```
p / (0.5 * 8.30 * 8.30)
```

(instead of the compressible-form `(p - 39011) / (0.5 * 0.472 * 248.08^2)`
we used for the transonic benchmark.)

Backup pattern:
- `results/lale_baseline_YYYYMMDD/`
- `results/lale_run_YYYYMMDD/`
- `results/lale_optimum_coldstart_YYYYMMDD/` (replay at the optimum shape)

### Stage 12 — Update the reports

- Add a "Case Study 2 — LALE" section to `docs/REPORT.md`
- Reference this file (`docs/LALE_PROBLEM.md`) for the detailed setup
- Add Oettershagen et al. 2017 to the References section
- Once results are in, add the same result tables + Cp overlay + shape delta plots to the report

---

## 5. Total effort estimate

| Stage | Hands-on effort | Wall time (compute) |
|---|---|---|
| 1 — airfoil prep | 30 min |  — |
| 2 — case_lale/ folder | 15 min |  — |
| 3 — mesh regen | 30 min | 2-3 min compute |
| 4 — FFD regen | 15 min |  — |
| 5 — solver / BC / schemes | 2-3 hours (new territory, expect iteration) |  — |
| 6 — dafoam_model.py preset | 45 min |  — |
| 7 — mdo/run_optimization.py branch | 15 min |  — |
| 8 — baseline primal | 15 min hands-on | 1-2 hours compute |
| 9 — gradient verify (optional) | 15 min hands-on | 3-4 hours compute |
| 10 — optimization (max-iter 10) | 30 min hands-on | 15-25 hours compute |
| 11 — post-processing | 1 hour hands-on | negligible |
| 12 — report update | 1-2 hours | — |
| **Total** | **~8-10 hours hands-on** | **~20-30 hours compute** |

---

## 6. Known simplifications and their implications

- **SA fully turbulent** cannot represent the laminar separation bubble.
  Absolute CD will be overpredicted by an estimated 20-50 counts vs XFOIL /
  wind-tunnel data. The optimizer's gradient direction remains defensible.
- **2D uniform loading** assumption means the C_L target is treated as
  section CL = wing CL. Finite-wing corrections would shift the section
  target up by 10-27 % but are neglected as standard in 2D airfoil MDO.
- **Fixed α = 4.5°** removes one degree of freedom from the optimizer.
  Shape alone must reach C_L = 0.975. If the physical shape cannot reach
  0.975 within the ± 2 % chord DV bounds, the optimizer will report
  infeasibility.
- **Blunt TE from `thicken_te()`** applies here too; the Cp plotter
  needs the same near-TE trim (x/c ≤ 0.98) documented in
  `docs/CP_PLOT_NOTES.md`.

---

## 7. Files to be created or modified

Live workflow files (all edits, no new files unless noted):

| File | Change type | What |
|---|---|---|
| `mesh/boundary_layer.py` | edit | add `LALE_500M` preset |
| `mesh/generate_cmesh.py` | edit | accept preset argument; use physical chord = 0.305 m for LALE |
| `mesh/airfoil_io.py` | possibly edit | small helper to split a Selig `.dat` into two `.profile` files |
| `airfoils/MH139-F.dat` | edit | strip HTML wrapper from top/bottom |
| `airfoils/mh139f_clean.dat` | **new** | cleaned coordinate file |
| `case_lale/` | **new** (copy of `case/`) | full LALE case tree |
| `case_lale/profiles/MH139F_SS.profile` | **new** | suction surface after resample |
| `case_lale/profiles/MH139F_PS.profile` | **new** | pressure surface after resample |
| `case_lale/system/controlDict` | rewrite | incompressible application + new forceCoeffs1 constants |
| `case_lale/system/fvSchemes` | rewrite | incompressible SIMPLE schemes |
| `case_lale/system/fvSolution` | rewrite | SIMPLE + relaxation |
| `case_lale/0_orig/U`, `p`, `nut`, `nuTilda` | rewrite | incompressible BCs, fixed α = 4.5° freestream |
| `case_lale/0_orig/T`, `alphat` | delete | not used in incompressible |
| `case_lale/constant/transportProperties` | rewrite | incompressible with ν = 1.5196e-5 |
| `case_lale/constant/turbulenceProperties` | copy | unchanged (RAS + SA) |
| `case_lale/dafoam_model.py` | edit | add LALE preset; solver conditional; drop patchV as DV |
| `mdo/run_optimization.py` | edit | design-space branch for LALE (only shape, no aoa) |

Docs:

| File | Change type |
|---|---|
| `docs/LALE_PROBLEM.md` (this file) | new |
| `docs/REPORT.md` | add "Case Study 2" section once results are in |
| `README.md` | add a one-liner + link to this file once results are in |

---

## 8. Reference

- Oettershagen, P., Melzer, A., Mantel, T., Rudin, K., Stastny, T.,
  Wawrzacz, B., Hinzmann, T., Leutenegger, S., Alexis, K., Siegwart, R.
  (2017). *Design of small hand-launched solar-powered UAVs: From concept
  study to a multi-day world endurance record flight*. Journal of Field
  Robotics 34(7), 1352-1385. DOI:10.1002/rob.21717.
  Local copy: `E:\ISAE\Applied AD I\oettershagen2017.pdf`
- Hepperle, M. (author of the MH series of low-Re airfoils):
  http://www.mh-aerotools.de/airfoils/
