# Run archive: case_lale_20260806_084558
Archived 2026-08-06T08:50:31 on f361d2502723

## Provenance

- Commit: `unknown` (branch `unknown`)
- Working tree dirty: **False**

## Software

- openfoam: `v2506`
- python: `3.10.8`
- host: `f361d2502723`
- dafoam: `unknown`
- openmdao: `3.26.0`
- gemseo: `unknown`
- pygeo: `1.13.0`
- mphys: `unknown`

## Run settings

- preset: `lale`
- algorithm: `PYOPTSPARSE_SLSQP`
- max_iter: `50`
- n_ranks: `6`
- wall_time_s: `31039.1`
- n_evaluations: `50`
- n_gradients: `27`

## Results

- CD_baseline: `0.026373675527976773`
- CD_optimum: `0.037010699635785245`
- CD_reduction_counts: `-106.37`
- CD_reduction_pct: `-40.33`
- CL_final: `0.9747047099857273`
- CL_target: `0.975`
- aoa_final_deg: `4.5`
- LD_final: `26.335754783821915`
- shape_norm_final: `0.044309242870229346`
- feasible: `True`

## Mesh

- `note        "nPoints:48782  nCells:24009  nFaces:96418  nInternalFaces:47636";`

## Resolved daOptions

```json
{
  "designSurfaces": [
    "wing"
  ],
  "solverName": "DASimpleFoam",
  "primalInitCondition": {
    "U": [
      2.523696230345413,
      0.19861920083503945,
      0.0
    ]
  },
  "primalBC": {
    "U0": {
      "variable": "U",
      "patches": [
        "inout"
      ],
      "value": [
        2.523696230345413,
        0.19861920083503945,
        0.0
      ]
    },
    "p0": {
      "variable": "p",
      "patches": [
        "inout"
      ],
      "value": [
        0.0
      ]
    },
    "nuTilda0": {
      "variable": "nuTilda",
      "patches": [
        "inout"
      ],
      "value": [
        4.5e-05
      ]
    },
    "useWallFunction": true,
    "transport:nu": 1.5196e-05
  },
  "primalMinIters": 3000,
  "primalFuncStdTol": {
    "stdTol": -1.0,
    "slopeTol": -1.0,
    "funcName": "CD",
    "nStepsFrac": 0.2
  },
  "function": {
    "CD": {
      "type": "force",
      "source": "patchToFace",
      "patches": [
        "wing"
      ],
      "directionMode": "parallelToFlow",
      "patchVelocityInputName": "patchV",
      "scale": 26.735705375257716
    },
    "CL": {
      "type": "force",
      "source": "patchToFace",
      "patches": [
        "wing"
      ],
      "directionMode": "normalToFlow",
      "patchVelocityInputName": "patchV",
      "scale": 26.735705375257716
    }
  },
  "adjEqnOption": {
    "gmresRelTol": 1e-10,
    "pcFillLevel": 1,
    "jacMatReOrdering": "rcm"
  },
  "normalizeStates": {
    "U": 2.5315,
    "p": 3.2042461249999996,
    "phi": 1.0,
    "nuTilda": 0.001
  },
  "primalMinResTol": 0.0001,
  "primalMinResTolDiff": 10000.0,
  "checkMeshThreshold": {
    "maxNonOrth": 75.0,
    "maxSkewness": 3.0,
    "maxAspectRatio": 200000.0
  },
  "inputInfo": {
    "aero_vol_coords": {
      "type": "volCoord",
      "components": [
        "solver",
        "function"
      ]
    },
    "patchV": {
      "type": "patchVelocity",
      "patches": [
        "inout"
      ],
      "flowAxis": "x",
      "normalAxis": "y",
      "components": [
        "solver",
        "function"
      ]
    }
  }
}
```

## Contents

| Directory | Holds |
|---|---|
| `config/` | numerics dictionaries, BCs, and the source that generated them |
| `geometry/` | FFD lattice, airfoil profiles |
| `mesh/polyMesh/` | mesh as actually used |
| `optimization/` | summary, GEMSEO database, plots, per-evaluation shapes, logs |
| `flow/` | reconstructed final flow field |

## Recovering a flow field (no re-solve)

DAFoam stores one solution directory per evaluation inside each `processor*`, named as an incrementing pseudo-time. List them with `ls flow/processor0/`.

```bash
cp -r flow/processor*  <case>/
cp -r mesh/polyMesh    <case>/constant/polyMesh
cp    config/system/*  <case>/system/
cd <case>
reconstructPar -time 0.0007     # or -latestTime, or -allTime
touch case.foam && paraview case.foam
```

## Re-solving an evaluation from its design vector

Use this when the flow field was not archived, or to regenerate at different settings.

```bash
cp -r mesh/polyMesh   <case>/constant/polyMesh
cp -r geometry/FFD    <case>/FFD
cp -r config/0_orig   <case>/0_orig
cp    config/system/* <case>/system/
cd <case> && cp -r 0_orig 0
mpirun -np 4 python run_at_shape.py --shape <archive>/optimization/shapes/eval0007_shape.npy
```

## Not present at archive time

- `constant/thermophysicalProperties`
