#!/usr/bin/env python3
"""OpenMDAO/MPhys/DAFoam problem for the RAE 2822 case.

Importable module that assembles the differentiable multiphysics model:

    dvs (shape, patchV) --> geometry (pyGeo FFD) --> IDWarp mesh warp
        --> DAFoam primal (DAHisaFoam, SA turbulence)
        --> functionals CD, CL  (+ geometric constraints thickcon/volcon/rcon)

Reverse mode produces the discrete adjoint state and total derivatives
d{CD, CL}/d{shape, aoa} at the cost of approximately one extra flow
solve per functional.

Adapted from the official DAFoam v4 RAE 2822 tutorial
(https://github.com/DAFoam/tutorials, RAE2822_Airfoil/runScript.py) and
restructured so that all three entry points share it:

    * ``run_primal.py``           - baseline primal
    * ``verify_gradients.py``     - adjoint gradient verification
    * ``run_at_shape.py``         - post-optimization replay
    * ``../mdo/run_optimization.py`` - full MDO scenario (GEMSEO)

Must be executed inside the DAFoam environment (Docker image
``dafoam/opt-packages`` or a native WSL install). See the top-level
README for setup instructions.
"""
import os

import numpy as np
import openmdao.api as om
from mphys.multipoint import Multipoint
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP

from dafoam.mphys import DAFoamBuilder, OptFuncs  # noqa: F401 (OptFuncs re-exported)

# =============================================================================
# Operating point presets
# =============================================================================
# "tutorial"  : the exact numbers of the official DAFoam tutorial. Use this
#               first -- it is known to converge with the shipped mesh/schemes.
# "benchmark" : He et al. (2019) / ADODG Case 2 conditions
#               M = 0.729, Re_c = 6.5e6, CL* = 0.824.
#               p is reduced so that Re matches at chord = 1 m with
#               mu = 1.8e-5 (see mesh/boundary_layer.py).
PRESETS = {
    "tutorial": dict(U0=248.0, p0=101325.0, T0=300.0, CL_target=0.7,
                     aoa0=2.52517169),
    "benchmark": dict(U0=248.08, p0=39011.0, T0=288.15, CL_target=0.824,
                      aoa0=2.8),
}

NUTILDA0 = 4.5e-5
A0 = 0.01  # reference area = chord(1 m) x span(0.01 m)


def make_options(preset: str = "tutorial"):
    """Build (daOptions, meshOptions, params) for a given operating point."""
    p = PRESETS[preset]
    U0, p0, T0, aoa0 = p["U0"], p["p0"], p["T0"], p["aoa0"]
    rho0 = p0 / T0 / 287.0
    Ux = float(U0 * np.cos(np.radians(aoa0)))
    Uy = float(U0 * np.sin(np.radians(aoa0)))

    daOptions = {
        "designSurfaces": ["wing"],
        # Density-based HiSA solver: the right tool for M ~ 0.73 with a shock.
        # (DASimpleFoam is incompressible; DARhoSimpleCFoam is the segregated
        #  compressible alternative -- see README "Solver choice".)
        "solverName": "DAHisaFoam",
        "primalInitCondition": {"U": [Ux, Uy, 0.0], "p": p0, "T": T0},
        "primalBC": {
            "charFarFieldBC": [Ux, Uy, 0.0, p0, T0],
            "useWallFunction": False,          # y+ ~ 1 mesh, resolve to wall
        },
        # Stop the primal when CD is statistically converged.
        # DAFoam v5 checks: std < stdTol AND |slope| < slopeTol.
        # WITHOUT primalMinIters set high, both checks trivially pass at
        # iter 1 (no variance yet) and the primal exits before the flow
        # has developed. Set primalMinIters=1000 so auto-stop can't fire
        # until we've iterated enough for the shock/BL to form.
        # slopeTol=1e-3 is the effective gate after minIters is reached.
        # stdTol=0.1 is deliberately loose (std over all iters is dominated
        # by early wild CD values and never reaches 1e-3).
        # (`tol` is the DAFoam v4 name, kept for backwards compat.)
        # OPTIMIZATION settings: middle ground between "cheap" (2500/0.1/1e-3)
        # and "verify-grade" (15000/1e-3/1e-6). Tighter slope+std reduces the
        # convergence-state variability that caused verify_gradients to fail;
        # 4000 minIters gives more room to settle. Cost per iter: ~45 min.
        "primalMinIters": 8000,
        "primalFuncStdTol": {
            "stdTol": 0.01,
            "slopeTol": 5e-5,
            "tol": 5e-5,
            "funcName": "CD",
            "nStepsFrac": 0.2,
        },
        "function": {
            "CD": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "parallelToFlow",
                "patchVelocityInputName": "patchV",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
            },
            "CL": {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": "normalToFlow",
                "patchVelocityInputName": "patchV",
                "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
            },
        },
        # ---- Phase 4: adjoint linear solve  [dR/dw]^T psi = dF/dw ----
        "adjStateOrdering": "cell",
        "adjEqnOption": {
            "gmresRelTol": 1.0e-6,
            "pcFillLevel": 1,
            "jacMatReOrdering": "natural",
            "gmresMaxIters": 2000,
            "gmresRestart": 2000,
        },
        "normalizeStates": {"U": U0, "p": p0, "T": T0, "nuTilda": NUTILDA0 * 10.0},
        # Quality gate applied EVERY iteration. Non-orth 70 and skewness 6 are
        # OK (our mesh: 60 and 0.40). AR limit bumped to 2e6 to accept our
        # y+=1 wall stack (AR ~1.6M at the wall is normal for wall-resolved
        # RANS; DAFoam's default 5000 is meant for wall-function meshes).
        # This matches the relaxed Phase 1 QC gate.
        "checkMeshThreshold": {
            "maxNonOrth": 70.0, "maxSkewness": 6.0, "maxAspectRatio": 2.0e6,
        },
        "inputInfo": {
            "aero_vol_coords": {"type": "volCoord",
                                "components": ["solver", "function"]},
            "patchV": {
                "type": "patchVelocity",
                "patches": ["inout"],
                "flowAxis": "x",
                "normalAxis": "y",
                "components": ["solver", "function"],
            },
        },
    }

    # ---- Phase 2.2: IDWarp mesh deformation ----
    # When the FFD points move, IDWarp propagates the surface displacement
    # into the volume with inverse-distance weighting, preserving the
    # boundary-layer stack (near-wall cells move almost rigidly).
    meshOptions = {
        "gridFile": os.getcwd(),
        "fileType": "OpenFOAM",
        "useRotations": False,
        "symmetryPlanes": [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.01], [0.0, 0.0, 1.0]],
        ],
    }
    return daOptions, meshOptions, p


# =============================================================================
# The MPhys model
# =============================================================================
class Top(Multipoint):
    """dvs -> FFD geometry -> mesh warp -> primal -> functions."""

    def initialize(self):
        self.options.declare("preset", default="tutorial")

    def setup(self):
        daOptions, meshOptions, self.params = make_options(self.options["preset"])
        self.daOptions = daOptions

        dafoam_builder = DAFoamBuilder(daOptions, meshOptions,
                                       scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.add_subsystem("mesh",
                           dafoam_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem("geometry",
                           OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))
        self.mphys_add_scenario(
            "scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder)
        )
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        self.connect("geometry.x_aero0", "scenario1.x_aero")

    def configure(self):
        U0, aoa0 = self.params["U0"], self.params["aoa0"]
        CL_target = self.params["CL_target"]

        points = self.mesh.mphys_get_surface_mesh()
        self.geometry.nom_add_discipline_coords("aero", points)
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # ---- Phase 2.1: shape design variables via shape functions ----
        # Interior FFD points: k=0/k=1 slaved (2D symmetry in z).
        # LE & TE points: j=0 and j=1 move in OPPOSITE directions so the
        # LE/TE camber line stays fixed (removes rigid-body modes).
        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        for i in range(1, pts.shape[0] - 1):
            for j in range(pts.shape[1]):
                shapes.append({pts[i, j, 0]: dir_y, pts[i, j, 1]: dir_y})
        for i in [0, pts.shape[0] - 1]:
            shapes.append({pts[i, 0, 0]: dir_y, pts[i, 0, 1]: dir_y,
                           pts[i, 1, 0]: -dir_y, pts[i, 1, 1]: -dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)
        self.n_shape = len(shapes)

        # ---- geometric constraints (evaluated on the FFD-embedded surface)
        leList = [[1e-3, 0.0, 1e-3], [1e-3, 0.0, 0.01 - 1e-3]]
        teList = [[0.961, 0.005, 1e-3], [0.961, 0.005, 0.01 - 1e-3]]
        self.geometry.nom_addThicknessConstraints2D("thickcon", leList, teList,
                                                    nSpan=2, nChord=10)
        self.geometry.nom_addVolumeConstraint("volcon", leList, teList,
                                              nSpan=2, nChord=10)
        self.geometry.nom_addLERadiusConstraints("rcon", leList, 2,
                                                 [0.0, 1.0, 0.0],
                                                 [-1.0, 0.0, 0.0])

        # ---- top-level design variables ----
        self.dvs.add_output("shape", val=np.zeros(self.n_shape))
        self.dvs.add_output("patchV", val=np.array([U0, aoa0]))
        self.connect("patchV", "scenario1.patchV")
        self.connect("shape", "geometry.shape")

        # These declarations are used by the pyOptSparse reference driver;
        # the GEMSEO route reads the same names via run_model/compute_totals.
        self.add_design_var("shape", lower=-1.0, upper=1.0, scaler=10.0)
        self.add_design_var("patchV", lower=[U0, 0.0], upper=[U0, 10.0],
                            scaler=0.1)
        self.add_objective("scenario1.aero_post.CD", scaler=10.0)
        self.add_constraint("scenario1.aero_post.CL", equals=CL_target)
        self.add_constraint("geometry.thickcon", lower=0.5, upper=3.0)
        self.add_constraint("geometry.volcon", lower=1.0)
        self.add_constraint("geometry.rcon", lower=0.8)


# canonical output names used everywhere downstream
OF = {
    "CD": "scenario1.aero_post.CD",
    "CL": "scenario1.aero_post.CL",
    "thickcon": "geometry.thickcon",
    "volcon": "geometry.volcon",
    "rcon": "geometry.rcon",
}
WRT = ["shape", "patchV"]


def build_problem(preset: str = "tutorial") -> om.Problem:
    """Assemble and set up the OpenMDAO problem in reverse (adjoint) mode."""
    prob = om.Problem()
    prob.model = Top(preset=preset)
    prob.setup(mode="rev")
    om.n2(prob, show_browser=False, outfile="mphys.html")
    return prob
