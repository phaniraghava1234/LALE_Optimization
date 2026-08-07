#!/usr/bin/env python3
"""OpenMDAO/MPhys/DAFoam problem for the MH139-F LALE case.

Importable module that assembles the differentiable multiphysics model:

    dv (shape) --> geometry (pyGeo FFD) --> IDWarp mesh warp
        --> DAFoam primal (DASimpleFoam, incompressible, SA turbulence)
        --> functionals CD, CL  (+ geometric constraints thickcon/volcon/rcon)

Reverse mode produces the discrete adjoint state and total derivatives
d{CD, CL}/d{shape} at the cost of approximately one extra flow solve
per functional.

LALE-specific vs the transonic case (case/dafoam_model.py):

    - Solver:            DASimpleFoam (incompressible, SIMPLE)
    - Alpha:             FIXED at 4.5 deg (not a design variable)
    - Freestream:        set entirely via 0_orig/U (no patchV DV)
    - CL target:         0.975 (from level-flight equilibrium of the
                         AtlantikSolar UAV in Oettershagen 2017)
    - Re per chord:      1.666e5 (500 m ISA, U = 8.30 m/s, c = 0.305 m
                         physical; represented in unit-chord CFD as
                         U = 2.5315 m/s, c = 1 m)

Entry points that consume this module:

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
# "lale" : Oettershagen et al. (2017), AtlantikSolar cruise condition.
#          Physical:  c = 0.305 m, U = 8.30 m/s, rho = 1.1673 kg/m^3,
#                     nu = 1.5196e-5 m^2/s, alpha = 4.5 deg, CL* = 0.975.
#          Unit-chord CFD:  U = 2.5315 m/s (= 8.30 * 0.305) at same Re.
PRESETS = {
    "lale": dict(
        U0=2.5315,
        aoa0=4.5,
        rho0=1.1673,
        nu=1.5196e-5,
        CL_target=0.975,
    ),
}

NUTILDA0 = 4.5e-5              # SA freestream seed = ~3 * nu
A0 = 0.01                      # reference area = chord(1) x 2D span(0.01)


def make_options(preset: str = "lale"):
    """Build (daOptions, meshOptions, params) for the LALE preset."""
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset!r}; valid: {list(PRESETS)}")
    p = PRESETS[preset]
    U0, aoa0, rho0 = p["U0"], p["aoa0"], p["rho0"]
    Ux = float(U0 * np.cos(np.radians(aoa0)))
    Uy = float(U0 * np.sin(np.radians(aoa0)))

    daOptions = {
        "designSurfaces": ["wing"],
        # Incompressible SIMPLE solver -- appropriate for M ~ 0.007 LALE
        # flow. Freestream velocity is set entirely by 0/U (fixed alpha).
        "solverName": "DASimpleFoam",
        # NOTE: writeDeformedFFDs / writeDeformedConstraints were tried here
        # (the AeroOpt regression test sets them) and coincided with a SEGV
        # inside the first adjoint linear solve. They are pure convenience --
        # shape evolution is already recoverable from the per-evaluation
        # shapes/*.npy archive -- so they stay off. Re-enable only if you
        # want to retest them in isolation.
        # Reset only U at the start of each primal (p is kinematic and
        # its reference is arbitrary). Prevents state contamination from
        # previous optimization iterations when the mesh warps.
        "primalInitCondition": {"U": [Ux, Uy, 0.0]},
        "primalBC": {
            # Matched to dafoam/tests/runRegTests_AeroOpt.py -- the regression
            # test that ships with THIS container's DAFoam commit and passes
            # here. (The online NACA0012 tutorial targets a newer mphys/DAFoam
            # stack and omits transport:nu; the version-matched test includes
            # it, so we do too.)
            "U0":       {"variable": "U",       "patches": ["inout"], "value": [Ux, Uy, 0.0]},
            "p0":       {"variable": "p",       "patches": ["inout"], "value": [0.0]},
            "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [NUTILDA0]},
            "useWallFunction": True,
            "transport:nu": p["nu"],
        },
        # CD-std/slope gate DEACTIVATED (-1), matching runRegTests_AeroOpt.py.
        # The reference relies purely on primalMinResTol, and that is the
        # right criterion for adjoint validity: the discrete adjoint
        # linearizes dR/dW at the converged state, so what matters is that
        # R(W) is actually small -- not that CD has stopped wandering.
        # Our previous setup exited on CD-std with residuals still ~1e-5.
        # Floor before ANY convergence gate can fire. With the loose 1e-4
        # residual gate below, a low floor let the primal exit at iteration
        # 633 -- before the flow was developed (CD needs ~3000-4000 iters to
        # settle on this mesh). The adjoint then linearizes about a transient
        # state. Keep this above the CD settling time.
        # Overridable so a cold-start replay can be run long enough to
        # actually converge without editing this file:
        #     DAFOAM_MIN_ITERS=15000 mpirun -np 4 python run_at_shape.py ...
        "primalMinIters": int(os.environ.get("DAFOAM_MIN_ITERS", 3000)),
        "primalFuncStdTol": {
            "stdTol": -1.0,          # -1 = deactivated
            "slopeTol": -1.0,        # -1 = deactivated
            "funcName": "CD",
            "nStepsFrac": 0.2,
        },
        "function": {
            # Canonical DAFoam force setup: parallelToFlow / normalToFlow
            # WITH patchVelocityInputName. This is what the transonic case,
            # the AeroOpt regression test, and every DAFoam tutorial use.
            # Using "fixedDirection" (which we had) skips the direction-vs-
            # patchV linearization; the adjoint chain then leaves the BC-
            # derivative rows uninitialized (NaN) even when patchV is in
            # inputInfo. Direction is still fixed at 4.5 deg AoA because
            # we hold patchV constant at (U0, aoa0) in the OpenMDAO layer.
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
        # Adjoint linear solve settings — matched to runRegTests_AeroOpt.py
        # (the version-matched test that passes on this container).
        "adjEqnOption": {
            "gmresRelTol": 1.0e-10,
            "pcFillLevel": 1,
            "jacMatReOrdering": "rcm",
        },
        # State normalization — matched to runRegTests_AeroOpt.py.
        # nuTilda uses the fixed 1e-3 scale, not NUTILDA0 * 10.
        "normalizeStates": {
            "U": U0,
            "p": U0 * U0 / 2.0,
            "phi": 1.0,
            "nuTilda": 1e-3,
        },
        # Production exit criterion for the optimization loop.
        #
        # The reference value is 1e-10, but this mesh plateaus at ~7.6e-5
        # (non-orthogonality 68 vs the reference O-mesh's 26), so 1e-10 is
        # unreachable and would reject every primal. 1e-4 sits just above the
        # plateau: combined with primalMinIters=3000, every evaluation runs a
        # comparable number of iterations and lands at a comparable level of
        # convergence.
        #
        # For gradient-based optimization, CONSISTENCY across evaluations
        # matters more than absolute convergence. A systematic discretization
        # bias largely cancels in the gradient, whereas a varying exit point
        # injects noise into the objective from one SLSQP iteration to the
        # next. That is also why the CD-std/slope gate stays deactivated
        # above -- it exits at a different iteration count for every shape.
        #
        # primalMinResTolDiff=1e4 puts the failure threshold at 1e-4*1e4 = 1.0,
        # permissive enough that an intermediate shape does not spuriously
        # abort the optimization.
        #
        # Tighten toward 1e-10 once mesh non-orthogonality is reduced.
        "primalMinResTol": 1.0e-4,
        "primalMinResTolDiff": 1e4,
        # Quality gate applied every iteration. Our LALE mesh has
        # non-orth ~68 and skewness ~1.0; give a small margin above.
        "checkMeshThreshold": {
            "maxNonOrth": 75.0,
            "maxSkewness": 3.0,
            "maxAspectRatio": 2.0e5,
        },
        # inputInfo mirrors DAFoam's AeroOpt regression test structure:
        # BOTH aero_vol_coords AND patchV are declared. patchV isn't just
        # a DV declaration -- it tells DAFoam how to linearize the velocity
        # BC at inout for the adjoint. Without it, adjoint matrix rows for
        # BC-derivatives are uninitialized (NaN at KSP iter 0).
        # We still hold alpha fixed at 4.5 deg in the MPhys layer above
        # (dafoam_discipline.py doesn't push patchV changes when fixed_aoa=True).
        "inputInfo": {
            "aero_vol_coords": {
                "type": "volCoord",
                "components": ["solver", "function"],
            },
            "patchV": {
                "type": "patchVelocity",
                "patches": ["inout"],
                "flowAxis": "x",
                "normalAxis": "y",
                "components": ["solver", "function"],
            },
        },
    }

    # ---- IDWarp mesh deformation ----------------------------------------
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
    """dv (shape) -> FFD geometry -> mesh warp -> primal -> functions."""

    def initialize(self):
        self.options.declare("preset", default="lale")

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
        CL_target = self.params["CL_target"]

        points = self.mesh.mphys_get_surface_mesh()
        self.geometry.nom_add_discipline_coords("aero", points)
        tri_points = self.mesh.mphys_get_triangulated_surface()
        self.geometry.nom_setConstraintSurface(tri_points)

        # ---- Shape design variables via FFD shape functions -----------
        # Interior FFD points: k=0/k=1 slaved (2D symmetry in z).
        # LE & TE points: j=0 and j=1 move in OPPOSITE directions so the
        # LE/TE camber line stays fixed (removes rigid-body y-translation
        # modes).
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

        # ---- top-level design variables (shape only, no aoa) ----------
        # patchV is declared as an input to DAFoam (see inputInfo above) so
        # the adjoint can linearize the velocity BC. Value is held constant
        # at freestream (U0, aoa0) -- NOT added as a design variable. This
        # gives the adjoint the BC-linearization info it needs while keeping
        # alpha fixed at 4.5 deg.
        U0, aoa0 = self.params["U0"], self.params["aoa0"]
        self.dvs.add_output("shape", val=np.zeros(self.n_shape))
        self.dvs.add_output("patchV", val=np.array([U0, aoa0]))
        self.connect("shape", "geometry.shape")
        self.connect("patchV", "scenario1.patchV")

        # These declarations are used by the pyOptSparse reference driver;
        # the GEMSEO route reads the same names via run_model/compute_totals.
        # patchV is NOT added as a design variable -- alpha stays fixed.
        self.add_design_var("shape", lower=-0.02, upper=0.02, scaler=10.0)
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
WRT = ["shape"]   # LALE case: shape is the only design variable


def build_problem(preset: str = "lale") -> om.Problem:
    """Assemble and set up the OpenMDAO problem in reverse (adjoint) mode."""
    prob = om.Problem()
    prob.model = Top(preset=preset)
    prob.setup(mode="rev")
    om.n2(prob, show_browser=False, outfile="mphys.html")
    return prob
