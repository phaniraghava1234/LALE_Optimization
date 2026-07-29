#!/usr/bin/env python3
"""GEMSEO discipline wrapping the OpenMDAO/MPhys/DAFoam problem.

Exposes the CFD problem defined in ``case/dafoam_model.py`` as a single
GEMSEO :class:`Discipline`. Each evaluation performs the following
sequence, driven from ``_run`` (primal) and ``_compute_jacobian``
(adjoint):

    1. Push the FFD shape design-variable vector and the angle of attack
       into the underlying OpenMDAO problem.
    2. Warp the volume mesh through IDWarp (called inside the model
       during ``run_model``).
    3. Run the primal RANS solve; extract CD, CL, and the geometric
       constraint values.
    4. On gradient requests, run the discrete adjoint per functional and
       propagate the sensitivities back to the design vector.
    5. Optionally append a summary row to ``summary_path`` after each
       primal evaluation.

Because GEMSEO sees an ordinary differentiable discipline, any of its
gradient-based algorithms may drive the scenario (SLSQP,
PYOPTSPARSE_SLSQP, NLOPT_SLSQP, and so on).

Written for GEMSEO 6.x. In GEMSEO 5.x the base class was named
``MDODiscipline``, ``io.data`` was ``local_data``,
``io.input_grammar.defaults`` was ``default_inputs``, and
``_run(input_data)`` was ``_run(self)`` followed by
``self.store_local_data(**out)``.
"""
from __future__ import annotations

import hashlib
import logging
import time

import numpy as np

from gemseo.core.discipline import Discipline

LOGGER = logging.getLogger(__name__)

OUTPUTS = ["CD", "CL", "thickcon", "volcon", "rcon"]


class DAFoamDiscipline(Discipline):
    """shape (n), aoa (1, degrees)  -->  CD, CL, thickcon, volcon, rcon."""

    def __init__(self, preset: str = "tutorial", summary_path: str | None = None):
        super().__init__(name="DAFoamRAE2822")

        # Build the (expensive) DAFoam problem ONCE and keep it alive
        # across all optimizer iterations.
        from dafoam_model import OF, build_problem, make_options

        self._OF = OF
        self.prob = build_problem(preset)
        _, _, self.params = make_options(preset)
        self.n_shape = self.prob.model.n_shape
        self.U0 = self.params["U0"]

        self.io.input_grammar.update_from_names(["shape", "aoa"])
        self.io.output_grammar.update_from_names(OUTPUTS)
        self.io.input_grammar.defaults = {
            "shape": np.zeros(self.n_shape),
            "aoa": np.array([self.params["aoa0"]]),
        }

        self._last_hash = None
        self._summary_path = summary_path
        self._eval_count = 0
        self._grad_count = 0
        self._t_start = time.time()
        # Baseline CD captured on first primal, used for delta-counts column.
        self._baseline_CD = None

    # ------------------------------------------------------------------ #
    def _push_and_solve(self, shape: np.ndarray, aoa: float) -> None:
        h = hashlib.md5(np.concatenate([shape, [aoa]]).tobytes()).hexdigest()
        if h == self._last_hash:
            return
        self.prob.set_val("shape", shape)
        self.prob.set_val("patchV", np.array([self.U0, aoa]))
        self.prob.run_model()
        self._last_hash = h

    def _run(self, input_data: dict) -> dict:
        shape = np.asarray(input_data["shape"], dtype=float)
        aoa = float(np.atleast_1d(input_data["aoa"])[0])
        LOGGER.info("DAFoam primal @ aoa=%.4f deg, |shape|=%.3e",
                    aoa, np.linalg.norm(shape))

        t0 = time.time()
        self._push_and_solve(shape, aoa)
        dt = time.time() - t0

        out = {
            name: np.atleast_1d(
                np.asarray(self.prob.get_val(self._OF[name]), dtype=float)
            )
            for name in OUTPUTS
        }

        self._eval_count += 1
        self._log_iter(shape, aoa, out, dt)
        return out

    # ------------------------------------------------------------------ #
    def _compute_jacobian(self, input_names=None, output_names=None) -> None:
        """Exact Jacobians from the discrete adjoint (Phase 4)."""
        shape = np.asarray(self.io.data["shape"], dtype=float)
        aoa = float(np.atleast_1d(self.io.data["aoa"])[0])
        self._push_and_solve(shape, aoa)

        of = [self._OF[n] for n in OUTPUTS]
        totals = self.prob.compute_totals(of=of, wrt=["shape", "patchV"])

        # GEMSEO 6 dropped `with_zeros` -- zero-init is the default now.
        self._init_jacobian(input_names, output_names)
        for name in OUTPUTS:
            key = self._OF[name]
            n_out = np.atleast_1d(self.io.data[name]).size
            self.jac[name]["shape"] = np.atleast_2d(
                totals[(key, "shape")]
            ).reshape(n_out, self.n_shape)
            d_patchV = np.atleast_2d(totals[(key, "patchV")]).reshape(n_out, 2)
            self.jac[name]["aoa"] = d_patchV[:, 1].reshape(n_out, 1)

        self._grad_count += 1

    # ------------------------------------------------------------------ #
    def _log_iter(self, shape, aoa, out, dt):
        """Append a per-evaluation row to summary_path (rank-0 only)."""
        if self._summary_path is None:
            return
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.rank != 0:
                return
        except ImportError:
            pass

        CD = float(out["CD"][0])
        CL = float(out["CL"][0])
        thick_min = float(np.min(out["thickcon"]))
        vol = float(out["volcon"][0])
        rcon = float(np.min(out["rcon"]))
        if self._baseline_CD is None:
            self._baseline_CD = CD
        # Derived: L/D, drag counts, delta counts vs baseline, CL error
        LD = CL / CD if CD > 0 else float("nan")
        counts = CD * 1e4
        dcounts = (CD - self._baseline_CD) * 1e4
        CL_target = self.params.get("CL_target", 0.0)
        CL_err = CL - CL_target
        elapsed = time.time() - self._t_start
        elapsed_h = int(elapsed // 3600)
        elapsed_m = int((elapsed % 3600) // 60)
        # Type: baseline/primal/gradient-triggered eval
        kind = "base" if self._eval_count == 1 else "eval"
        with open(self._summary_path, "a") as f:
            f.write(
                f"| {self._eval_count:4d} "
                f"| {kind} "
                f"| {np.linalg.norm(shape):9.4e} "
                f"| {aoa:7.4f} "
                f"| {CD:.6f} "
                f"| {counts:6.1f} "
                f"| {dcounts:+7.1f} "
                f"| {CL:.6f} "
                f"| {CL_err:+.4f} "
                f"| {LD:6.2f} "
                f"| {thick_min:.4f} "
                f"| {vol:.4f} "
                f"| {rcon:.4f} "
                f"| {dt/60:5.1f} "
                f"| {elapsed_h:02d}h{elapsed_m:02d}m |\n"
            )
