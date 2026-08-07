#!/usr/bin/env python3
"""GEMSEO discipline wrapping the OpenMDAO/MPhys/DAFoam problem.

Exposes the CFD problem defined by the case-local ``dafoam_model.py``
as a single GEMSEO :class:`Discipline`. Each evaluation performs:

    1. Push the FFD shape design-variable vector (and, if the case
       exposes it, the angle of attack) into the underlying OpenMDAO
       problem.
    2. Warp the volume mesh through IDWarp (called inside the model
       during ``run_model``).
    3. Run the primal RANS solve; extract CD, CL, and the geometric
       constraint values.
    4. On gradient requests, run the discrete adjoint per functional and
       propagate the sensitivities back to the design vector.
    5. Optionally append a summary row to ``summary_path`` after each
       primal evaluation.

Two operating modes are supported:

    fixed_aoa=False (default, transonic case)
        Inputs: shape (n), aoa (1 deg). aoa is a design variable.
        DAFoam has a ``patchV`` input exposing [U, alpha] to the primal
        BC, and the adjoint returns d(F)/d(patchV) alongside d(F)/d(shape).

    fixed_aoa=True (LALE case, alpha hard-constrained at 4.5 deg)
        Inputs: shape (n) only. Alpha is set by the case BC files
        (0/U) and is NOT a design variable. DAFoam has no ``patchV``
        input; the adjoint returns d(F)/d(shape) only.

Written for GEMSEO 6.x. In GEMSEO 5.x the base class was ``MDODiscipline``,
``io.data`` was ``local_data``, ``io.input_grammar.defaults`` was
``default_inputs``, and ``_run(input_data)`` was ``_run(self)`` followed
by ``self.store_local_data(**out)``.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os
import shutil
import time

import numpy as np

from gemseo.core.discipline import Discipline

LOGGER = logging.getLogger(__name__)

OUTPUTS = ["CD", "CL", "thickcon", "volcon", "rcon"]


def _copy_dir(src: str, dst: str) -> bool:
    """Copy a directory tree, contents only.

    Deliberately avoids ``shutil.copytree``/``copy2``: they preserve
    permissions and timestamps, and ``chmod``/``utime`` raise ``EPERM`` on a
    Windows-mounted filesystem (DrvFs / 9p under Docker Desktop). One failed
    metadata call would otherwise abort the whole copy mid-run.
    """
    ok = False
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        try:
            os.makedirs(target, exist_ok=True)
        except OSError:
            continue
        for name in files:
            try:
                shutil.copyfile(os.path.join(root, name),
                                os.path.join(target, name))
                ok = True
            except OSError:
                pass
    return ok


def _latest_time_dir(proc_dir: str) -> str | None:
    """Most recently written numeric time directory inside a processor dir.

    DAFoam names the working solution after the final iteration count (e.g.
    ``3001``) and only renames it to a compact pseudo-time (``0.0001`` ...)
    when an adjoint runs. Since line-search evaluations have no adjoint, the
    directory name is not predictable -- so pick by modification time rather
    than by name.
    """
    newest, newest_mtime = None, -1.0
    try:
        entries = os.listdir(proc_dir)
    except OSError:
        return None
    for name in entries:
        path = os.path.join(proc_dir, name)
        if name == "constant" or not os.path.isdir(path):
            continue
        try:
            float(name)          # numeric time directories only
        except ValueError:
            continue
        mtime = os.path.getmtime(path)
        if mtime > newest_mtime:
            newest, newest_mtime = path, mtime
    return newest


class DAFoamDiscipline(Discipline):
    """DAFoam multiphysics problem wrapped as a GEMSEO discipline.

    Parameters
    ----------
    preset : str
        Preset name passed through to the case-local
        ``dafoam_model.build_problem(preset)``.
    summary_path : str or None
        Optional markdown file to append a per-evaluation summary row to.
    fixed_aoa : bool
        If True, alpha is hard-constrained by the case BC files and NOT
        a design variable. The discipline exposes only ``shape`` as an
        input. If False (default), alpha is a design variable driven
        through the DAFoam ``patchV`` input.
    shape_dir : str or None
        Directory for the per-evaluation design-vector archive.
    save_dir : str or None
        Directory for the per-evaluation *full state* archive: deformed mesh
        points, flow fields and FFD control points, one subdirectory per
        evaluation.

        This exists because DAFoam only preserves a solution when it runs an
        adjoint -- the rename lives inside ``solve_linear``. Line-search
        evaluations therefore have their flow field overwritten by the next
        one. Saving here instead, from a hook that fires on *every* primal,
        decouples preservation from the optimizer's gradient schedule.

        It also removes an ambiguity when re-evaluating a design point later:
        with the mesh saved, a replay loads the exact geometry rather than
        recreating it by re-warping, so any disagreement is attributable to
        the flow solve alone.
    """

    def __init__(self, preset: str = "tutorial",
                 summary_path: str | None = None,
                 fixed_aoa: bool = False,
                 shape_dir: str | None = None,
                 save_dir: str | None = None):
        super().__init__(name="DAFoamDiscipline")

        # Build the (expensive) DAFoam problem ONCE and keep it alive
        # across all optimizer iterations.
        from dafoam_model import OF, build_problem, make_options

        self._OF = OF
        self.prob = build_problem(preset)
        _, _, self.params = make_options(preset)
        self.n_shape = self.prob.model.n_shape
        self.U0 = self.params["U0"]
        self._fixed_aoa = fixed_aoa
        self._fixed_aoa_value = float(self.params["aoa0"])  # only used if fixed

        # Grammar and default inputs depend on the mode.
        if self._fixed_aoa:
            self.io.input_grammar.update_from_names(["shape"])
            self.io.input_grammar.defaults = {
                "shape": np.zeros(self.n_shape),
            }
        else:
            self.io.input_grammar.update_from_names(["shape", "aoa"])
            self.io.input_grammar.defaults = {
                "shape": np.zeros(self.n_shape),
                "aoa": np.array([self.params["aoa0"]]),
            }
        self.io.output_grammar.update_from_names(OUTPUTS)

        self._last_hash = None
        self._summary_path = summary_path
        self._eval_count = 0
        self._grad_count = 0
        self._t_start = time.time()
        # Baseline CD captured on first primal, used for delta-counts column.
        self._baseline_CD = None

        # Per-evaluation archives, both written from _log_iter so they fire on
        # every primal rather than only on the ones the optimizer requests a
        # gradient for. Written incrementally, so unlike opt_history.h5 a
        # mid-run crash costs nothing.
        self._shape_dir = shape_dir
        self._save_dir = save_dir
        if self._is_rank0():
            for d in (self._shape_dir, self._save_dir):
                if d is not None:
                    os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_rank0() -> bool:
        """True on MPI rank 0, or when running serially."""
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD.rank == 0
        except ImportError:
            return True

    # ------------------------------------------------------------------ #
    def _save_state(self, tag: str) -> None:
        """Archive everything needed to re-run this evaluation.

        Copies, into ``<save_dir>/<tag>/``:

        * ``processorN/`` -- the decomposed flow field AND
          ``polyMesh/points`` for this shape. Mesh topology never changes
          under FFD deformation (only vertex positions move), so the
          invariant ``constant/polyMesh`` is archived once per run instead.
        * ``ffd_coef.npy`` -- deformed FFD control points, straight from
          pyGeo as an (N, 3) array.

        Every primal leaves a complete solution directory, so this always has
        something to copy. From ``DASimpleFoam.C::solvePrimal``, after the
        SIMPLE loop::

            mesh.write();                  // unconditional
            this->writeAssociatedFields(); // unconditional

        and ``pyDAFoam.renameSolution`` only ever calls ``shutil.move`` -- it
        never creates a directory. So the write is done by the primal, which
        has no knowledge of whether an adjoint will follow; only the *rename*
        is tied to the adjoint. Line-search evaluations therefore do write
        their solution, and simply get overwritten by the next primal because
        nothing gave them a unique name. Copying here, immediately after the
        solve, catches them before that happens.

        Failures are logged, never raised: losing one evaluation's archive
        must not abort a ten-hour optimization.
        """
        if self._save_dir is None:
            return
        dest = os.path.join(self._save_dir, tag)

        for proc in sorted(glob.glob("processor*")):
            if not os.path.isdir(proc):
                continue
            src = _latest_time_dir(proc)
            if src is None:
                LOGGER.warning("[%s] no time directory in %s", tag, proc)
                continue
            _copy_dir(src, os.path.join(dest, os.path.basename(proc)))

        # FFD control points. Taken as a raw array rather than written via
        # DAFoam's writeDeformedFFDs, which segfaulted the adjoint solve.
        try:
            coef = self.prob.model.geometry.DVGeo.FFD.coef
            os.makedirs(dest, exist_ok=True)
            np.save(os.path.join(dest, "ffd_coef.npy"), np.asarray(coef))
        except Exception as exc:                       # noqa: BLE001
            LOGGER.warning("FFD control points not saved: %s", exc)

    # ------------------------------------------------------------------ #
    def _push_and_solve(self, shape: np.ndarray, aoa: float) -> None:
        h = hashlib.md5(np.concatenate([shape, [aoa]]).tobytes()).hexdigest()
        if h == self._last_hash:
            return
        self.prob.set_val("shape", shape)
        if not self._fixed_aoa:
            self.prob.set_val("patchV", np.array([self.U0, aoa]))
        self.prob.run_model()
        self._last_hash = h

    def _run(self, input_data: dict) -> dict:
        shape = np.asarray(input_data["shape"], dtype=float)
        if self._fixed_aoa:
            aoa = self._fixed_aoa_value
        else:
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
        """Exact Jacobians from the discrete adjoint."""
        shape = np.asarray(self.io.data["shape"], dtype=float)
        if self._fixed_aoa:
            aoa = self._fixed_aoa_value
        else:
            aoa = float(np.atleast_1d(self.io.data["aoa"])[0])
        self._push_and_solve(shape, aoa)

        of = [self._OF[n] for n in OUTPUTS]
        wrt = ["shape"] if self._fixed_aoa else ["shape", "patchV"]
        totals = self.prob.compute_totals(of=of, wrt=wrt)

        # GEMSEO 6 dropped `with_zeros` -- zero-init is the default now.
        self._init_jacobian(input_names, output_names)
        for name in OUTPUTS:
            key = self._OF[name]
            n_out = np.atleast_1d(self.io.data[name]).size
            self.jac[name]["shape"] = np.atleast_2d(
                totals[(key, "shape")]
            ).reshape(n_out, self.n_shape)
            if not self._fixed_aoa:
                d_patchV = np.atleast_2d(
                    totals[(key, "patchV")]
                ).reshape(n_out, 2)
                self.jac[name]["aoa"] = d_patchV[:, 1].reshape(n_out, 1)

        self._grad_count += 1

    # ------------------------------------------------------------------ #
    def _log_iter(self, shape, aoa, out, dt):
        """Archive this evaluation and append a summary row (rank-0 only)."""
        if not self._is_rank0():
            return

        tag = f"eval{self._eval_count:04d}"

        # ---- design-vector archive -----------------------------------
        # Tiny (~0.5 kB). Replay a design point later with
        #   mpirun -np 4 python run_at_shape.py --shape eval0007_shape.npy
        if self._shape_dir is not None:
            np.save(os.path.join(self._shape_dir, f"{tag}_shape.npy"), shape)
            if not self._fixed_aoa:
                np.save(os.path.join(self._shape_dir, f"{tag}_aoa.npy"),
                        np.array([aoa]))

        # ---- full-state archive: mesh points, flow fields, FFD --------
        self._save_state(tag)

        # ---- summary row ---------------------------------------------
        if self._summary_path is None:
            return

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
