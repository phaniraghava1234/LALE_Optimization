#!/usr/bin/env python3
"""Snapshot a completed optimization run into ``results/<run_name>/``.

Every optimization overwrites the case directory in place: ``processor*``
holds only the last evaluation's flow field, ``opt_summary.md`` and
``opt_history.h5`` are rewritten, and the mesh/config files are whatever
they happened to be at launch. This module freezes all of it so a run can
be reproduced, compared against another run, or written up months later.

What gets archived
------------------
``manifest.json`` / ``MANIFEST.md``
    Provenance: timestamp, git commit and dirty state, host, MPI ranks,
    DAFoam and OpenFOAM versions, the fully-resolved ``daOptions`` dict,
    mesh statistics, optimizer settings, and the headline results.
``config/``
    Everything that defines the numerics: ``system/*``, the fluid and
    turbulence dictionaries, the ``0_orig`` boundary conditions, plus the
    Python that generated them (``dafoam_model.py``, the mesh generator,
    the FFD generator).
``geometry/``
    FFD lattice and the airfoil profile files.
``mesh/``
    ``constant/polyMesh`` as actually used (already gzipped by OpenFOAM;
    ~1-2 MB for a 24k-cell 2D case).
``optimization/``
    ``opt_summary.md``, ``opt_history.h5`` (the GEMSEO database, which
    carries every evaluation's design vector, functions and gradients),
    the SLSQP convergence plots, the per-evaluation ``shapes/*.npy``
    archive, and the run logs.
``flow/`` (optional, ``include_flow=True``)
    Reconstructed final flow field. Skipped by default because it is by
    far the largest item and any evaluation can be regenerated from its
    archived shape vector via ``run_at_shape.py``.

Usage
-----
Called automatically at the end of ``run_optimization.py``. Also usable
standalone to archive a run after the fact::

    python ../mdo/archive_run.py --case . --name lale_run1 --include-flow
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import platform
import shutil
import subprocess

# Files copied into config/, relative to the case directory. Missing
# entries are skipped silently -- not every preset writes every file.
_CONFIG_FILES = [
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
    "system/decomposeParDict",
    "system/blockMeshDict",
    "constant/transportProperties",
    "constant/turbulenceProperties",
    "constant/thermophysicalProperties",
    "dafoam_model.py",
    "run_primal.py",
    "verify_gradients.py",
    "run_at_shape.py",
]

# Copied from the repo root (case_dir/..) so the mesh and FFD are
# reproducible from the archive alone.
_REPO_FILES = [
    "mesh/generate_cmesh.py",
    "mesh/boundary_layer.py",
    "mesh/airfoil_io.py",
    "mesh/split_selig_dat.py",
    "ffd/generate_ffd.py",
    "mdo/run_optimization.py",
    "mdo/dafoam_discipline.py",
]

_OPT_GLOBS = [
    "opt_summary.md",
    "opt_history.h5",
    "opt_history*.png",
    "*.log",
    "opt_log.txt",
    "logCheckMesh.txt",
]


def _run(cmd: list[str], cwd: str | None = None) -> str:
    """Best-effort command capture; returns '' on any failure."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return ""


def _git_info(repo_dir: str) -> dict:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir)
    status = _run(["git", "status", "--porcelain"], cwd=repo_dir)
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        # A dirty tree means the archived source may differ from the commit,
        # which is exactly why the source files are copied in too.
        "dirty": bool(status),
        "dirty_files": status.splitlines()[:50] if status else [],
    }


def _versions() -> dict:
    info = {
        "openfoam": os.environ.get("WM_PROJECT_VERSION", "unknown"),
        "python": platform.python_version(),
        "host": platform.node(),
    }
    for mod in ("dafoam", "openmdao", "gemseo", "pygeo", "mphys"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "unknown"
    return info


def _mesh_stats(case_dir: str) -> dict:
    """Cell/point/face counts read from the polyMesh owner header."""
    stats: dict = {}
    boundary = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if os.path.exists(boundary):
        try:
            with open(boundary) as f:
                stats["patches"] = [
                    ln.strip() for ln in f
                    if ln.strip() and not ln.startswith(("/", "|", "\\", " "))
                ][:20]
        except Exception:
            pass
    # `note` line in owner.gz carries nPoints/nCells/nFaces
    owner = os.path.join(case_dir, "constant", "polyMesh", "owner.gz")
    if os.path.exists(owner):
        try:
            import gzip
            with gzip.open(owner, "rt", errors="ignore") as f:
                for _ in range(30):
                    line = f.readline()
                    if "nCells" in line:
                        stats["note"] = line.strip()
                        break
        except Exception:
            pass
    return stats


def _copy(src: str, dst: str, failed: list | None = None) -> bool:
    """Copy a file or directory tree, creating parents.

    Uses ``shutil.copyfile`` (contents only) rather than ``copy2``/``copytree``
    because those preserve permissions and timestamps, and ``chmod``/``utime``
    raise ``EPERM`` on a Windows-mounted filesystem (DrvFs / 9p under Docker
    Desktop). Directory trees are walked manually for the same reason --
    ``copytree`` calls ``copystat`` on each directory even with a custom
    ``copy_function``.

    Individual failures are recorded in ``failed`` and do not propagate, so one
    unreadable file cannot abandon the whole archive.

    Returns True if anything was copied.
    """
    if not os.path.exists(src):
        return False

    def _note(path: str, exc: Exception) -> None:
        if failed is not None:
            failed.append(f"{path}: {exc}")

    try:
        if os.path.isdir(src):
            any_ok = False
            for root, _dirs, files in os.walk(src):
                rel = os.path.relpath(root, src)
                target = dst if rel == "." else os.path.join(dst, rel)
                try:
                    os.makedirs(target, exist_ok=True)
                except Exception as e:
                    _note(root, e)
                    continue
                for name in files:
                    try:
                        shutil.copyfile(os.path.join(root, name),
                                        os.path.join(target, name))
                        any_ok = True
                    except Exception as e:
                        _note(os.path.join(root, name), e)
            return any_ok

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return True
    except Exception as e:
        _note(src, e)
        return False


def archive_run(case_dir: str,
                run_name: str | None = None,
                results_root: str | None = None,
                metadata: dict | None = None,
                include_flow: bool = False) -> str:
    """Freeze a completed run. Returns the archive directory path.

    Parameters
    ----------
    case_dir
        The OpenFOAM case that was optimized (e.g. ``case_lale``).
    run_name
        Archive subdirectory name. Defaults to ``<case>_<YYYYmmdd_HHMMSS>``.
    results_root
        Where archives live. Defaults to ``<case_dir>/../results``.
    metadata
        Extra provenance merged into the manifest -- preset, algorithm,
        resolved daOptions, timings, headline results.
    include_flow
        Also copy reconstructed time directories. Large; off by default
        since any evaluation is reproducible from its archived shape
        vector via ``run_at_shape.py``.
    """
    case_dir = os.path.abspath(case_dir)
    repo_dir = os.path.dirname(case_dir)
    case_name = os.path.basename(case_dir)

    if results_root is None:
        results_root = os.path.join(repo_dir, "results")
    if run_name is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{case_name}_{stamp}"

    dest = os.path.join(results_root, run_name)
    os.makedirs(dest, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    failed: list[str] = []

    def cp(src: str, dst: str, label: str) -> None:
        (copied if _copy(src, dst, failed) else missing).append(label)

    # ---- config: numerics + the code that produced them ---------------
    for rel in _CONFIG_FILES:
        cp(os.path.join(case_dir, rel),
           os.path.join(dest, "config", rel), rel)
    cp(os.path.join(case_dir, "0_orig"),
       os.path.join(dest, "config", "0_orig"), "0_orig/")
    for rel in _REPO_FILES:
        cp(os.path.join(repo_dir, rel),
           os.path.join(dest, "config", "src", rel), rel)

    # ---- geometry ------------------------------------------------------
    for rel in ("FFD", "profiles"):
        cp(os.path.join(case_dir, rel),
           os.path.join(dest, "geometry", rel), rel + "/")

    # ---- mesh ----------------------------------------------------------
    cp(os.path.join(case_dir, "constant", "polyMesh"),
       os.path.join(dest, "mesh", "polyMesh"), "constant/polyMesh/")

    # ---- optimization outputs -----------------------------------------
    for pattern in _OPT_GLOBS:
        for src in glob.glob(os.path.join(case_dir, pattern)):
            base = os.path.basename(src)
            cp(src, os.path.join(dest, "optimization", base), base)
    cp(os.path.join(case_dir, "shapes"),
       os.path.join(dest, "optimization", "shapes"), "shapes/")
    # Per-evaluation full state (deformed mesh, flow fields, FFD points).
    cp(os.path.join(case_dir, "saved"),
       os.path.join(dest, "optimization", "saved"), "saved/")

    # ---- flow fields ---------------------------------------------------
    # DAFoam renames each primal's solution to an incrementing pseudo-time
    # (0.0001, 0.0002, ...) inside processor*/, so the decomposed case holds
    # one directory per evaluation rather than just the last. Archiving
    # processor* wholesale -- including processor*/constant/polyMesh, which
    # reconstructPar needs -- means any evaluation's flow field can be
    # recovered from the archive later.
    if include_flow:
        for src in sorted(glob.glob(os.path.join(case_dir, "processor*"))):
            base = os.path.basename(src)
            if os.path.isdir(src):
                cp(src, os.path.join(dest, "flow", base), f"flow/{base}/")
        # Anything already reconstructed at the top level.
        for src in glob.glob(os.path.join(case_dir, "[0-9]*")):
            base = os.path.basename(src)
            if os.path.isdir(src) and base not in ("0", "0_orig"):
                cp(src, os.path.join(dest, "flow", base), f"flow/{base}/")

    # ---- manifest ------------------------------------------------------
    manifest = {
        "run_name": run_name,
        "case": case_name,
        "archived_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(repo_dir),
        "versions": _versions(),
        "mesh": _mesh_stats(case_dir),
        "files_archived": sorted(set(copied)),
        "files_missing": sorted(set(missing)),
        "copy_failures": failed[:50],
        "include_flow": include_flow,
    }
    if metadata:
        manifest.update(metadata)

    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    _write_manifest_md(os.path.join(dest, "MANIFEST.md"), manifest)
    return dest


def _write_manifest_md(path: str, m: dict) -> None:
    """Human-readable companion to manifest.json."""
    def section(title: str) -> str:
        return f"\n## {title}\n\n"

    lines = [f"# Run archive: {m['run_name']}\n",
             f"Archived {m['archived_at']} on {m['versions'].get('host')}\n"]

    lines.append(section("Provenance"))
    g = m["git"]
    lines.append(f"- Commit: `{g['commit']}` (branch `{g['branch']}`)\n")
    lines.append(f"- Working tree dirty: **{g['dirty']}**\n")
    if g["dirty"]:
        lines.append("- Uncommitted at archive time — the copies under "
                     "`config/` are the authoritative record:\n")
        for f in g["dirty_files"][:20]:
            lines.append(f"  - `{f}`\n")

    lines.append(section("Software"))
    for k, v in m["versions"].items():
        lines.append(f"- {k}: `{v}`\n")

    if any(k in m for k in ("preset", "algorithm", "max_iter", "n_ranks")):
        lines.append(section("Run settings"))
        for k in ("preset", "algorithm", "max_iter", "n_ranks",
                  "wall_time_s", "n_evaluations", "n_gradients"):
            if k in m:
                lines.append(f"- {k}: `{m[k]}`\n")

    if "results" in m:
        lines.append(section("Results"))
        for k, v in m["results"].items():
            lines.append(f"- {k}: `{v}`\n")

    if m.get("mesh", {}).get("note"):
        lines.append(section("Mesh"))
        lines.append(f"- `{m['mesh']['note']}`\n")

    if "daOptions" in m:
        lines.append(section("Resolved daOptions"))
        lines.append("```json\n")
        lines.append(json.dumps(m["daOptions"], indent=2, default=str))
        lines.append("\n```\n")

    lines.append(section("Contents"))
    lines.append("| Directory | Holds |\n|---|---|\n")
    lines.append("| `config/` | numerics dictionaries, BCs, and the source "
                 "that generated them |\n")
    lines.append("| `geometry/` | FFD lattice, airfoil profiles |\n")
    lines.append("| `mesh/polyMesh/` | mesh as actually used |\n")
    lines.append("| `optimization/` | summary, GEMSEO database, plots, "
                 "per-evaluation shapes, logs |\n")
    if m.get("include_flow"):
        lines.append("| `flow/` | reconstructed final flow field |\n")

    if m.get("include_flow"):
        lines.append(section("Recovering a flow field (no re-solve)"))
        lines.append("DAFoam stores one solution directory per evaluation "
                     "inside each `processor*`, named as an incrementing "
                     "pseudo-time. List them with `ls flow/processor0/`.\n\n")
        lines.append("```bash\n")
        lines.append("cp -r flow/processor*  <case>/\n")
        lines.append("cp -r mesh/polyMesh    <case>/constant/polyMesh\n")
        lines.append("cp    config/system/*  <case>/system/\n")
        lines.append("cd <case>\n")
        lines.append("reconstructPar -time 0.0007     # or -latestTime, or -allTime\n")
        lines.append("touch case.foam && paraview case.foam\n")
        lines.append("```\n")

    lines.append(section("Re-solving an evaluation from its design vector"))
    lines.append("Use this when the flow field was not archived, or to "
                 "regenerate at different settings.\n\n")
    lines.append("```bash\n")
    lines.append("cp -r mesh/polyMesh   <case>/constant/polyMesh\n")
    lines.append("cp -r geometry/FFD    <case>/FFD\n")
    lines.append("cp -r config/0_orig   <case>/0_orig\n")
    lines.append("cp    config/system/* <case>/system/\n")
    lines.append("cd <case> && cp -r 0_orig 0\n")
    lines.append("mpirun -np 4 python run_at_shape.py "
                 "--shape <archive>/optimization/shapes/eval0007_shape.npy\n")
    lines.append("```\n")

    if m.get("files_missing"):
        lines.append(section("Not present at archive time"))
        for f in m["files_missing"]:
            lines.append(f"- `{f}`\n")

    if m.get("copy_failures"):
        lines.append(section("Copy failures"))
        lines.append("These files could not be copied. The archive is "
                     "otherwise complete.\n\n")
        for f in m["copy_failures"]:
            lines.append(f"- `{f}`\n")

    with open(path, "w") as fh:
        fh.write("".join(lines))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default=".", help="case directory to archive")
    ap.add_argument("--name", default=None, help="archive name")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--include-flow", action="store_true",
                    help="also copy reconstructed time directories (large)")
    args = ap.parse_args()

    dest = archive_run(case_dir=args.case, run_name=args.name,
                       results_root=args.results_root,
                       include_flow=args.include_flow)
    print(f"[archive] wrote {dest}")


if __name__ == "__main__":
    main()
