#!/usr/bin/env python3
"""Minimal reader for ASCII OpenFOAM polyMesh files, gzipped or plain.

Exists so that archived runs can be post-processed on a workstation without
the DAFoam container (pyvista / OpenFOAM are not required). Only the subset
needed to pull a boundary patch surface out of an archive is implemented:
points, faces, the boundary dictionary, and processor point addressing.
"""
from __future__ import annotations

import gzip
import os
import re
from functools import lru_cache

import numpy as np

__all__ = [
    "read_points",
    "read_faces",
    "read_boundary",
    "read_labels",
    "patch_faces",
    "patch_point_ids",
    "patch_face_owners",
    "ordered_slice_ids",
    "read_scalar_internal",
    "reconstruct_saved_scalar",
    "reconstruct_points",
    "reconstruct_saved_points",
]


def _open(path: str):
    """Open `path`, transparently handling a `.gz` twin."""
    if os.path.exists(path):
        return open(path, "rb")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rb")
    raise FileNotFoundError(f"neither {path} nor {path}.gz exists")


def _payload(path: str) -> str:
    """Return the file body with the FoamFile header stripped."""
    with _open(path) as fh:
        txt = fh.read().decode("utf-8", errors="replace")
    # Drop the banner comment and the FoamFile { ... } dictionary.
    end = txt.find("}", txt.find("FoamFile"))
    return txt[end + 1:] if end != -1 else txt


@lru_cache(maxsize=64)
def read_points(path: str) -> np.ndarray:
    """Read a vectorField into an (N, 3) array."""
    body = _payload(path)
    vals = re.findall(r"\(([^()]*)\)", body)
    pts = np.array([[float(v) for v in s.split()] for s in vals if s.strip()])
    return pts.reshape(-1, 3)


@lru_cache(maxsize=64)
def read_faces(path: str) -> list[np.ndarray]:
    """Read a faceList as a list of point-index arrays."""
    body = _payload(path)
    return [np.fromstring(s, dtype=int, sep=" ")
            for s in re.findall(r"\d+\(([^()]*)\)", body)]


@lru_cache(maxsize=64)
def read_labels(path: str) -> np.ndarray:
    """Read a labelList into a 1-D integer array."""
    body = _payload(path)
    start = body.find("(")
    stop = body.rfind(")")
    return np.fromstring(body[start + 1:stop], dtype=int, sep=" ")


@lru_cache(maxsize=64)
def read_boundary(path: str) -> dict[str, dict[str, int]]:
    """Read the boundary dictionary as {patch: {nFaces, startFace}}."""
    body = _payload(path)
    out: dict[str, dict[str, int]] = {}
    for name, block in re.findall(r"(\w+)\s*\{([^{}]*)\}", body):
        n = re.search(r"nFaces\s+(\d+)", block)
        s = re.search(r"startFace\s+(\d+)", block)
        if n and s:
            out[name] = {"nFaces": int(n.group(1)),
                         "startFace": int(s.group(1))}
    return out


@lru_cache(maxsize=64)
def patch_faces(mesh_dir: str, patch: str) -> list[np.ndarray]:
    """The point-index arrays of every face in `patch`."""
    bnd = read_boundary(os.path.join(mesh_dir, "boundary"))
    if patch not in bnd:
        raise KeyError(f"patch {patch!r} not in {sorted(bnd)}")
    faces = read_faces(os.path.join(mesh_dir, "faces"))
    start = bnd[patch]["startFace"]
    return faces[start:start + bnd[patch]["nFaces"]]


def patch_point_ids(mesh_dir: str, patch: str) -> np.ndarray:
    """Unique, sorted global point indices belonging to `patch`."""
    return np.unique(np.concatenate(patch_faces(mesh_dir, patch)))


def ordered_slice_ids(mesh_dir: str, patch: str,
                      points: np.ndarray) -> np.ndarray:
    """Global point ids of `patch` on its lowest z plane, ordered around the
    contour by face connectivity.

    An angular sort fails on a cambered section near the trailing edge, where
    upper and lower surfaces subtend nearly the same angle. Walking the edge
    graph is exact regardless of shape.
    """
    faces = patch_faces(mesh_dir, patch)
    ids = np.unique(np.concatenate(faces))
    z0 = points[ids][:, 2].min()
    on_slice = {int(i) for i in ids if abs(points[i, 2] - z0) < 1e-9}

    # Each slab face contributes exactly one edge lying in the z0 plane.
    adj: dict[int, list[int]] = {}
    for f in faces:
        e = [int(p) for p in f if int(p) in on_slice]
        if len(e) != 2:
            continue
        adj.setdefault(e[0], []).append(e[1])
        adj.setdefault(e[1], []).append(e[0])

    if not adj:
        raise ValueError(f"no in-plane edges found for patch {patch!r}")

    # Walk the loop. Start at the leading edge (minimum x) for a stable phase.
    start = min(adj, key=lambda i: points[i, 0])
    order = [start]
    prev, cur = None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            break               # open contour (blunt TE leaves a gap)
        nxt = nxt[0]
        if nxt == start:
            break               # closed the loop
        order.append(nxt)
        prev, cur = cur, nxt
    return np.array(order, dtype=int)


@lru_cache(maxsize=64)
def read_scalar_internal(path: str) -> np.ndarray:
    """Read the internalField of a volScalarField as a 1-D array.

    Handles both `uniform <v>` and `nonuniform List<scalar> N ( ... )`.
    """
    body = _payload(path)
    i = body.find("internalField")
    if i == -1:
        raise ValueError(f"no internalField in {path}")
    seg = body[i:]
    m = re.match(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", seg)
    if m:
        return np.array([float(m.group(1))])
    start = seg.find("(")
    stop = seg.find(")", start)
    return np.fromstring(seg[start + 1:stop], dtype=float, sep=" ")


def patch_face_owners(mesh_dir: str, patch: str) -> np.ndarray:
    """Owner cell index for each face of `patch`.

    Boundary faces have no neighbour, so the owner is the adjacent interior
    cell -- which is what a zeroGradient wall value resolves to.
    """
    bnd = read_boundary(os.path.join(mesh_dir, "boundary"))
    if patch not in bnd:
        raise KeyError(f"patch {patch!r} not in {sorted(bnd)}")
    owner = read_labels(os.path.join(mesh_dir, "owner"))
    start = bnd[patch]["startFace"]
    return owner[start:start + bnd[patch]["nFaces"]]


def reconstruct_saved_scalar(eval_dir: str, flow_dir: str, field: str,
                             n_cells: int) -> np.ndarray:
    """Global cell array of a volScalarField from a `saved/evalNNNN` snapshot."""
    procs = sorted(d for d in os.listdir(eval_dir)
                   if d.startswith("processor") and d[9:].isdigit())
    glob = np.full(n_cells, np.nan)
    for p in procs:
        vals = read_scalar_internal(os.path.join(eval_dir, p, field))
        addr = read_labels(os.path.join(flow_dir, p, "constant", "polyMesh",
                                        "cellProcAddressing"))
        glob[addr] = vals[0] if vals.size == 1 else vals
    if np.isnan(glob).any():
        raise ValueError(f"some cells of {field} were never assigned")
    return glob


def reconstruct_saved_points(eval_dir: str, flow_dir: str,
                             n_points: int) -> np.ndarray:
    """Global point array for a `saved/evalNNNN` snapshot.

    Those snapshots store `processorN/polyMesh/points` with no enclosing time
    directory; the addressing still lives in the decomposed constant/ tree.
    """
    procs = sorted(d for d in os.listdir(eval_dir)
                   if d.startswith("processor") and d[9:].isdigit())
    glob = np.full((n_points, 3), np.nan)
    for p in procs:
        pts = read_points(os.path.join(eval_dir, p, "polyMesh", "points"))
        addr = read_labels(os.path.join(flow_dir, p, "constant", "polyMesh",
                                        "pointProcAddressing"))
        glob[addr] = pts
    if np.isnan(glob[:, 0]).any():
        raise ValueError("some global points were never assigned")
    return glob


def reconstruct_points(flow_dir: str, time_name: str,
                       n_points: int) -> np.ndarray:
    """Rebuild the global point array from a decomposed case.

    Each `processorN/<time>/polyMesh/points` holds that rank's deformed
    points; `processorN/constant/polyMesh/pointProcAddressing` maps local
    index -> global index.
    """
    procs = sorted(
        d for d in os.listdir(flow_dir)
        if d.startswith("processor") and d[9:].isdigit()
    )
    if not procs:
        raise FileNotFoundError(f"no processor* directories under {flow_dir}")

    glob = np.full((n_points, 3), np.nan)
    for p in procs:
        pts = read_points(
            os.path.join(flow_dir, p, time_name, "polyMesh", "points"))
        addr = read_labels(
            os.path.join(flow_dir, p, "constant", "polyMesh",
                         "pointProcAddressing"))
        if len(addr) != len(pts):
            raise ValueError(
                f"{p}: {len(pts)} points but {len(addr)} addressing entries")
        glob[addr] = pts

    missing = int(np.isnan(glob[:, 0]).sum())
    if missing:
        raise ValueError(f"{missing} global points were never assigned")
    return glob
