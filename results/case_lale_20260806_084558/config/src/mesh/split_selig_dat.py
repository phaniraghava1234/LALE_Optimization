#!/usr/bin/env python3
"""Convert a Selig-format airfoil .dat file into two .profile files.

Selig .dat format:
    line 1: airfoil name / header
    lines 2..N: x y coordinate pairs, ordered TE upper -> LE -> TE lower

Output:
    <name>_SS.profile   suction (upper) surface, ordered LE -> TE
    <name>_PS.profile   pressure (lower) surface, ordered LE -> TE

Both files use two space-separated columns matching the existing RAE 2822
profile files that `mesh/airfoil_io.py:read_profile_pair` expects.

Usage:
    python mesh/split_selig_dat.py airfoils/MH139-F.dat \\
           --out-dir case_lale/profiles --stem MH139F
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np


def parse_selig_dat(path: str) -> tuple[str, np.ndarray]:
    """Read a Selig .dat file, tolerating stray HTML tags.

    Returns
    -------
    name : str
        The airfoil name from the header line (best effort).
    coords : (N, 2) ndarray
        Coordinates in file order.
    """
    header = ""
    coords: list[list[float]] = []
    coord_re = re.compile(
        r"^\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
    )
    with open(path) as f:
        for line in f:
            # Strip any HTML wrapper (<pre>, <div>, </div>, </pre>, class="...")
            clean = re.sub(r"<[^>]*>", "", line).strip()
            if not clean:
                continue
            m = coord_re.match(clean)
            if m:
                coords.append([float(m.group(1)), float(m.group(2))])
            elif not header:
                header = clean
    return header, np.array(coords)


def split_upper_lower(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split Selig-ordered coordinates into upper and lower surfaces.

    Assumes the input walks TE upper -> LE -> TE lower. Splits at LE
    (index of minimum x). Both output arrays are ordered LE -> TE.
    """
    i_le = int(np.argmin(coords[:, 0]))
    upper_te2le = coords[: i_le + 1]        # includes LE
    lower_le2te = coords[i_le:]             # includes LE
    upper_le2te = upper_te2le[::-1]         # reverse to LE -> TE
    return upper_le2te, lower_le2te


def write_profile(path: str, xy: np.ndarray) -> None:
    with open(path, "w") as f:
        for x, y in xy:
            f.write(f"  {x:.8f}  {y:.8f}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dat", help="Path to the Selig .dat file")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for the two .profile files")
    p.add_argument("--stem", required=True,
                   help="Filename stem, e.g. MH139F -> writes "
                        "<stem>_SS.profile and <stem>_PS.profile")
    args = p.parse_args()

    name, coords = parse_selig_dat(args.dat)
    print(f"Parsed {len(coords)} coordinate points from {args.dat}")
    print(f"Header: {name!r}")
    print(f"x range: [{coords[:, 0].min():.6f}, {coords[:, 0].max():.6f}]")
    print(f"y range: [{coords[:, 1].min():.6f}, {coords[:, 1].max():.6f}]")

    upper, lower = split_upper_lower(coords)
    print(f"Upper surface (SS): {len(upper)} points, LE -> TE")
    print(f"Lower surface (PS): {len(lower)} points, LE -> TE")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss_path = out_dir / f"{args.stem}_SS.profile"
    ps_path = out_dir / f"{args.stem}_PS.profile"
    write_profile(str(ss_path), upper)
    write_profile(str(ps_path), lower)
    print(f"[wrote] {ss_path}")
    print(f"[wrote] {ps_path}")


if __name__ == "__main__":
    main()
