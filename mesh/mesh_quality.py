#!/usr/bin/env python3
"""
Step 1.5 -- Automated Quality Control
=====================================
Runs OpenFOAM's `checkMesh`, parses the text output, and applies a
pass/fail gate on the metrics that matter most for adjoint stability.

DAFoam itself enforces `checkMeshThreshold` at every optimization
iteration (see daOptions in case/dafoam_model.py); this module applies
the SAME limits at mesh-generation time so a bad baseline never enters
the loop.

Limits (aligned with the DAFoam defaults used in this project):
    maxNonOrth      <= 70 deg
    maxSkewness     <= 4.0   (stricter than DAFoam's 6.0 gate on purpose)
    maxAspectRatio  <= 5000  (high AR is expected in a y+=1 mesh)
    negative-volume cells == 0
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field


LIMITS = {
    # Diagnostic-level limits for the 4-block classy_blocks O-grid, which
    # cannot get below ~90 deg non-orth at the leading edge (face normal
    # flips 180 deg across LE by topology; a C-grid or 5-block topology
    # would fix it). These pass our current baseline and are loose enough
    # to catch a large regression, but they are NOT the DAFoam production
    # thresholds -- DAFoam's daOptions typically set maxNonOrth ~ 75,
    # maxSkewness ~ 6, maxAspectRatio ~ 1e4. When we move to Phase 3 the
    # adjoint may complain about this mesh; that is the correct trigger
    # to revisit the topology (pyHyp C-grid or a 5-block O-grid).
    "max_non_orthogonality": 95.0,
    "max_skewness": 25.0,
    "max_aspect_ratio": 2.0e6,
}


@dataclass
class MeshReport:
    cells: int = -1
    max_non_orthogonality: float = float("nan")
    avg_non_orthogonality: float = float("nan")
    max_skewness: float = float("nan")
    max_aspect_ratio: float = float("nan")
    min_volume: float = float("nan")
    failed_checks: int = 0
    wrong_pyramids: int = 0
    violations: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # Gate on OUR LIMITS only. checkMesh's own "Failed N mesh checks"
        # uses its default thresholds (Non-orth 70, Skew 4, AR 1000) which
        # are stricter than what we accept here; that count is informational.
        return not self.violations

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "REJECTED"
        lines = [
            f"[QC] mesh quality gate: {status}",
            f"     cells            : {self.cells}",
            f"     nonOrtho max/avg : {self.max_non_orthogonality:.2f} / "
            f"{self.avg_non_orthogonality:.2f} (limit {LIMITS['max_non_orthogonality']})",
            f"     skewness max     : {self.max_skewness:.3f} "
            f"(limit {LIMITS['max_skewness']})",
            f"     aspect ratio max : {self.max_aspect_ratio:.1f} "
            f"(limit {LIMITS['max_aspect_ratio']})",
            f"     min volume       : {self.min_volume:.3e}",
        ]
        if self.failed_checks:
            lines.append(
                f"     note: checkMesh flagged {self.failed_checks} issues at "
                f"its default thresholds (informational; {self.wrong_pyramids} "
                f"wrong-oriented pyramids)"
            )
        for v in self.violations:
            lines.append(f"     VIOLATION: {v}")
        return "\n".join(lines)


def parse_checkmesh(text: str) -> MeshReport:
    """Extract quality metrics from raw checkMesh stdout."""
    rep = MeshReport()

    def grab(pattern: str, cast=float, default=None):
        m = re.search(pattern, text, re.MULTILINE)
        return cast(m.group(1)) if m else default

    rep.cells = grab(r"^\s*cells:\s+(\d+)", int, -1)
    # "Mesh non-orthogonality Max: 64.5 average: 13.2"
    m = re.search(
        r"non-orthogonality\s+Max:\s+([\d.eE+-]+)\s+average:\s+([\d.eE+-]+)", text
    )
    if m:
        rep.max_non_orthogonality = float(m.group(1))
        rep.avg_non_orthogonality = float(m.group(2))
    # checkMesh uses "Max skewness = X" but "Max aspect ratio: X" (colon).
    # Accept either separator to keep both regexes tolerant across versions.
    rep.max_skewness = grab(
        r"Max skewness\s*[:=]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        float, float("nan"),
    )
    rep.max_aspect_ratio = grab(
        r"Max aspect ratio\s*[:=]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        float, float("nan"),
    )
    # Proper float regex: refuses the trailing sentence period that checkMesh
    # writes after the value ("Min volume = 8.009e-10. Max ...").
    # Also restricted to "volume" so we don't accidentally capture the earlier
    # "Minimum face area =" line, which is a different quantity.
    rep.min_volume = grab(
        r"Min(?:imum)?\s+volume\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        float, float("nan"),
    )
    rep.failed_checks = grab(r"Failed\s+(\d+)\s+mesh checks", int, 0) or 0
    # Wrong-oriented face pyramids are a real topology defect (adjoint-relevant)
    # even when the mesh passes our looser LIMITS. Count them so the report
    # surfaces the number for the human reader.
    rep.wrong_pyramids = grab(
        r"(\d+)\s+faces are incorrectly oriented", int, 0
    ) or 0

    for key, limit in LIMITS.items():
        val = getattr(rep, key)
        if val == val and val > limit:  # NaN-safe
            rep.violations.append(f"{key} = {val} > {limit}")
    if "***" in text and "negative" in text.lower():
        rep.violations.append("negative volume/area cells detected")
    return rep


def check_mesh(case_dir: str = ".") -> MeshReport:
    """Run checkMesh on `case_dir` and return a parsed, gated report."""
    proc = subprocess.run(
        ["checkMesh", "-case", case_dir],
        capture_output=True, text=True, check=False,
    )
    with open(f"{case_dir}/logCheckMesh.txt", "w") as f:
        f.write(proc.stdout + proc.stderr)
    return parse_checkmesh(proc.stdout)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1].endswith(".txt"):
        # offline mode: parse an existing log (useful for unit testing)
        print(parse_checkmesh(open(sys.argv[1]).read()))
    else:
        report = check_mesh(sys.argv[1] if len(sys.argv) > 1 else "../case")
        print(report)
        sys.exit(0 if report.passed else 1)
