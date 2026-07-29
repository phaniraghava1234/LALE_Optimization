#!/usr/bin/env python3
"""
Produce clean line plots of the SLSQP benchmark run's convergence history
from the values recorded in `opt_summary.md`. Replaces GEMSEO's default
heatmap-style plots with plots that are readable in a report.

Usage:
    python docs/make_history_plots.py

Writes 4 PNGs into results/benchmark_run_20260729/plots/:
    history_CD.png
    history_CL.png
    history_constraints.png
    history_merit_feasibility.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np

# ---- data lifted from case/opt_summary.md (benchmark run) -----------
evals   = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
CD      = np.array([0.018385, 0.019400, 0.016417, 0.016149, 0.015942,
                    0.018022, 0.015251, 0.025661, 0.014713, 0.022346])
CL      = np.array([0.788461, 0.826844, 0.798586, 0.814745, 0.822368,
                    0.784408, 0.825190, 0.675661, 0.826245, 0.721383])
thick_m = np.array([1.0000, 0.9988, 0.9953, 0.9960, 0.9941,
                    0.9871, 0.9926, 0.9790, 0.9909, 0.9824])
volcon  = np.array([1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                    1.0001, 1.0000, 1.0004, 1.0001, 1.0000])
rcon_m  = np.array([1.0000, 1.0010, 1.0051, 1.0040, 1.0050,
                    1.0083, 1.0057, 1.0125, 1.0066, 1.0101])
aoa     = np.array([2.8000, 2.9339, 2.6375, 2.6868, 2.6752,
                    2.4607, 2.6294, 2.1363, 2.5680, 2.2563])

# Constants
CL_TARGET       = 0.824
THICKCON_LOWER  = 0.5
VOLCON_LOWER    = 1.0
RCON_LOWER      = 0.8

CD_COUNTS       = CD * 1e4
CL_ERROR        = CL - CL_TARGET
OPT_EVAL        = 9    # reported optimum

# ---- output path ---------------------------------------------------
OUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "results", "benchmark_run_20260729", "plots"))
os.makedirs(OUT_DIR, exist_ok=True)


def annotate_optimum(ax, x, y, label):
    ax.axvline(OPT_EVAL, color="red", ls="--", lw=1, alpha=0.6,
               label=f"Eval {OPT_EVAL} = reported optimum")
    ax.plot([OPT_EVAL], [y[OPT_EVAL - 1]], "ro", ms=8, zorder=5)
    ax.annotate(label, xy=(OPT_EVAL, y[OPT_EVAL - 1]),
                xytext=(OPT_EVAL - 3, y[OPT_EVAL - 1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
                fontsize=9)


# ============ 1. CD history ============================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(evals, CD_COUNTS, "-o", lw=2, ms=7, color="#1f4e79",
        label="CD (drag counts)")
ax.axhline(CD_COUNTS[0], color="#7f7f7f", ls=":",
           label=f"Baseline = {CD_COUNTS[0]:.1f}")
ax.axhspan(123, 135, color="#a2d5c6", alpha=0.35,
           label="He 2019 optimum range (123-135)")
annotate_optimum(ax, evals, CD_COUNTS,
                 f"Eval 9\n{CD_COUNTS[OPT_EVAL-1]:.1f} cts")
ax.set_xlabel("SLSQP evaluation")
ax.set_ylabel("Drag coefficient CD (counts)")
ax.set_title("Objective history — Drag coefficient")
ax.grid(True, alpha=0.4)
ax.set_xticks(evals)
ax.legend(loc="upper left")
fig.tight_layout()
p = os.path.join(OUT_DIR, "history_CD.png")
fig.savefig(p, dpi=140)
print(f"[wrote] {p}")

# ============ 2. CL history ============================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(evals, CL, "-o", lw=2, ms=7, color="#c0392b", label="CL (achieved)")
ax.axhline(CL_TARGET, color="#0f6c26", ls="--", lw=2,
           label=f"Target CL* = {CL_TARGET}")
ax.fill_between(evals, CL_TARGET - 0.01, CL_TARGET + 0.01,
                color="#a2d5c6", alpha=0.35,
                label="GEMSEO eq. tolerance ±0.01")
annotate_optimum(ax, evals, CL, f"Eval 9\nCL = {CL[OPT_EVAL-1]:.4f}")
ax.set_xlabel("SLSQP evaluation")
ax.set_ylabel("Lift coefficient CL")
ax.set_title("Lift-equality-constraint tracking")
ax.grid(True, alpha=0.4)
ax.set_xticks(evals)
ax.legend(loc="lower left")
fig.tight_layout()
p = os.path.join(OUT_DIR, "history_CL.png")
fig.savefig(p, dpi=140)
print(f"[wrote] {p}")

# ============ 3. Constraint history ====================================
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

axes[0].plot(evals, thick_m, "-o", color="#1f4e79", lw=2, ms=6)
axes[0].axhline(THICKCON_LOWER, color="red", ls="--",
                label=f"lower bound = {THICKCON_LOWER}")
axes[0].set_ylabel("thickcon min")
axes[0].set_title("Inequality constraints (all must stay above dashed line)")
axes[0].axvline(OPT_EVAL, color="red", ls="--", lw=1, alpha=0.6)
axes[0].grid(True, alpha=0.4)
axes[0].legend(loc="lower left")

axes[1].plot(evals, volcon, "-o", color="#0f6c26", lw=2, ms=6)
axes[1].axhline(VOLCON_LOWER, color="red", ls="--",
                label=f"lower bound = {VOLCON_LOWER}")
axes[1].set_ylabel("volcon")
axes[1].axvline(OPT_EVAL, color="red", ls="--", lw=1, alpha=0.6)
axes[1].grid(True, alpha=0.4)
axes[1].legend(loc="lower left")

axes[2].plot(evals, rcon_m, "-o", color="#8e44ad", lw=2, ms=6)
axes[2].axhline(RCON_LOWER, color="red", ls="--",
                label=f"lower bound = {RCON_LOWER}")
axes[2].set_ylabel("rcon min")
axes[2].axvline(OPT_EVAL, color="red", ls="--", lw=1, alpha=0.6)
axes[2].set_xlabel("SLSQP evaluation")
axes[2].set_xticks(evals)
axes[2].grid(True, alpha=0.4)
axes[2].legend(loc="lower left")

fig.tight_layout()
p = os.path.join(OUT_DIR, "history_constraints.png")
fig.savefig(p, dpi=140)
print(f"[wrote] {p}")

# ============ 4. Merit function + Feasibility norm ====================
# Feasibility violation (L1 norm across all constraints):
#   eq CL   contributes max(|CL - CL*| - eq_tol, 0)
#   ineq thickcon contributes max(THICKCON_LOWER - thick_m, 0)
#   ineq volcon   contributes max(VOLCON_LOWER   - volcon,  0)
#   ineq rcon     contributes max(RCON_LOWER     - rcon_m,  0)
# For SLSQP, "feasibility" is |violations|. Zero = feasible.
eq_tol = 0.01
feas = (
    np.maximum(np.abs(CL_ERROR) - eq_tol, 0.0) +
    np.maximum(THICKCON_LOWER - thick_m, 0.0) +
    np.maximum(VOLCON_LOWER   - volcon,  0.0) +
    np.maximum(RCON_LOWER     - rcon_m,  0.0)
)

# Merit function (L1 penalty). Common textbook form:
#   phi(x, rho) = f(x) + rho * ||g_violation(x)||_1
# rho chosen so obj and penalty are of comparable scale.
rho = 0.1
merit = CD + rho * (np.abs(CL_ERROR)                # equality treated as ineq
                    + np.maximum(THICKCON_LOWER - thick_m, 0.0)
                    + np.maximum(VOLCON_LOWER   - volcon,  0.0)
                    + np.maximum(RCON_LOWER     - rcon_m,  0.0))
merit_counts = merit * 1e4

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

axes[0].plot(evals, merit_counts, "-o", color="#c0392b", lw=2, ms=7,
             label=f"Merit  = CD + ρ·violation, ρ={rho}")
axes[0].plot(evals, CD_COUNTS,   "--s", color="#1f4e79", lw=1.5, ms=6,
             label="CD only (objective)")
axes[0].axvline(OPT_EVAL, color="red", ls="--", lw=1, alpha=0.6,
                label=f"Eval {OPT_EVAL} = optimum")
axes[0].set_ylabel("counts")
axes[0].set_title("L1 merit function vs pure objective")
axes[0].grid(True, alpha=0.4)
axes[0].legend()

axes[1].plot(evals, feas, "-o", color="#8e44ad", lw=2, ms=7,
             label="Constraint violation norm")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].axvline(OPT_EVAL, color="red", ls="--", lw=1, alpha=0.6)
axes[1].set_xlabel("SLSQP evaluation")
axes[1].set_ylabel("|violation|_1")
axes[1].set_title("Feasibility norm (0 = fully feasible)")
axes[1].set_xticks(evals)
axes[1].grid(True, alpha=0.4)
axes[1].legend()

fig.tight_layout()
p = os.path.join(OUT_DIR, "history_merit_feasibility.png")
fig.savefig(p, dpi=140)
print(f"[wrote] {p}")

# ============ Print a small summary ====================================
print("\n=== Summary ===")
print(f"Baseline (eval 1) : CD = {CD_COUNTS[0]:.1f} cts, CL = {CL[0]:.4f}")
print(f"Optimum  (eval {OPT_EVAL}): "
      f"CD = {CD_COUNTS[OPT_EVAL-1]:.1f} cts, CL = {CL[OPT_EVAL-1]:.4f}")
print(f"Feasibility at Eval {OPT_EVAL}: {feas[OPT_EVAL-1]:.4e} (0 = feasible)")
print(f"Merit    at Eval {OPT_EVAL}: {merit_counts[OPT_EVAL-1]:.2f} counts")
