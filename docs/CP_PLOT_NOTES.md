# Notes: Extracting Cp Along the Airfoil Surface

Working notes on how the Cp distribution is extracted from a converged
DAFoam solution and turned into a clean two-curve plot (upper and lower
surface). Documents a classification issue near the trailing edge and
how it was resolved.

---

## 1. Simple explanation (with a picture)

We have ~854 surface points from ParaView. Each point has an `(x, y)`
position and a `Cp` value. To draw a Cp-vs-x plot we must first decide,
for every point, whether it belongs to the **upper** or **lower**
surface of the airfoil.

Near the trailing edge, this is harder than it sounds.

### Sketch of the near-TE region

```
y
^
|                                UPPER SURFACE
|  \___                          (slopes downward
|      \___                       toward y ≈ 0)
|          \___
|              \___
+ - - - - - - - - - - - - - - - -  chord line (y ≈ 0)
|              ___/
|          ___/
|      ___/                       LOWER SURFACE
|  ___/                            (slopes upward
|                                   toward y ≈ 0)
|
+---+----+----+----+---> x/c
   0.85 0.9  0.95  1.0
```

| x/c | upper y | lower y | Vertical gap |
|---|---:|---:|---:|
| 0.85 | +0.030 | −0.020 | 0.050 |
| 0.95 | +0.020 | −0.010 | 0.030 |
| 0.99 | +0.005 | −0.005 | 0.010 |

Everything is fine at x/c = 0.85 — plenty of separation between upper
and lower. But by x/c = 0.99, upper and lower are only a centimetre
apart (on a 1-metre chord). Any classification rule that gets confused
by "small y" will fail here.

### The bad rule (used first)

> **Compare each point's y to the leading-edge y-value.
> If y > y_LE → upper. Otherwise → lower.**

This uses a single number (y_LE, say +0.0003) as the reference
for the whole airfoil.

- A lower-surface point at x = 0.95 with y = +0.0005 is > 0.0003 →
  incorrectly classified as UPPER.
- Effect: some real lower points end up on the upper curve.
- Visually: a **zigzag** as the plotter tries to draw a straight line
  through a mess of alternating true-upper and mislabelled points.

### The good rule (final version)

> **Draw a straight line from the LE to the TE.
> If a point is above that line → upper. Otherwise → lower.**

The reference is no longer a single scalar; it's a slanted line that
adapts to your position along the chord:

```
y_chord(x) = y_LE + (y_TE − y_LE) × (x − x_LE) / (x_TE − x_LE)
```

Now the classifier says "above the chord line, whatever x you're at" —
which is robust to camber, small LE/TE offsets, and most of the airfoil.

### Why we still trim

Even the chord-line rule fails in the very last percent of chord, where
upper and lower are so close to the chord line itself (within ~2 mm)
that numerical noise dominates the comparison. We drop those points
entirely.

**Trim value chosen**: x/c ≤ 0.98. Discards the last 2% chord, keeps
everything else clean.

Note: 0.98 is *not* the position where the blunt TE physically starts —
the blunt base is the last 1% (x/c ≈ 0.99–1.0). The 0.98 cutoff is the
point past which even the good rule can no longer separate upper from
lower reliably.

---

## 2. Technical explanation

### Data source

ParaView reads the OpenFOAM case and applies the following filter
chain:

```
case.foam
  └── Calculator1        Mach = |U| / sqrt(1.4 * 287 * T)
      └── Calculator2    Cp   = (p - p_inf) / (0.5 * rho_inf * U_inf^2)
          └── ExtractBlock1        (patch/wing only, 427 cells × 2 z stations)
              └── CellDatatoPointData1
                  └── (Save Data → cp_eval9.csv, 854 rows)
```

The CSV columns are `Cp, Mach, Points:0, Points:1, Points:2, T, U:0,
U:1, U:2, nuTilda, nut, p`. Position columns are `Points:0` (x),
`Points:1` (y), `Points:2` (z).

### Filtering to one spanwise station

The wing patch has two z-planes at z = 0 and z = 0.01 (the 2D case is
one cell thick). We keep only `z = 0`, reducing 854 → 427 points.

### Classification problem: upper vs lower

The 427 points form a closed loop tracing the airfoil surface. We
want to split into two open curves ordered by x. For any given
x-position on the airfoil, one point sits on the upper surface and one
on the lower.

Difficulty arises where |y_upper − y_lower| is small (near the LE, and
especially near the blunt TE base).

### Rule 1: LE-y threshold

```python
upper_mask = y > y[argmin(x)]
```

Fails near TE for the reason described above: a single scalar y-threshold
is not local to where the point is along the chord.

### Rule 2: angle around centroid

```python
theta = np.arctan2(y - yc, x - xc)
order = np.argsort(theta)
# split at argmin(x) and argmax(x) in the reordered array
```

Concept: walk the closed boundary CCW using polar angle. Split at LE
(π) and TE (0). Fails at the blunt TE where multiple points cluster at
very similar angles — the split index landed in the middle of the TE
cluster, mixing upper and lower TE points into one segment.

### Rule 3: chord-line signed offset (used in the final version)

```python
i_LE5  = np.argsort(x)[:5]
i_TE5  = np.argsort(x)[-5:]
x_LE, y_LE = x[i_LE5].mean(), y[i_LE5].mean()
x_TE, y_TE = x[i_TE5].mean(), y[i_TE5].mean()

y_chord = y_LE + (y_TE - y_LE) * (x - x_LE) / (x_TE - x_LE)
upper_mask = y > y_chord
lower_mask = y < y_chord
```

Advantages:

- **Local threshold**: y_chord(x) evolves along the chord, so a slight
  overall y-tilt doesn't cause blanket misclassification.
- **Camber-agnostic**: the chord line is defined by the actual LE and
  TE positions, not by the y = 0 line.
- **Robust to point noise**: using the 5-point average at each end
  smooths out any single mispositioned point.

### Blunt-TE trim

The `mesh/airfoil_io.py:thicken_te()` function inflates the
last ~1% chord into a small vertical strip so the mesh generator (which
struggles with knife-edge geometry) can produce a valid block topology.
The strip physically occupies x/c ≈ 0.99 − 1.00.

However, even before the physical strip, the airfoil surfaces are close
enough to the chord line that the signed offset (y − y_chord) is
smaller than typical solver-output noise. The chord-line rule breaks
down not at the base itself but slightly earlier.

Empirical cutoff: **`X_TE_CUT = 0.98`** removes just enough of the
near-TE region that the remaining data classifies cleanly.

### Interpretation for the report

The last 2% of chord (x/c > 0.98) is **not** physically representative
of the airfoil pressure distribution as designed. It is a mixture of
(a) real near-wake pressure and (b) mesh-topology artefacts from the
blunt-TE strip. Discarding it in Cp visualisations is the correct
scientific choice.

For anyone reproducing this work with an **O-mesh** and **knife-edge
TE** (as in He et al. 2019), the blunt-TE zone does not exist and the
trim is unnecessary. This is a deviation between our workflow and the
paper's.

### File map

| File | Role |
|---|---|
| `mesh/airfoil_io.py:thicken_te()` | source of the blunt TE (necessary for `classy_blocks`) |
| `case/system/controlDict:forceCoeffs1` | writes per-iter integrated Cd/Cl (independent of the Cp plot) |
| `docs/plot_cp.py` | script that reads the CSV and produces the plot |
| `results/benchmark_eval9_coldstart_20260729/cp_eval9.csv` | ParaView export |
| `results/benchmark_eval9_coldstart_20260729/cp_eval9.png` | final plot |

---

## 3. TL;DR

- Near the airfoil TE, upper and lower surface points are close in y —
  any simple "compare y to a threshold" rule breaks down.
- Fix (Part A): **use a chord line as the local threshold**. Adapts along x.
- Fix (Part B): **trim x/c > 0.98**. The last 2% is ambiguous no matter
  what rule you use — that's the mesh-topology zone from the blunt TE.
- Result: a clean two-curve Cp plot that shows the LE suction peak,
  supersonic plateau, shock foot, and pressure recovery, exactly like
  the classic transonic-airfoil textbook picture.
