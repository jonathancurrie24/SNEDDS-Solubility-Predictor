"""
mixture_doe.py — A UI-agnostic core for constrained ternary-mixture design & modeling.

Built for small-data formulation work (e.g. SNEDDS), where you have only a
handful of experimental points on three components that sum to 100%.

Design goals
------------
* No UI, no notebook, no widget dependencies. Pure functions + small classes.
  Wire Jupyter / Streamlit / Dash / a CLI on top of this without touching it.
* No shapely, no sklearn. Everything here is numpy.
* Statistically honest: fits *Scheffe mixture models* (no intercept — the
  components are collinear because they sum to a constant), and refuses to
  fit models the data cannot support.

The three components are referred to as c1, c2, c3 throughout and always sum
to `TOTAL` (default 100). Vertex/point tuples are in (c1, c2, c3) order unless
a function name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TOTAL = 100.0
_EPS = 1e-9

Point = Tuple[float, float, float]


# ----------------------------------------------------------------------------
# Coordinate transforms (ternary <-> 2D cartesian, for area / geometry / plots)
# ----------------------------------------------------------------------------
def ternary_to_cartesian(c1: float, c2: float, c3: float) -> Tuple[float, float]:
    """Map a composition summing to TOTAL onto the 2D plane."""
    x = c2 + 0.5 * c3
    y = (np.sqrt(3.0) / 2.0) * c3
    return x, y


def cartesian_to_ternary(x: float, y: float) -> Point:
    """Inverse of ternary_to_cartesian."""
    c3 = y / (np.sqrt(3.0) / 2.0)
    c2 = x - 0.5 * c3
    c1 = TOTAL - c2 - c3
    return (c1, c2, c3)


def _polygon_area(points_xy: np.ndarray) -> float:
    """Shoelace area of an ordered set of 2D points."""
    x, y = points_xy[:, 0], points_xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _order_ccw(points: Sequence[Point]) -> List[Point]:
    """Order (c1,c2,c3) points counter-clockwise for clean polygon drawing."""
    pts = np.array(points, dtype=float)
    centroid = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return [tuple(points[i]) for i in np.argsort(ang)]


# ----------------------------------------------------------------------------
# Constraints: the feasible region defined by per-component min/max bounds
# ----------------------------------------------------------------------------
@dataclass
class MixtureConstraints:
    """
    Per-component lower/upper bounds on a 3-component mixture.

    bounds maps 'c1' | 'c2' | 'c3' -> (low, high). Everything downstream
    (vertices, feasibility, DOE candidates) derives from this one object, so
    there is a single source of truth instead of the four cells each re-reading
    slider state.
    """
    bounds: Dict[str, Tuple[float, float]]
    names: Dict[str, str] = field(
        default_factory=lambda: {"c1": "C1", "c2": "C2", "c3": "C3"}
    )
    total: float = TOTAL
    # Optional hard formulation limits ("advisability" range in the old code).
    hard_min: Optional[float] = None
    hard_max: Optional[float] = None

    # -- basic tests --------------------------------------------------------
    def contains(self, point: Sequence[float], tol: float = 1e-6) -> bool:
        """Is a (c1,c2,c3) composition inside the box constraints and on-simplex?"""
        c = dict(zip(("c1", "c2", "c3"), point))
        if abs(sum(point) - self.total) > 1e-4:
            return False
        return all(
            self.bounds[k][0] - tol <= c[k] <= self.bounds[k][1] + tol
            for k in ("c1", "c2", "c3")
        )

    def reachable_range(self, key: str) -> Tuple[float, float]:
        """
        Effective range a component can actually take, given the *other two*
        components' bounds. A bound outside this range is 'dead' (does nothing).
        This is the genuinely useful diagnostic from your original cell 1.
        """
        others = [k for k in ("c1", "c2", "c3") if k != key]
        lo_self, hi_self = self.bounds[key]
        lo_o = sum(self.bounds[o][0] for o in others)
        hi_o = sum(self.bounds[o][1] for o in others)
        return (max(lo_self, self.total - hi_o), min(hi_self, self.total - lo_o))

    def feasibility_report(self) -> "FeasibilityReport":
        return _build_feasibility_report(self)

    # -- geometry -----------------------------------------------------------
    def vertices(self) -> List[Point]:
        """
        Corners of the feasible polygon: intersections of pairs of constraint
        lines that satisfy every bound and lie on the simplex. This is the
        de-duplicated version of the function that appeared in three of your
        four cells.
        """
        b = self.bounds
        lines = [
            ("c1", b["c1"][0]), ("c1", b["c1"][1]),
            ("c2", b["c2"][0]), ("c2", b["c2"][1]),
            ("c3", b["c3"][0]), ("c3", b["c3"][1]),
        ]
        cand: List[Point] = []
        for (n1, v1), (n2, v2) in combinations(lines, 2):
            if n1 == n2:
                continue
            fixed = {n1: v1, n2: v2}
            free = [k for k in ("c1", "c2", "c3") if k not in fixed][0]
            comp = dict(fixed)
            comp[free] = self.total - v1 - v2
            cand.append((comp["c1"], comp["c2"], comp["c3"]))

        valid = []
        for p in cand:
            if self.contains(p, tol=_EPS):
                valid.append(tuple(np.round(p, 5)))
        return _order_ccw(sorted(set(valid))) if valid else []

    def area(self) -> float:
        """Area of the feasible region in cartesian units (0 if degenerate)."""
        verts = self.vertices()
        if len(verts) < 3:
            return 0.0
        xy = np.array([ternary_to_cartesian(*v) for v in verts])
        return _polygon_area(xy)

    def side_lengths(self) -> List[float]:
        """
        Cartesian side lengths of the feasible polygon in the order returned
        by vertices() (i.e. CCW). Empty list if the region isn't a polygon.
        """
        verts = self.vertices()
        if len(verts) < 3:
            return []
        xy = np.array([ternary_to_cartesian(*v) for v in verts])
        n = len(xy)
        return [float(np.linalg.norm(xy[(i + 1) % n] - xy[i])) for i in range(n)]

    def is_equilateral(self, rel_tol: float = 0.02) -> bool:
        """
        True iff the feasible region is a triangle with three equal-length
        sides (within `rel_tol` of the mean side length). This is the
        precondition for treating the vertices as pseudo-components in an
        apex-style linear Scheffe prediction.
        """
        sides = self.side_lengths()
        if len(sides) != 3:
            return False
        s = np.array(sides)
        mean = s.mean()
        return bool(mean > _EPS and (s.max() - s.min()) / mean <= rel_tol)


@dataclass
class FeasibilityReport:
    feasible: bool
    sum_min: float
    sum_max: float
    reachable: Dict[str, Tuple[float, float]]
    binding: List[str]
    inactive: List[str]
    violations: List[str]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        head = "FEASIBLE" if self.feasible else "INFEASIBLE"
        lines = [f"[{head}]  sum(min)={self.sum_min:.1f}  sum(max)={self.sum_max:.1f}"]
        for k, (lo, hi) in self.reachable.items():
            lines.append(f"  reachable {k}: [{lo:.1f}, {hi:.1f}]")
        for v in self.violations:
            lines.append(f"  ! {v}")
        for i in self.inactive:
            lines.append(f"  ~ inactive: {i}")
        return "\n".join(lines)


def _build_feasibility_report(c: MixtureConstraints) -> FeasibilityReport:
    b, names, total = c.bounds, c.names, c.total
    sum_min = sum(b[k][0] for k in ("c1", "c2", "c3"))
    sum_max = sum(b[k][1] for k in ("c1", "c2", "c3"))
    violations: List[str] = []

    if sum_min > total + _EPS:
        violations.append(
            f"Sum of minimums is {sum_min:.1f}% (must be <= {total:.0f}). "
            f"Lower one minimum by >= {sum_min - total:.1f}%."
        )
    if sum_max < total - _EPS:
        violations.append(
            f"Sum of maximums is {sum_max:.1f}% (must be >= {total:.0f}). "
            f"Raise one maximum by >= {total - sum_max:.1f}%."
        )

    reachable, binding, inactive = {}, [], []
    DEAD = 0.5
    for k in ("c1", "c2", "c3"):
        lo, hi = c.reachable_range(k)
        reachable[k] = (lo, hi)
        set_lo, set_hi = b[k]
        if lo > hi + _EPS:
            violations.append(
                f"{names[k]}: reachable range collapses to [{lo:.1f}, {hi:.1f}] "
                f"- bounds inconsistent with the other two components."
            )
            continue
        if set_lo < lo - DEAD:
            inactive.append(f"{names[k]} min ({set_lo:g}%) -> real floor {lo:.0f}%")
        else:
            binding.append(f"{names[k]} >= {set_lo:g}%")
        if set_hi > hi + DEAD:
            inactive.append(f"{names[k]} max ({set_hi:g}%) -> real cap {hi:.0f}%")
        else:
            binding.append(f"{names[k]} <= {set_hi:g}%")

    # Optional hard formulation limits.
    if c.hard_min is not None or c.hard_max is not None:
        for k in ("c1", "c2", "c3"):
            lo, hi = b[k]
            if c.hard_min is not None and lo < c.hard_min - _EPS:
                violations.append(
                    f"{names[k]} minimum ({lo:g}%) below allowed floor {c.hard_min:g}%."
                )
            if c.hard_max is not None and hi > c.hard_max + _EPS:
                violations.append(
                    f"{names[k]} maximum ({hi:g}%) above allowed ceiling {c.hard_max:g}%."
                )

    return FeasibilityReport(
        feasible=not violations,
        sum_min=sum_min, sum_max=sum_max,
        reachable=reachable, binding=binding,
        inactive=inactive, violations=violations,
    )


# ----------------------------------------------------------------------------
# Design of experiments: where to place your handful of points
# ----------------------------------------------------------------------------
# Scheffe model term counts, so we can guard n vs p.
_MODEL_TERMS = {"linear": 3, "quadratic": 6, "special_cubic": 7}


def _scheffe_expand(X: np.ndarray, degree: str) -> np.ndarray:
    """
    Build the Scheffe model matrix (NO intercept — that is the whole point).

    linear:         x1, x2, x3
    quadratic:      + x1x2, x1x3, x2x3
    special_cubic:  + x1x2x3
    """
    X = np.asarray(X, dtype=float) / TOTAL  # work in proportions for conditioning
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    cols = [x1, x2, x3]
    if degree in ("quadratic", "special_cubic"):
        cols += [x1 * x2, x1 * x3, x2 * x3]
    if degree == "special_cubic":
        cols += [x1 * x2 * x3]
    return np.column_stack(cols)


def candidate_points(
    c: MixtureConstraints,
    include_edges: bool = True,
    include_centroid: bool = True,
) -> List[Point]:
    """
    Candidate set for a constrained-mixture design (McLean-Anderson style):
    the extreme vertices of the feasible region, plus optionally the midpoint
    of each edge and the overall centroid. These are exactly the points a
    D-optimal design will want to choose from.
    """
    verts = c.vertices()
    if not verts:
        return []
    pts = list(verts)
    if include_edges and len(verts) >= 2:
        for a, b in zip(verts, verts[1:] + verts[:1]):
            mid = tuple((np.array(a) + np.array(b)) / 2.0)
            pts.append(mid)
    if include_centroid:
        pts.append(tuple(np.mean(verts, axis=0)))
    # de-duplicate
    uniq = {tuple(np.round(p, 4)) for p in pts}
    return [tuple(map(float, p)) for p in uniq]


def d_optimal_design(
    c: MixtureConstraints,
    n_points: int,
    degree: str = "linear",
    seed: Optional[int] = 0,
) -> Dict[str, object]:
    """
    Pick `n_points` from the candidate set to maximise D-efficiency
    (det of the information matrix) for the chosen Scheffe model, using a
    Fedorov-style exchange. This replaces the random 20k-iteration
    quadrilateral search with something deterministic and principled.

    Returns a dict with the chosen points and the model it is optimal for.
    Raises ValueError if the model is not identifiable with n_points.
    """
    if degree not in _MODEL_TERMS:
        raise ValueError(f"degree must be one of {list(_MODEL_TERMS)}")
    p = _MODEL_TERMS[degree]
    if n_points < p:
        raise ValueError(
            f"A '{degree}' Scheffe model has {p} terms; you asked for "
            f"{n_points} points. You need at least {p} (and ideally a few "
            f"more, to estimate error). Use degree='linear' for small budgets."
        )

    cand = candidate_points(c)
    C = _scheffe_expand(np.array(cand), degree)
    m = len(cand)
    if m < n_points:
        raise ValueError(
            f"Only {m} candidate points exist in this region but {n_points} "
            f"were requested. Loosen the constraints or lower n_points."
        )

    rng = np.random.default_rng(seed)

    def logdet(idx):
        M = C[idx]
        sign, ld = np.linalg.slogdet(M.T @ M)
        return ld if sign > 0 else -np.inf

    # Start from a random identifiable subset.
    best = list(rng.choice(m, size=n_points, replace=False))
    best_val = logdet(best)
    for _ in range(200):  # restarts
        cur = list(rng.choice(m, size=n_points, replace=False))
        cur_val = logdet(cur)
        improved = True
        while improved:
            improved = False
            for i in range(n_points):
                for j in range(m):
                    if j in cur:
                        continue
                    trial = cur.copy()
                    trial[i] = j
                    val = logdet(trial)
                    if val > cur_val + 1e-9:
                        cur, cur_val = trial, val
                        improved = True
        if cur_val > best_val:
            best, best_val = cur, cur_val

    chosen = [tuple(np.round(cand[i], 2)) for i in best]
    return {
        "points": chosen,
        "degree": degree,
        "n_terms": p,
        "residual_df": n_points - p,
        "log_det": best_val,
        "n_candidates": m,
    }


# ----------------------------------------------------------------------------
# The model: Scheffe mixture regression with small-n-honest diagnostics
# ----------------------------------------------------------------------------
class ScheffeModel:
    """
    Scheffe canonical mixture-model regression (no intercept).

    Handles the collinearity that makes an ordinary intercept model invalid on
    mixture data, and reports diagnostics (R^2, adjusted R^2, residual df)
    appropriate for a handful of points.
    """

    def __init__(self, degree: str = "linear"):
        if degree not in _MODEL_TERMS:
            raise ValueError(f"degree must be one of {list(_MODEL_TERMS)}")
        self.degree = degree
        self.coef_: Optional[np.ndarray] = None
        self.n_: int = 0

    @property
    def n_terms(self) -> int:
        return _MODEL_TERMS[self.degree]

    def fit(self, X, y) -> "ScheffeModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        if n < self.n_terms:
            raise ValueError(
                f"'{self.degree}' needs >= {self.n_terms} points; got {n}. "
                f"Fit a lower-degree model or collect more data."
            )
        M = _scheffe_expand(X, self.degree)
        self.coef_, *_ = np.linalg.lstsq(M, y, rcond=None)
        self.n_ = n
        self._X, self._y = X, y
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        return _scheffe_expand(X, self.degree) @ self.coef_

    def summary(self) -> Dict[str, object]:
        """R^2, adjusted R^2, residual df, coefficients, and an honesty flag.

        Leave-one-out cross-validation was removed: with the design sizes
        this app targets (n ≈ 4-8, p ≥ 3), refitting on n-1 points routinely
        produces saturated or ill-conditioned models, so a LOO-RMSE
        derived from those refits is more misleading than informative.
        External validation points (Step 5 of the UI) are the honest
        generalisation check instead.
        """
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        y, X = self._y, self._X
        n, p = self.n_, self.n_terms
        yhat = self.predict(X)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # adjusted R^2 for a no-intercept model uses df = n - p
        if n - p > 0 and ss_tot > 0:
            adj_r2 = 1.0 - (ss_res / (n - p)) / (ss_tot / (n - 1))
        else:
            adj_r2 = float("nan")

        return {
            "degree": self.degree,
            "n_points": n,
            "n_terms": p,
            "residual_df": n - p,
            "coefficients": {i: float(b) for i, b in enumerate(self.coef_)},
            "r2": r2,
            "adj_r2": adj_r2,
            "observed": [float(v) for v in y],
            "fitted_values": [float(v) for v in yhat],
            "trustworthy": (n - p) >= 2,
            "note": (
                "Saturated or near-saturated fit: R^2 will look perfect but "
                "means nothing. Collect more points or drop model degree."
                if (n - p) < 2 else
                "Enough residual df for a basic error estimate. Add validation "
                "points (Step 5) for an honest generalisation check."
            ),
        }


def recommend_degree(n_points: int) -> str:
    """Highest Scheffe degree that leaves at least ~2 residual df for n points."""
    for deg in ("special_cubic", "quadratic", "linear"):
        if n_points - _MODEL_TERMS[deg] >= 2:
            return deg
    return "linear"


# ----------------------------------------------------------------------------
# Prediction grid inside the feasible region (for plotting a surface)
# ----------------------------------------------------------------------------
def feasible_grid(c: MixtureConstraints, step: float = 1.0) -> np.ndarray:
    """(N,3) array of on-simplex compositions inside the constraints."""
    pts = []
    n = int(round(c.total / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            comp = (i * step, j * step, c.total - i * step - j * step)
            if c.contains(comp, tol=1e-6):
                pts.append(comp)
    return np.array(pts) if pts else np.empty((0, 3))
