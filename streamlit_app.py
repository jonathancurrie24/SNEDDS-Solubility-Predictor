"""
SNEDDS Solubility Prediction App — constrained-mixture DOE and Scheffe
modeling for self-nanoemulsifying drug delivery systems.

Install: pip install streamlit numpy matplotlib mpltern pandas
Run:     streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from mixture_doe import (
    MixtureConstraints,
    ScheffeModel,
    d_optimal_design,
    recommend_degree,
    feasible_grid,
    candidate_points,
    _scheffe_expand,   # used to build the prediction surface
)

# ============================================================================
# PAGE CONFIG & SESSION STATE
# ============================================================================
st.set_page_config(
    page_title="SNEDDS Solubility Prediction App",
    page_icon="△",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_DEFAULTS = {
    "comp_names": ["Oil", "Surfactant", "Co-Surfactant"],
    "comp_mins":  [30.0, 10.0, 20.0],
    "comp_maxs":  [60.0, 40.0, 50.0],
    "budget": 6,
    "design_points": [],
    "solubilities": [],
    "fit_result": None,
    "manual_mode": False,
    "apex_mode": False,
    "drug_name": "Drug",
    "title_color": "#1f77b4",
    # ---- Ternary plot element toggles (Step "Plot controls") --------------
    "show_training_points": True,
    "show_legend": True,
    "show_apex_labels": True,
    "show_boundary": True,
    # ---- Validation points (Step 5) --------------------------------------
    # Kept separate from design_points/solubilities so they never affect the
    # fit or the ternary plot — they only feed the validation parity plot.
    "validation_points": [],
    "validation_measurements": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# HELPERS
# ============================================================================
# Model-supported range per single component. Bounds outside this range are
# not physically prevented — the user may set them anywhere in 0-100 — but
# the app warns clearly and the feasibility report flags them as violations.
MIN_SINGLE_COMPONENT = 10.0
MAX_SINGLE_COMPONENT = 80.0


def _apex_predict(points: np.ndarray, vertices, readings) -> np.ndarray:
    """
    Linear interpolation over the feasible triangle using barycentric
    coordinates in the pseudo-component space.

    For each point P (composition summing to 100) we solve
        w1*V1 + w2*V2 + w3*V3 = P,   w1 + w2 + w3 = 1
    for the barycentric weights (w1, w2, w3), then predict
        y_hat(P) = w1*y1 + w2*y2 + w3*y3.

    At P = Vi this gives w = e_i and y_hat = y_i exactly, which is the whole
    point of the pseudo-component read-off. The previous formula
    ((P/100) @ readings) treated the ORIGINAL components as pseudo-components
    and was only correct when the vertices sat at (100,0,0), (0,100,0),
    (0,0,100) — i.e. the unconstrained simplex corners.

    points:   (N, 3) compositions
    vertices: length-3 iterable of 3-tuples (the feasible-triangle corners)
    readings: length-3 iterable of measured responses at those vertices
    """
    V = np.asarray(vertices, dtype=float)   # (3, 3)
    y = np.asarray(readings, dtype=float)   # (3,)
    P = np.atleast_2d(np.asarray(points, dtype=float))  # (N, 3)

    # 2 composition equations + 1 sum-to-one constraint = 3x3 system per point.
    # Solve once for all N via a single matrix inverse.
    A = np.array([
        [V[0, 0], V[1, 0], V[2, 0]],
        [V[0, 1], V[1, 1], V[2, 1]],
        [1.0,     1.0,     1.0    ],
    ])
    rhs = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])  # (N, 3)
    W = rhs @ np.linalg.inv(A).T                                # (N, 3) weights
    return W @ y                                                # (N,)


def predict_from_fit(points, fit_result) -> np.ndarray:
    """
    Predict solubility at arbitrary compositions using a stored fit_result.

    Handles both apex mode (barycentric interpolation over the feasible
    triangle vertices) and Scheffé mode (multiplication of the model matrix
    by the fitted coefficient vector). Used by the validation-points step to
    score user-supplied test compositions against the fitted model without
    ever re-touching the training data.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if fit_result["degree"] == "apex":
        V = np.array(fit_result["apex_vertices"], dtype=float)
        y = np.array(fit_result["apex_readings"], dtype=float)
        return _apex_predict(points, V, y)
    coef_dict = fit_result["coef"]
    coef = np.array([coef_dict[i] for i in sorted(coef_dict)])
    return _scheffe_expand(points, fit_result["degree"]) @ coef


def get_constraints() -> MixtureConstraints:
    return MixtureConstraints(
        bounds={
            "c1": (st.session_state.comp_mins[0], st.session_state.comp_maxs[0]),
            "c2": (st.session_state.comp_mins[1], st.session_state.comp_maxs[1]),
            "c3": (st.session_state.comp_mins[2], st.session_state.comp_maxs[2]),
        },
        names={
            "c1": st.session_state.comp_names[0],
            "c2": st.session_state.comp_names[1],
            "c3": st.session_state.comp_names[2],
        },
        hard_min=MIN_SINGLE_COMPONENT,
        hard_max=MAX_SINGLE_COMPONENT,
    )


def _order_vertices_by_dominant_component(verts):
    """
    Return the 3 vertices ordered so vertex i is the one where component i
    (c1, c2, c3) is largest. Makes the apex fit align with the user's
    mental model: coef[0] is the reading at the 'high-c1 corner', etc.
    """
    verts = list(verts)
    order, used = [], set()
    for comp_idx in range(3):
        best = max(
            (i for i in range(len(verts)) if i not in used),
            key=lambda i: verts[i][comp_idx],
        )
        order.append(verts[best])
        used.add(best)
    return order


def _style_ternary_axes(ax, names, drug, title_color):
    """House-style formatting: tick2 positions, dashed grid, black spines."""
    for axis_name in ("taxis", "laxis", "raxis"):
        getattr(ax, axis_name).set_ticks_position("tick2")
        getattr(ax, axis_name).set_label_position("tick2")

    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.set_major_locator(MultipleLocator(10.0))
        axis.set_minor_locator(MultipleLocator(5.0))

    # Make the tick marks themselves actually visible. Without this, mpltern
    # draws the tick locations but the marks have effectively zero length.
    for which_axis in ("t", "l", "r"):
        ax.tick_params(axis=which_axis, which="major",
                       direction="out", length=6, width=1.2, colors="black")
        ax.tick_params(axis=which_axis, which="minor",
                       direction="out", length=3, width=0.8, colors="black")

    # matplotlib's ax.grid(axis=...) only accepts 'x', 'y', or 'both' — the
    # ternary axis names 't','l','r' silently no-op there. Call grid() on each
    # ternary axis object directly instead; those are real Axis instances and
    # their grid() respects which='major'/'minor' plus styling kwargs. Pass
    # visible=True explicitly so the second (minor) call can't toggle the
    # major grid back off.
    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.grid(visible=True, which="major",
                  linestyle="--", linewidth=0.8, color="gray", alpha=0.7)
        axis.grid(visible=True, which="minor",
                  linestyle=":", linewidth=0.5, color="gray", alpha=0.4)
    # Force the grid to sit above the tripcolor surface.
    ax.set_axisbelow(False)

    ax.set_tlabel(f"% w/w {names['c1']}")
    ax.set_llabel(f"% w/w {names['c3']}")
    ax.set_rlabel(f"% w/w {names['c2']}")

    ax.set_title(drug, pad=20, color=title_color, fontweight="bold")

    for side in ("tside", "lside", "rside"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(2)


def plot_parity(observed, fitted, loo=None,
                val_observed=None, val_predicted=None,
                color="#1f77b4", drug=""):
    """
    Observed-vs-predicted parity plot.

    Open squares: in-sample fitted values (training-time predictions).
    Filled circles: leave-one-out predictions, when available.
    Red triangles: external validation points (`val_observed` vs
    `val_predicted`), when provided. Validation predictions come from the
    fitted model — they are held-out data, not part of the fit — so their
    scatter around the 1:1 line is the honest generalisation check that
    LOO-RMSE is trying (with much less data) to approximate.

    A metrics box in the upper-left shows in-sample MAE / RMSE / R² (and
    LOO-RMSE when available); a second box in the lower-right shows the
    same three metrics computed on the validation set.
    """
    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    loo_arr = np.asarray(loo, dtype=float) if loo is not None else None
    val_obs = np.asarray(val_observed, dtype=float) if val_observed is not None else None
    val_pred = np.asarray(val_predicted, dtype=float) if val_predicted is not None else None
    have_val = val_obs is not None and val_pred is not None and len(val_obs) > 0

    fig, ax = plt.subplots(figsize=(6, 6))

    # square axes around all data (training + LOO + validation)
    stack = [observed, fitted]
    if loo_arr is not None:
        stack.append(loo_arr)
    if have_val:
        stack.extend([val_obs, val_pred])
    all_vals = np.concatenate(stack)
    lo, hi = float(all_vals.min()), float(all_vals.max())
    span = hi - lo
    buf = span * 0.1 if span > 0 else max(abs(hi), 1.0) * 0.1
    limits = [lo - buf, hi + buf]

    ax.plot(limits, limits, "k--", alpha=0.6, zorder=3)

    ax.scatter(
        observed, fitted,
        marker="s", facecolors="none", edgecolors=color,
        s=80, linewidths=1.5, zorder=4,
        label="In-sample fit",
    )
    if loo_arr is not None:
        ax.scatter(
            observed, loo_arr,
            marker="o", color=color,
            s=120, edgecolors="white", linewidths=1.5, zorder=5,
            label="LOO prediction",
        )
    if have_val:
        ax.scatter(
            val_obs, val_pred,
            marker="^", color="#d62728",
            s=140, edgecolors="white", linewidths=1.5, zorder=6,
            label="Validation",
        )

    # in-sample metrics
    resid = observed - fitted
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    lines = [f"MAE   {mae:.2f}", f"RMSE  {rmse:.2f}", f"R²    {r2:.3f}"]
    if loo_arr is not None:
        loo_rmse = float(np.sqrt(np.mean((observed - loo_arr) ** 2)))
        lines.append(f"LOO   {loo_rmse:.2f}")
    props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=11, family="monospace",
        verticalalignment="top", bbox=props, zorder=7,
    )

    # validation metrics (separate box so it's obvious which set they came from)
    if have_val:
        v_resid = val_obs - val_pred
        v_mae = float(np.mean(np.abs(v_resid)))
        v_rmse = float(np.sqrt(np.mean(v_resid ** 2)))
        v_ss_tot = float(np.sum((val_obs - val_obs.mean()) ** 2))
        v_r2 = 1.0 - float(np.sum(v_resid ** 2)) / v_ss_tot if v_ss_tot > 0 else float("nan")
        v_lines = [
            f"Val n = {len(val_obs)}",
            f"MAE   {v_mae:.2f}",
            f"RMSE  {v_rmse:.2f}",
            f"R²    {v_r2:.3f}" if np.isfinite(v_r2) else "R²    n/a",
        ]
        v_props = dict(boxstyle="round", facecolor="#fff5f5", alpha=0.9, edgecolor="#d62728")
        ax.text(
            0.95, 0.05, "\n".join(v_lines),
            transform=ax.transAxes, fontsize=11, family="monospace",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=v_props, zorder=7,
        )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed (mg/g)")
    ax.set_ylabel("Predicted (mg/g)")
    ax.set_title(
        f"{drug} — parity" if drug else "Parity",
        color=color, fontweight="bold", pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right" if not have_val else "upper right",
              frameon=True, fontsize=9)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.5)

    plt.tight_layout()
    return fig


def plot_ternary(constraints, design_pts=None, fit_result=None,
                 drug="", title_color="#1f77b4", apex_mode=False,
                 show_training_points=True, show_legend=True,
                 show_apex_labels=True, show_boundary=True):
    """
    Standard mode: feasible outline + heatmap restricted to the feasible region.
    Apex mode:     heatmap over the feasible region built from the three
                   vertex readings (no extrapolation beyond the design box)
                   + feasible-triangle outline + vertex value labels.

    The four `show_*` flags are the user-facing plot toggles: hide the
    training-point markers, the legend, the apex value annotations, or the
    feasible-region boundary line without touching the underlying data.
    """
    try:
        import mpltern  # noqa: F401  (registers the 'ternary' projection)
    except ImportError:
        st.error("mpltern not installed. Add `mpltern` to requirements.txt.")
        return None

    fig = plt.figure(figsize=(8, 7))
    # ternary_sum=100 is the key fix: without it mpltern defaults to a 0-1
    # coordinate range and MultipleLocator(10) produces zero in-range ticks,
    # so no tick marks, no numeric labels, and no grid ever draw. Data is
    # still positioned by ratio, which is why the heatmap and markers looked
    # right while everything else was blank.
    ax = fig.add_subplot(111, projection="ternary", ternary_sum=100.0)
    _style_ternary_axes(ax, constraints.names, drug or "SNEDDS Solubility Prediction App", title_color)

    verts = constraints.vertices()

    # -- prediction surface -------------------------------------------------
    if fit_result is not None:
        coef_dict = fit_result["coef"]
        coef = np.array([coef_dict[i] for i in sorted(coef_dict)])
        deg = fit_result["degree"]

        if deg == "apex":
            # Barycentric interpolation w.r.t. the three feasible-triangle
            # vertices. The previous formula (grid/100) @ coef treated the
            # ORIGINAL components as pseudo-components — which is only correct
            # if the vertices sit at pure-component corners (100,0,0) etc.
            # For any interior feasible triangle it gives wrong values
            # everywhere, including at the vertices themselves (where the
            # prediction must equal the measured reading).
            grid_pts = feasible_grid(constraints, step=2.0)
            V = fit_result.get("apex_vertices")
            y = fit_result.get("apex_readings")
            if V is None or y is None:
                # Backward compatibility with older fit_result payloads.
                V = design_pts
                y = [coef_dict[i] for i in range(3)]
            preds = _apex_predict(grid_pts, V, y) if len(grid_pts) else np.array([])
        else:
            grid_pts = feasible_grid(constraints, step=2.0)
            preds = _scheffe_expand(grid_pts, deg) @ coef if len(grid_pts) else np.array([])

        if len(grid_pts) > 0:
            tri = ax.tripcolor(
                grid_pts[:, 0], grid_pts[:, 2], grid_pts[:, 1], preds,
                cmap="viridis", shading="gouraud",
                zorder=1,
            )
            # Iso-solubility contour lines on top of the viridis fill. Wrapped
            # in try/except because on very small feasible regions the
            # triangulation can degenerate and tricontour will raise.
            try:
                cs = ax.tricontour(
                    grid_pts[:, 0], grid_pts[:, 2], grid_pts[:, 1], preds,
                    levels=8, colors="white", linewidths=0.8, alpha=0.85,
                    zorder=2,
                )
                ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
            except (ValueError, RuntimeError):
                pass
            cbar = plt.colorbar(tri, ax=ax, fraction=0.046, pad=0.1)
            cbar.set_label("Predicted solubility (mg/g)", rotation=270, labelpad=20)

    # -- feasible-region boundary (on top of the surface) ------------------
    if show_boundary and len(verts) >= 3:
        loop = verts + [verts[0]]
        t_vals = [p[0] for p in loop]
        l_vals = [p[2] for p in loop]
        r_vals = [p[1] for p in loop]
        boundary_color = "white" if fit_result is not None else "red"
        ax.plot(t_vals, l_vals, r_vals, color=boundary_color,
                linewidth=2.5, label="Feasible region")

    # -- design points ------------------------------------------------------
    if show_training_points and design_pts:
        pts_array = np.array(design_pts, dtype=float)
        ax.scatter(
            pts_array[:, 0], pts_array[:, 2], pts_array[:, 1],
            s=100, facecolors="none", edgecolors="black", linewidths=2,
            label="Design points", zorder=5,
        )

    # -- apex-mode vertex annotations --------------------------------------
    if apex_mode and show_apex_labels and fit_result is not None and len(verts) == 3:
        coef_dict = fit_result["coef"]
        coef = [coef_dict[i] for i in sorted(coef_dict)]
        ordered = _order_vertices_by_dominant_component(verts)

        # Offset each label OUTWARD from the polygon centroid so it doesn't
        # sit on top of the vertex marker or the neighbouring labels. We
        # shift each vertex along the (vertex - centroid) direction in
        # ternary coords by a small amount — mpltern's transform is affine,
        # so the shifted coords don't need to sum to 100; they just place
        # the text a bit outside the triangle.
        centroid = np.mean(ordered, axis=0)
        # ~5 percentage points of ternary "reach" clears the marker and
        # the label bbox at this figure size; tune here if you change the
        # feasible-region size dramatically.
        SHIFT = 6.0

        text_props = dict(
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="ivory", ec="black",
                      lw=1, alpha=0.9),
            zorder=10,
        )
        for v, val in zip(ordered, coef):
            direction = np.array(v) - centroid
            norm = np.linalg.norm(direction) or 1.0
            shifted = np.array(v) + SHIFT * direction / norm
            # mpltern's ax.text takes (t, l, r) which in this file's
            # convention is (c1, c3, c2).
            ax.text(shifted[0], shifted[2], shifted[1],
                    f"{val:.2f}", **text_props)

    if show_legend:
        # Only render the legend if there's at least one labeled artist,
        # otherwise matplotlib emits a UserWarning and draws an empty box.
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    return fig


# ============================================================================
# PAGE LAYOUT
# ============================================================================
st.markdown("## △ SNEDDS Solubility Prediction App")
st.markdown("Design, sample, and model a three-component formulation space.")

# ---- Top-of-page mode toggle ---------------------------------------------
mode_col1, mode_col2 = st.columns([1, 3])
with mode_col1:
    st.session_state.apex_mode = st.toggle(
        "Apex prediction mode",
        value=st.session_state.apex_mode,
        help=(
            "Uses the 3 vertices of the feasible region as pseudo-components. "
            "Coefficients are read directly from the observations at those "
            "vertices. The linear Scheffe surface is shown across the feasible "
            "region only (no extrapolation beyond the design box). Requires "
            "the feasible region to be an equilateral triangle."
        ),
    )
with mode_col2:
    if st.session_state.apex_mode:
        st.caption(
            "Apex mode active: the design is the 3 vertices, the fit is the "
            "linear Scheffe read-off, and the plot covers the feasible region."
        )

# ---- Detect config changes that should invalidate stale state ------------
_cfg_signature = (
    tuple(st.session_state.comp_mins),
    tuple(st.session_state.comp_maxs),
    st.session_state.apex_mode,
)
if st.session_state.get("_last_cfg") != _cfg_signature:
    st.session_state._last_cfg = _cfg_signature
    if st.session_state.apex_mode:
        # Apex design is derived from the constraints; drop stale points
        # from an earlier session or the other mode.
        st.session_state.design_points = []
        st.session_state.solubilities = []
        st.session_state.fit_result = None

col_left, col_right = st.columns([1.6, 1], gap="large")

# ---------------------------------------------------------------------------
# LEFT COLUMN - Controls
# ---------------------------------------------------------------------------
with col_left:
    # ---- Step 1: Components & Constraints ----
    st.subheader("1. Components & constraints")
    st.caption(
        "The three components below fill 100% of the formulation. Any "
        "**co-solvent** (e.g. ethanol, PG) must be held at the **same "
        "proportion across every run** — it isn't part of the mixture design."
    )

    # Persistent reminder of the caps so users see the boundary conditions
    # BEFORE they type values, not only after they exceed them.
    st.info(
        f"🔒 **Model-supported range: "
        f"{MIN_SINGLE_COMPONENT:.0f}%–{MAX_SINGLE_COMPONENT:.0f}% per component.** "
        f"Bounds outside this range are flagged below; the mixture math still "
        f"runs, but predictions outside the calibrated range are extrapolations."
    )

    c1_col, c2_col, c3_col = st.columns(3)
    for idx, col in enumerate((c1_col, c2_col, c3_col)):
        with col:
            st.session_state.comp_names[idx] = st.text_input(
                f"Component {idx+1} name",
                value=st.session_state.comp_names[idx],
                key=f"name_{idx}",
            )
            st.session_state.comp_mins[idx] = st.number_input(
                f"C{idx+1} min (%)",
                value=float(st.session_state.comp_mins[idx]),
                min_value=0.0, max_value=100.0, step=1.0,
                key=f"min_{idx}",
                help=(
                    f"Model-supported floor is {MIN_SINGLE_COMPONENT:.0f}%. "
                    f"You can type a lower number, but you'll be warned."
                ),
            )
            st.session_state.comp_maxs[idx] = st.number_input(
                f"C{idx+1} max (%)",
                value=float(st.session_state.comp_maxs[idx]),
                min_value=0.0, max_value=100.0, step=1.0,
                key=f"max_{idx}",
                help=(
                    f"Model-supported ceiling is {MAX_SINGLE_COMPONENT:.0f}%. "
                    f"You can type a higher number, but you'll be warned."
                ),
            )

            # Per-component status caption. Show one of:
            #   ✅ Inside model range
            #   🔒 At the model floor / ceiling (edge of validated range)
            #   ⚠ Outside model range (extrapolation)
            # Also flag when min > max, since that's an easy typo.
            lo, hi = st.session_state.comp_mins[idx], st.session_state.comp_maxs[idx]
            notes = []
            if lo > hi:
                notes.append(f":red[⚠ min ({lo:.0f}%) > max ({hi:.0f}%)]")
            if lo < MIN_SINGLE_COMPONENT:
                notes.append(
                    f":red[⚠ min {lo:.0f}% is below the model floor "
                    f"({MIN_SINGLE_COMPONENT:.0f}%) — extrapolating]"
                )
            elif lo == MIN_SINGLE_COMPONENT:
                notes.append(
                    f":orange[🔒 min sits at the model floor "
                    f"({MIN_SINGLE_COMPONENT:.0f}%)]"
                )
            if hi > MAX_SINGLE_COMPONENT:
                notes.append(
                    f":red[⚠ max {hi:.0f}% is above the model ceiling "
                    f"({MAX_SINGLE_COMPONENT:.0f}%) — extrapolating]"
                )
            elif hi == MAX_SINGLE_COMPONENT:
                notes.append(
                    f":orange[🔒 max sits at the model ceiling "
                    f"({MAX_SINGLE_COMPONENT:.0f}%)]"
                )
            if not notes and lo <= hi:
                notes.append(":green[✅ inside model range]")
            for n in notes:
                st.caption(n)

    # ---- Model-range violation banner (only when actually violated) ------
    range_violations = []
    for i, nm in enumerate(st.session_state.comp_names):
        lo = st.session_state.comp_mins[i]
        hi = st.session_state.comp_maxs[i]
        if lo < MIN_SINGLE_COMPONENT:
            range_violations.append(
                f"**{nm}** min is {lo:.0f}% (model floor: {MIN_SINGLE_COMPONENT:.0f}%)"
            )
        if hi > MAX_SINGLE_COMPONENT:
            range_violations.append(
                f"**{nm}** max is {hi:.0f}% (model ceiling: {MAX_SINGLE_COMPONENT:.0f}%)"
            )
    if range_violations:
        st.warning(
            f"Model is validated for {MIN_SINGLE_COMPONENT:.0f}%–"
            f"{MAX_SINGLE_COMPONENT:.0f}% per component. Some bounds are "
            f"outside this range:\n\n" + "\n".join(f"- {v}" for v in range_violations)
        )

    dcol1, dcol2 = st.columns([2, 1])
    with dcol1:
        st.session_state.drug_name = st.text_input(
            "Plot title (e.g. drug name)",
            value=st.session_state.drug_name, key="drug_input",
        )
    with dcol2:
        st.session_state.title_color = st.color_picker(
            "Title color", value=st.session_state.title_color, key="color_input",
        )

    constraints = get_constraints()
    feas = constraints.feasibility_report()

    if feas.feasible:
        st.success("Feasible region OK")
    else:
        st.error("Infeasible - fix the following:")
        for v in feas.violations:
            st.write(f"• {v}")

    if feas.binding:
        st.info("Binding constraints: " + ", ".join(feas.binding))
    if feas.inactive:
        st.caption("Inactive bounds: " + "; ".join(feas.inactive))

    # ---- Effective (reachable) range per component -----------------------
    # This is the OTHER kind of cap: even if you set component A max to 60%,
    # if the other two components' minimums add up to 50%, A's real ceiling
    # is 50% — the extra 10% you set is dead. We surface this explicitly
    # because it's the single most common source of "why can't I pick that
    # composition?" confusion.
    reach_rows = []
    for i, key in enumerate(("c1", "c2", "c3")):
        set_lo = st.session_state.comp_mins[i]
        set_hi = st.session_state.comp_maxs[i]
        r_lo, r_hi = feas.reachable[key]
        # A bound is "capped by other components" when the reachable end is
        # tighter than the user's set end by more than 0.5 pp.
        floor_capped = r_lo > set_lo + 0.5
        ceil_capped = r_hi < set_hi - 0.5
        status = []
        if floor_capped:
            status.append(f"floor lifted to {r_lo:.0f}%")
        if ceil_capped:
            status.append(f"ceiling clipped to {r_hi:.0f}%")
        if not status:
            status.append("bounds are the real limits")
        reach_rows.append({
            "Component": st.session_state.comp_names[i],
            "You set": f"[{set_lo:.0f}, {set_hi:.0f}]%",
            "Actually reachable": f"[{r_lo:.0f}, {r_hi:.0f}]%",
            "Status": "; ".join(status),
        })
    any_capped = any("lifted" in r["Status"] or "clipped" in r["Status"]
                     for r in reach_rows)
    with st.expander(
        ("🔒 Effective ranges (some bounds capped by other components)"
         if any_capped else "Effective ranges"),
        expanded=any_capped,
    ):
        st.caption(
            "'Actually reachable' is what each component can really take once "
            "the other two components' bounds are respected. If it's tighter "
            "than what you set, the extra headroom is dead — no design point "
            "can use it."
        )
        st.table(pd.DataFrame(reach_rows))

    # ---- Apex-mode geometry check ----------------------------------------
    apex_ready = False
    if st.session_state.apex_mode and feas.feasible:
        sides = constraints.side_lengths()
        if len(sides) != 3:
            st.error(
                f"Apex mode requires exactly 3 vertices; the current region "
                f"has {len(sides)}. Tighten a bound so one constraint stops "
                f"being active."
            )
        elif not constraints.is_equilateral(rel_tol=0.02):
            mean_s = float(np.mean(sides))
            spread = (max(sides) - min(sides)) / mean_s * 100
            st.error(
                f"Apex mode requires an equilateral feasible triangle. Sides "
                f"are {sides[0]:.2f}, {sides[1]:.2f}, {sides[2]:.2f} "
                f"(spread {spread:.1f}% of mean; tolerance is 2%). "
                f"Adjust bounds so the three sides match."
            )
        else:
            apex_ready = True
            st.success(
                f"Equilateral feasible triangle detected "
                f"(side ≈ {float(np.mean(sides)):.2f}). Apex fit available."
            )

    st.divider()

    # ---- Step 2: Design ----
    st.subheader("2. Experimental design")

    if st.session_state.apex_mode:
        # =============== APEX MODE ===============
        st.caption(
            "In apex mode the design is fixed to the 3 vertices of the "
            "feasible triangle. Enter one measured solubility per vertex; "
            "the linear Scheffe coefficients ARE those readings."
        )

        if apex_ready:
            ordered_verts = _order_vertices_by_dominant_component(
                constraints.vertices()
            )
            # (Re)seed design points if they don't match the current geometry.
            needs_reseed = (
                len(st.session_state.design_points) != 3
                or [tuple(np.round(p, 2)) for p in st.session_state.design_points]
                    != [tuple(np.round(v, 2)) for v in ordered_verts]
            )
            if needs_reseed:
                st.session_state.design_points = [
                    tuple(float(x) for x in v) for v in ordered_verts
                ]
                if len(st.session_state.solubilities) != 3:
                    st.session_state.solubilities = [0.0, 0.0, 0.0]
                st.session_state.fit_result = None

            cnames = st.session_state.comp_names
            vertex_rows = [
                {
                    "Vertex": f"V{i+1} (high {cnames[i]})",
                    cnames[0]: round(v[0], 2),
                    cnames[1]: round(v[1], 2),
                    cnames[2]: round(v[2], 2),
                    "Solubility (mg/g)": s,
                }
                for i, (v, s) in enumerate(zip(
                    st.session_state.design_points, st.session_state.solubilities
                ))
            ]
            edit_df = pd.DataFrame(vertex_rows)
            edited = st.data_editor(
                edit_df,
                use_container_width=True,
                num_rows="fixed",
                key="apex_editor",
                disabled=["Vertex", cnames[0], cnames[1], cnames[2]],
                column_config={
                    "Solubility (mg/g)": st.column_config.NumberColumn(step=0.1),
                },
            )
            st.session_state.solubilities = [
                float(x) for x in edited["Solubility (mg/g)"].tolist()
            ]
        else:
            st.info("Waiting for a valid equilateral triangle above.")

    else:
        # =============== NON-APEX MODES ===============
        st.session_state.manual_mode = st.toggle(
            "Choose points manually",
            value=st.session_state.manual_mode,
            help=(
                "Default is D-optimal auto-generation. Turn on to enter "
                "compositions yourself."
            ),
        )
        degree = st.selectbox(
            "Model degree",
            ["auto", "linear", "quadratic", "special_cubic"],
            key="degree_sel",
        )

        if not st.session_state.manual_mode:
            # ---- AUTO (D-OPTIMAL) ----
            col_d1, col_d2 = st.columns([1, 1.2], gap="small")
            with col_d1:
                st.session_state.budget = st.selectbox(
                    "Budget (points)", [4, 5, 6, 7, 8], index=2, key="budget_sel",
                )
            with col_d2:
                st.write("")
                if st.button("Suggest points", key="design_btn", use_container_width=True):
                    if not feas.feasible:
                        st.error("Fix constraint violations first.")
                    else:
                        deg = recommend_degree(st.session_state.budget) if degree == "auto" else degree
                        try:
                            res = d_optimal_design(constraints, st.session_state.budget, deg)
                            st.session_state.design_points = [tuple(map(float, p)) for p in res["points"]]
                            st.session_state.solubilities = [0.0] * len(res["points"])
                            st.session_state.fit_result = None
                            st.success(
                                f"Placed {len(res['points'])} points for a '{deg}' model "
                                f"({res['residual_df']} residual df)."
                            )
                        except ValueError as e:
                            st.error(f"Design failed: {e}")

            # ---- Editor for entering measured solubilities against the
            #      auto-generated compositions. This block was missing
            #      previously — auto mode placed the points but gave the user
            #      no way to type in the response values, so Step 3 always
            #      failed with "solubilities are all zero". Compositions are
            #      locked (they came from the D-optimal search); only the
            #      response column is editable.
            if st.session_state.design_points:
                st.caption(
                    "Enter the measured solubility for each suggested run. "
                    "Compositions are locked to the D-optimal design; only "
                    "the response column is editable."
                )
                cnames = st.session_state.comp_names
                auto_rows = [
                    {
                        "Run": f"R{i+1}",
                        cnames[0]: round(p[0], 2),
                        cnames[1]: round(p[1], 2),
                        cnames[2]: round(p[2], 2),
                        "Solubility (mg/g)": s,
                    }
                    for i, (p, s) in enumerate(zip(
                        st.session_state.design_points,
                        st.session_state.solubilities,
                    ))
                ]
                edited_auto = st.data_editor(
                    pd.DataFrame(auto_rows),
                    use_container_width=True,
                    num_rows="fixed",
                    key="auto_editor",
                    disabled=["Run", cnames[0], cnames[1], cnames[2]],
                    column_config={
                        "Solubility (mg/g)": st.column_config.NumberColumn(step=0.1),
                    },
                )
                new_sols = [float(x) for x in edited_auto["Solubility (mg/g)"].tolist()]
                # Only invalidate the fit if the user actually changed a value;
                # otherwise every rerun would wipe an already-fitted model.
                if new_sols != st.session_state.solubilities:
                    st.session_state.solubilities = new_sols
                    st.session_state.fit_result = None
        else:
            # ---- MANUAL ----
            st.caption(
                "Enter each composition as (C1, C2, C3). Rows are validated "
                "against the constraints; only feasible rows contribute to the fit."
            )

            n_manual = st.number_input(
                "Number of runs", min_value=3, max_value=20,
                value=max(3, len(st.session_state.design_points) or 4),
                step=1, key="manual_n",
            )

            current = st.session_state.design_points
            if len(current) != n_manual:
                new_rows = list(current)
                while len(new_rows) < n_manual:
                    mids = [
                        (st.session_state.comp_mins[i] + st.session_state.comp_maxs[i]) / 2
                        for i in range(3)
                    ]
                    total = sum(mids) or 1.0
                    mids = [m * 100.0 / total for m in mids]
                    new_rows.append(tuple(mids))
                new_rows = new_rows[:n_manual]
                st.session_state.design_points = [tuple(map(float, p)) for p in new_rows]

                sols = list(st.session_state.solubilities)
                while len(sols) < n_manual:
                    sols.append(0.0)
                st.session_state.solubilities = sols[:n_manual]

            cnames = st.session_state.comp_names
            edit_df = pd.DataFrame(
                [
                    {
                        cnames[0]: p[0],
                        cnames[1]: p[1],
                        cnames[2]: p[2],
                        "Solubility (mg/g)": s,
                    }
                    for p, s in zip(st.session_state.design_points, st.session_state.solubilities)
                ]
            )
            edited = st.data_editor(
                edit_df,
                use_container_width=True,
                num_rows="fixed",
                key="manual_editor",
                column_config={
                    cnames[0]: st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1),
                    cnames[1]: st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1),
                    cnames[2]: st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1),
                    "Solubility (mg/g)": st.column_config.NumberColumn(step=0.1),
                },
            )
            st.session_state.design_points = [
                (float(r[cnames[0]]), float(r[cnames[1]]), float(r[cnames[2]]))
                for _, r in edited.iterrows()
            ]
            st.session_state.solubilities = [float(x) for x in edited["Solubility (mg/g)"].tolist()]

            flags = []
            for p in st.session_state.design_points:
                s = sum(p)
                if abs(s - 100.0) > 0.5:
                    flags.append(f"Σ = {s:.1f}% (should be 100)")
                elif not constraints.contains(p, tol=0.5):
                    flags.append("outside constraints")
                else:
                    flags.append("OK")
            n_ok = sum(f == "OK" for f in flags)
            if n_ok == len(flags):
                st.success(f"All {n_ok} rows feasible.")
            else:
                bad = [f"row {i+1}: {f}" for i, f in enumerate(flags) if f != "OK"]
                st.warning("Some rows need attention → " + "; ".join(bad))

            st.session_state.fit_result = None  # invalidate old fit when editing

    st.divider()

    # ---- Step 3: Fit ----
    st.subheader("3. Fit the model")

    if st.session_state.apex_mode:
        if apex_ready and st.button("Fit apex model", key="fit_apex_btn"):
            y = np.array(st.session_state.solubilities, dtype=float)
            if np.all(y == 0):
                st.warning("All solubilities are 0 - enter your measurements first.")
            else:
                # Apex mode is a barycentric interpolation over the feasible
                # triangle in pseudo-component space: the coefficients (β1,β2,β3)
                # are the readings at V1,V2,V3, and predictions at any interior
                # point P come from the barycentric weights of P w.r.t. the
                # vertices. At each vertex the weights collapse to a basis
                # vector, so the fitted value equals the measured value exactly.
                # We stash the vertex compositions on the fit_result so the
                # plot function can evaluate the surface without re-deriving
                # them from state.
                V = np.array(st.session_state.design_points, dtype=float)
                fitted = _apex_predict(V, V, y).tolist()  # equal to y by construction
                st.session_state.fit_result = {
                    "coef": {i: float(y[i]) for i in range(3)},
                    "degree": "apex",
                    "apex_vertices": [tuple(float(x) for x in v) for v in V],
                    "apex_readings": [float(v) for v in y],
                    "summary": {
                        "degree": "apex",
                        "n_points": 3,
                        "n_terms": 3,
                        "residual_df": 0,
                        "coefficients": {i: float(y[i]) for i in range(3)},
                        "r2": float("nan"),
                        "adj_r2": float("nan"),
                        "loo_rmse": float("nan"),
                        "observed": [float(v) for v in y],
                        "fitted_values": fitted,
                        "loo_predictions": None,
                        "trustworthy": False,
                        "vertex_values": [float(v) for v in y],
                    },
                }
                st.success("Apex model built. Coefficients = vertex readings.")
    elif st.session_state.design_points:
        if st.button("Fit model", key="fit_btn"):
            X_all = np.array(st.session_state.design_points, dtype=float)
            y_all = np.array(st.session_state.solubilities, dtype=float)

            if st.session_state.manual_mode:
                mask = np.array([
                    abs(sum(p) - 100.0) <= 0.5 and constraints.contains(p, tol=0.5)
                    for p in st.session_state.design_points
                ])
                X, y = X_all[mask], y_all[mask]
            else:
                X, y = X_all, y_all

            if len(X) == 0:
                st.error("No feasible rows to fit.")
            elif np.all(y == 0):
                st.warning("All solubilities are 0 - enter your measurements first.")
            else:
                degree = st.session_state.get("degree_sel", "auto")
                deg = recommend_degree(len(X)) if degree == "auto" else degree
                try:
                    model = ScheffeModel(deg).fit(X, y)
                    summary = model.summary()
                    st.session_state.fit_result = {
                        "coef": summary["coefficients"],
                        "degree": deg,
                        "summary": summary,
                    }
                    st.success(f"Fitted '{deg}' model on {len(X)} runs.")
                except ValueError as e:
                    st.error(f"Fit failed: {e}")
    else:
        st.info("Add design points above first.")

    st.divider()

    # ---- Step 4: Diagnostics ----
    st.subheader("4. Fit diagnostics")

    if st.session_state.fit_result is not None:
        s = st.session_state.fit_result["summary"]

        if s["degree"] == "apex":
            st.warning("Apex read-off (not a statistical fit)")
            st.info(
                "The linear Scheffe surface is defined by the three vertex "
                "readings, so there are 0 residual degrees of freedom and no "
                "error estimate. Interpret the coefficients as 'what each "
                "vertex measured' and the surface as a linear interpolation of "
                "those three values (extrapolated beyond the feasible triangle "
                "for context). Confidence in the surface = confidence in the "
                "three individual measurements."
            )
            cnames = list(st.session_state.comp_names)
            coef = st.session_state.fit_result["coef"]
            rows = [
                {
                    "Vertex": f"V{i+1} (high {cnames[i]})",
                    "Reading (mg/g)": f"{coef[i]:.2f}",
                }
                for i in range(3)
            ]
            st.table(pd.DataFrame(rows))

            if s.get("observed") and s.get("fitted_values"):
                st.pyplot(
                    plot_parity(
                        s["observed"], s["fitted_values"],
                        loo=s.get("loo_predictions"),
                        color=st.session_state.title_color,
                        drug=st.session_state.drug_name,
                    ),
                    use_container_width=True,
                )
        else:
            df = s["residual_df"]

            if df < 2 or not np.isfinite(s["loo_rmse"]):
                verdict, level = "Under-determined", "warning"
                note = (
                    f"Only {df} residual degrees of freedom - not enough to "
                    "check the fit. R² will look near-perfect no matter what. "
                    "Add points or drop to a simpler model."
                )
            elif s["adj_r2"] <= 0 or not np.isfinite(s["adj_r2"]):
                verdict, level = "Poor fit", "error"
                note = (
                    f"Adjusted R² is {s['adj_r2']:.3f} - explains no more than "
                    "the mean. Try a different degree or widen the design region."
                )
            else:
                verdict, level = "Trustworthy", "success"
                note = (
                    f"{df} residual df and positive adjusted R². "
                    f"Trust LOO-RMSE ({s['loo_rmse']:.2f}) as the real accuracy "
                    "indicator, not R²."
                )

            {"success": st.success, "warning": st.warning, "error": st.error}[level](verdict)

            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{s['r2']:.3f}")
            m2.metric("Adj R²", f"{s['adj_r2']:.3f}")
            m3.metric("LOO-RMSE",
                      f"{s['loo_rmse']:.2f}" if np.isfinite(s['loo_rmse']) else "n/a")

            st.info(note)

            cnames = list(st.session_state.comp_names)
            coef_labels = list(cnames)
            if s["degree"] in ("quadratic", "special_cubic"):
                coef_labels += [
                    f"{cnames[0]}·{cnames[1]}",
                    f"{cnames[0]}·{cnames[2]}",
                    f"{cnames[1]}·{cnames[2]}",
                ]
            if s["degree"] == "special_cubic":
                coef_labels += [f"{cnames[0]}·{cnames[1]}·{cnames[2]}"]

            coef_vals = [s["coefficients"][i] for i in sorted(s["coefficients"])]
            coef_str = " | ".join(f"{n} {v:+.2f}" for n, v in zip(coef_labels, coef_vals))
            st.code(f"Scheffe ({s['degree']}): {coef_str}", language="text")

            if s.get("observed") and s.get("fitted_values"):
                st.pyplot(
                    plot_parity(
                        s["observed"], s["fitted_values"],
                        loo=s.get("loo_predictions"),
                        color=st.session_state.title_color,
                        drug=st.session_state.drug_name,
                    ),
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# RIGHT COLUMN - Plot
# ---------------------------------------------------------------------------
with col_right:
    st.subheader("Ternary plot")

    # ---- Plot element toggles ----
    # Grouped in an expander so they don't consume vertical space by default,
    # but discoverable via the "Plot controls" summary. Every toggle here is
    # cosmetic — none of them touch design_points, solubilities, or fits.
    with st.expander("Plot controls", expanded=False):
        pc1, pc2 = st.columns(2)
        with pc1:
            st.session_state.show_training_points = st.checkbox(
                "Show training / design points",
                value=st.session_state.show_training_points,
                help="Hide the black-outlined circles marking each experiment.",
            )
            st.session_state.show_boundary = st.checkbox(
                "Show design-space boundary line",
                value=st.session_state.show_boundary,
                help="Hide the polygon outlining the feasible region.",
            )
        with pc2:
            st.session_state.show_legend = st.checkbox(
                "Show legend",
                value=st.session_state.show_legend,
            )
            st.session_state.show_apex_labels = st.checkbox(
                "Show apex value labels",
                value=st.session_state.show_apex_labels,
                help=(
                    "Apex mode only — hide the ivory boxes showing each "
                    "vertex's measured solubility."
                ),
            )

    fig = plot_ternary(
        constraints,
        design_pts=st.session_state.design_points or None,
        fit_result=st.session_state.fit_result,
        drug=st.session_state.drug_name,
        title_color=st.session_state.title_color,
        apex_mode=st.session_state.apex_mode,
        show_training_points=st.session_state.show_training_points,
        show_legend=st.session_state.show_legend,
        show_apex_labels=st.session_state.show_apex_labels,
        show_boundary=st.session_state.show_boundary,
    )
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Plot will appear here once mpltern is installed / design points are placed.")

# ---------------------------------------------------------------------------
# STEP 5 — VALIDATION POINTS (both apex and Scheffé fits)
# ---------------------------------------------------------------------------
# External validation is the honest generalisation check the small-n LOO
# metric is trying to approximate. The user supplies a handful of
# compositions they've already measured but that were NOT part of the fit;
# we predict at those compositions with the current model, overlay the
# predicted-vs-observed pairs on the parity plot as red triangles, and
# report validation MAE / RMSE / R². Nothing here touches the training
# data, the ternary plot, or the fit itself.
st.divider()
st.subheader("5. Validation points (optional)")

if st.session_state.fit_result is None:
    st.info(
        "Fit the model in Step 3 first. Once a fit exists, you can enter "
        "external validation compositions here and see how the model "
        "performs on held-out data."
    )
else:
    st.caption(
        "Enter compositions you've already measured that were **not** part "
        "of the fit. Each row must sum to 100% and fall inside the feasible "
        "region to score against the model. Predictions come from the "
        "current fit; the ternary plot is unaffected."
    )

    n_val = st.number_input(
        "Number of validation points",
        min_value=0, max_value=30,
        value=len(st.session_state.validation_points),
        step=1, key="n_val_input",
    )

    # Grow / shrink the validation buffers to match n_val, seeding new rows
    # with a feasible interior guess (midpoint of set bounds, renormalised).
    cur_pts = list(st.session_state.validation_points)
    cur_meas = list(st.session_state.validation_measurements)
    while len(cur_pts) < n_val:
        mids = [
            (st.session_state.comp_mins[i] + st.session_state.comp_maxs[i]) / 2.0
            for i in range(3)
        ]
        total = sum(mids) or 1.0
        cur_pts.append(tuple(m * 100.0 / total for m in mids))
        cur_meas.append(0.0)
    cur_pts = cur_pts[:n_val]
    cur_meas = cur_meas[:n_val]
    st.session_state.validation_points = [tuple(map(float, p)) for p in cur_pts]
    st.session_state.validation_measurements = [float(x) for x in cur_meas]

    if n_val > 0:
        cnames = st.session_state.comp_names
        val_df = pd.DataFrame(
            [
                {
                    "Run": f"V{i+1}",
                    cnames[0]: p[0],
                    cnames[1]: p[1],
                    cnames[2]: p[2],
                    "Measured (mg/g)": m,
                }
                for i, (p, m) in enumerate(zip(
                    st.session_state.validation_points,
                    st.session_state.validation_measurements,
                ))
            ]
        )
        val_edited = st.data_editor(
            val_df,
            use_container_width=True,
            num_rows="fixed",
            key="validation_editor",
            disabled=["Run"],
            column_config={
                cnames[0]: st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.1),
                cnames[1]: st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.1),
                cnames[2]: st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.1),
                "Measured (mg/g)": st.column_config.NumberColumn(step=0.1),
            },
        )
        st.session_state.validation_points = [
            (float(r[cnames[0]]), float(r[cnames[1]]), float(r[cnames[2]]))
            for _, r in val_edited.iterrows()
        ]
        st.session_state.validation_measurements = [
            float(x) for x in val_edited["Measured (mg/g)"].tolist()
        ]

        # Row-level feasibility flags (same rules as the manual design editor).
        val_flags = []
        for p in st.session_state.validation_points:
            s = sum(p)
            if abs(s - 100.0) > 0.5:
                val_flags.append(f"Σ = {s:.1f}%")
            elif not constraints.contains(p, tol=0.5):
                val_flags.append("outside constraints")
            else:
                val_flags.append("OK")
        n_val_ok = sum(f == "OK" for f in val_flags)
        if n_val_ok == n_val:
            st.success(f"All {n_val} validation rows feasible.")
        else:
            bad = [f"row {i+1}: {f}" for i, f in enumerate(val_flags) if f != "OK"]
            st.warning("Some validation rows will be skipped → " + "; ".join(bad))

        if st.button("Evaluate validation set", key="validate_btn"):
            X_val = np.array(st.session_state.validation_points, dtype=float)
            y_val = np.array(st.session_state.validation_measurements, dtype=float)
            mask = np.array([
                abs(sum(p) - 100.0) <= 0.5 and constraints.contains(p, tol=0.5)
                for p in st.session_state.validation_points
            ])
            if not mask.any():
                st.error("No feasible validation rows to evaluate.")
            elif np.all(y_val[mask] == 0):
                st.warning(
                    "All validation measurements are 0 — enter your "
                    "measured solubilities first."
                )
            else:
                X_use, y_use = X_val[mask], y_val[mask]
                y_pred = predict_from_fit(X_use, st.session_state.fit_result)
                # Overlay validation on top of the training parity plot so
                # the two sets are directly comparable on the same 1:1 line.
                summary = st.session_state.fit_result["summary"]
                st.pyplot(
                    plot_parity(
                        summary["observed"],
                        summary["fitted_values"],
                        loo=summary.get("loo_predictions"),
                        val_observed=y_use.tolist(),
                        val_predicted=y_pred.tolist(),
                        color=st.session_state.title_color,
                        drug=st.session_state.drug_name,
                    ),
                    use_container_width=True,
                )
                # Numeric summary alongside the plot for quick reference.
                resid = y_use - y_pred
                v_mae = float(np.mean(np.abs(resid)))
                v_rmse = float(np.sqrt(np.mean(resid ** 2)))
                v_ss_tot = float(np.sum((y_use - y_use.mean()) ** 2))
                v_r2 = 1.0 - float(np.sum(resid ** 2)) / v_ss_tot if v_ss_tot > 0 else float("nan")
                vm1, vm2, vm3, vm4 = st.columns(4)
                vm1.metric("Val n", f"{len(y_use)}")
                vm2.metric("Val MAE", f"{v_mae:.2f}")
                vm3.metric("Val RMSE", f"{v_rmse:.2f}")
                vm4.metric(
                    "Val R²",
                    f"{v_r2:.3f}" if np.isfinite(v_r2) else "n/a",
                )
    else:
        st.caption("Set the number of validation points above to begin.")
