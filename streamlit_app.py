"""
Streamlit app for Mixture Studio - constrained-mixture DOE and Scheffe modeling.

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
    "comp_names": ["Captex 300", "Kolliphor RH40", "Capmul MCM C8"],
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
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# HELPERS
# ============================================================================
# Hard formulation limit: no single component may exceed this fraction.
# Surfaced in three places so the constraint is impossible to miss:
#   1. The number_input max_value below, so the user physically can't type past it.
#   2. Passed to MixtureConstraints as hard_max so the feasibility report flags it.
#   3. Drawn as a dashed guide line on the ternary plot.
MAX_SINGLE_COMPONENT = 80.0


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
    ax.taxis.set_ticks_position("tick2")
    ax.laxis.set_ticks_position("tick2")
    ax.raxis.set_ticks_position("tick2")
    ax.taxis.set_label_position("tick2")
    ax.laxis.set_label_position("tick2")
    ax.raxis.set_label_position("tick2")

    for axis in (ax.taxis, ax.laxis, ax.raxis):
        axis.set_major_locator(MultipleLocator(10.0))
        axis.set_minor_locator(MultipleLocator(5.0))

    ax.grid(axis="t", linestyle="--", alpha=0.6, color="gray")
    ax.grid(axis="l", linestyle="--", alpha=0.6, color="gray")
    ax.grid(axis="r", linestyle="--", alpha=0.6, color="gray")

    ax.set_tlabel(f"% w/w {names['c1']}")
    ax.set_llabel(f"% w/w {names['c3']}")
    ax.set_rlabel(f"% w/w {names['c2']}")

    ax.set_title(drug, pad=20, color=title_color, fontweight="bold")

    for side in ("tside", "lside", "rside"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(2)


def _full_simplex_grid(step: float = 2.0) -> np.ndarray:
    """
    (N,3) array covering the entire ternary at the given step.
    Used for the apex-mode prediction surface, which extrapolates the
    linear model beyond the feasible region on purpose.
    """
    pts = []
    n = int(round(100.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((i * step, j * step, 100.0 - i * step - j * step))
    return np.array(pts, dtype=float)


def plot_parity(observed, fitted, loo=None, color="#1f77b4", drug=""):
    """
    Observed-vs-predicted parity plot.
    Open squares: in-sample fitted values (training-time predictions).
    Filled circles: leave-one-out predictions, when available.

    A metrics box (MAE, RMSE, R² on the training fit, plus LOO-RMSE when we
    have it) sits in the upper-left.
    """
    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    loo_arr = np.asarray(loo, dtype=float) if loo is not None else None

    fig, ax = plt.subplots(figsize=(6, 6))

    # square axes around all data
    stack = [observed, fitted] + ([loo_arr] if loo_arr is not None else [])
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
        verticalalignment="top", bbox=props, zorder=6,
    )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed (mg/mL)")
    ax.set_ylabel("Predicted (mg/mL)")
    ax.set_title(
        f"{drug} — parity" if drug else "Parity",
        color=color, fontweight="bold", pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.5)

    plt.tight_layout()
    return fig


def plot_ternary(constraints, design_pts=None, fit_result=None,
                 drug="", title_color="#1f77b4", apex_mode=False):
    """
    Standard mode: feasible outline + heatmap restricted to the feasible region.
    Apex mode:     heatmap across the FULL simplex from the three vertex
                   readings + feasible-triangle outline + vertex value labels.
    """
    try:
        import mpltern  # noqa: F401  (registers the 'ternary' projection)
    except ImportError:
        st.error("mpltern not installed. Add `mpltern` to requirements.txt.")
        return None

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="ternary")
    _style_ternary_axes(ax, constraints.names, drug or "Mixture Studio", title_color)

    verts = constraints.vertices()

    # -- hard-cap guide lines (80% max per component) ----------------------
    # Each line is where one component equals the cap; the other two split
    # the remaining 20% between them. Drawn thin and dashed so they read as
    # a *limit* rather than as data, and always visible so the constraint is
    # obvious even before the user tightens their bounds against it.
    cap = getattr(constraints, "hard_max", None)
    if cap is not None and 0 < cap < constraints.total:
        rem = constraints.total - cap
        cap_lines = {
            "c1": [(cap, rem, 0.0), (cap, 0.0, rem)],
            "c2": [(rem, cap, 0.0), (0.0, cap, rem)],
            "c3": [(rem, 0.0, cap), (0.0, rem, cap)],
        }
        cap_label_used = False
        for k, (a, b) in cap_lines.items():
            ax.plot(
                [a[0], b[0]], [a[2], b[2]], [a[1], b[1]],
                linestyle="--", color="#c0392b", linewidth=1.2, alpha=0.7,
                zorder=2,
                label=None if cap_label_used else f"{cap:.0f}% cap",
            )
            cap_label_used = True

    # -- prediction surface -------------------------------------------------
    if fit_result is not None:
        coef_dict = fit_result["coef"]
        coef = np.array([coef_dict[i] for i in sorted(coef_dict)])
        deg = fit_result["degree"]

        if deg == "apex":
            grid_pts = _full_simplex_grid(step=2.0)
            preds = (grid_pts / 100.0) @ coef
        else:
            grid_pts = feasible_grid(constraints, step=2.0)
            preds = _scheffe_expand(grid_pts, deg) @ coef if len(grid_pts) else np.array([])

        if len(grid_pts) > 0:
            tri = ax.tripcolor(
                grid_pts[:, 0], grid_pts[:, 2], grid_pts[:, 1], preds,
                cmap="viridis", shading="gouraud",
            )
            cbar = plt.colorbar(tri, ax=ax, fraction=0.046, pad=0.1)
            cbar.set_label("Predicted solubility (mg/mL)", rotation=270, labelpad=20)

    # -- feasible-region boundary (on top of the surface) ------------------
    if len(verts) >= 3:
        loop = verts + [verts[0]]
        t_vals = [p[0] for p in loop]
        l_vals = [p[2] for p in loop]
        r_vals = [p[1] for p in loop]
        boundary_color = "white" if fit_result is not None else "red"
        ax.plot(t_vals, l_vals, r_vals, color=boundary_color,
                linewidth=2.5, label="Feasible region")

    # -- design points ------------------------------------------------------
    if design_pts:
        pts_array = np.array(design_pts, dtype=float)
        ax.scatter(
            pts_array[:, 0], pts_array[:, 2], pts_array[:, 1],
            s=100, facecolors="none", edgecolors="black", linewidths=2,
            label="Design points", zorder=5,
        )

    # -- apex-mode vertex annotations --------------------------------------
    if apex_mode and fit_result is not None and len(verts) == 3:
        coef_dict = fit_result["coef"]
        coef = [coef_dict[i] for i in sorted(coef_dict)]
        ordered = _order_vertices_by_dominant_component(verts)
        text_props = dict(
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="ivory", ec="black",
                      lw=1, alpha=0.85),
        )
        for v, val in zip(ordered, coef):
            ax.text(v[0], v[2], v[1], f"{val:.2f}", **text_props)

    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    return fig


# ============================================================================
# PAGE LAYOUT
# ============================================================================
st.markdown("## △ Mixture Studio")
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
            "vertices, and the linear Scheffe surface is extrapolated across "
            "the full simplex. Requires the feasible region to be an "
            "equilateral triangle."
        ),
    )
with mode_col2:
    if st.session_state.apex_mode:
        st.caption(
            "Apex mode active: the design is the 3 vertices, the fit is the "
            "linear Scheffe read-off, and the plot covers the full simplex."
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
    st.warning(
        f"**Model limit: no single component may exceed "
        f"{MAX_SINGLE_COMPONENT:.0f}%.** Max inputs are capped at this value."
    )

    # Clamp any stored max above the hard cap (e.g. from an old session or
    # an imported CSV) so the sliders/inputs never present an illegal state.
    for i in range(3):
        if st.session_state.comp_maxs[i] > MAX_SINGLE_COMPONENT:
            st.session_state.comp_maxs[i] = MAX_SINGLE_COMPONENT

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
                min_value=0.0, max_value=MAX_SINGLE_COMPONENT, step=1.0,
                key=f"min_{idx}",
            )
            st.session_state.comp_maxs[idx] = st.number_input(
                f"C{idx+1} max (%)",
                value=float(st.session_state.comp_maxs[idx]),
                min_value=0.0, max_value=MAX_SINGLE_COMPONENT, step=1.0,
                key=f"max_{idx}",
                help=f"Hard cap: {MAX_SINGLE_COMPONENT:.0f}%",
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
                    "Solubility (mg/mL)": s,
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
                    "Solubility (mg/mL)": st.column_config.NumberColumn(step=0.1),
                },
            )
            st.session_state.solubilities = [
                float(x) for x in edited["Solubility (mg/mL)"].tolist()
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
                        "Solubility (mg/mL)": s,
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
                    "Solubility (mg/mL)": st.column_config.NumberColumn(step=0.1),
                },
            )
            st.session_state.design_points = [
                (float(r[cnames[0]]), float(r[cnames[1]]), float(r[cnames[2]]))
                for _, r in edited.iterrows()
            ]
            st.session_state.solubilities = [float(x) for x in edited["Solubility (mg/mL)"].tolist()]

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
                # In apex mode the linear-Scheffe coefficients ARE the vertex
                # readings, matching the pseudo-component interpretation used
                # in the original notebook code. The fitted values are the
                # linear-Scheffe predictions at each vertex in ORIGINAL
                # coordinates; they only equal the readings when a vertex sits
                # at a pure component, so the parity plot will show the mild
                # mismatch caused by extrapolating from pseudo-components.
                V = np.array(st.session_state.design_points, dtype=float)
                betas = y  # coefficients = vertex readings
                fitted = ((V / 100.0) @ betas).tolist()
                st.session_state.fit_result = {
                    "coef": {i: float(y[i]) for i in range(3)},
                    "degree": "apex",
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
                    "Reading (mg/mL)": f"{coef[i]:.2f}",
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
    fig = plot_ternary(
        constraints,
        design_pts=st.session_state.design_points or None,
        fit_result=st.session_state.fit_result,
        drug=st.session_state.drug_name,
        title_color=st.session_state.title_color,
        apex_mode=st.session_state.apex_mode,
    )
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Plot will appear here once mpltern is installed / design points are placed.")

# ---------------------------------------------------------------------------
# EXPORT / IMPORT
# ---------------------------------------------------------------------------
st.divider()
col_exp, col_imp = st.columns(2)

with col_exp:
    st.subheader("Export data")
    if st.session_state.design_points and st.session_state.solubilities:
        export_df = pd.DataFrame(
            {
                st.session_state.comp_names[0]: [p[0] for p in st.session_state.design_points],
                st.session_state.comp_names[1]: [p[1] for p in st.session_state.design_points],
                st.session_state.comp_names[2]: [p[2] for p in st.session_state.design_points],
                "Solubility (mg/mL)": st.session_state.solubilities,
            }
        )
        st.download_button(
            "Download as CSV",
            data=export_df.to_csv(index=False),
            file_name="mixture_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No data to export yet.")

with col_imp:
    st.subheader("Import data")
    uploaded = st.file_uploader(
        "Upload CSV: first 3 columns are components, last column is solubility."
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            comp_cols = df.columns[:3].tolist()
            sol_col = df.columns[-1]
            st.session_state.comp_names = comp_cols
            st.session_state.design_points = [tuple(map(float, p)) for p in df[comp_cols].values]
            st.session_state.solubilities = [float(x) for x in df[sol_col].tolist()]
            st.session_state.fit_result = None
            st.success("Data imported. Adjust names and constraints above if needed.")
        except Exception as e:
            st.error(f"Import failed: {e}")
