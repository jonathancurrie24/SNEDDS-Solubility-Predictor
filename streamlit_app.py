"""
Streamlit app for Mixture Studio — constrained-mixture DOE and Scheffé modeling.

Install: pip install streamlit numpy matplotlib mpltern pandas
Run:     streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    page_title="SNEDDS Solubility Predictor App",
    page_icon="△",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_DEFAULTS = {
    "comp_names": ["Surfactant", "Oil", "Co-Surfactant"],
    "comp_mins":  [30.0, 10.0, 20.0],
    "comp_maxs":  [60.0, 40.0, 50.0],
    "budget": 6,
    "design_points": [],
    "solubilities": [],
    "fit_result": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# HELPERS
# ============================================================================
def get_constraints() -> MixtureConstraints:
    """Build MixtureConstraints from session state."""
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
    )


def plot_ternary(constraints, design_pts=None, fit_result=None):
    """Ternary plot: feasible region, optional heatmap surface, design points."""
    try:
        import mpltern  # noqa: F401  (registers the 'ternary' projection)
    except ImportError:
        st.error("mpltern not installed. Add `mpltern` to requirements.txt.")
        return None

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="ternary")

    names = constraints.names
    ax.set_tlabel(f"{names['c1']} (%)", fontsize=11)
    ax.set_llabel(f"{names['c3']} (%)", fontsize=11)
    ax.set_rlabel(f"{names['c2']} (%)", fontsize=11)

    for axis in [ax.taxis, ax.laxis, ax.raxis]:
        axis.set_major_locator(mticker.MultipleLocator(10.0))
        axis.set_minor_locator(mticker.MultipleLocator(5.0))
    ax.grid(True, alpha=0.3)

    # Feasible-region boundary  (vertices() is a METHOD, not a free function)
    verts = constraints.vertices()
    if len(verts) >= 3:
        loop = verts + [verts[0]]
        t_vals = [p[0] for p in loop]
        l_vals = [p[2] for p in loop]
        r_vals = [p[1] for p in loop]
        ax.plot(t_vals, l_vals, r_vals, "r-", linewidth=2.5, label="Feasible region")

    # Optional heatmap surface
    if fit_result is not None:
        coef_dict = fit_result["coef"]
        # coefficients dict is keyed by integer index; put them back in order
        coef = np.array([coef_dict[i] for i in sorted(coef_dict)])
        deg = fit_result["degree"]
        grid_pts = feasible_grid(constraints, step=2.0)
        if len(grid_pts) > 0:
            M = _scheffe_expand(grid_pts, deg)
            preds = M @ coef
            t_grid = grid_pts[:, 0]
            l_grid = grid_pts[:, 2]
            r_grid = grid_pts[:, 1]
            tri = ax.tripcolor(
                t_grid, l_grid, r_grid, preds,
                cmap="viridis", shading="gouraud",
            )
            cbar = plt.colorbar(tri, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Predicted solubility (mg/mL)", rotation=270, labelpad=20)

    # Design points on top
    if design_pts:
        pts_array = np.array(design_pts, dtype=float)
        ax.scatter(
            pts_array[:, 0], pts_array[:, 2], pts_array[:, 1],
            s=100, facecolors="none", edgecolors="black", linewidths=2,
            label="Design points", zorder=5,
        )

    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    return fig


# ============================================================================
# PAGE LAYOUT
# ============================================================================
st.markdown("## △ Mixture Studio")
st.markdown("Design, sample, and model a three-component formulation space.")

col_left, col_right = st.columns([1.6, 1], gap="large")

# ---------------------------------------------------------------------------
# LEFT COLUMN — Controls
# ---------------------------------------------------------------------------
with col_left:
    # ---- Step 1: Components & Constraints ----
    st.subheader("1. Components & constraints")

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
            )
            st.session_state.comp_maxs[idx] = st.number_input(
                f"C{idx+1} max (%)",
                value=float(st.session_state.comp_maxs[idx]),
                min_value=0.0, max_value=100.0, step=1.0,
                key=f"max_{idx}",
            )

    constraints = get_constraints()
    feas = constraints.feasibility_report()

    if feas.feasible:
        st.success("Feasible region OK")
    else:
        st.error("Infeasible — fix the following:")
        for v in feas.violations:            # <-- was `v.t`; violations are plain strings
            st.write(f"• {v}")

    if feas.binding:
        st.info("Binding constraints: " + ", ".join(feas.binding))
    if feas.inactive:
        st.caption("Inactive bounds: " + "; ".join(feas.inactive))

    st.divider()

    # ---- Step 2: Design ----
    st.subheader("2. Experimental design")

    col_d1, col_d2, col_d3 = st.columns([1, 1, 1.2], gap="small")
    with col_d1:
        st.session_state.budget = st.selectbox(
            "Budget (points)", [4, 5, 6, 7, 8], index=2, key="budget_sel"
        )
    with col_d2:
        degree = st.selectbox(
            "Model degree",
            ["auto", "linear", "quadratic", "special_cubic"],
            key="degree_sel",
        )
    with col_d3:
        st.write("")  # vertical alignment
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
                        f"({res['residual_df']} residual df)."   # <-- was res['df']
                    )
                except ValueError as e:
                    st.error(f"Design failed: {e}")

    st.divider()

    # ---- Step 3: Measure & fit ----
    st.subheader("3. Measure & fit")

    if st.session_state.design_points:
        # Show the runs as a table
        pts_df = pd.DataFrame(
            st.session_state.design_points,
            columns=st.session_state.comp_names,
        ).round(2)
        pts_df.index = pts_df.index + 1
        pts_df.index.name = "Run"
        st.dataframe(pts_df, use_container_width=True)

        st.write("**Enter measured solubility for each run (mg/mL):**")

        # One number_input per run, laid out in a compact row
        n_runs = len(st.session_state.design_points)
        # widen if many runs
        cols = st.columns(min(n_runs, 4))
        # ensure the list is the right length (protects against stale imports)
        if len(st.session_state.solubilities) != n_runs:
            st.session_state.solubilities = [0.0] * n_runs

        for j in range(n_runs):
            with cols[j % len(cols)]:
                st.session_state.solubilities[j] = st.number_input(
                    f"Run {j+1}",
                    value=float(st.session_state.solubilities[j] or 0.0),
                    step=0.1,
                    key=f"sol_{j}",
                )

        if st.button("Fit model", key="fit_btn"):
            X = np.array(st.session_state.design_points, dtype=float)
            y = np.array(st.session_state.solubilities, dtype=float)

            if np.all(y == 0):
                st.warning("All solubilities are 0 — enter your measurements first.")
            else:
                deg = recommend_degree(len(X)) if degree == "auto" else degree
                try:
                    model = ScheffeModel(deg).fit(X, y)
                    summary = model.summary()
                    st.session_state.fit_result = {
                        "coef": summary["coefficients"],
                        "degree": deg,
                        "summary": summary,
                    }
                    st.success("Model fitted.")
                except ValueError as e:
                    st.error(f"Fit failed: {e}")
    else:
        st.info("Suggest design points above first.")

    st.divider()

    # ---- Step 4: Diagnostics ----
    st.subheader("4. Fit diagnostics")

    if st.session_state.fit_result is not None:
        s = st.session_state.fit_result["summary"]
        df = s["residual_df"]

        if df < 2 or not np.isfinite(s["loo_rmse"]):
            verdict, level = "Under-determined", "warning"
            note = (
                f"Only {df} residual degrees of freedom — not enough to check the fit. "
                "R² will look near-perfect no matter what. Add points or drop to a simpler model."
            )
        elif s["adj_r2"] <= 0 or not np.isfinite(s["adj_r2"]):
            verdict, level = "Poor fit", "error"
            note = (
                f"Adjusted R² is {s['adj_r2']:.3f} — explains no more than the mean. "
                "These components may not drive the response with this model. "
                "Try a different degree or widen the design region."
            )
        else:
            verdict, level = "Trustworthy", "success"
            note = (
                f"{df} residual df and positive adjusted R². "
                f"Trust LOO-RMSE ({s['loo_rmse']:.2f}) as the real accuracy indicator, not R²."
            )

        {"success": st.success, "warning": st.warning, "error": st.error}[level](verdict)

        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{s['r2']:.3f}")
        m2.metric("Adj R²", f"{s['adj_r2']:.3f}")
        m3.metric("LOO-RMSE", f"{s['loo_rmse']:.2f}" if np.isfinite(s['loo_rmse']) else "n/a")

        st.info(note)

        # Coefficient names
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
        st.code(f"Scheffé ({s['degree']}): {coef_str}", language="text")

# ---------------------------------------------------------------------------
# RIGHT COLUMN — Plot
# ---------------------------------------------------------------------------
with col_right:
    st.subheader("Ternary plot")
    fig = plot_ternary(
        constraints,
        design_pts=st.session_state.design_points or None,
        fit_result=st.session_state.fit_result,
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
