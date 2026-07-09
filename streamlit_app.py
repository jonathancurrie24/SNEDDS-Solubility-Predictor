"""
Streamlit app for Mixture Studio — constrained-mixture DOE and Scheffé modeling.

Install: pip install streamlit numpy matplotlib mpltern pandas

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import StringIO

from mixture_doe import (
    MixtureConstraints, ScheffeModel, d_optimal_design, recommend_degree,
    feasible_grid, candidate_points, vertices
)

# ============================================================================
# PAGE CONFIG & SESSION STATE
# ============================================================================
st.set_page_config(
    page_title="Mixture Studio",
    page_icon="△",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state for constraints, design, and data
if "comp_names" not in st.session_state:
    st.session_state.comp_names = ["Surfactant", "Oil", "Co-Surfactant"]
if "comp_mins" not in st.session_state:
    st.session_state.comp_mins = [30.0, 10.0, 20.0]
if "comp_maxs" not in st.session_state:
    st.session_state.comp_maxs = [60.0, 40.0, 50.0]
if "budget" not in st.session_state:
    st.session_state.budget = 6
if "design_points" not in st.session_state:
    st.session_state.design_points = []
if "solubilities" not in st.session_state:
    st.session_state.solubilities = []
if "fit_result" not in st.session_state:
    st.session_state.fit_result = None


# ============================================================================
# HELPERS
# ============================================================================
def get_constraints():
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
    """Draw the ternary plot with feasible region, design points, and optionally a surface."""
    try:
        import mpltern
    except ImportError:
        st.error("mpltern not found. Install with: pip install mpltern")
        return None

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="ternary")

    names = constraints.names
    ax.set_tlabel(f"{names['c1']} (%)", fontsize=11)
    ax.set_llabel(f"{names['c3']} (%)", fontsize=11)
    ax.set_rlabel(f"{names['c2']} (%)", fontsize=11)

    # Grid
    for axis in [ax.taxis, ax.laxis, ax.raxis]:
        axis.set_major_locator(mticker.MultipleLocator(10.0))
        axis.set_minor_locator(mticker.MultipleLocator(5.0))
    ax.grid(True, alpha=0.3)

    # Feasible region boundary
    verts = vertices(constraints.bounds)
    if len(verts) >= 3:
        loop = verts + [verts[0]]
        t_vals = [p[0] for p in loop]
        l_vals = [p[2] for p in loop]
        r_vals = [p[1] for p in loop]
        ax.plot(t_vals, l_vals, r_vals, "r-", linewidth=2.5, label="Feasible Region")

    # Draw heatmap surface if fitted
    if fit_result is not None:
        coef, deg = fit_result["coef"], fit_result["degree"]
        grid_pts = feasible_grid(constraints, step=2.0)
        if len(grid_pts) > 0:
            from mixture_doe import expand as scheffe_expand
            M = scheffe_expand(grid_pts, deg)
            preds = M @ coef
            t_grid = [p[0] for p in grid_pts]
            l_grid = [p[2] for p in grid_pts]
            r_grid = [p[1] for p in grid_pts]
            tri = ax.tripcolor(t_grid, l_grid, r_grid, preds, cmap="viridis", shading="gouraud")
            cbar = plt.colorbar(tri, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Predicted Solubility (mg/mL)", rotation=270, labelpad=20)

    # Design points
    if design_pts is not None and len(design_pts) > 0:
        pts_array = np.array(design_pts)
        ax.scatter(
            pts_array[:, 0], pts_array[:, 2], pts_array[:, 1],
            s=100, facecolors="none", edgecolors="black", linewidths=2,
            label="Design Points", zorder=5
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

# ============================================================================
# LEFT COLUMN: Controls
# ============================================================================
with col_left:
    # ---- Step 1: Components & Constraints ----
    st.subheader("1. Components & Constraints")

    c1_col, c2_col, c3_col = st.columns(3)
    with c1_col:
        st.session_state.comp_names[0] = st.text_input(
            "Component 1 name", value=st.session_state.comp_names[0], key="c1_name"
        )
        st.session_state.comp_mins[0] = st.number_input(
            "C1 min (%)", value=st.session_state.comp_mins[0], min_value=0.0, max_value=100.0, key="c1_min"
        )
        st.session_state.comp_maxs[0] = st.number_input(
            "C1 max (%)", value=st.session_state.comp_maxs[0], min_value=0.0, max_value=100.0, key="c1_max"
        )

    with c2_col:
        st.session_state.comp_names[1] = st.text_input(
            "Component 2 name", value=st.session_state.comp_names[1], key="c2_name"
        )
        st.session_state.comp_mins[1] = st.number_input(
            "C2 min (%)", value=st.session_state.comp_mins[1], min_value=0.0, max_value=100.0, key="c2_min"
        )
        st.session_state.comp_maxs[1] = st.number_input(
            "C2 max (%)", value=st.session_state.comp_maxs[1], min_value=0.0, max_value=100.0, key="c2_max"
        )

    with c3_col:
        st.session_state.comp_names[2] = st.text_input(
            "Component 3 name", value=st.session_state.comp_names[2], key="c3_name"
        )
        st.session_state.comp_mins[2] = st.number_input(
            "C3 min (%)", value=st.session_state.comp_mins[2], min_value=0.0, max_value=100.0, key="c3_min"
        )
        st.session_state.comp_maxs[2] = st.number_input(
            "C3 max (%)", value=st.session_state.comp_maxs[2], min_value=0.0, max_value=100.0, key="c3_max"
        )

    constraints = get_constraints()
    feas = constraints.feasibility_report()

    # Feasibility summary
    if feas.feasible:
        st.success("✓ Feasible", icon="✓")
    else:
        st.error("✗ Infeasible", icon="✗")
        for v in feas.violations:
            st.error(f"  • {v.t}", icon="⚠")

    if feas.binding:
        st.info(f"Binding constraints: {', '.join(feas.binding)}")

    st.divider()

    # ---- Step 2: Experimental Design ----
    st.subheader("2. Experimental Design")

    col_d1, col_d2, col_d3 = st.columns([1, 1, 1.2], gap="small")
    with col_d1:
        st.session_state.budget = st.selectbox("Budget (points)", [4, 5, 6, 7, 8], index=2, key="budget_sel")
    with col_d2:
        degree = st.selectbox("Model degree", ["auto", "linear", "quadratic", "special_cubic"], key="degree_sel")
    with col_d3:
        if st.button("Suggest points", key="design_btn"):
            if not feas.feasible:
                st.error("Fix constraint violations first.")
            else:
                if degree == "auto":
                    deg = recommend_degree(st.session_state.budget)
                else:
                    deg = degree
                try:
                    res = d_optimal_design(constraints, st.session_state.budget, deg)
                    st.session_state.design_points = res["points"]
                    st.session_state.solubilities = [""] * len(res["points"])
                    st.session_state.fit_result = None
                    st.success(
                        f"✓ Placed {len(res['points'])} points for a {deg} model "
                        f"({res['df']} residual df)."
                    )
                except ValueError as e:
                    st.error(f"Design failed: {e}")

    st.divider()

    # ---- Step 3: Measure & Fit ----
    st.subheader("3. Measure & Fit")

    if st.session_state.design_points:
        pts_df = pd.DataFrame(
            st.session_state.design_points,
            columns=[c.replace("_", " ").title() for c in ["c1", "c2", "c3"]]
        )
        pts_df.index = pts_df.index + 1
        pts_df.index.name = "Run"

        st.write("**Enter measured solubility for each run:**")

        # Build input table using columns
        sol_cols = st.columns([0.5] + [1] * len(st.session_state.design_points))
        with sol_cols[0]:
            st.write("**Run**")
            for i in range(len(st.session_state.design_points)):
                st.write(str(i + 1))

        for j, col_obj in enumerate(sol_cols[1:]):
            with col_obj:
                st.write(f"**Sol. {j+1}**")
                s = st.number_input(
                    f"Run {j+1} solubility (mg/mL)",
                    value=float(st.session_state.solubilities[j]) if st.session_state.solubilities[j] else 0.0,
                    step=0.1,
                    key=f"sol_{j}",
                    label_visibility="collapsed"
                )
                st.session_state.solubilities[j] = s

        if st.button("Fit model", key="fit_btn"):
            # Validate all inputs are filled
            if all(s != "" and s != 0 for s in st.session_state.solubilities):
                X = np.array(st.session_state.design_points, dtype=float)
                y = np.array([float(s) for s in st.session_state.solubilities], dtype=float)

                # Auto-select degree based on budget
                if degree == "auto":
                    deg = recommend_degree(len(X))
                else:
                    deg = degree

                try:
                    model = ScheffeModel(deg)
                    model.fit(X, y)
                    summary = model.summary()
                    st.session_state.fit_result = {
                        "coef": summary["coefficients"],
                        "degree": deg,
                        "summary": summary,
                    }
                    st.success("✓ Model fitted.")
                except ValueError as e:
                    st.error(f"Fit failed: {e}")
            else:
                st.warning("Enter solubility for all runs before fitting.")

    else:
        st.info("Suggest design points above first.")

    st.divider()

    # ---- Step 4: Diagnostics ----
    st.subheader("4. Fit Diagnostics")

    if st.session_state.fit_result is not None:
        s = st.session_state.fit_result["summary"]
        n, p, df = s["n_points"], s["n_terms"], s["residual_df"]

        # Verdict: under-determined / poor fit / trustworthy
        if df < 2 or not np.isfinite(s["loo_rmse"]):
            verdict, color = "Under-determined", "warning"
            note = (
                f"Only {df} residual degrees of freedom — not enough to check the fit. "
                f"R² will look near-perfect no matter what. Add points or drop to a simpler model."
            )
        elif s["adj_r2"] <= 0 or not np.isfinite(s["adj_r2"]):
            verdict, color = "Poor fit", "error"
            note = (
                f"Adjusted R² is {s['adj_r2']:.3f} — explains no more than the average. "
                f"These components may not drive solubility linearly. Try a different model or wider design."
            )
        else:
            verdict, color = "Trustworthy", "success"
            note = (
                f"{df} residual degrees of freedom and positive adj R². "
                f"Trust **LOO-RMSE ({s['loo_rmse']:.2f})** as the real accuracy — "
                f"it's what to expect on a new formulation."
            )

        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            if color == "success":
                st.success(verdict)
            elif color == "warning":
                st.warning(verdict)
            else:
                st.error(verdict)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("R²", f"{s['r2']:.3f}")
        with col_m2:
            st.metric("Adj R²", f"{s['adj_r2']:.3f}")
        with col_m3:
            st.metric("LOO-RMSE", f"{s['loo_rmse']:.2f}")

        st.info(note)

        # Coefficients
        coef_names = [
            st.session_state.comp_names[0],
            st.session_state.comp_names[1],
            st.session_state.comp_names[2],
        ]
        if s["degree"] in ("quadratic", "special_cubic"):
            coef_names += [
                f"{coef_names[0]}·{coef_names[1]}",
                f"{coef_names[0]}·{coef_names[2]}",
                f"{coef_names[1]}·{coef_names[2]}",
            ]
        if s["degree"] == "special_cubic":
            coef_names += [f"{coef_names[0]}·{coef_names[1]}·{coef_names[2]}"]

        coef_str = " | ".join(
            f"{name} {c:+.2f}" for name, c in zip(coef_names, s["coefficients"].values())
        )
        st.code(f"Scheffé ({s['degree']}): {coef_str}", language="text")

# ============================================================================
# RIGHT COLUMN: Plot
# ============================================================================
with col_right:
    st.subheader("Ternary Plot")
    fig = plot_ternary(
        constraints,
        design_pts=st.session_state.design_points if st.session_state.design_points else None,
        fit_result=st.session_state.fit_result,
    )
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Plot will appear here once you suggest design points.")

# ============================================================================
# EXPORT / IMPORT
# ============================================================================
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
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="mixture_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No data to export yet.")

with col_imp:
    st.subheader("Import data")
    uploaded_file = st.file_uploader("Upload CSV with columns: component names + 'Solubility (mg/mL)'")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Assume first 3 columns are components, last is solubility
            comp_cols = df.columns[:3].tolist()
            sol_col = df.columns[-1]
            pts = df[comp_cols].values.tolist()
            sols = df[sol_col].tolist()
            st.session_state.comp_names = comp_cols
            st.session_state.design_points = [tuple(p) for p in pts]
            st.session_state.solubilities = sols
            st.success("✓ Data imported. Edit component names and constraints above if needed.")
        except Exception as e:
            st.error(f"Import failed: {e}")
