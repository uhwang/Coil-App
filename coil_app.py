import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import csv
import coil_geom as cg

pi1 = np.pi
pi2 = np.pi * 2


# =========================================================================
# 1. Electromagnetic Solver (Biot–Savart Fast Numerical Integration)
# =========================================================================
def analyze_magnetic_field(xx, yy, current=1.0, grid_res=60, z_height=0.1):

    """
    Computes 3D magnetic field using vectorized Biot–Savart integration.
    """

    mu_0 = 4 * np.pi * 1e-7

    x_margin = 0.25 * (np.max(xx) - np.min(xx)) if len(xx) > 0 else 1.0
    y_margin = 0.25 * (np.max(yy) - np.min(yy)) if len(yy) > 0 else 1.0

    x_range = np.linspace(np.min(xx) - x_margin, np.max(xx) + x_margin, grid_res)
    y_range = np.linspace(np.min(yy) - y_margin, np.max(yy) + y_margin, grid_res)
    X, Y = np.meshgrid(x_range, y_range)

    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(X)

    for i in range(len(xx) - 1):

        xs = (xx[i] + xx[i + 1]) / 2.0
        ys = (yy[i] + yy[i + 1]) / 2.0

        dx = xx[i + 1] - xx[i]
        dy = yy[i + 1] - yy[i]

        rx = X - xs
        ry = Y - ys
        rz = z_height

        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        r_mag = np.where(r_mag < 1e-9, 1e-9, r_mag)

        cx = dy * rz
        cy = -dx * rz
        cz = dx * ry - dy * rx

        factor = (mu_0 * current) / (4 * np.pi * (r_mag**3))

        Bx += cx * factor
        By += cy * factor
        Bz += cz * factor

    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    return X, Y, Bx, By, Bz, B_mag


# =========================================================================
# 2. Geometry Generator
# =========================================================================
class AdvancedCoilSimulator:

    def generate_geometry(self, coil_t, radius, axlen, bxlen, r_dist, p_dist, ncoil, curv_sim, shap_sim):

        if coil_t == cg.coil_util._coil_type_circle:
            coil = cg.CircleCoil(r=radius, r_dist=r_dist, p_dist=p_dist, ncoil=ncoil)

        elif coil_t == cg.coil_util._coil_type_ellipse:
            coil = cg.EllipseCoil(axlen=axlen, bxlen=bxlen, r_dist=r_dist, p_dist=p_dist, ncoil=ncoil)

        elif coil_t == cg.coil_util._coil_type_ellipse_curvature:
            coil = cg.EllipseCoilCurvature(
                axlen=axlen, bxlen=bxlen,
                r_dist=r_dist, p_dist=p_dist,
                ncoil=ncoil, target=curv_sim
            )

        elif coil_t == cg.coil_util._coil_type_ellipse_shape:
            coil = cg.EllipseCoilShape(
                axlen=axlen, bxlen=bxlen,
                r_dist=r_dist, p_dist=p_dist,
                ncoil=ncoil, target=shap_sim
            )

        return coil


# =========================================================================
# 3. Streamlit Setup
# =========================================================================
st.set_page_config(page_title="Planar Coil EM Analyzer", layout="wide")

st.title("🧲 Planar Coil Design & Electromagnetic Analysis System")
st.caption("Real-time physics-based feedback for coil geometry optimization")
st.divider()

tab1, tab2 = st.tabs([
    "📐 1. Geometry Design",
    "📊 2. Electromagnetic Analysis"
])


if "coil_x" not in st.session_state:
    st.session_state["coil_x"] = None

if "coil_y" not in st.session_state:
    st.session_state["coil_y"] = None


# =========================================================================
# 4. TAB 1 - Geometry Design
# =========================================================================
with tab1:

    st.header("Geometry Parameter Tuning")

    st.write("Design coil geometry using the control panel below.")

    main_col_left, main_col_right = st.columns([1, 4])

    with main_col_left:

        st.subheader("Design Parameters")

        coil_type = st.selectbox(
            "Coil Type",
            [
                cg.coil_util._coil_type_circle,
                cg.coil_util._coil_type_ellipse,
                cg.coil_util._coil_type_ellipse_curvature,
                cg.coil_util._coil_type_ellipse_shape
            ]
        )

        ncoil = st.slider("ncoil", 2, 50, 5, 1, key="ncoil_slider")

        if coil_type == cg.coil_util._coil_type_circle:
            radius = st.slider("Radius [mm]", 0.1, 9.0, 2.0, 0.1)
            axlen, bxlen = 0, 0
        else:
            radius = 0
            axlen = st.slider("Ax Length [mm]", 0.1, 9.0, 1.0, 0.1)
            bxlen = st.slider("Bx Length [mm]", 0.1, 9.0, 2.0, 0.1)

        r_dist = st.slider("r_dist", 0.0, 9.0, 2.5, 0.1)
        p_dist = st.slider("p_dist", -1.0, 1.0, 0.4, 0.1)

        curv_sim = 0.4
        if coil_type == cg.coil_util._coil_type_ellipse_curvature:
            curv_sim = st.slider("Curvature Similarity", 0.0, 1.0, 0.4, 0.05)

        shap_sim = 0.4
        if coil_type == cg.coil_util._coil_type_ellipse_shape:
            shap_sim = st.slider("Shape Similarity", 0.0, 1.0, 0.4, 0.05)

        sim = AdvancedCoilSimulator()

        coil = sim.generate_geometry(
            coil_type,
            radius,
            axlen,
            bxlen,
            r_dist,
            p_dist,
            ncoil,
            curv_sim,
            shap_sim
        )

    with main_col_right:

        st.subheader("Geometry Preview")

        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(14, 5),
            gridspec_kw={'width_ratios': [1, 2]}
        )

        if coil_type == cg.coil_util._coil_type_circle:
            ax_len, bx_len = coil.r, coil.r
        else:
            ax_len, bx_len = coil.axlen, coil.bxlen

        pnt = np.linspace(0, pi2, 50)

        x1 = coil.c1x + ax_len * np.cos(pnt)
        y1 = coil.c1y + bx_len * np.sin(pnt)

        x2 = coil.c2x + ax_len * np.cos(pnt)
        y2 = coil.c2y + bx_len * np.sin(pnt)

        x3 = coil.P[2] + coil.r_fillet * np.cos(pnt)
        y3 = coil.P[3] + coil.r_fillet * np.sin(pnt)

        ax1.plot(x1, y1, 'r-')
        ax1.plot(x2, y2, 'g-')
        ax1.plot(x3, y3, 'b-')

        # =========================================================
        # Transition ellipse (curvature/shape models only)
        # =========================================================
        if coil_type == cg.coil_util._coil_type_ellipse_curvature or \
        coil_type == cg.coil_util._coil_type_ellipse_shape:
        
            x4 = coil.xc_s + coil.a_s * np.cos(pnt)
            y4 = coil.yc_s + coil.b_s * np.sin(pnt)
        
            ax1.plot(x4, y4, 'm-', alpha=0.8)
            ax1.plot(coil.xc_s, coil.yc_s, marker='o', color='m', markersize=4)
        
        # =========================================================
        # Transition arrows (IMPORTANT - previously missing)
        # =========================================================
        ax1.annotate(
            '',
            xy=(coil.P1[0], coil.P1[1]),
            xytext=(coil.P1[2], coil.P1[3]),
            arrowprops=dict(
                arrowstyle="<-",
                color="purple",
                lw=1.8,
                shrinkA=0, shrinkB=0
            )
        )
        
        ax1.annotate(
            '',
            xy=(coil.P2[0], coil.P2[1]),
            xytext=(coil.P2[2], coil.P2[3]),
            arrowprops=dict(
                arrowstyle="<-",
                color="purple",
                lw=1.8,
                shrinkA=0, shrinkB=0
            )
        )
        ax1.set_title("Base Unit")
        ax1.grid(True)
        ax1.set_aspect('equal')

        cc = coil.create_geom()
        xx, yy = cc.x, cc.y

        st.session_state["coil_x"] = xx
        st.session_state["coil_y"] = yy

        ax2.plot(xx, yy, 'r-')
        ax2.set_title(f"Full Coil (Turns: {ncoil})")
        ax2.grid(True)
        ax2.set_aspect('equal')

        st.pyplot(fig, width="stretch")


# =========================================================================
# 5. TAB 2 - Electromagnetic Analysis
# =========================================================================
with tab2:

    st.header("Electromagnetic Field Mapping & Design Validation")

    xx_input = st.session_state["coil_x"]
    yy_input = st.session_state["coil_y"]

    if xx_input is None or yy_input is None:
        st.warning("No coil geometry generated yet. Please go to Geometry Design tab.")
    else:

        st.markdown("### Simulation Conditions")

        c1, c2, c3 = st.columns(3)

        with c1:
            current = st.number_input("Current [A]", value=2.0, step=0.5)

        with c2:
            z_height = st.slider("Observation Height [mm]", 0.05, 2.0, 0.1, 0.05)

        with c3:
            grid_res = st.slider("Mesh Resolution", 40, 150, 60, 5)

        st.divider()

        with st.spinner("Solving Biot–Savart equation..."):
            X, Y, Bx, By, Bz, B_mag = analyze_magnetic_field(
                xx_input,
                yy_input,
                current,
                grid_res,
                z_height
            )

        st.subheader("Magnetic Field Visualization")

        fig_main, ax_main = plt.subplots(figsize=(13, 6.5))

        contour = ax_main.contourf(
            X, Y,
            B_mag * 1e6,
            levels=45,
            cmap='inferno'
        )

        fig_main.colorbar(contour, ax=ax_main, orientation='horizontal')

        ax_main.streamplot(X, Y, Bx, By, linewidth=0.8, density=1.1)

        ax_main.plot(xx_input, yy_input, color='green', linewidth=1.8)

        ax_main.set_xlabel("X (mm)")
        ax_main.set_ylabel("Y (mm)")
        ax_main.set_aspect('equal')
        ax_main.grid(True, linestyle=':', alpha=0.3)

        fig_main.subplots_adjust(
            left=0.06,
            right=0.98,
            top=0.95,
            bottom=0.22
        )

        st.pyplot(fig_main, use_container_width=True)

        st.divider()

        metric_col, export_col = st.columns(2)

        with metric_col:

            st.subheader("Design Metrics")

            out_rad = np.max(np.sqrt(xx_input**2 + yy_input**2))
            in_rad = np.min(np.sqrt(xx_input**2 + yy_input**2))

            aspect_ratio = in_rad / out_rad if out_rad > 0 else 0
            total_length = np.sum(np.sqrt(np.diff(xx_input)**2 + np.diff(yy_input)**2))
            max_b_field = np.max(B_mag) * 1e6

            st.metric("Outer Radius", f"{out_rad:.2f} mm")
            st.metric("Aspect Ratio", f"{aspect_ratio:.2f}")
            st.metric("Wire Length", f"{total_length:.1f} mm")
            st.metric("Peak B-field", f"{max_b_field:.1f} µT")

        with export_col:

            st.subheader("Export")

            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["X_mm", "Y_mm"])

            for x_val, y_val in zip(xx_input, yy_input):
                writer.writerow([x_val, y_val])

            st.download_button(
                "Download CSV",
                csv_buffer.getvalue().encode("utf-8"),
                file_name="coil_geometry.csv",
                mime="text/csv"
            )

            img_buffer = io.BytesIO()
            fig_main.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")

            st.download_button(
                "Download PNG",
                img_buffer.getvalue(),
                file_name="field_map.png",
                mime="image/png"
            )