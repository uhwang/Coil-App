import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import csv
import coil_geom as cg

pi1 = np.pi
pi2 = np.pi * 2

# =========================================================================
# 단위 정의
# =========================================================================
LENGTH_UNITS = {
    "mm (밀리미터)": {"scale": 1.0,    "to_m": 1e-3,  "symbol": "mm"},
    "cm (센티미터)": {"scale": 0.1,    "to_m": 1e-2,  "symbol": "cm"},
    "m  (미터)":     {"scale": 1e-3,   "to_m": 1.0,   "symbol": "m"},
    "µm (마이크로)": {"scale": 1e3,    "to_m": 1e-6,  "symbol": "µm"},
    "in (인치)":     {"scale": 1/25.4, "to_m": 0.0254,"symbol": "in"},
}

BFIELD_UNITS = {
    "µT (마이크로테슬라)": 1e6,
    "mT (밀리테슬라)":     1e3,
    "T  (테슬라)":         1e0,
    "G  (가우스)":         1e4,
}


# =========================================================================
# 1. Electromagnetic Solver (Biot–Savart Fast Numerical Integration)
# =========================================================================
def analyze_magnetic_field(xx_mm, yy_mm, current=1.0, grid_res=60, z_height_mm=0.1):
    """
    입력: xx_mm, yy_mm, z_height_mm → 내부에서 m 변환 후 계산
    반환: X, Y (mm), Bx, By, Bz, B_mag (Tesla)
    """
    mu_0 = 4 * np.pi * 1e-7

    x_margin = 0.25 * (np.max(xx_mm) - np.min(xx_mm)) if len(xx_mm) > 0 else 1.0
    y_margin = 0.25 * (np.max(yy_mm) - np.min(yy_mm)) if len(yy_mm) > 0 else 1.0

    x_range = np.linspace(np.min(xx_mm) - x_margin, np.max(xx_mm) + x_margin, grid_res)
    y_range = np.linspace(np.min(yy_mm) - y_margin, np.max(yy_mm) + y_margin, grid_res)
    X, Y = np.meshgrid(x_range, y_range)   # mm

    X_m = X * 1e-3
    Y_m = Y * 1e-3
    rz  = z_height_mm * 1e-3

    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(X)

    for i in range(len(xx_mm) - 1):
        xs = (xx_mm[i] + xx_mm[i+1]) / 2.0 * 1e-3
        ys = (yy_mm[i] + yy_mm[i+1]) / 2.0 * 1e-3
        dx = (xx_mm[i+1] - xx_mm[i]) * 1e-3
        dy = (yy_mm[i+1] - yy_mm[i]) * 1e-3

        rx = X_m - xs
        ry = Y_m - ys

        r_mag = np.sqrt(rx**2 + ry**2 + rz**2)
        r_mag = np.where(r_mag < 1e-12, 1e-12, r_mag)

        cx = dy * rz
        cy = -dx * rz
        cz = dx * ry - dy * rx

        factor = (mu_0 * current) / (4 * np.pi * r_mag**3)
        Bx += cx * factor
        By += cy * factor
        Bz += cz * factor

    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    return X, Y, Bx, By, Bz, B_mag


# =========================================================================
# 2. Geometry Generator
# =========================================================================
class AdvancedCoilSimulator:
    def generate_geometry(self, coil_t, radius, axlen, bxlen, r_dist, p_dist,
                          ncoil, curv_sim, shap_sim):
        if coil_t == cg.coil_util._coil_type_circle:
            return cg.CircleCoil(r=radius, r_dist=r_dist, p_dist=p_dist, ncoil=ncoil)
        elif coil_t == cg.coil_util._coil_type_ellipse:
            return cg.EllipseCoil(axlen=axlen, bxlen=bxlen, r_dist=r_dist,
                                  p_dist=p_dist, ncoil=ncoil)
        elif coil_t == cg.coil_util._coil_type_ellipse_curvature:
            return cg.EllipseCoilCurvature(axlen=axlen, bxlen=bxlen, r_dist=r_dist,
                                           p_dist=p_dist, ncoil=ncoil, target=curv_sim)
        elif coil_t == cg.coil_util._coil_type_ellipse_shape:
            return cg.EllipseCoilShape(axlen=axlen, bxlen=bxlen, r_dist=r_dist,
                                       p_dist=p_dist, ncoil=ncoil, target=shap_sim)


# =========================================================================
# 3. Streamlit Setup
# =========================================================================
st.set_page_config(page_title="Planar Coil EM Analyzer", layout="wide")
st.title("🧲 Planar Coil Design & Electromagnetic Analysis System")
st.caption("Real-time physics-based feedback for coil geometry optimization")
st.divider()

tab1, tab2 = st.tabs(["📐 1. Geometry Design", "📊 2. Electromagnetic Analysis"])

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

        # ── 길이 단위 선택 ──────────────────────────────────────────
        sel_len_unit = st.selectbox(
            "Length Unit",
            list(LENGTH_UNITS.keys()),
            index=0,
            key="len_unit_tab1"
        )
        lu = LENGTH_UNITS[sel_len_unit]   # {"scale", "to_m", "symbol"}
        ls = lu["symbol"]                 # 표시용 기호

        # 내부 계산은 항상 mm 기준이므로,
        # 사용자 입력값 → mm 변환 계수 = 1 / scale (mm 기준 scale)
        # scale: 1mm = scale * [선택단위]  →  1[선택단위] = (1/scale) mm
        to_mm = 1.0 / lu["scale"]         # 선택 단위 → mm

        coil_type = st.selectbox(
            "Coil Type",
            [
                cg.coil_util._coil_type_circle,
                cg.coil_util._coil_type_ellipse,
                cg.coil_util._coil_type_ellipse_curvature,
                cg.coil_util._coil_type_ellipse_shape,
            ]
        )

        ncoil = st.slider("ncoil", 2, 50, 5, 1, key="ncoil_slider")

        if coil_type == cg.coil_util._coil_type_circle:
            radius_u = st.slider(f"Radius [{ls}]", 0.1, 9.0, 2.0, 0.1)
            radius = radius_u * to_mm   # → mm
            axlen, bxlen = 0, 0
        else:
            radius = 0
            axlen_u = st.slider(f"Ax Length [{ls}]", 0.1, 9.0, 1.0, 0.1)
            bxlen_u = st.slider(f"Bx Length [{ls}]", 0.1, 9.0, 2.0, 0.1)
            axlen = axlen_u * to_mm     # → mm
            bxlen = bxlen_u * to_mm     # → mm

        r_dist_u = st.slider(f"r_dist [{ls}]", 0.0, 9.0, 2.5, 0.1)
        r_dist   = r_dist_u * to_mm    # → mm
        p_dist   = st.slider("p_dist", -1.0, 1.0, 0.4, 0.1)

        curv_sim = 0.4
        if coil_type == cg.coil_util._coil_type_ellipse_curvature:
            curv_sim = st.slider("Curvature Similarity", 0.0, 1.0, 0.4, 0.05)

        shap_sim = 0.4
        if coil_type == cg.coil_util._coil_type_ellipse_shape:
            shap_sim = st.slider("Shape Similarity", 0.0, 1.0, 0.4, 0.05)

        sim  = AdvancedCoilSimulator()
        coil = sim.generate_geometry(coil_type, radius, axlen, bxlen,
                                     r_dist, p_dist, ncoil, curv_sim, shap_sim)

    with main_col_right:
        st.subheader("Geometry Preview")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                        gridspec_kw={'width_ratios': [1, 2]})

        ax_len = coil.r    if coil_type == cg.coil_util._coil_type_circle else coil.axlen
        bx_len = coil.r    if coil_type == cg.coil_util._coil_type_circle else coil.bxlen

        # 플롯은 선택 단위로 표시 (mm 값 × scale)
        sc = lu["scale"]
        pnt = np.linspace(0, pi2, 50)

        x1 = (coil.c1x + ax_len * np.cos(pnt)) * sc
        y1 = (coil.c1y + bx_len * np.sin(pnt)) * sc
        x2 = (coil.c2x + ax_len * np.cos(pnt)) * sc
        y2 = (coil.c2y + bx_len * np.sin(pnt)) * sc
        x3 = (coil.P[2] + coil.r_fillet * np.cos(pnt)) * sc
        y3 = (coil.P[3] + coil.r_fillet * np.sin(pnt)) * sc

        ax1.plot(x1, y1, 'r-')
        ax1.plot(x2, y2, 'g-')
        ax1.plot(x3, y3, 'b-')

        if coil_type in (cg.coil_util._coil_type_ellipse_curvature,
                         cg.coil_util._coil_type_ellipse_shape):
            x4 = (coil.xc_s + coil.a_s * np.cos(pnt)) * sc
            y4 = (coil.yc_s + coil.b_s * np.sin(pnt)) * sc
            ax1.plot(x4, y4, 'm-', alpha=0.8)
            ax1.plot(coil.xc_s * sc, coil.yc_s * sc, marker='o', color='m', markersize=4)

        ax1.annotate('',
            xy=(coil.P1[0]*sc, coil.P1[1]*sc),
            xytext=(coil.P1[2]*sc, coil.P1[3]*sc),
            arrowprops=dict(arrowstyle="<-", color="purple", lw=1.8, shrinkA=0, shrinkB=0))
        ax1.annotate('',
            xy=(coil.P2[0]*sc, coil.P2[1]*sc),
            xytext=(coil.P2[2]*sc, coil.P2[3]*sc),
            arrowprops=dict(arrowstyle="<-", color="purple", lw=1.8, shrinkA=0, shrinkB=0))

        ax1.set_title("Base Unit")
        ax1.set_xlabel(f"X ({ls})")
        ax1.set_ylabel(f"Y ({ls})")
        ax1.grid(True)
        ax1.set_aspect('equal')

        cc = coil.create_geom()
        xx_mm, yy_mm = cc.x, cc.y   # 항상 mm

        st.session_state["coil_x"] = xx_mm
        st.session_state["coil_y"] = yy_mm

        ax2.plot(xx_mm * sc, yy_mm * sc, 'r-')
        ax2.set_title(f"Full Coil (Turns: {ncoil})")
        ax2.set_xlabel(f"X ({ls})")
        ax2.set_ylabel(f"Y ({ls})")
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

        disp_col1, disp_col2 = st.columns(2)

        # ── 길이 단위 선택 (Tab 2) ───────────────────────────────────
        with disp_col1:
            sel_len_unit2 = st.selectbox(
                "Length Display Unit",
                list(LENGTH_UNITS.keys()),
                index=0,
                key="len_unit_tab2"
            )
            lu2 = LENGTH_UNITS[sel_len_unit2]
            ls2 = lu2["symbol"]
            sc2 = lu2["scale"]   # mm → 선택 단위

        # ── B-field 단위 선택 ────────────────────────────────────────
        with disp_col2:
            sel_b_unit = st.selectbox(
                "B-field Display Unit",
                list(BFIELD_UNITS.keys()),
                index=0
            )
            b_scale  = BFIELD_UNITS[sel_b_unit]
            b_symbol = sel_b_unit.split()[0]

        st.divider()

        with st.spinner("Solving Biot–Savart equation..."):
            X, Y, Bx, By, Bz, B_mag = analyze_magnetic_field(
                xx_input, yy_input, current, grid_res, z_height
            )

        st.subheader("Magnetic Field Visualization")

        fig_main, ax_main = plt.subplots(figsize=(13, 6.5))

        # 플롯 좌표: mm → 선택 단위
        X_disp = X * sc2
        Y_disp = Y * sc2

        contour = ax_main.contourf(X_disp, Y_disp, B_mag * b_scale, levels=45, cmap='inferno')
        cbar = fig_main.colorbar(contour, ax=ax_main, orientation='horizontal')
        cbar.set_label(f"B-field magnitude [{b_symbol}]")

        ax_main.streamplot(X_disp, Y_disp, Bx, By, linewidth=0.8, density=1.1)
        ax_main.plot(xx_input * sc2, yy_input * sc2, color='green', linewidth=1.8)

        ax_main.set_xlabel(f"X ({ls2})")
        ax_main.set_ylabel(f"Y ({ls2})")
        ax_main.set_aspect('equal')
        ax_main.grid(True, linestyle=':', alpha=0.3)

        fig_main.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.22)
        st.pyplot(fig_main, use_container_width=True)

        st.divider()

        metric_col, export_col = st.columns(2)

        with metric_col:
            st.subheader("Design Metrics")

            coil_width   = (np.max(xx_input) - np.min(xx_input)) * sc2
            coil_height  = (np.max(yy_input) - np.min(yy_input)) * sc2
            aspect_ratio = coil_height / coil_width if coil_width > 0 else 0
            total_length = np.sum(np.sqrt(np.diff(xx_input)**2 + np.diff(yy_input)**2)) * sc2
            max_b_field  = np.max(B_mag)  * b_scale
            mean_b_field = np.mean(B_mag) * b_scale

            st.metric("Coil Width",          f"{coil_width:.4g} {ls2}")
            st.metric("Coil Height",         f"{coil_height:.4g} {ls2}")
            st.metric("Aspect Ratio (H/W)",  f"{aspect_ratio:.3f}")
            st.metric("Wire Length",         f"{total_length:.4g} {ls2}")
            st.metric("Peak B-field",        f"{max_b_field:.3g} {b_symbol}")
            st.metric("Mean B-field",        f"{mean_b_field:.3g} {b_symbol}")

        with export_col:
            st.subheader("Export")

            # CSV는 항상 mm 기준으로 저장 (원본 보존)
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["X_mm", "Y_mm"])
            for x_val, y_val in zip(xx_input, yy_input):
                writer.writerow([x_val, y_val])

            st.download_button(
                "Download CSV (mm)",
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
