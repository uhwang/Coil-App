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
    "mm (밀리미터)": {"scale": 1.0,         "to_m": 1e-3,   "symbol": "mm"},
    "cm (센티미터)": {"scale": 0.1,         "to_m": 1e-2,   "symbol": "cm"},
    "m  (미터)":     {"scale": 1e-3,        "to_m": 1.0,    "symbol": "m"},
    "µm (마이크로)": {"scale": 1e3,         "to_m": 1e-6,   "symbol": "µm"},
    "in (인치)":     {"scale": 1.0/25.4,    "to_m": 0.0254, "symbol": "in"},
}

BFIELD_UNITS = {
    "µT (마이크로테슬라)": 1e6,
    "mT (밀리테슬라)":     1e3,
    "T  (테슬라)":         1e0,
    "G  (가우스)":         1e4,
}


# =========================================================================
# 1. Electromagnetic Solver (Biot–Savart)
# =========================================================================
def analyze_magnetic_field(xx_mm, yy_mm, current=1.0, grid_res=60, z_height_mm=0.1):
    """
    입력: xx_mm, yy_mm, z_height_mm (모두 mm)
    반환: X, Y (mm 격자), Bx, By, Bz, B_mag (Tesla)
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

        factor = (mu_0 * current) / (4 * np.pi * r_mag**3)
        Bx += (dy * rz)          * factor
        By += (-dx * rz)         * factor
        Bz += (dx * ry - dy * rx) * factor

    return X, Y, Bx, By, Bz, np.sqrt(Bx**2 + By**2 + Bz**2)


# =========================================================================
# 2. Geometry Generator
# =========================================================================
class AdvancedCoilSimulator:
    def generate_geometry(self, coil_t, radius, axlen, bxlen,
                          r_dist, p_dist, ncoil, curv_sim, shap_sim):
        if coil_t == cg.coil_util._coil_type_circle:
            return cg.CircleCoil(r=radius, r_dist=r_dist, p_dist=p_dist, ncoil=ncoil)
        elif coil_t == cg.coil_util._coil_type_ellipse:
            return cg.EllipseCoil(axlen=axlen, bxlen=bxlen,
                                  r_dist=r_dist, p_dist=p_dist, ncoil=ncoil)
        elif coil_t == cg.coil_util._coil_type_ellipse_curvature:
            return cg.EllipseCoilCurvature(axlen=axlen, bxlen=bxlen,
                                           r_dist=r_dist, p_dist=p_dist,
                                           ncoil=ncoil, target=curv_sim)
        elif coil_t == cg.coil_util._coil_type_ellipse_shape:
            return cg.EllipseCoilShape(axlen=axlen, bxlen=bxlen,
                                       r_dist=r_dist, p_dist=p_dist,
                                       ncoil=ncoil, target=shap_sim)


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

    main_col_left, main_col_right = st.columns([1, 4])

    with main_col_left:
        st.subheader("Design Parameters")

        # ── 길이 단위 선택 ──────────────────────────────────
        sel_len = st.selectbox("Length Unit", list(LENGTH_UNITS.keys()),
                               index=0, key="len_unit_tab1")
        lu   = LENGTH_UNITS[sel_len]
        ls   = lu["symbol"]       # 표시 기호 (예: "cm")
        sc   = lu["scale"]        # mm → 선택 단위 (표시용)
        to_mm = 1.0 / sc          # 선택 단위 → mm (내부 계산용)

        # 슬라이더 범위/기본값을 선택 단위로 자동 변환하는 헬퍼
        def make_slider(label, min_mm, max_mm, def_mm, step_mm, key=None):
            kwargs = dict(key=key) if key else {}
            return st.slider(
                label,
                float(min_mm * sc),
                float(max_mm * sc),
                float(def_mm * sc),
                float(step_mm * sc),
                **kwargs
            )

        coil_type = st.selectbox("Coil Type", [
            cg.coil_util._coil_type_circle,
            cg.coil_util._coil_type_ellipse,
            cg.coil_util._coil_type_ellipse_curvature,
            cg.coil_util._coil_type_ellipse_shape,
        ])

        ncoil = st.slider("ncoil", 2, 50, 5, 1, key="ncoil_slider")

        if coil_type == cg.coil_util._coil_type_circle:
            radius = make_slider(f"Radius [{ls}]", 0.1, 9.0, 2.0, 0.1) * to_mm
            axlen, bxlen = 0, 0
        else:
            radius = 0
            axlen  = make_slider(f"Ax Length [{ls}]", 0.1, 9.0, 1.0, 0.1) * to_mm
            bxlen  = make_slider(f"Bx Length [{ls}]", 0.1, 9.0, 2.0, 0.1) * to_mm

        r_dist = make_slider(f"r_dist [{ls}]", 0.0, 9.0, 2.5, 0.1) * to_mm  # 실제 길이
        p_dist = st.slider("p_dist", -1.0, 1.0, 0.4, 0.1)                  # 무차원

        curv_sim = 0.4
        if coil_type == cg.coil_util._coil_type_ellipse_curvature:
            curv_sim = st.slider("Curvature Similarity", 0.0, 1.0, 0.4, 0.05)

        shap_sim = 0.4
        if coil_type == cg.coil_util._coil_type_ellipse_shape:
            shap_sim = st.slider("Shape Similarity", 0.0, 1.0, 0.4, 0.05)

        coil = AdvancedCoilSimulator().generate_geometry(
            coil_type, radius, axlen, bxlen,
            r_dist, p_dist, ncoil, curv_sim, shap_sim
        )

    with main_col_right:
        st.subheader("Geometry Preview")

        ax_len = coil.r if coil_type == cg.coil_util._coil_type_circle else coil.axlen
        bx_len = coil.r if coil_type == cg.coil_util._coil_type_circle else coil.bxlen
        pnt    = np.linspace(0, pi2, 50)

        # 모든 좌표: mm × sc → 선택 단위로 변환
        def cvt(v): return v * sc

        # ── 위: Base Unit ─────────────────────────────────────
        # rcParams로 글자 크기 전역 설정 (Streamlit 렌더링 환경에서도 확실히 적용)
        plt.rcParams.update({
            'font.size':       8,
            'axes.titlesize':  9,
            'axes.labelsize':  8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
        })
        fig1, ax1 = plt.subplots(figsize=(4, 4))

        ax1.plot(cvt(coil.c1x + ax_len * np.cos(pnt)),
                 cvt(coil.c1y + bx_len * np.sin(pnt)), 'r-')
        ax1.plot(cvt(coil.c2x + ax_len * np.cos(pnt)),
                 cvt(coil.c2y + bx_len * np.sin(pnt)), 'g-')
        ax1.plot(cvt(coil.P[2] + coil.r_fillet * np.cos(pnt)),
                 cvt(coil.P[3] + coil.r_fillet * np.sin(pnt)), 'b-')

        if coil_type in (cg.coil_util._coil_type_ellipse_curvature,
                         cg.coil_util._coil_type_ellipse_shape):
            ax1.plot(cvt(coil.xc_s + coil.a_s * np.cos(pnt)),
                     cvt(coil.yc_s + coil.b_s * np.sin(pnt)), 'm-', alpha=0.8)
            ax1.plot(cvt(coil.xc_s), cvt(coil.yc_s),
                     marker='o', color='m', markersize=4)

        ax1.annotate('',
            xy    =(cvt(coil.P1[0]), cvt(coil.P1[1])),
            xytext=(cvt(coil.P1[2]), cvt(coil.P1[3])),
            arrowprops=dict(arrowstyle="<-", color="purple",
                            lw=1.8, shrinkA=0, shrinkB=0))
        ax1.annotate('',
            xy    =(cvt(coil.P2[0]), cvt(coil.P2[1])),
            xytext=(cvt(coil.P2[2]), cvt(coil.P2[3])),
            arrowprops=dict(arrowstyle="<-", color="purple",
                            lw=1.8, shrinkA=0, shrinkB=0))

        ax1.set_title("Base Unit")
        ax1.set_xlabel(f"X ({ls})")
        ax1.set_ylabel(f"Y ({ls})")
        ax1.grid(True)
        ax1.set_aspect('equal')
        #st.pyplot(fig1, use_container_width=False)
        st.pyplot(fig1, width="content")
        plt.rcParams.update(plt.rcParamsDefault)  # 전역 설정 복원

        # ── 아래: Full Coil ───────────────────────────────────
        cc = coil.create_geom()
        xx_mm, yy_mm = cc.x, cc.y   # 항상 mm로 저장

        st.session_state["coil_x"] = xx_mm
        st.session_state["coil_y"] = yy_mm

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(cvt(xx_mm), cvt(yy_mm), 'r-')
        ax2.set_title(f"Full Coil (Turns: {ncoil})")
        ax2.set_xlabel(f"X ({ls})")
        ax2.set_ylabel(f"Y ({ls})")
        ax2.grid(True)
        ax2.set_aspect('equal')
        #st.pyplot(fig2, use_container_width=True)
        st.pyplot(fig2, width="stretch")


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
        st.markdown("### Display Units")

        u_col1, u_col2 = st.columns(2)

        with u_col1:
            sel_len2  = st.selectbox("Length Display Unit", list(LENGTH_UNITS.keys()),
                                     index=0, key="len_unit_tab2")
            lu2  = LENGTH_UNITS[sel_len2]
            ls2  = lu2["symbol"]
            sc2  = lu2["scale"]        # mm → 선택 단위
            to_mm2 = 1.0 / sc2         # 선택 단위 → mm

        with u_col2:
            sel_b    = st.selectbox("B-field Display Unit", list(BFIELD_UNITS.keys()), index=0)
            b_scale  = BFIELD_UNITS[sel_b]
            b_symbol = sel_b.split()[0]

        st.markdown("### Simulation Conditions")
        c1, c2, c3 = st.columns(3)

        with c1:
            current = st.number_input("Current [A]", value=2.0, step=0.5)
        with c2:
            # z_height 슬라이더도 선택 단위 적용 (0.05mm ~ 2mm 범위를 변환)
            z_height_u = st.slider(
                f"Observation Height [{ls2}]",
                float(0.05 * sc2), float(2.0 * sc2),
                float(0.1  * sc2), float(0.05 * sc2)
            )
            z_height_mm = z_height_u * to_mm2   # → mm (내부 계산용)
        with c3:
            grid_res = st.slider("Mesh Resolution", 40, 150, 60, 5)

        st.divider()

        with st.spinner("Solving Biot–Savart equation..."):
            X, Y, Bx, By, Bz, B_mag = analyze_magnetic_field(
                xx_input, yy_input, current, grid_res, z_height_mm
            )

        st.subheader("Magnetic Field Visualization")

        fig_main, ax_main = plt.subplots(figsize=(13, 6.5))

        # 플롯 좌표: mm → 선택 단위
        X_d = X * sc2
        Y_d = Y * sc2

        contour = ax_main.contourf(X_d, Y_d, B_mag * b_scale, levels=45, cmap='inferno')
        cbar = fig_main.colorbar(contour, ax=ax_main, orientation='horizontal')
        cbar.set_label(f"B-field magnitude [{b_symbol}]")

        ax_main.streamplot(X_d, Y_d, Bx, By, linewidth=0.8, density=1.1)
        ax_main.plot(xx_input * sc2, yy_input * sc2, color='green', linewidth=1.8)

        ax_main.set_xlabel(f"X ({ls2})")
        ax_main.set_ylabel(f"Y ({ls2})")
        ax_main.set_aspect('equal')
        ax_main.grid(True, linestyle=':', alpha=0.3)
        fig_main.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.22)
        #st.pyplot(fig_main, use_container_width=True)
        st.pyplot(fig_main, width="stretch")

        st.divider()

        metric_col, export_col = st.columns(2)

        with metric_col:
            st.subheader("Design Metrics")

            coil_w = (np.max(xx_input) - np.min(xx_input)) * sc2
            coil_h = (np.max(yy_input) - np.min(yy_input)) * sc2
            ar     = coil_h / coil_w if coil_w > 0 else 0
            wlen   = np.sum(np.sqrt(np.diff(xx_input)**2 + np.diff(yy_input)**2)) * sc2

            st.metric("Coil Width",          f"{coil_w:.4g} {ls2}")
            st.metric("Coil Height",         f"{coil_h:.4g} {ls2}")
            st.metric("Aspect Ratio (H/W)",  f"{ar:.3f}")
            st.metric("Wire Length",         f"{wlen:.4g} {ls2}")
            st.metric("Peak B-field",        f"{np.max(B_mag)  * b_scale:.3g} {b_symbol}")
            st.metric("Mean B-field",        f"{np.mean(B_mag) * b_scale:.3g} {b_symbol}")

        with export_col:
            st.subheader("Export")

            # CSV는 항상 mm 기준 저장 (원본 보존)
            csv_buf = io.StringIO()
            writer  = csv.writer(csv_buf)
            writer.writerow(["X_mm", "Y_mm"])
            for xv, yv in zip(xx_input, yy_input):
                writer.writerow([xv, yv])

            st.download_button("Download CSV (mm)",
                               csv_buf.getvalue().encode("utf-8"),
                               file_name="coil_geometry.csv", mime="text/csv")

            img_buf = io.BytesIO()
            fig_main.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
            st.download_button("Download PNG", img_buf.getvalue(),
                               file_name="field_map.png", mime="image/png")
