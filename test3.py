
import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION AND UI STYLING
# ============================================================

st.set_page_config(
    page_title="Footing Bearing Capacity Checker",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-size: 18px;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.3rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.025em;
    }

    h2 {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.35rem !important;
        font-weight: 650 !important;
    }

    p, label, .stMarkdown {
        font-size: 1.05rem !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 1.05rem !important;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        min-height: 2.8rem;
        font-size: 1.05rem;
        font-weight: 650;
        border-radius: 9px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Footing Breadth Optimisation and Bearing Capacity Checker")

G = 9.80665
GAMMA_W = 9.81

SOILS = {
    "Loose sand": "Drained",
    "Medium-dense sand": "Drained",
    "Dense sand": "Drained",
    "Sand with fines": "Drained",
    "Soft clay": "Undrained",
    "Stiff clay": "Undrained",
    "Silty soil": "Drained",
    "Custom soil": "Drained"
}


# ============================================================
# 2. UNIT WEIGHT CALCULATION
# ============================================================

def unit_weight_from_cylinder(
    sample_mass_kg,
    radius_m,
    height_m,
    water_content_pct
):
    volume = math.pi * radius_m**2 * height_m

    if volume <= 0 or sample_mass_kg <= 0:
        raise ValueError("Mass and cylinder dimensions must be positive.")

    gamma_bulk = sample_mass_kg * G / volume / 1000

    water_content = water_content_pct / 100
    gamma_dry = gamma_bulk / (1 + water_content)

    return volume, gamma_bulk, gamma_dry


# ============================================================
# 3. DIRECT SHEAR TEST REGRESSION
# ============================================================

def shear_test_regression(data):

    clean = data.dropna(
        subset=[
            "Normal stress (kPa)",
            "Peak shear stress (kPa)"
        ]
    )

    if len(clean) < 2:
        raise ValueError("At least two complete shear tests are required.")

    x = clean["Normal stress (kPa)"].to_numpy(dtype=float)
    y = clean["Peak shear stress (kPa)"].to_numpy(dtype=float)

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Stress values cannot be negative.")

    if len(np.unique(x)) < 2:
        raise ValueError("Use at least two different normal stresses.")

    slope, intercept = np.polyfit(x, y, 1)

    if slope < 0:
        raise ValueError("Regression produced a negative friction slope.")

    phi = math.degrees(math.atan(slope))
    cohesion = max(0.0, float(intercept))

    predicted = slope * x + intercept

    ss_res = np.sum((y - predicted)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot == 0 and ss_res == 0:
        r2 = 1.0
    elif ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - ss_res / ss_tot

    return cohesion, phi, slope, r2


# ============================================================
# 4. BEARING CAPACITY FACTORS
# ============================================================

def bearing_factors(phi_deg, method):

    if phi_deg < 0 or phi_deg >= 45:
        raise ValueError("Friction angle must be between 0° and 45°.")

    if abs(phi_deg) < 1e-9:
        return 5.14, 1.0, 0.0

    p = math.radians(phi_deg)

    nq = (
        math.exp(math.pi * math.tan(p))
        * math.tan(math.radians(45) + p / 2)**2
    )

    nc = (nq - 1) / math.tan(p)

    if method == "European Eurocode 7":
        ng = 2 * (nq - 1) * math.tan(p)
    else:
        ng = 2 * (nq + 1) * math.tan(p)

    return nc, nq, ng


# ============================================================
# 5. EFFECTIVE SURCHARGE AND UNIT WEIGHT
# ============================================================

def effective_surcharge(gamma_bulk, gamma_sat, gwt, df):

    gamma_sub = gamma_sat - GAMMA_W

    if gamma_sub <= 0:
        raise ValueError("Saturated unit weight must exceed 9.81 kN/m³.")

    if gwt >= df:
        return gamma_bulk * df

    return (
        gamma_bulk * gwt
        + gamma_sub * (df - gwt)
    )


def average_effective_gamma(
    gamma_bulk,
    gamma_sat,
    gwt,
    df,
    breadth
):

    gamma_sub = gamma_sat - GAMMA_W

    if gamma_sub <= 0:
        raise ValueError("Invalid saturated unit weight.")

    base = df
    bottom = df + breadth

    if gwt <= base:
        return gamma_sub

    if gwt >= bottom:
        return gamma_bulk

    dry = gwt - base
    submerged = bottom - gwt

    return (
        gamma_bulk * dry
        + gamma_sub * submerged
    ) / breadth


# ============================================================
# 6. FOOTING SHAPE FACTORS
# ============================================================

def shape_factors(method, b, l, phi, footing_type):

    if footing_type == "Strip":
        return 1.0, 1.0, 1.0

    ratio = b / l

    if method == "Indian Standards":

        if math.isclose(b, l, rel_tol=1e-6):
            return 1.3, 1.2, 0.8

        return (
            1 + 0.2 * ratio,
            1 + 0.2 * ratio,
            1 - 0.4 * ratio
        )

    if method == "European Eurocode 7":

        _, nq, _ = bearing_factors(phi, method)

        sq = 1 + ratio * math.sin(math.radians(phi))
        sg = 1 - 0.3 * ratio

        if abs(phi) < 1e-9:
            sc = 1 + 0.2 * ratio
        else:
            sc = (sq * nq - 1) / (nq - 1)

        return sc, sq, sg

    nc, nq, _ = bearing_factors(phi, method)

    if abs(phi) < 1e-9:
        sc = 1 + ratio / 5
        sq = 1.0
    else:
        sc = 1 + ratio * nq / nc
        sq = 1 + ratio * math.tan(math.radians(phi))

    return sc, sq, 1 - 0.4 * ratio


# ============================================================
# 7. GLOBAL FACTOR OF SAFETY CALCULATION
# ============================================================

def run_global_factor_check(
    method,
    b,
    l,
    load,
    df,
    gamma,
    gamma_sat,
    phi,
    c_eff,
    cu,
    condition,
    gwt,
    fos,
    footing_type
):

    q = effective_surcharge(gamma, gamma_sat, gwt, df)

    gamma_eff = average_effective_gamma(
        gamma, gamma_sat, gwt, df, b
    )

    sc, sq, sg = shape_factors(
        method, b, l, phi, footing_type
    )

    nc, nq, ng = bearing_factors(phi, method)

    if condition == "Undrained":

        nc, nq, ng = 5.14, 1.0, 0.0

        qnu = cu * nc * sc

    else:

        nphi = (
            math.tan(math.radians(45 + phi / 2))**2
            if phi > 0 else 1.0
        )

        dc = 1 + 0.2 * (df / b) * math.sqrt(nphi)

        dq = (
            1 + 0.1 * (df / b) * nphi
            if phi > 10 else 1.0
        )

        dg = dq

        qnu = (
            c_eff * nc * sc * dc
            + q * (nq - 1) * sq * dq
            + 0.5 * gamma_eff * b * ng * sg * dg
        )

    area = b * l

    gross_ultimate = qnu + q
    qgross = load / area
    qnet = qgross - q

    qsafe = qnu / fos

    return {
        "Footing type": footing_type,
        "Breadth (m)": b,
        "Length (m)": l,
        "Area (m²)": area,
        "Surcharge (kPa)": q,
        "Gross applied pressure (kPa)": qgross,
        "Net applied pressure (kPa)": qnet,
        "Gross ultimate capacity (kPa)": gross_ultimate,
        "Net ultimate capacity (kPa)": qnu,
        "Net safe bearing capacity (kPa)": qsafe,
        "Nc": nc,
        "Nq": nq,
        "Nγ": ng,
        "Factor of safety": fos,
        "Result": "PASS" if qnet <= qsafe else "FAIL"
    }


# ============================================================
# 8. EUROCODE-STYLE DESIGN RESISTANCE SCREENING
# ============================================================

def run_eurocode_screen(
    b,
    l,
    load,
    df,
    gamma,
    gamma_sat,
    phi_k,
    c_k,
    cu,
    condition,
    gwt,
    gamma_phi,
    gamma_c,
    gamma_R,
    footing_type
):

    q = effective_surcharge(gamma, gamma_sat, gwt, df)

    gamma_eff = average_effective_gamma(
        gamma, gamma_sat, gwt, df, b
    )

    if condition == "Undrained":

        cu_d = cu / gamma_c

        sc, sq, sg = shape_factors(
            "European Eurocode 7", b, l, 0, footing_type
        )

        # Undrained gross resistance, including surcharge.
        resistance_before_R = (
            q + (math.pi + 2) * cu_d * sc
        )

        nc, nq, ng = 5.14, 1.0, 0.0

    else:

        phi_d = math.degrees(
            math.atan(
                math.tan(math.radians(phi_k)) / gamma_phi
            )
        )

        c_d = c_k / gamma_c

        nc, nq, ng = bearing_factors(
            phi_d, "European Eurocode 7"
        )

        sc, sq, sg = shape_factors(
            "European Eurocode 7",
            b,
            l,
            phi_d,
            footing_type
        )

        resistance_before_R = (
            c_d * nc * sc
            + q * nq * sq
            + 0.5 * gamma_eff * b * ng * sg
        )

    resistance_d = resistance_before_R / gamma_R

    area = b * l

    applied_gross = load / area
    applied_net = applied_gross - q

    net_ultimate = resistance_before_R - q
    net_design_resistance = resistance_d - q

    return {
        "Footing type": footing_type,
        "Breadth (m)": b,
        "Length (m)": l,
        "Area (m²)": area,
        "Surcharge (kPa)": q,
        "Gross applied pressure (kPa)": applied_gross,
        "Net applied pressure (kPa)": applied_net,
        "Gross ultimate capacity (kPa)": resistance_before_R,
        "Net ultimate capacity (kPa)": net_ultimate,
        "Net safe bearing capacity (kPa)": net_design_resistance,
        "Design resistance (kPa)": resistance_d,
        "Resistance before γR (kPa)": resistance_before_R,
        "Nc": nc,
        "Nq": nq,
        "Nγ": ng,
        "Result": (
            "PASS"
            if applied_gross <= resistance_d
            else "FAIL"
        )
    }


# ============================================================
# 9. SOIL UNIT WEIGHT FROM CYLINDRICAL SAMPLE
# ============================================================

st.header("1. Soil unit weight")

st.write(
    "Enter the measured sample mass and cylinder dimensions."
)

c1, c2 = st.columns(2)

with c1:

    sample_mass = st.number_input(
        "Mass of soil sample (kg)",
        min_value=0.001,
        value=1.44,
        step=0.01
    )

    radius = st.number_input(
        "Internal radius of cylinder (m)",
        min_value=0.001,
        value=0.0500,
        step=0.005,
        format="%.4f"
    )

with c2:

    height = st.number_input(
        "Height of soil in cylinder (m)",
        min_value=0.001,
        value=0.1000,
        step=0.005,
        format="%.4f"
    )

    water_content = st.number_input(
        "Water content, w (%)",
        min_value=0.0,
        max_value=200.0,
        value=8.80,
        step=0.1
    )

volume, gamma_bulk, gamma_dry = unit_weight_from_cylinder(
    sample_mass,
    radius,
    height,
    water_content
)

st.markdown("**Calculated properties**")

co1, co2, co3 = st.columns(3)

with co1:
    st.metric("Sample volume", f"{volume:.6f} m³")

with co2:
    st.metric("Bulk unit weight, γ", f"{gamma_bulk:.2f} kN/m³")

with co3:
    st.metric("Dry unit weight, γd", f"{gamma_dry:.2f} kN/m³")

st.caption(
    "Use the internal cylinder radius and actual soil-filled height."
)


# ============================================================
# 10. DIRECT SHEAR TEST DATA
# ============================================================

st.divider()

st.header("2. Direct shear test data")

st.write(
    "For drained/effective-stress interpretation, the app fits "
    "the Mohr–Coulomb failure envelope: "
    "τpeak = c′ + σ′n tan φ′. "
    "Enter results from multiple tests at different normal stresses."
)

default_tests = pd.DataFrame({
    "Normal stress (kPa)": [50.0, 100.0, 150.0],
    "Peak shear force (N)": [103.923, 207.846, 311.765],
    "Shear area (m²)": [0.0036, 0.0036, 0.0036]
})

tests = st.data_editor(
    default_tests,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Normal stress (kPa)": st.column_config.NumberColumn(
            min_value=0.0,
            format="%.3f"
        ),
        "Peak shear force (N)": st.column_config.NumberColumn(
            min_value=0.0,
            format="%.3f"
        ),
        "Shear area (m²)": st.column_config.NumberColumn(
            min_value=0.000001,
            format="%.4f"
        )
    }
)

tests_used = tests.copy()

tests_used["Peak shear stress (kPa)"] = (
    tests_used["Peak shear force (N)"]
    / tests_used["Shear area (m²)"]
    / 1000
)

st.markdown("**Calculated peak shear stresses**")

st.dataframe(
    tests_used,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Normal stress (kPa)": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "Peak shear force (N)": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "Shear area (m²)": st.column_config.NumberColumn(
            format="%.4f"
        ),
        "Peak shear stress (kPa)": st.column_config.NumberColumn(
            format="%.3f"
        )
    }
)

try:

    c_calc, phi_calc, slope, r2 = shear_test_regression(
        tests_used
    )

    st.success(
        f"Regression results: c′ = {c_calc:.2f} kPa | "
        f"φ′ = {phi_calc:.2f}° | R² = {r2:.4f}"
    )

except ValueError as e:

    c_calc, phi_calc, r2 = 0.0, 30.0, None

    st.warning(f"Effective strength not calculated: {e}")


# ============================================================
# 11. UNDRAINED SHEAR STRENGTH
# ============================================================

st.divider()

st.header("3. Undrained shear strength")

st.write(
    "Enter cu from an appropriate undrained test. "
    "Do not automatically treat drained direct-shear results "
    "as undrained strength."
)

cu_source = st.radio(
    "Source for cu",
    [
        "Enter cu measured from UU triaxial / vane / other suitable undrained test",
        "Estimate cu from undrained direct-shear tests assuming φu = 0"
    ]
)

if cu_source.startswith("Enter"):

    cu_input = st.number_input(
        "Measured undrained shear strength cu (kPa)",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

else:

    st.warning(
        "Use this approximation only when the tests were actually "
        "conducted under appropriate undrained conditions and φu = 0 "
        "is justified. Do not use drained effective-stress test results."
    )

    cu_input = float(
        tests_used["Peak shear stress (kPa)"].mean()
    )

    st.metric(
        "Estimated cu",
        f"{cu_input:.2f} kPa"
    )


# ============================================================
# 12. SOIL CONDITION AND DESIGN FRAMEWORK
# ============================================================

st.divider()

st.header("4. Soil condition and design framework")

soil_type = st.selectbox(
    "Soil description",
    list(SOILS.keys()),
    index=1
)

condition = st.selectbox(
    "Analysis condition",
    ["Drained", "Undrained"],
    index=0 if SOILS[soil_type] == "Drained" else 1
)

if condition == "Drained":

    phi_use = phi_calc
    c_use = c_calc

    st.info(
        f"Using test-derived effective parameters: "
        f"φ′ = {phi_use:.2f}° | c′ = {c_use:.2f} kPa."
    )

else:

    phi_use = 0.0
    c_use = 0.0

    st.info(
        f"Using cu = {cu_input:.2f} kPa and φu = 0°."
    )

method = st.selectbox(
    "Calculation framework",
    [
        "Indian Standards",
        "European Eurocode 7",
        "American practice"
    ]
)


# ============================================================
# 13. FOOTING TYPE
# ============================================================

st.divider()

st.header("5. Footing type and geometry")

footing_type = st.radio(
    "Select footing type",
    ["Square", "Rectangular", "Strip"],
    horizontal=True
)

if footing_type == "Square":

    st.info(
        "Square footing selected: L = B automatically."
    )

elif footing_type == "Rectangular":

    st.info(
        "Enter length and breadth independently."
    )

else:

    st.info(
        "Strip footing is treated as infinitely long. "
        "The applied load will be entered per metre run (kN/m)."
    )


# ============================================================
# 14. FOUNDATION AND GROUNDWATER INPUTS
# ============================================================

st.divider()

st.header("6. Foundation and groundwater inputs")

# All foundation inputs are arranged in one horizontal row.

a, bcol, ccol, dcol = st.columns(4)

with a:

    if footing_type == "Rectangular":

        length = st.number_input(
            "Footing length L (m)",
            min_value=0.1,
            value=1.50,
            step=0.25
        )

    elif footing_type == "Square":

        st.metric("Footing length L", "L = B")

    else:

        st.metric("Footing length L", "Infinite")

with bcol:

    df = st.number_input(
        "Foundation depth Df (m)",
        min_value=0.0,
        value=1.50,
        step=0.25
    )

with ccol:

    gwt = st.number_input(
        "Groundwater depth (m)",
        min_value=0.0,
        value=3.00,
        step=0.25
    )

with dcol:

    gamma_sat = st.number_input(
        "Saturated unit weight γsat (kN/m³)",
        min_value=9.82,
        max_value=35.0,
        value=20.0,
        step=0.5
    )


# ============================================================
# 15. FACTOR OF SAFETY / PARTIAL FACTORS
# ============================================================

st.divider()

if method == "European Eurocode 7":

    st.header("7. Eurocode partial factors")

    f1, f2, f3 = st.columns(3)

    with f1:

        gamma_phi = st.number_input(
            "Partial factor γφ",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.05
        )

    with f2:

        gamma_c = st.number_input(
            "Partial factor γc",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.05
        )

    with f3:

        gamma_R = st.number_input(
            "Resistance factor γR",
            min_value=1.0,
            max_value=2.0,
            value=1.0,
            step=0.05
        )

else:

    st.header("7. Factor of safety")

    fos = st.number_input(
        "Global factor of safety",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.25
    )

st.caption(
    f"Bulk unit weight used in the calculation is derived from "
    f"the cylinder: γ = {gamma_bulk:.3f} kN/m³."
)


# ============================================================
# 16. APPLIED LOAD
# ============================================================

st.divider()

st.header("8. Applied Load")

# Vertical load gets its own separate full-width line.

if footing_type == "Strip":

    load_label = (
        "Vertical design action Vd (kN/m)"
        if method == "European Eurocode 7"
        else "Vertical service load P (kN/m)"
    )

else:

    load_label = (
        "Vertical design action Vd (kN)"
        if method == "European Eurocode 7"
        else "Vertical service load P (kN)"
    )

load = st.number_input(
    load_label,
    min_value=0.1,
    value=900.0,
    step=50.0
)


# ============================================================
# 17. TRIAL FOOTING BREADTHS
# ============================================================

st.divider()

st.header("9. Trial footing breadths")

analysis = st.radio(
    "Analysis type",
    ["Single breadth", "Multiple trial breadths"],
    horizontal=True
)

if analysis == "Single breadth":

    breadth = st.number_input(
        "Footing breadth B (m)",
        min_value=0.1,
        value=1.50,
        step=0.25
    )

    breadths = [breadth]

else:

    x1, x2, x3 = st.columns(3)

    with x1:

        b_start = st.number_input(
            "Starting B (m)",
            min_value=0.1,
            value=1.50,
            step=0.25
        )

    with x2:

        b_end = st.number_input(
            "Ending B (m)",
            min_value=0.1,
            value=4.00,
            step=0.25
        )

    with x3:

        b_step = st.number_input(
            "Increment (m)",
            min_value=0.01,
            value=0.50,
            step=0.1
        )

    if b_end < b_start:

        st.error(
            "Ending breadth must be greater than starting breadth."
        )

        st.stop()

    breadths = []

    current = b_start

    while current <= b_end + 1e-9:

        breadths.append(round(current, 4))
        current += b_step


# ============================================================
# 18. RUN BEARING CAPACITY CHECK
# ============================================================

st.divider()

if st.button(
    "Run bearing-capacity check",
    type="primary",
    use_container_width=True
):

    if condition == "Drained":

        valid_tests = tests_used.dropna(
            subset=[
                "Normal stress (kPa)",
                "Peak shear stress (kPa)"
            ]
        )

        if len(valid_tests) < 2:

            st.error(
                "At least two complete direct-shear tests are required."
            )

            st.stop()

    if condition == "Undrained" and cu_input <= 0:

        st.error(
            "A positive and justified undrained shear strength cu is required."
        )

        st.stop()

    if gamma_sat <= GAMMA_W:

        st.error(
            "Saturated unit weight must exceed 9.81 kN/m³."
        )

        st.stop()

    if method != "European Eurocode 7" and fos <= 0:

        st.error("Factor of safety must be positive.")
        st.stop()

    output = []

    try:

        for breadth in breadths:

            # Geometry selection.

            if footing_type == "Square":

                effective_length = breadth

            elif footing_type == "Rectangular":

                effective_length = length

            else:

                effective_length = 1.0

            if footing_type == "Rectangular" and breadth > effective_length:

                st.warning(
                    "Breadth exceeds length. Check the rectangular footing "
                    "orientation and dimensions."
                )

            if method == "European Eurocode 7":

                row = run_eurocode_screen(
                    breadth,
                    effective_length,
                    load,
                    df,
                    gamma_bulk,
                    gamma_sat,
                    phi_use,
                    c_use,
                    cu_input,
                    condition,
                    gwt,
                    gamma_phi,
                    gamma_c,
                    gamma_R,
                    footing_type
                )

            else:

                row = run_global_factor_check(
                    method,
                    breadth,
                    effective_length,
                    load,
                    df,
                    gamma_bulk,
                    gamma_sat,
                    phi_use,
                    c_use,
                    cu_input,
                    condition,
                    gwt,
                    fos,
                    footing_type
                )

            output.append(row)

        results = pd.DataFrame(output)

        # ====================================================
        # 19. FINAL RESULTS
        # ====================================================

        st.divider()

        st.header("10. Bearing-capacity results")

        st.caption(
            f"{footing_type} footing | "
            f"{len(results)} trial breadth(s)"
        )

        for _, result in results.iterrows():

            with st.container(border=True):

                if footing_type == "Square":

                    title = (
                        f"Square footing: "
                        f"B = L = {result['Breadth (m)']:.2f} m"
                    )

                elif footing_type == "Rectangular":

                    title = (
                        f"Rectangular footing: "
                        f"B = {result['Breadth (m)']:.2f} m "
                        f"| L = {result['Length (m)']:.2f} m"
                    )

                else:

                    title = (
                        f"Strip footing: "
                        f"B = {result['Breadth (m)']:.2f} m"
                    )

                st.markdown(f"### {title}")

                # First horizontal row.

                r1, r2, r3 = st.columns(3)

                with r1:

                    st.metric(
                        "Gross applied pressure",
                        f"{result['Gross applied pressure (kPa)']:.2f} kPa"
                    )

                with r2:

                    st.metric(
                        "Net applied pressure",
                        f"{result['Net applied pressure (kPa)']:.2f} kPa"
                    )

                with r3:

                    st.metric(
                        "Net ultimate capacity",
                        f"{result['Net ultimate capacity (kPa)']:.2f} kPa"
                    )

                # Second horizontal row.

                r4, r5 = st.columns(2)

                with r4:

                    st.metric(
                        "Net safe bearing capacity",
                        f"{result['Net safe bearing capacity (kPa)']:.2f} kPa"
                    )

                with r5:

                    if result["Result"] == "PASS":

                        st.success("PASS")

                    else:

                        st.error("FAIL")



                # Additional calculation details.

                with st.expander(
                    "Calculation factors and details"
                ):

                    d1, d2, d3 = st.columns(3)

                    with d1:
                        st.metric("Nc", f"{result['Nc']:.3f}")

                    with d2:
                        st.metric("Nq", f"{result['Nq']:.3f}")

                    with d3:
                        st.metric("Nγ", f"{result['Nγ']:.3f}")

                    if method == "European Eurocode 7":

                        st.metric(
                            "Resistance before γR",
                            f"{result['Resistance before γR (kPa)']:.2f} kPa"
                        )

        # ====================================================
        # 20. OPTIMUM BREADTH
        # ====================================================

        st.divider()

        st.header("11. Optimum footing breadth")

        passed = results[
            results["Result"] == "PASS"
        ]

        if len(passed) > 0:

            optimum = passed.sort_values(
                "Breadth (m)"
            ).iloc[0]

            st.success(
                f"Smallest passing trial breadth: "
                f"{optimum['Breadth (m)']:.2f} m"
            )

            if footing_type == "Square":

                st.write(
                    f"Trial dimensions: "
                    f"{optimum['Breadth (m)']:.2f} m × "
                    f"{optimum['Breadth (m)']:.2f} m"
                )

            elif footing_type == "Rectangular":

                st.write(
                    f"Trial dimensions: "
                    f"{optimum['Breadth (m)']:.2f} m × "
                    f"{optimum['Length (m)']:.2f} m"
                )

            else:

                st.write(
                    f"Trial strip breadth: "
                    f"{optimum['Breadth (m)']:.2f} m"
                )

        else:

            st.error(
                "None of the selected trial breadths passed "
                "the bearing-capacity check."
            )

        # ====================================================
        # 21. DOWNLOAD RESULTS
        # ====================================================

        st.divider()

        st.header("12. Download results")

        d1, d2 = st.columns(2)

        with d1:

            st.download_button(
                "Download bearing-capacity results",
                results.to_csv(index=False).encode("utf-8"),
                "bearing_capacity_results.csv",
                "text/csv",
                use_container_width=True
            )

        with d2:

            st.download_button(
                "Download direct-shear data",
                tests_used.to_csv(index=False).encode("utf-8"),
                "direct_shear_results.csv",
                "text/csv",
                use_container_width=True
            )

    except ValueError as err:

        st.error(str(err))


# ============================================================
# 22. EQUATIONS AND INTERPRETATION
# ============================================================

st.divider()

with st.expander("Equations and interpretation"):

    st.markdown(r"""
### Unit weight

\[
V=\pi r^2h
\]

\[
\gamma=\frac{mg}{V}
\]

\[
\gamma_d=\frac{\gamma}{1+w}
\]

### Mohr–Coulomb failure envelope

\[
\tau_f=c'+\sigma'_n\tan\phi'
\]

### General bearing-capacity equation

\[
q_{ult}=
c'N_cs_cd_c+
qN_qs_qd_q+
\frac{1}{2}\gamma BN_\gamma s_\gamma d_\gamma
\]

### Net ultimate bearing capacity

\[
q_{nu}=q_{ult}-q
\]

### Net safe bearing capacity

\[
q_{ns}=\frac{q_{nu}}{FOS}
\]

### Applied pressure

For square and rectangular footings:

\[
q_{gross}=\frac{P}{BL}
\]

For strip footings:

\[
q_{gross}=\frac{P}{B}
\]

where strip footing load P is in kN/m.
""")

st.warning(
    "IMPORTANT: This application provides preliminary academic "
    "bearing-capacity screening. The simplified calculation methods "
    "are not complete verified implementations of IS 6403, Eurocode 7 "
    "or American foundation-design standards. Confirm the applicable "
    "standard, design approach, groundwater corrections, shape and "
    "depth factors, eccentricity, inclination, settlement and other "
    "geotechnical requirements before using results for construction."
)
