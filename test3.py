

import math

import numpy as np

import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt



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



# ============================================================
# MAIN MENU / FULL-PAGE NAVIGATION
# ============================================================
PAGES = ["Bearing Capacity", "Settlement", "Combined Graph"]
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

if st.session_state.active_page == "Home":
    st.title("Footing Breadth Optimisation and Bearing Capacity Checker")
    st.write("Select a module to open it as a full page.")
    nav_col1, nav_col2, nav_col3 = st.columns(3, gap="large")
    with nav_col1:
        with st.container(border=True):
            st.subheader("1. Bearing Capacity")
            st.caption("Inputs, calculations, checks and results")
            if st.button("Open Bearing Capacity", key="open_bearing", use_container_width=True):
                st.session_state.active_page = "Bearing Capacity"
                st.rerun()
    with nav_col2:
        with st.container(border=True):
            st.subheader("2. Settlement")
            st.caption("Immediate, primary and secondary settlement")
            if st.button("Open Settlement", key="open_settlement", use_container_width=True):
                st.session_state.active_page = "Settlement"
                st.rerun()
    with nav_col3:
        with st.container(border=True):
            st.subheader("3. Combined Graph")
            st.caption("Bearing capacity and settlement vs footing breadth")
            if st.button("Open Combined Graph", key="open_graph", use_container_width=True):
                st.session_state.active_page = "Combined Graph"
                st.rerun()
else:
    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.title(st.session_state.active_page)
    with top_right:
        if st.button("← Main Menu", key="back_to_menu", use_container_width=True):
            st.session_state.active_page = "Home"
            st.rerun()



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



if st.session_state.active_page == "Bearing Capacity":
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

        width='stretch',

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

        width='stretch',

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

        width='stretch'

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
            st.session_state["bearing_results"] = results.copy()



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

                    width='stretch'

                )



            with d2:



                st.download_button(

                    "Download direct-shear data",

                    tests_used.to_csv(index=False).encode("utf-8"),

                    "direct_shear_results.csv",

                    "text/csv",

                    width='stretch'

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



    \\[

    V=\pi r^2h

    \\]



    \\[

    \gamma=\frac{mg}{V}

    \\]



    \\[

    \gamma_d=\frac{\gamma}{1+w}

    \\]



    ### Mohr–Coulomb failure envelope



    \\[

    \tau_f=c'+\sigma'_n\tan\phi'

    \\]



    ### General bearing-capacity equation



    \\[

    q_{ult}=

    c'N_cs_cd_c+

    qN_qs_qd_q+

    \frac{1}{2}\gamma BN_\gamma s_\gamma d_\gamma

    \\]



    ### Net ultimate bearing capacity



    \\[

    q_{nu}=q_{ult}-q

    \\]



    ### Net safe bearing capacity



    \\[

    q_{ns}=\frac{q_{nu}}{FOS}

    \\]



    ### Applied pressure



    For square and rectangular footings:



    \\[

    q_{gross}=\frac{P}{BL}

    \\]



    For strip footings:



    \\[

    q_{gross}=\frac{P}{B}

    \\]



    where strip footing load P is in kN/m.

    """)





    # ============================================================



if st.session_state.active_page == "Settlement":
    st.title("FOUNDATION SETTLEMENT CALCULATOR")

    st.subheader(

        "Immediate, Primary Consolidation and Secondary Settlement"

    )







    # =========================================================

    # 1. SOIL PRESETS

    # =========================================================



    SOILS = {

        "Loose sand": {

            "gamma": 17.0, "Es": 10000.0, "nu": 0.30,

            "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,

            "e0": 0.65, "ep": 0.65, "sigma_pc": 100.0

        },

        "Medium-dense sand": {

            "gamma": 18.0, "Es": 25000.0, "nu": 0.30,

            "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,

            "e0": 0.60, "ep": 0.60, "sigma_pc": 100.0

        },

        "Dense sand": {

            "gamma": 19.0, "Es": 50000.0, "nu": 0.28,

            "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,

            "e0": 0.50, "ep": 0.50, "sigma_pc": 100.0

        },

        "Baseline sand with fines": {

            "gamma": 18.0, "Es": 20000.0, "nu": 0.30,

            "Cc": 0.15, "Cr": 0.03, "C_alpha": 0.003,

            "e0": 0.65, "ep": 0.60, "sigma_pc": 100.0

        },

        "Soft clay": {

            "gamma": 16.5, "Es": 4000.0, "nu": 0.45,

            "Cc": 0.35, "Cr": 0.07, "C_alpha": 0.025,

            "e0": 1.00, "ep": 0.90, "sigma_pc": 50.0

        },

        "Stiff clay": {

            "gamma": 19.0, "Es": 30000.0, "nu": 0.35,

            "Cc": 0.20, "Cr": 0.04, "C_alpha": 0.010,

            "e0": 0.65, "ep": 0.60, "sigma_pc": 200.0

        },

        "Silty soil": {

            "gamma": 18.0, "Es": 12000.0, "nu": 0.32,

            "Cc": 0.12, "Cr": 0.025, "C_alpha": 0.006,

            "e0": 0.70, "ep": 0.65, "sigma_pc": 100.0

        },

        "Layered soil": {

            "gamma": 17.0, "Es": 10000.0, "nu": 0.30,

            "Cc": 0.35, "Cr": 0.07, "C_alpha": 0.025,

            "e0": 1.00, "ep": 0.90, "sigma_pc": 50.0

        }

    }



    # =========================================================

    # 2. CALCULATION FUNCTIONS

    # =========================================================



    def calculate_net_pressure(P, B, L, gamma, Df):

        area = B * L

        q_gross = P / area

        q_surcharge = gamma * Df

        q_net = max(0.0, q_gross - q_surcharge)

        return q_gross, q_net





    def calculate_immediate(q_net, B, Es, nu, Is):

        """Simplified elastic immediate settlement. Output in metres."""

        return q_net * B * (1 - nu**2) * Is / Es





    def calculate_stress_increase(q_net, B, L, z):

        """Approximate 2:1 stress distribution. Output in kPa."""

        return q_net * B * L / ((B + z) * (L + z))





    def calculate_compression_index(e1, e2, sigma1, sigma2):

        """

        Calculate compression index from two e-log effective stress readings.

        Returns None for invalid inputs.

        """

        if sigma1 <= 0 or sigma2 <= 0 or sigma2 <= sigma1:

            return None



        if e1 < 0 or e2 < 0 or e1 < e2:

            return None



        denominator = np.log10(sigma2 / sigma1)



        if denominator <= 0:

            return None



        return (e1 - e2) / denominator





    def calculate_primary(

        H, sigma0, delta_sigma, sigma_pc, Cc, Cr, e0

    ):

        """

        One-dimensional e-log effective stress calculation.



        NC: virgin compression index Cc.

        OC: recompression index Cr below sigma_pc,

            compression index Cc above sigma_pc.



        Output in metres.

        """



        if H <= 0 or sigma0 <= 0 or e0 <= -1:

            return 0.0



        final_stress = sigma0 + delta_sigma



        if final_stress <= sigma0:

            return 0.0



        if sigma_pc <= sigma0:

            settlement = (

                Cc * H / (1 + e0)

                * np.log10(final_stress / sigma0)

            )



        elif final_stress <= sigma_pc:

            settlement = (

                Cr * H / (1 + e0)

                * np.log10(final_stress / sigma0)

            )



        else:

            S_recompression = (

                Cr * H / (1 + e0)

                * np.log10(sigma_pc / sigma0)

            )



            e_pc = (

                e0 - Cr * np.log10(sigma_pc / sigma0)

            )



            if e_pc <= -1:

                raise ValueError(

                    "Calculated void ratio at preconsolidation "

                    "stress is invalid. Check input parameters."

                )



            S_virgin = (

                Cc * H / (1 + e_pc)

                * np.log10(final_stress / sigma_pc)

            )



            settlement = S_recompression + S_virgin



        return max(0.0, settlement)





    def calculate_secondary(H, C_alpha, ep, t1, t2):

        """Secondary compression settlement. Output in metres."""



        if H <= 0 or C_alpha <= 0 or ep <= -1:

            return 0.0



        if t1 <= 0 or t2 <= t1:

            return 0.0



        return (

            C_alpha * H / (1 + ep)

            * np.log10(t2 / t1)

        )





    def calculate_layer_settlement(

        layers, q_net, B, L, t1, t2,

        use_primary=True, use_secondary=True

    ):

        """Sum primary and secondary settlement across layers."""



        primary_total = 0.0

        secondary_total = 0.0

        layer_results = []



        for _, layer in layers.iterrows():



            H = float(layer["Thickness H (m)"])

            z = float(layer["Centre depth z (m)"])



            sigma0 = float(

                layer["Initial effective stress (kPa)"]

            )



            sigma_pc = float(

                layer["Preconsolidation stress (kPa)"]

            )



            Cc = float(layer["Cc"])

            Cr = float(layer["Cr"])

            C_alpha = float(layer["C_alpha"])

            e0 = float(layer["Initial void ratio e0"])

            ep = float(layer["End-primary void ratio ep"])



            delta_sigma = calculate_stress_increase(

                q_net, B, L, z

            )



            Sc = 0.0

            Ss = 0.0



            if use_primary and bool(layer["Include primary"]):

                Sc = calculate_primary(

                    H, sigma0, delta_sigma,

                    sigma_pc, Cc, Cr, e0

                )



            if use_secondary and bool(layer["Include secondary"]):

                Ss = calculate_secondary(

                    H, C_alpha, ep, t1, t2

                )



            primary_total += Sc

            secondary_total += Ss



            layer_results.append({

                "Layer": layer["Layer"],

                "Stress increase (kPa)": delta_sigma,

                "Cc": Cc,

                "Cr": Cr,

                "Primary settlement (mm)": Sc * 1000,

                "Secondary settlement (mm)": Ss * 1000

            })



        return primary_total, secondary_total, layer_results





    # =========================================================

    # 3. FOUNDATION INPUTS

    # =========================================================



    st.header("SECTION A - FOUNDATION DETAILS")



    c1, c2, c3 = st.columns(3)



    with c1:

        P = st.number_input(

            "Service vertical load P (kN)",

            min_value=1.0,

            value=900.0

        )



        Df = st.number_input(

            "Foundation depth Df (m)",

            min_value=0.0,

            value=1.5

        )



    with c2:

        L = st.number_input(

            "Footing length L (m)",

            min_value=0.1,

            value=2.0

        )



        settlement_limit = st.number_input(

            "Allowable total settlement (mm)",

            min_value=0.1,

            value=25.0

        )



    with c3:

        B_start = st.number_input(

            "Minimum breadth (m)",

            min_value=0.1,

            value=1.5

        )



        B_end = st.number_input(

            "Maximum breadth (m)",

            min_value=0.2,

            value=4.0

        )



        B_step = st.number_input(

            "Breadth increment (m)",

            min_value=0.05,

            value=0.5

        )





    # =========================================================

    # 4. SOIL INPUTS

    # =========================================================



    st.header("SECTION B - SOIL PARAMETERS")



    soil_name = st.selectbox(

        "Select soil scenario",

        list(SOILS.keys())

    )



    preset = SOILS[soil_name]



    c1, c2, c3 = st.columns(3)



    with c1:

        gamma = st.number_input(

            "Bulk unit weight γ (kN/m³)",

            min_value=1.0,

            value=float(preset["gamma"])

        )



        Es = st.number_input(

            "Elastic modulus Es (kPa)",

            min_value=1.0,

            value=float(preset["Es"])

        )



    with c2:

        nu = st.number_input(

            "Poisson's ratio ν",

            min_value=0.0,

            max_value=0.49,

            value=float(preset["nu"])

        )



        Is = st.number_input(

            "Immediate settlement influence factor Is",

            min_value=0.01,

            value=1.12

        )



    with c3:

        water_table = st.number_input(

            "Water table depth below ground (m)",

            min_value=0.0,

            value=3.0

        )



        gamma_w = st.number_input(

            "Water unit weight (kN/m³)",

            min_value=1.0,

            value=9.81

        )



    st.caption(

        "The water-table depth is recorded but does not automatically "

        "calculate effective stresses. Enter groundwater-consistent "

        "initial effective stresses for each consolidation layer."

    )





    # =========================================================

    # 5. SOIL LAYER INPUTS

    # =========================================================



    st.header("SECTION C - COMPRESSIBLE SOIL LAYERS")



    st.write(

        "Depth to layer centre is measured below the footing base. "

        "Enter each layer's properties separately."

    )



    if soil_name == "Layered soil":



        default_layers = pd.DataFrame([

            {

                "Layer": "Upper loose sand",

                "Thickness H (m)": 0.5,

                "Centre depth z (m)": 0.25,

                "Initial effective stress (kPa)": 25.0,

                "Preconsolidation stress (kPa)": 100.0,

                "C_alpha": 0.0,

                "Initial void ratio e0": 0.65,

                "End-primary void ratio ep": 0.65,

                "Include primary": False,

                "Include secondary": False

            },

            {

                "Layer": "Lower soft clay",

                "Thickness H (m)": 3.0,

                "Centre depth z (m)": 2.0,

                "Initial effective stress (kPa)": 45.0,

                "Preconsolidation stress (kPa)": 60.0,

                "C_alpha": 0.025,

                "Initial void ratio e0": 1.0,

                "End-primary void ratio ep": 0.90,

                "Include primary": True,

                "Include secondary": True

            }

        ])



    else:



        default_layers = pd.DataFrame([

            {

                "Layer": "Layer 1",

                "Thickness H (m)": 3.0,

                "Centre depth z (m)": 1.5,

                "Initial effective stress (kPa)": 50.0,

                "Preconsolidation stress (kPa)": preset["sigma_pc"],

                "C_alpha": preset["C_alpha"],

                "Initial void ratio e0": preset["e0"],

                "End-primary void ratio ep": preset["ep"],

                "Include primary": preset["Cc"] > 0,

                "Include secondary": preset["C_alpha"] > 0

            }

        ])



    edited_layers = st.data_editor(

        default_layers,

        num_rows="dynamic",

        use_container_width=True,

        key="layers_editor"

    )



    layers = edited_layers.copy()





    # =========================================================

    # 6. Cc AND Cr LABORATORY INPUTS

    # =========================================================



    st.subheader("Cc and Cr Calculation from Consolidation Test")





    def make_test_defaults(layer_df, soil_preset):

        """Create illustrative editable test readings for each layer."""



        rows = []



        for _, layer in layer_df.iterrows():



            e0 = float(layer["Initial void ratio e0"])

            pc = float(layer["Preconsolidation stress (kPa)"])



            name = str(layer["Layer"])



            if "sand" in name.lower():

                cc_default = 0.0

                cr_default = 0.0

            else:

                cc_default = float(soil_preset["Cc"])

                cr_default = float(soil_preset["Cr"])



            # Virgin compression readings: both stresses above sigma_pc.

            cc_s1 = pc * 1.2

            cc_s2 = pc * 12.0



            cc_e1 = e0

            cc_e2 = e0 - cc_default * np.log10(cc_s2 / cc_s1)



            # Recompression readings: both stresses below sigma_pc.

            cr_s1 = pc * 0.1

            cr_s2 = pc * 0.8



            cr_e1 = e0

            cr_e2 = e0 - cr_default * np.log10(cr_s2 / cr_s1)



            rows.append({

                "Layer": name,

                "Cc e1": cc_e1,

                "Cc e2": cc_e2,

                "Cc σ1' (kPa)": cc_s1,

                "Cc σ2' (kPa)": cc_s2,

                "Cr e1": cr_e1,

                "Cr e2": cr_e2,

                "Cr σ1' (kPa)": cr_s1,

                "Cr σ2' (kPa)": cr_s2

            })



        return pd.DataFrame(rows)





    test_defaults = make_test_defaults(layers, preset)



    test_inputs = st.data_editor(

        test_defaults,

        num_rows="fixed",

        use_container_width=True,

        key="consolidation_test_editor",

        disabled=["Layer"]

    )



    # Calculate Cc and Cr for every layer.



    calculated_cc = []

    calculated_cr = []

    test_errors = []



    for index, row in test_inputs.iterrows():



        layer_name = row["Layer"]



        cc = calculate_compression_index(

            float(row["Cc e1"]),

            float(row["Cc e2"]),

            float(row["Cc σ1' (kPa)"]),

            float(row["Cc σ2' (kPa)"])

        )



        cr = calculate_compression_index(

            float(row["Cr e1"]),

            float(row["Cr e2"]),

            float(row["Cr σ1' (kPa)"]),

            float(row["Cr σ2' (kPa)"])

        )



        # Check whether the readings belong to the correct curve portions.

        pc = float(

            layers.iloc[index]["Preconsolidation stress (kPa)"]

        )



        cc_s1 = float(row["Cc σ1' (kPa)"])

        cc_s2 = float(row["Cc σ2' (kPa)"])



        cr_s1 = float(row["Cr σ1' (kPa)"])

        cr_s2 = float(row["Cr σ2' (kPa)"])



        if cc is None:

            test_errors.append(

                f"{layer_name}: Invalid Cc readings. Check e1, e2, "

                "and ensure σ2' is greater than σ1'."

            )



        elif not (cc_s1 > pc and cc_s2 > pc):

            test_errors.append(

                f"{layer_name}: Cc stresses should both be above "

                "preconsolidation stress."

            )



        if cr is None:

            test_errors.append(

                f"{layer_name}: Invalid Cr readings. Check e1, e2, "

                "and ensure σ2' is greater than σ1'."

            )



        elif not (cr_s1 < pc and cr_s2 < pc):

            test_errors.append(

                f"{layer_name}: Cr stresses should both be below "

                "preconsolidation stress."

            )



        calculated_cc.append(cc)

        calculated_cr.append(cr)





    # Display calculated indices.



    st.subheader("Calculated Compression Indices")



    index_results = pd.DataFrame({

        "Layer": test_inputs["Layer"],

        "Calculated Cc": calculated_cc,

        "Calculated Cr": calculated_cr

    })



    st.dataframe(

        index_results,

        use_container_width=True

    )



    if test_errors:

        for error in test_errors:

            st.error(error)



    # Add the calculated indices to the layer table.

    # These values are used directly in settlement calculations.



    layers["Cc"] = calculated_cc

    layers["Cr"] = calculated_cr



    st.success(

        "Cc and Cr are calculated from the entered laboratory readings "

        "and are automatically linked to the primary consolidation calculation."

    )





    # =========================================================

    # 7. TIME INPUTS

    # =========================================================



    st.header("SECTION D - CONSOLIDATION TIME")



    c1, c2 = st.columns(2)



    with c1:

        t1 = st.number_input(

            "Time at end of primary consolidation t1",

            min_value=0.001,

            value=1.0

        )



    with c2:

        t2 = st.number_input(

            "Final evaluation time t2",

            min_value=0.001,

            value=10.0

        )



    st.caption(

        "Use the same time unit for both values, such as years."

    )





    # =========================================================

    # 8. VALIDATION

    # =========================================================



    required_columns = [

        "Layer",

        "Thickness H (m)",

        "Centre depth z (m)",

        "Initial effective stress (kPa)",

        "Preconsolidation stress (kPa)",

        "Cc", "Cr", "C_alpha",

        "Initial void ratio e0",

        "End-primary void ratio ep",

        "Include primary",

        "Include secondary"

    ]



    valid = True



    if not all(col in layers.columns for col in required_columns):

        st.error("Layer table is missing required columns.")

        valid = False



    elif layers.empty:

        st.error("Enter at least one compressible layer.")

        valid = False



    else:



        numeric_columns = [

            "Thickness H (m)",

            "Centre depth z (m)",

            "Initial effective stress (kPa)",

            "Preconsolidation stress (kPa)",

            "Cc", "Cr", "C_alpha",

            "Initial void ratio e0",

            "End-primary void ratio ep"

        ]



        if layers[numeric_columns].isnull().any().any():

            st.error("Fill all numerical layer properties.")

            valid = False



        elif (layers["Thickness H (m)"] <= 0).any():

            st.error("Layer thickness must be positive.")

            valid = False



        elif (layers["Centre depth z (m)"] < 0).any():

            st.error("Layer centre depth cannot be negative.")

            valid = False



        elif (layers["Initial effective stress (kPa)"] <= 0).any():

            st.error("Initial effective stress must be positive.")

            valid = False



        elif (layers["Preconsolidation stress (kPa)"] <= 0).any():

            st.error("Preconsolidation stress must be positive.")

            valid = False



        elif (layers[["Cc", "Cr", "C_alpha"]] < 0).any().any():

            st.error("Compression indices cannot be negative.")

            valid = False



        elif (layers["Initial void ratio e0"] <= -1).any():

            st.error("Initial void ratio must be greater than -1.")

            valid = False



        elif (layers["End-primary void ratio ep"] <= -1).any():

            st.error("End-primary void ratio must be greater than -1.")

            valid = False



    if test_errors:

        valid = False



    if t2 <= t1:

        st.error("Final time t2 must be greater than t1.")

        valid = False



    if B_end < B_start:

        st.error("Maximum breadth must exceed minimum breadth.")

        valid = False



    if B_step <= 0:

        st.error("Breadth increment must be positive.")

        valid = False





    # =========================================================

    # 10. INDIVIDUAL CALCULATORS

    # =========================================================



    st.header("SECTION F - CALCULATE INDIVIDUAL SETTLEMENTS")



    calc_B = st.number_input(

        "Breadth for individual calculations (m)",

        min_value=0.1,

        value=2.0,

        key="individual_B"

    )



    if valid:



        qgross_ind, qnet_ind = calculate_net_pressure(

            P, calc_B, L, gamma, Df

        )



        st.write(

            f"Gross pressure: **{qgross_ind:.2f} kPa**"

        )



        st.write(

            f"Net pressure: **{qnet_ind:.2f} kPa**"

        )



        with st.expander("Calculate Immediate Settlement", expanded=True):



            if st.button(

                "Calculate Immediate Settlement",

                key="btn_immediate"

            ):



                Si = calculate_immediate(

                    qnet_ind, calc_B, Es, nu, Is

                )



                st.success(

                    f"Immediate settlement = {Si * 1000:.2f} mm"

                )



        with st.expander("Calculate Primary Consolidation", expanded=True):



            if st.button(

                "Calculate Primary Settlement",

                key="btn_primary"

            ):



                Sc, _, layer_details = calculate_layer_settlement(

                    layers, qnet_ind, calc_B, L,

                    t1, t2,

                    use_primary=True,

                    use_secondary=False

                )



                st.success(

                    f"Primary consolidation settlement = {Sc * 1000:.2f} mm"

                )



                st.dataframe(

                    pd.DataFrame(layer_details),

                    use_container_width=True

                )



        with st.expander("Calculate Secondary Consolidation", expanded=True):



            if st.button(

                "Calculate Secondary Settlement",

                key="btn_secondary"

            ):



                _, Ss, layer_details = calculate_layer_settlement(

                    layers, qnet_ind, calc_B, L,

                    t1, t2,

                    use_primary=False,

                    use_secondary=True

                )



                st.success(

                    f"Secondary settlement = {Ss * 1000:.2f} mm"

                )



                st.dataframe(

                    pd.DataFrame(layer_details),

                    use_container_width=True

                )



    else:

        st.warning(

            "Correct the input errors before performing settlement calculations."

        )





    # =========================================================

    # 11. BREADTH-WISE SETTLEMENT ANALYSIS

    # =========================================================



    st.header("SECTION G - SETTLEMENT FOR DIFFERENT FOOTING BREADTHS")



    st.write(

        "This analysis calculates all settlement components for each "

        "trial breadth using the same selected footing length."

    )



    if st.button("Run Complete Breadth Analysis"):



        if not valid:



            st.error("Correct the input errors before running analysis.")



        else:



            breadths = np.arange(

                B_start,

                B_end + B_step * 0.5,

                B_step

            )



            results = []



            for B in breadths:



                qgross, qnet = calculate_net_pressure(

                    P, B, L, gamma, Df

                )



                Si = calculate_immediate(

                    qnet, B, Es, nu, Is

                )



                Sc, Ss, layer_details = calculate_layer_settlement(

                    layers, qnet, B, L,

                    t1, t2,

                    use_primary=True,

                    use_secondary=True

                )



                Si_mm = Si * 1000

                Sc_mm = Sc * 1000

                Ss_mm = Ss * 1000



                total = Si_mm + Sc_mm + Ss_mm



                status = (

                    "PASS"

                    if total <= settlement_limit

                    else "FAIL"

                )



                results.append({

                    "Breadth (m)": B,

                    "Length (m)": L,

                    "Area (m²)": B * L,

                    "Gross pressure (kPa)": qgross,

                    "Net pressure (kPa)": qnet,

                    "Immediate (mm)": Si_mm,

                    "Primary (mm)": Sc_mm,

                    "Secondary (mm)": Ss_mm,

                    "Total settlement (mm)": total,

                    "Allowable (mm)": settlement_limit,

                    "Status": status

                })



            df = pd.DataFrame(results)



            st.session_state["settlement_results"] = df





    # =========================================================

    # 12. DISPLAY RESULTS

    # =========================================================



    if "settlement_results" in st.session_state:



        df = st.session_state["settlement_results"]



        st.subheader("Breadth-wise Settlement Results")



        st.dataframe(

            df.style.format({

                "Breadth (m)": "{:.2f}",

                "Length (m)": "{:.2f}",

                "Area (m²)": "{:.2f}",

                "Gross pressure (kPa)": "{:.2f}",

                "Net pressure (kPa)": "{:.2f}",

                "Immediate (mm)": "{:.2f}",

                "Primary (mm)": "{:.2f}",

                "Secondary (mm)": "{:.2f}",

                "Total settlement (mm)": "{:.2f}",

                "Allowable (mm)": "{:.2f}"

            }),

            use_container_width=True

        )



        passing = df[df["Status"] == "PASS"]



        st.subheader("Preliminary Settlement-Based Breadth")



        if not passing.empty:



            optimum = passing.iloc[0]



            c1, c2, c3 = st.columns(3)



            c1.metric(

                "Smallest passing breadth",

                f"{optimum['Breadth (m)']:.2f} m"

            )



            c2.metric(

                "Total settlement",

                f"{optimum['Total settlement (mm)']:.2f} mm"

            )



            c3.metric(

                "Footing area",

                f"{optimum['Area (m²)']:.2f} m²"

            )



            st.success(

                "The selected breadth satisfies the specified "

                "settlement limit within the trial range."

            )



        else:



            st.error(

                "No trial breadth satisfies the specified settlement limit."

            )



        # Settlement graph



        st.subheader("Settlement versus Footing Breadth")



        fig, ax = plt.subplots(figsize=(10, 5))



        ax.plot(

            df["Breadth (m)"],

            df["Immediate (mm)"],

            marker="o",

            label="Immediate settlement"

        )



        ax.plot(

            df["Breadth (m)"],

            df["Primary (mm)"],

            marker="s",

            label="Primary consolidation"

        )



        ax.plot(

            df["Breadth (m)"],

            df["Secondary (mm)"],

            marker="^",

            label="Secondary consolidation"

        )



        ax.plot(

            df["Breadth (m)"],

            df["Total settlement (mm)"],

            marker="D",

            linewidth=2,

            label="Total settlement"

        )



        ax.axhline(

            settlement_limit,

            linestyle="--",

            label="Allowable settlement"

        )



        ax.set_xlabel("Footing breadth (m)")

        ax.set_ylabel("Settlement (mm)")

        ax.set_title("Settlement versus Footing Breadth")

        ax.grid(True)

        ax.legend()



        st.pyplot(fig)



        # Component graph



        st.subheader("Settlement Component Comparison")



        fig2, ax2 = plt.subplots(figsize=(10, 5))



        x = np.arange(len(df))

        width = 0.25



        ax2.bar(

            x - width,

            df["Immediate (mm)"],

            width,

            label="Immediate"

        )



        ax2.bar(

            x,

            df["Primary (mm)"],

            width,

            label="Primary"

        )



        ax2.bar(

            x + width,

            df["Secondary (mm)"],

            width,

            label="Secondary"

        )



        ax2.set_xticks(x)

        ax2.set_xticklabels(

            [f"{b:.2f}" for b in df["Breadth (m)"]]

        )



        ax2.set_xlabel("Footing breadth (m)")

        ax2.set_ylabel("Settlement (mm)")

        ax2.set_title("Individual Settlement Components")

        ax2.grid(axis="y")

        ax2.legend()



        st.pyplot(fig2)



        # CSV export



        st.subheader("Download Results")



        csv = df.to_csv(index=False).encode("utf-8")



        st.download_button(

            label="Download Settlement Results CSV",

            data=csv,

            file_name="settlement_results.csv",

            mime="text/csv"

        )


# ============================================================
# COMBINED BEARING CAPACITY AND SETTLEMENT GRAPH (COLUMN 3)
# ============================================================
if st.session_state.active_page == "Combined Graph":
    bearing_df = st.session_state.get("bearing_results")
    settlement_df = st.session_state.get("settlement_results")

    if bearing_df is None and settlement_df is None:
        st.info("Run the bearing-capacity check and the settlement breadth analysis to display the combined graph.")
    else:
        fig_combined, ax_capacity = plt.subplots(figsize=(8, 5))
        ax_settlement = ax_capacity.twinx()
        plotted = False

        if bearing_df is not None and not bearing_df.empty:
            bearing_plot = bearing_df.sort_values("Breadth (m)")
            ax_capacity.plot(
                bearing_plot["Breadth (m)"],
                bearing_plot["Net safe bearing capacity (kPa)"],
                marker="o", linewidth=2, label="Net safe bearing capacity"
            )
            ax_capacity.plot(
                bearing_plot["Breadth (m)"],
                bearing_plot["Net applied pressure (kPa)"],
                marker="s", linestyle="--", label="Net applied pressure"
            )
            plotted = True

        if settlement_df is not None and not settlement_df.empty:
            settlement_plot = settlement_df.sort_values("Breadth (m)")
            ax_settlement.plot(
                settlement_plot["Breadth (m)"],
                settlement_plot["Total settlement (mm)"],
                marker="^", color="tab:red", linewidth=2,
                label="Total settlement"
            )
            plotted = True

        if plotted:
            ax_capacity.set_xlabel("Footing breadth, B (m)")
            ax_capacity.set_ylabel("Bearing capacity / applied pressure (kPa)")
            ax_settlement.set_ylabel("Total settlement (mm)")
            ax_capacity.set_title("Bearing Capacity and Settlement vs Footing Breadth")
            ax_capacity.grid(True, alpha=0.3)

            h1, l1 = ax_capacity.get_legend_handles_labels()
            h2, l2 = ax_settlement.get_legend_handles_labels()
            ax_capacity.legend(h1 + h2, l1 + l2, loc="best")
            fig_combined.tight_layout()
            st.pyplot(fig_combined)
            st.caption("Left axis: bearing capacity and applied pressure (kPa). Right axis: total settlement (mm). The two vertical axes are separate because the quantities have different units.")
        else:
            st.info("No valid breadth-wise results are available yet.")
