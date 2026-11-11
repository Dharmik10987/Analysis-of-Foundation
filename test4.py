
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Foundation Settlement Calculator",
    page_icon="🏗️",
    layout="wide"
)

st.title("Foundation Settlement Calculator")
st.caption(
    "Academic calculation tool: immediate, primary consolidation "
    "and secondary settlement."
)

st.warning(
    "Use verified project-specific soil parameters wherever available. "
    "Preset values are illustrative assumptions, not measured results. "
    "This tool is not a substitute for a complete geotechnical design."
)

# ---------------------------------------------------------
# SOIL PRESETS
# ---------------------------------------------------------

SOILS = {
    "Loose sand": {
        "gamma": 17.0, "Es": 10000, "nu": 0.30,
        "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,
        "e0": 0.65, "ep": 0.65,
        "sigma_pc": 100.0, "soil_type": "Granular"
    },
    "Medium-dense sand": {
        "gamma": 18.0, "Es": 25000, "nu": 0.30,
        "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,
        "e0": 0.60, "ep": 0.60,
        "sigma_pc": 100.0, "soil_type": "Granular"
    },
    "Dense sand": {
        "gamma": 19.0, "Es": 50000, "nu": 0.28,
        "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,
        "e0": 0.50, "ep": 0.50,
        "sigma_pc": 100.0, "soil_type": "Granular"
    },
    "Baseline sand with fines": {
        "gamma": 18.0, "Es": 20000, "nu": 0.30,
        "Cc": 0.15, "Cr": 0.03, "C_alpha": 0.003,
        "e0": 0.65, "ep": 0.60,
        "sigma_pc": 100.0, "soil_type": "Granular"
    },
    "Soft clay": {
        "gamma": 16.5, "Es": 4000, "nu": 0.45,
        "Cc": 0.35, "Cr": 0.07, "C_alpha": 0.025,
        "e0": 1.00, "ep": 0.90,
        "sigma_pc": 50.0, "soil_type": "Fine-grained"
    },
    "Stiff clay": {
        "gamma": 19.0, "Es": 30000, "nu": 0.35,
        "Cc": 0.20, "Cr": 0.04, "C_alpha": 0.010,
        "e0": 0.65, "ep": 0.60,
        "sigma_pc": 200.0, "soil_type": "Fine-grained"
    },
    "Silty soil": {
        "gamma": 18.0, "Es": 12000, "nu": 0.32,
        "Cc": 0.12, "Cr": 0.025, "C_alpha": 0.006,
        "e0": 0.70, "ep": 0.65,
        "sigma_pc": 100.0, "soil_type": "Fine-grained"
    },
    "Layered soil": {
        "gamma": 17.0, "Es": 10000, "nu": 0.30,
        "Cc": 0.35, "Cr": 0.07, "C_alpha": 0.025,
        "e0": 1.00, "ep": 0.90,
        "sigma_pc": 50.0, "soil_type": "Layered"
    }
}

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

st.header("1. Foundation details")

c1, c2, c3 = st.columns(3)

with c1:
    load = st.number_input(
        "Service vertical load P (kN)",
        min_value=1.0, value=900.0
    )
    depth = st.number_input(
        "Foundation depth Df (m)",
        min_value=0.0, value=1.5
    )

with c2:
    length = st.number_input(
        "Footing length L (m)",
        min_value=0.1, value=2.0
    )
    settlement_limit = st.number_input(
        "Allowable total settlement (mm)",
        min_value=0.1, value=25.0
    )

with c3:
    breadth_start = st.number_input(
        "Minimum breadth (m)",
        min_value=0.1, value=1.5
    )
    breadth_end = st.number_input(
        "Maximum breadth (m)",
        min_value=0.2, value=4.0
    )

    breadth_step = st.number_input(
        "Breadth increment (m)",
        min_value=0.05, value=0.5
    )

st.header("2. Soil selection")

soil_name = st.selectbox("Select soil scenario", list(SOILS.keys()))
preset = SOILS[soil_name]

st.info(
    "Preset parameters are starting assumptions only. "
    "Edit them to match your adopted soil profile."
)

c1, c2, c3 = st.columns(3)

with c1:
    gamma = st.number_input(
        "Bulk unit weight γ (kN/m³)",
        min_value=1.0, value=float(preset["gamma"])
    )
    Es = st.number_input(
        "Elastic modulus Es (kPa)",
        min_value=1.0, value=float(preset["Es"])
    )

with c2:
    nu = st.number_input(
        "Poisson's ratio ν",
        min_value=0.0, max_value=0.49,
        value=float(preset["nu"])
    )
    influence = st.number_input(
        "Immediate settlement influence factor Is",
        min_value=0.01, value=1.12
    )

with c3:
    water_table = st.number_input(
        "Water table depth below ground (m)",
        min_value=0.0, value=3.0
    )
    water_unit_weight = st.number_input(
        "Water unit weight (kN/m³)",
        min_value=1.0, value=9.81
    )

st.header("3. Consolidation layer properties")

st.write(
    "Enter the properties of the compressible layer or layers. "
    "Depth z is measured from the footing base to the centre "
    "of each layer."
)

default_layers = pd.DataFrame([
    {
        "Layer": "Layer 1",
        "Thickness H (m)": 2.0,
        "Centre depth z (m)": 1.0,
        "Initial effective stress (kPa)": 30.0,
        "Preconsolidation stress (kPa)": preset["sigma_pc"],
        "Cc": preset["Cc"],
        "Cr": preset["Cr"],
        "C_alpha": preset["C_alpha"],
        "Initial void ratio e0": preset["e0"],
        "End-primary void ratio ep": preset["ep"],
        "Include primary": preset["Cc"] > 0,
        "Include secondary": preset["C_alpha"] > 0,
        "State": "Overconsolidated"
    }
])

if soil_name == "Layered soil":
    default_layers = pd.DataFrame([
        {
            "Layer": "Upper sand",
            "Thickness H (m)": 0.5,
            "Centre depth z (m)": 0.25,
            "Initial effective stress (kPa)": 25.0,
            "Preconsolidation stress (kPa)": 100.0,
            "Cc": 0.0, "Cr": 0.0, "C_alpha": 0.0,
            "Initial void ratio e0": 0.65,
            "End-primary void ratio ep": 0.65,
            "Include primary": False,
            "Include secondary": False,
            "State": "Normally consolidated"
        },
        {
            "Layer": "Lower clay",
            "Thickness H (m)": 3.0,
            "Centre depth z (m)": 2.0,
            "Initial effective stress (kPa)": 45.0,
            "Preconsolidation stress (kPa)": 60.0,
            "Cc": 0.35, "Cr": 0.07, "C_alpha": 0.025,
            "Initial void ratio e0": 1.0,
            "End-primary void ratio ep": 0.90,
            "Include primary": True,
            "Include secondary": True,
            "State": "Overconsolidated"
        }
    ])

edited_layers = st.data_editor(
    default_layers,
    num_rows="dynamic",
    use_container_width=True,
    key="layer_editor"
)

st.header("4. Consolidation time parameters")

c1, c2 = st.columns(2)

with c1:
    t1 = st.number_input(
        "Time at end of primary consolidation t1",
        min_value=0.001, value=1.0
    )

with c2:
    t2 = st.number_input(
        "Final evaluation time t2",
        min_value=0.001, value=10.0
    )

if t2 <= t1:
    st.error("t2 must be greater than t1 for secondary settlement.")
    st.stop()

st.caption(
    "Use the same time unit for t1 and t2, for example years."
)

st.header("5. Calculation options")

include_immediate = st.checkbox(
    "Include immediate settlement", value=True
)

include_primary = st.checkbox(
    "Include primary consolidation settlement", value=True
)

include_secondary = st.checkbox(
    "Include secondary settlement", value=True
)

# ---------------------------------------------------------
# CALCULATION FUNCTIONS
# ---------------------------------------------------------

def immediate_settlement(q_net, B, Es, nu, Is):
    """
    Simplified elastic immediate settlement estimate.
    Returns settlement in metres.
    """
    return q_net * B * (1 - nu**2) * Is / Es


def stress_increase_2to1(q_net, B, L, z):
    """
    Simplified 2:1 stress distribution.
    """
    denominator = (B + z) * (L + z)
    return q_net * B * L / denominator


def primary_settlement(
    H, sigma0, delta_sigma, sigma_pc,
    Cc, Cr, e0
):
    """
    One-dimensional e-log sigma' primary consolidation model.

    NC: Cc throughout.
    OC: Cr below preconsolidation stress and Cc above it.
    """
    if H <= 0 or sigma0 <= 0 or e0 <= -1:
        return 0.0

    final_stress = sigma0 + delta_sigma

    if final_stress <= sigma0:
        return 0.0

    if sigma_pc <= sigma0:
        # Soil is normally consolidated at the initial stress.
        settlement = (
            Cc * H / (1 + e0)
            * np.log10(final_stress / sigma0)
        )

    elif final_stress <= sigma_pc:
        # Entire stress increment is recompression.
        settlement = (
            Cr * H / (1 + e0)
            * np.log10(final_stress / sigma0)
        )

    else:
        # Recompression up to sigma_pc, then virgin compression.
        recompression = (
            Cr * H / (1 + e0)
            * np.log10(sigma_pc / sigma0)
        )

        # Approximate void ratio at preconsolidation stress.
        e_pc = e0 - Cr * np.log10(sigma_pc / sigma0)

        virgin_compression = (
            Cc * H / (1 + e_pc)
            * np.log10(final_stress / sigma_pc)
        )

        settlement = recompression + virgin_compression

    return max(0.0, settlement)


def secondary_settlement(H, C_alpha, ep, t1, t2):
    """
    Secondary compression settlement.
    C_alpha is the secondary compression index in e-log(time).
    """
    if H <= 0 or C_alpha <= 0 or ep <= -1:
        return 0.0

    return (
        C_alpha * H / (1 + ep)
        * np.log10(t2 / t1)
    )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

if breadth_end < breadth_start:
    st.error("Maximum breadth must be greater than minimum breadth.")
    st.stop()

if breadth_step <= 0:
    st.error("Breadth increment must be positive.")
    st.stop()

if edited_layers.empty:
    st.error("Enter at least one soil layer.")
    st.stop()

required_columns = [
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

if not all(col in edited_layers.columns for col in required_columns):
    st.error("The layer table is missing required columns.")
    st.stop()

numeric_columns = [
    "Thickness H (m)",
    "Centre depth z (m)",
    "Initial effective stress (kPa)",
    "Preconsolidation stress (kPa)",
    "Cc", "Cr", "C_alpha",
    "Initial void ratio e0",
    "End-primary void ratio ep"
]

if edited_layers[numeric_columns].isnull().any().any():
    st.error("Fill all numerical layer properties.")
    st.stop()

if (edited_layers["Thickness H (m)"] <= 0).any():
    st.error("Layer thickness must be positive.")
    st.stop()

if (edited_layers["Initial effective stress (kPa)"] <= 0).any():
    st.error("Initial effective stress must be positive.")
    st.stop()

if (edited_layers["Initial void ratio e0"] <= -1).any():
    st.error("Initial void ratio must be greater than -1.")
    st.stop()

if (edited_layers["End-primary void ratio ep"] <= -1).any():
    st.error("End-primary void ratio must be greater than -1.")
    st.stop()

# ---------------------------------------------------------
# BREADTH TRIALS
# ---------------------------------------------------------

breadths = np.arange(
    breadth_start,
    breadth_end + breadth_step * 0.5,
    breadth_step
)

results = []

# Ground-level surcharge at footing base.
q_surcharge = gamma * depth

# This is a simplified gross-to-net pressure calculation.
for B in breadths:

    area = B * length
    q_gross = load / area
    q_net = q_gross - q_surcharge

    if q_net < 0:
        q_net = 0.0

    # Immediate settlement
    Si = 0.0

    if include_immediate:
        Si = immediate_settlement(
            q_net, B, Es, nu, influence
        )

    # Primary and secondary settlement
    Sc = 0.0
    Ss = 0.0

    for _, layer in edited_layers.iterrows():

        H = float(layer["Thickness H (m)"])
        z = float(layer["Centre depth z (m)"])

        sigma0 = float(
            layer["Initial effective stress (kPa)"]
        )

        sigma_pc = float(
            layer["Preconsolidation stress (kPa)"]
        )

        delta_sigma = stress_increase_2to1(
            q_net, B, length, z
        )

        if include_primary and bool(layer["Include primary"]):

            Sc += primary_settlement(
                H=H,
                sigma0=sigma0,
                delta_sigma=delta_sigma,
                sigma_pc=sigma_pc,
                Cc=float(layer["Cc"]),
                Cr=float(layer["Cr"]),
                e0=float(layer["Initial void ratio e0"])
            )

        if include_secondary and bool(layer["Include secondary"]):

            Ss += secondary_settlement(
                H=H,
                C_alpha=float(layer["C_alpha"]),
                ep=float(layer["End-primary void ratio ep"]),
                t1=t1,
                t2=t2
            )

    Si_mm = Si * 1000
    Sc_mm = Sc * 1000
    Ss_mm = Ss * 1000

    total_mm = Si_mm + Sc_mm + Ss_mm

    status = (
        "PASS" if total_mm <= settlement_limit else "FAIL"
    )

    results.append({
        "Breadth B (m)": B,
        "Length L (m)": length,
        "Area (m²)": area,
        "Gross pressure (kPa)": q_gross,
        "Net pressure (kPa)": q_net,
        "Immediate (mm)": Si_mm,
        "Primary (mm)": Sc_mm,
        "Secondary (mm)": Ss_mm,
        "Total settlement (mm)": total_mm,
        "Allowable (mm)": settlement_limit,
        "Status": status
    })

df = pd.DataFrame(results)

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

st.header("6. Settlement results")

st.dataframe(
    df.style.format({
        "Breadth B (m)": "{:.2f}",
        "Length L (m)": "{:.2f}",
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

if not passing.empty:

    optimum = passing.iloc[0]

    st.success(
        f"Smallest passing trial breadth: "
        f"{optimum['Breadth B (m)']:.2f} m"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Selected breadth",
        f"{optimum['Breadth B (m)']:.2f} m"
    )

    c2.metric(
        "Total settlement",
        f"{optimum['Total settlement (mm)']:.2f} mm"
    )

    c3.metric(
        "Footing area",
        f"{optimum['Area (m²)']:.2f} m²"
    )

else:
    st.error(
        "No trial breadth satisfies the selected settlement limit. "
        "Increase the trial range or review the design assumptions."
    )

# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

st.header("7. Settlement versus footing breadth")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    df["Breadth B (m)"],
    df["Immediate (mm)"],
    marker="o",
    label="Immediate"
)

ax.plot(
    df["Breadth B (m)"],
    df["Primary (mm)"],
    marker="s",
    label="Primary consolidation"
)

ax.plot(
    df["Breadth B (m)"],
    df["Secondary (mm)"],
    marker="^",
    label="Secondary"
)

ax.plot(
    df["Breadth B (m)"],
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

ax.set_xlabel("Footing breadth B (m)")
ax.set_ylabel("Settlement (mm)")
ax.set_title("Settlement versus footing breadth")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

st.header("8. Export")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download settlement results (CSV)",
    data=csv,
    file_name="settlement_results.csv",
    mime="text/csv"
)

# ---------------------------------------------------------
# NOTES
# ---------------------------------------------------------

with st.expander("Calculation assumptions and limitations"):

    st.markdown("""
    **Immediate settlement**

    The elastic equation used is:

    \[
    S_i = \frac{q_n B(1-\\nu^2)I_s}{E_s}
    \]

    **Primary consolidation**

    The calculation uses a simplified one-dimensional e-log effective
    stress relationship. The recompression index Cr is used below
    preconsolidation pressure and Cc above it.

    **Stress distribution**

    Stress increase is estimated using the simplified 2:1 method:

    \[
    \\Delta\\sigma_z =
    \\frac{q_nBL}{(B+z)(L+z)}
    \]

    **Secondary settlement**

    \[
    S_s =
    \\frac{C_\\alpha H}{1+e_p}
    \\log_{10}\\left(\\frac{t_2}{t_1}\\right)
    \]

    C-alpha must be appropriate for the soil and time interval.
    Secondary settlement is not automatically relevant to every soil.

    **Important limitations**

    - Groundwater is displayed as an input but is not automatically
      used to calculate each layer's initial effective stress.
    - Enter initial effective stresses consistent with the groundwater
      profile and soil layering.
    - The 2:1 stress distribution is approximate.
    - The immediate settlement equation is simplified and does not
      replace a complete layered elastic analysis.
    - This app does not calculate bearing capacity.
    - It does not calculate time rate of consolidation or degree of
      consolidation.
    - The secondary calculation assumes the specified t1 represents
      the end of primary consolidation.
    - The first passing trial is only the smallest passing value in
      the selected trial grid.
    - A structural footing design and bearing-capacity check are still
      required before selecting a foundation.
    """)
