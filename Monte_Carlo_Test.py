import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import triang
import io

# NEW: Import CoolProp
from CoolProp.CoolProp import PropsSI

# --------------------------------------------------------------------------------
# 1. PAGE CONFIG AND TITLE
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Pressure Drop Monte Carlo Simulation",
    layout="wide"
)

st.title("Monte Carlo Simulation for Pressure Drop (Using CoolProp)")

# --------------------------------------------------------------------------------
# INSTRUCTIONS AND LA TE X EQUATIONS
# --------------------------------------------------------------------------------
st.markdown("""
## How to Use This Program

1. In the sidebar, select a fluid (Water, Oxygen, Nitrogen, parahydrogen).  

2. Enter the fluid pressure (in kPa) and temperature (in K). The program will 
   automatically query CoolProp for the fluid's density (kg/m³) and viscosity 
   (Pa·s). These values appear as the default distribution means in the next step.

3. Proceed to set distributions and their parameters for any uncertain pipe 
   dimensions, roughness, mass flow rate, etc.

4. Click "Run Simulation" to:
   - Generate random samples for each uncertain parameter.
   - Compute the resulting pressure drop distribution via Darcy-Weisbach.
   - Plot a histogram and a CDF of the simulated pressure drops.
   - Show correlation coefficients for a quick sensitivity analysis.

5. Download results in an Excel file with multiple sheets:
   - Simulation Data
   - Summary statistics
   - Sensitivity analysis
""")

if st.checkbox("### Show Equations Used"):

    st.latex(r"""
    Q = \frac{\dot{m}}{\rho}
    """)

    st.latex(r"""
    Re = \frac{\rho \, v \, D}{\mu}
    """)

    st.latex(r"""
    f = 0.25 \Big/ \left[\log_{10}\!\Bigl(\frac{\epsilon}{3.7\,D} 
    + \frac{5.74}{Re^{0.9}}\Bigr)\right]^2 
    \quad (\text{Swamee-Jain})
    """)

    st.latex(r"""
    h_f = f\,\frac{L}{D}\,\frac{v^2}{2g}
    """)

    st.latex(r"""
    h_\mathrm{fittings} = K_\mathrm{total}\,\frac{v^2}{2g}
    """)

    st.latex(r"""
    h_\mathrm{total} = h_f + h_\mathrm{fittings} + \Delta z
    """)

    st.latex(r"""
    \Delta P = \rho\,g\,h_\mathrm{total}
    """)

# --------------------------------------------------------------------------------
# 2. SIDEBAR: FLUID SELECTION & COOLPROP INPUTS
# --------------------------------------------------------------------------------
st.sidebar.header("Fluid Selection")

# 2A) Let user select from relevant fluids in CoolProp nomenclature
fluid_options = ["Water", "Oxygen", "Nitrogen", "parahydrogen"]
selected_fluid = st.sidebar.selectbox("Select Fluid:", fluid_options)

# 2B) User inputs for Pressure/Temperature
pressure_kPa = st.sidebar.number_input("Fluid Pressure (kPa)", value=101.325, step=10.0)
temperature_K = st.sidebar.number_input("Fluid Temperature (K)", value=300.0, step=10.0)

# 2C) Query CoolProp for density (kg/m³) and viscosity (Pa·s)
#    We convert that viscosity to mPa·s (or cP) to stay consistent with the UI.
try:
    calc_density = PropsSI("D", "T", temperature_K, "P", pressure_kPa * 1000.0, selected_fluid)  # kg/m³
    calc_viscosity = PropsSI("V", "T", temperature_K, "P", pressure_kPa * 1000.0, selected_fluid)  # Pa·s
except Exception as e:
    # If something goes wrong (e.g., fluid not valid at that P,T), handle gracefully
    st.sidebar.error(f"Error retrieving CoolProp data: {e}")
    calc_density = 1.0
    calc_viscosity = 1e-3

# Convert Pa·s -> mPa·s
calc_viscosity_mPas = calc_viscosity * 1e3

st.sidebar.write(f"CoolProp Density: {calc_density:.2f} kg/m³")
st.sidebar.write(f"CoolProp Viscosity: {calc_viscosity_mPas:.4f} mPa·s")

# --------------------------------------------------------------------------------
# 3. UNIT SELECTION FOR DIAMETER & LENGTH
# --------------------------------------------------------------------------------
units_selected = st.sidebar.radio(
    "Pipe Dimension Units",
    ["Inches", "Meters"]
)

def length_label(base_label):
    return f"{base_label} ({'inches' if units_selected=='Inches' else 'meters'})"

# --------------------------------------------------------------------------------
# 4. GET DISTRIBUTION FUNCTION
# --------------------------------------------------------------------------------
def get_distribution(dist_name, mean, std_dev, min_val=None, max_val=None):
    if dist_name == 'Normal':
        return lambda size: np.random.normal(mean, std_dev, size)
    elif dist_name == 'Uniform':
        return lambda size: np.random.uniform(min_val, max_val, size)
    elif dist_name == 'Triangular':
        c = (mean - min_val) / (max_val - min_val)
        return lambda size: triang.rvs(c, loc=min_val, scale=(max_val - min_val), size=size)
    else:
        # Deterministic
        return lambda size: np.full(size, mean)

# --------------------------------------------------------------------------------
# 5. FLUID PROPERTIES DISTRIBUTIONS (BASED ON COOLPROP MEANS)
# --------------------------------------------------------------------------------
st.sidebar.subheader("Fluid Properties (with CoolProp)")

rho_dist = st.sidebar.selectbox("Density Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=0)
rho_mean = st.sidebar.number_input(
    "Fluid Density Mean (kg/m³)",
    value=float(f"{calc_density:.2f}"),   # default from CoolProp
    format="%.2f"
)
rho_std = st.sidebar.number_input(
    "Fluid Density Std Dev (kg/m³)",
    value=0.0 if rho_dist != 'Normal' else 10.0,
    format="%.2f"
)
rho_min = st.sidebar.number_input(
    "Fluid Density Min (kg/m³)",
    value=rho_mean * 0.95 if rho_dist in ['Uniform', 'Triangular'] else 0.0,
    format="%.2f"
)
rho_max = st.sidebar.number_input(
    "Fluid Density Max (kg/m³)",
    value=rho_mean * 1.05 if rho_dist in ['Uniform', 'Triangular'] else 0.0,
    format="%.2f"
)

mu_dist = st.sidebar.selectbox("Viscosity Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=0)
mu_mean = st.sidebar.number_input(
    "Fluid Viscosity Mean (mPa·s)",
    value=float(f"{calc_viscosity_mPas:.3f}"),  # from CoolProp
    format="%.3f"
)
mu_std = st.sidebar.number_input(
    "Fluid Viscosity Std Dev (mPa·s)",
    value=0.0 if mu_dist != 'Normal' else 0.1,
    format="%.3f"
)
mu_min = st.sidebar.number_input(
    "Fluid Viscosity Min (mPa·s)",
    value=mu_mean * 0.95 if mu_dist in ['Uniform', 'Triangular'] else 0.0,
    format="%.3f"
)
mu_max = st.sidebar.number_input(
    "Fluid Viscosity Max (mPa·s)",
    value=mu_mean * 1.05 if mu_dist in ['Uniform', 'Triangular'] else 0.0,
    format="%.3f"
)

# --------------------------------------------------------------------------------
# 6. OTHER INPUTS (PIPE, ROUGHNESS, FLOW, ETC.)
# --------------------------------------------------------------------------------
st.sidebar.subheader("Pipe Properties")

# Diameter
D_dist = st.sidebar.selectbox(
    "Diameter Distribution", 
    ['Deterministic', 'Normal', 'Uniform', 'Triangular'], 
    index=0
)
D_mean = st.sidebar.number_input(length_label("Pipe Diameter Mean"), value=2.0 if units_selected=="Inches" else 0.05, format="%.4f")
D_std = st.sidebar.number_input(length_label("Pipe Diameter Std Dev"), value=0.0, format="%.4f")
D_min = st.sidebar.number_input(length_label("Pipe Diameter Min"),
                                value=D_mean * 0.95 if D_dist in ['Uniform','Triangular'] else 0.0,
                                format="%.4f")
D_max = st.sidebar.number_input(length_label("Pipe Diameter Max"),
                                value=D_mean * 1.05 if D_dist in ['Uniform','Triangular'] else 0.0,
                                format="%.4f")

# Length
L_dist = st.sidebar.selectbox("Length Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=0)
L_mean = st.sidebar.number_input(length_label("Pipe Length Mean"), value=18.0 if units_selected=="Inches" else 6.0, format="%.2f")
L_std = st.sidebar.number_input(length_label("Pipe Length Std Dev"), value=0.0, format="%.2f")
L_min = st.sidebar.number_input(length_label("Pipe Length Min"),
                                value=L_mean*0.9 if L_dist in ['Uniform','Triangular'] else 0.0,
                                format="%.2f")
L_max = st.sidebar.number_input(length_label("Pipe Length Max"),
                                value=L_mean*1.1 if L_dist in ['Uniform','Triangular'] else 0.0,
                                format="%.2f")

# Roughness
epsilon_dist = st.sidebar.selectbox("Roughness Distribution", ['Deterministic','Normal','Uniform','Triangular'], index=0)
epsilon_mean = st.sidebar.number_input("Pipe Roughness Mean (m)", value=0.0001, format="%.6f")
epsilon_std = st.sidebar.number_input("Pipe Roughness Std Dev (m)", value=0.0, format="%.6f")
epsilon_min = st.sidebar.number_input("Pipe Roughness Min (m)", value=0.0, format="%.6f")
epsilon_max = st.sidebar.number_input("Pipe Roughness Max (m)", value=0.00015, format="%.6f")

# Flow
st.sidebar.subheader("Flow Properties")
mass_flow_dist = st.sidebar.selectbox("Mass Flow Rate Distribution", ['Deterministic','Normal','Uniform','Triangular'], index=0)
mass_flow_mean = st.sidebar.number_input("Mass Flow Rate Mean (kg/s)", value=3.0, format="%.4f")
mass_flow_std = st.sidebar.number_input("Mass Flow Rate Std Dev (kg/s)",
                                        value=0.25 if mass_flow_dist=='Normal' else 0.0, format="%.4f")
mass_flow_min = st.sidebar.number_input("Mass Flow Rate Min (kg/s)",
                                        value=2.0 if mass_flow_dist in ['Uniform','Triangular'] else 0.0, format="%.4f")
mass_flow_max = st.sidebar.number_input("Mass Flow Rate Max (kg/s)",
                                        value=4.0 if mass_flow_dist in ['Uniform','Triangular'] else 0.0, format="%.4f")

# Fittings & valves
st.sidebar.subheader("Fittings and Valves")
num_fittings = st.sidebar.number_input("Number of Fittings", min_value=0, value=2, format="%d")
num_valves = st.sidebar.number_input("Number of Valves", min_value=0, value=1, format="%d")

st.sidebar.markdown("**Fitting K Values**")
fitting_k = st.sidebar.number_input("K Value per Fitting", value=0.5, format="%.2f")

st.sidebar.markdown("**Valve K Values**")
valve_k = st.sidebar.number_input("K Value per Valve", value=10.0, format="%.2f")

# Elevation
st.sidebar.subheader("Elevation Change")
elevation_dist = st.sidebar.selectbox("Elevation Change Distribution", ['Deterministic','Normal','Uniform','Triangular'], index=0)
elevation_mean = st.sidebar.number_input("Elevation Change Mean (m)", value=0.0, format="%.2f")
elevation_std = st.sidebar.number_input("Elevation Change Std Dev (m)", value=0.0, format="%.2f")
elevation_min = st.sidebar.number_input("Elevation Change Min (m)",
                                        value=-3.0 if elevation_dist in ['Uniform','Triangular'] else 0.0, 
                                        format="%.2f")
elevation_max = st.sidebar.number_input("Elevation Change Max (m)",
                                        value=3.0 if elevation_dist in ['Uniform','Triangular'] else 0.0,
                                        format="%.2f")

# Gravity
st.sidebar.subheader("Gravity Acceleration")
gravity = st.sidebar.number_input("Acceleration due to Gravity (m/s²)", value=9.81, format="%.4f")

# Simulation
st.sidebar.header("Simulation Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1000, max_value=50000, value=5000, step=500, format="%d")
confidence_level = st.sidebar.slider("Confidence Interval (%)", min_value=90, max_value=99, value=95, step=1)

# Pressure Unit selection
st.sidebar.header("Display Units")
unit_selection = st.sidebar.selectbox("Select Pressure Unit", ['Pascals (Pa)', 'Pounds per Square Inch Absolute (psia)'], index=0)

# Run simulation button
run_simulation = st.sidebar.button("Run Simulation")

# --------------------------------------------------------------------------------
# HELPER: CONVERT PA TO PSIA
# --------------------------------------------------------------------------------
def pa_to_psia(pa_values):
    # 1 Pa = 0.000145038 psi (approx.)
    return pa_values * 0.000145038

# --------------------------------------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------------------------------------
if run_simulation:
    st.header("Simulation Results")
    st.subheader("Generating Random Samples...")

    # Build sampler functions
    rho_sampler = get_distribution(rho_dist, rho_mean, rho_std, rho_min, rho_max)
    mu_sampler  = get_distribution(mu_dist, mu_mean, mu_std, mu_min, mu_max)
    D_sampler   = get_distribution(D_dist, D_mean, D_std, D_min, D_max)
    L_sampler   = get_distribution(L_dist, L_mean, L_std, L_min, L_max)
    e_sampler   = get_distribution(epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max)
    mf_sampler  = get_distribution(mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max)
    elev_sampler= get_distribution(elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max)

    # Generate numeric samples
    rho_samples = rho_sampler(num_simulations)
    mu_samples_mPa = mu_sampler(num_simulations)   # in mPa·s
    D_input_samples = D_sampler(num_simulations)
    L_input_samples = L_sampler(num_simulations)
    epsilon_samples = e_sampler(num_simulations)
    mass_flow_samples = mf_sampler(num_simulations)
    elevation_samples = elev_sampler(num_simulations)

    # Convert mPa·s => Pa·s
    mu_samples = mu_samples_mPa * 1e-3

    # Convert diameter, length if needed
    if units_selected == "Inches":
        D_samples = D_input_samples * 0.0254
        L_samples = L_input_samples * 0.0254
    else:
        D_samples = D_input_samples
        L_samples = L_input_samples

    # Clip to avoid zeros or negative
    rho_samples = np.clip(rho_samples, a_min=1e-6, a_max=None)
    mu_samples  = np.clip(mu_samples, a_min=1e-12, a_max=None)
    D_samples   = np.clip(D_samples,   a_min=1e-6,  a_max=None)
    L_samples   = np.clip(L_samples,   a_min=1e-6,  a_max=None)
    epsilon_samples = np.clip(epsilon_samples, a_min=0.0, a_max=None)
    mass_flow_samples = np.clip(mass_flow_samples, a_min=1e-6, a_max=None)

    st.success("Random samples generated successfully.")

    # --------------------------------------------------------------------------------
    # CALCULATE PRESSURE DROPS
    # --------------------------------------------------------------------------------
    st.subheader("Performing Calculations...")

    # 1) Volumetric flow
    Q_samples = mass_flow_samples / rho_samples
    # 2) Cross-sectional area
    A_samples = np.pi * (D_samples/2) ** 2
    # 3) Velocity
    v_samples = Q_samples / A_samples
    # 4) Reynolds number
    Re_samples = (rho_samples * v_samples * D_samples) / mu_samples

    # 5) Friction factor (Swamee-Jain) with laminar correction
    with np.errstate(divide='ignore', invalid='ignore'):
        f_samples = 0.25 / (np.log10((epsilon_samples/(3.7*D_samples)) + (5.74/(Re_samples**0.9))))**2
    laminar_flow = (Re_samples <= 2000)
    f_samples[laminar_flow] = 64.0 / Re_samples[laminar_flow]

    # 6) Head loss (pipe)
    head_loss_pipe = f_samples * (L_samples/D_samples) * (v_samples**2) / (2.0*gravity)

    # 7) Fittings & Valves
    K_fittings = num_fittings * fitting_k
    K_valves   = num_valves * valve_k
    K_total    = K_fittings + K_valves
    head_loss_fittings = K_total * (v_samples**2) / (2.0*gravity)

    # 8) Elevation
    head_loss_elevation = elevation_samples

    # 9) Total head loss
    head_loss_total = head_loss_pipe + head_loss_fittings + head_loss_elevation

    # 10) Pressure drop in Pascals
    deltaP_samples = rho_samples * gravity * head_loss_total

    # Convert to psia if needed
    if unit_selection == 'Pounds per Square Inch Absolute (psia)':
        deltaP_samples = pa_to_psia(deltaP_samples)
        pressure_unit = 'psia'
    else:
        pressure_unit = 'Pa'

    st.success("Calculations completed.")

    # --------------------------------------------------------------------------------
    # RESULTS & STATISTICS
    # --------------------------------------------------------------------------------
    st.subheader("Results")

    mean_deltaP = np.mean(deltaP_samples)
    median_deltaP = np.median(deltaP_samples)
    std_deltaP = np.std(deltaP_samples)
    ci_lower = np.percentile(deltaP_samples, (100 - confidence_level)/2)
    ci_upper = np.percentile(deltaP_samples, confidence_level + (100 - confidence_level)/2)

    st.write(f"**Mean Pressure Drop:** {mean_deltaP:.4f} {pressure_unit}")
    st.write(f"**Median Pressure Drop:** {median_deltaP:.4f} {pressure_unit}")
    st.write(f"**Standard Deviation:** {std_deltaP:.4f} {pressure_unit}")
    st.write(f"**{confidence_level}% Confidence Interval:** "
             f"[{ci_lower:.4f}, {ci_upper:.4f}] {pressure_unit}")

    # --------------------------------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------------------------------
    st.subheader("Pressure Drop Distribution")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(deltaP_samples, bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel(f'Pressure Drop ({pressure_unit})')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Pressure Drop (Monte Carlo)')
    st.pyplot(fig)

    # --------------------------------------------------------------------------------
    # CDF
    # --------------------------------------------------------------------------------
    st.subheader("Cumulative Distribution Function (CDF)")
    sorted_deltaP = np.sort(deltaP_samples)
    cdf = np.arange(1, num_simulations+1)/num_simulations
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(sorted_deltaP, cdf, color='darkblue')
    ax2.set_xlabel(f'Pressure Drop ({pressure_unit})')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('CDF of Pressure Drop')
    ax2.grid(True)
    st.pyplot(fig2)

    # --------------------------------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # --------------------------------------------------------------------------------
    st.subheader("Sensitivity Analysis")
    data = pd.DataFrame({
        'Pressure Drop': deltaP_samples,
        'Density (kg/m³)': rho_samples,
        'Viscosity (Pa·s)': mu_samples,
        f'Diameter ({units_selected})': D_input_samples,
        f'Length ({units_selected})': L_input_samples,
        'Roughness (m)': epsilon_samples,
        'Mass Flow Rate (kg/s)': mass_flow_samples,
        'Elevation Change (m)': elevation_samples
    })

    corr_matrix = data.corr()
    pressure_drop_corr = corr_matrix['Pressure Drop'].drop('Pressure Drop')
    st.write("Correlation with Pressure Drop:")
    st.dataframe(pressure_drop_corr)

    fig3, ax3 = plt.subplots(figsize=(10,6))
    pressure_drop_corr.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_title('Sensitivity Analysis')
    st.pyplot(fig3)

    # --------------------------------------------------------------------------------
    # PREPARE FOR EXCEL DOWNLOAD
    # --------------------------------------------------------------------------------
    st.success("Simulation and analysis completed successfully.")
    st.subheader("Download Simulation Results")

    simulation_data_sheet = data.copy()
    summary_df = pd.DataFrame({
        'Statistic': ['Mean','Median','Std Dev',f'{confidence_level}% CI Lower',f'{confidence_level}% CI Upper'],
        'Value': [mean_deltaP, median_deltaP, std_deltaP, ci_lower, ci_upper],
        'Units': [pressure_unit]*5
    })

    sensitivity_df = pd.DataFrame(pressure_drop_corr).reset_index()
    sensitivity_df.columns = ['Parameter','Correlation with Pressure Drop']

    # Function to build Excel with embedded figures
    def to_excel(sim_data, summary_data, sensitivity_data, hist_fig, cdf_fig, sens_fig):
        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
            # Write sheets
            sim_data.to_excel(writer, sheet_name='Simulation Data', index=False)
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            sensitivity_data.to_excel(writer, sheet_name='Sensitivity', index=False)

            # Insert images into Summary
            workbook = writer.book
            summary_ws = writer.sheets['Summary']

            # Histogram
            png_hist = io.BytesIO()
            hist_fig.savefig(png_hist, format='png', bbox_inches='tight')
            png_hist.seek(0)
            summary_ws.insert_image('B15', 'Histogram', {'image_data': png_hist})

            # CDF
            png_cdf = io.BytesIO()
            cdf_fig.savefig(png_cdf, format='png', bbox_inches='tight')
            png_cdf.seek(0)
            summary_ws.insert_image('B40', 'CDF', {'image_data': png_cdf})

            # Sensitivity
            png_sens = io.BytesIO()
            sens_fig.savefig(png_sens, format='png', bbox_inches='tight')
            png_sens.seek(0)
            summary_ws.insert_image('B65', 'Sensitivity', {'image_data': png_sens})

        return out_xlsx.getvalue()

    # Build Excel file
    excel_file = to_excel(simulation_data_sheet, summary_df, sensitivity_df, fig, fig2, fig3)

    st.download_button(
        label="Download Results (Excel)",
        data=excel_file,
        file_name='simulation_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

else:
    st.write("Adjust parameters in the sidebar and click **Run Simulation** to begin.")