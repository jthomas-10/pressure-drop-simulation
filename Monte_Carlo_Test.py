import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import triang, uniform

# Set the page configuration
st.set_page_config(page_title="Pressure Drop Monte Carlo Simulation", layout="wide")

# Title
st.title("Monte Carlo Simulation for Liquid Pressure Drop Analysis")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Define a function to get distributions
def get_distribution(dist_name, mean, std_dev, min_val=None, max_val=None):
    if dist_name == 'Normal':
        return lambda size: np.random.normal(mean, std_dev, size)
    elif dist_name == 'Uniform':
        return lambda size: np.random.uniform(min_val, max_val, size)
    elif dist_name == 'Triangular':
        c = (mean - min_val) / (max_val - min_val)
        return lambda size: triang.rvs(c, loc=min_val, scale=(max_val - min_val), size=size)
    else:
        return lambda size: np.full(size, mean)  # Deterministic

# Fluid properties
st.sidebar.subheader("Fluid Properties")
rho_dist = st.sidebar.selectbox("Density Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=1)
rho_mean = st.sidebar.number_input("Fluid Density Mean (kg/m³)", value=1100.0, format="%.2f")
rho_std = st.sidebar.number_input("Fluid Density Std Dev (kg/m³)", value=10.0 if rho_dist == 'Normal' else 0.0, format="%.2f")
rho_min = st.sidebar.number_input("Fluid Density Min (kg/m³)", value=950.0 if rho_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")
rho_max = st.sidebar.number_input("Fluid Density Max (kg/m³)", value=1050.0 if rho_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")

mu_dist = st.sidebar.selectbox("Viscosity Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=0)
mu_mean = st.sidebar.number_input("Fluid Viscosity Mean (μPa·s)", value=195.0, format="%.2f")
mu_std = st.sidebar.number_input("Fluid Viscosity Std Dev (μPa·s)", value=0.0 if mu_dist == 'Normal' else 0.0, format="%.2f")
mu_min = st.sidebar.number_input("Fluid Viscosity Min (μPa·s)", value=193.0 if mu_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")
mu_max = st.sidebar.number_input("Fluid Viscosity Max (μPa·s)", value=197.0 if mu_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")

# Pipe properties
st.sidebar.subheader("Pipe Properties")
D_dist = st.sidebar.selectbox("Diameter Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=0)
D_mean = st.sidebar.number_input("Pipe Diameter Mean (inches)", value=2.0, format="%.3f")
D_std = st.sidebar.number_input("Pipe Diameter Std Dev (inches)", value=0.0 if D_dist == 'Normal' else 0.0, format="%.3f")
D_min = st.sidebar.number_input("Pipe Diameter Min (inches)", value=1.91 if D_dist in ['Uniform', 'Triangular'] else 0.0, format="%.3f")
D_max = st.sidebar.number_input("Pipe Diameter Max (inches)", value=2.09 if D_dist in ['Uniform', 'Triangular'] else 0.0, format="%.3f")

L_dist = st.sidebar.selectbox("Length Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=0)
L_mean = st.sidebar.number_input("Pipe Length Mean (m)", value=18.0, format="%.2f")
L_std = st.sidebar.number_input("Pipe Length Std Dev (m)", value=0.0 if L_dist == 'Normal' else 0.0, format="%.2f")
L_min = st.sidebar.number_input("Pipe Length Min (m)", value=16.0 if L_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")
L_max = st.sidebar.number_input("Pipe Length Max (m)", value=20.0 if L_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")

epsilon_dist = st.sidebar.selectbox("Roughness Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=0)
epsilon_mean = st.sidebar.number_input("Pipe Roughness Mean (m)", value=0.0001, format="%.6f")
epsilon_std = st.sidebar.number_input("Pipe Roughness Std Dev (m)", value=0.0 if epsilon_dist == 'Normal' else 0.0, format="%.6f")
epsilon_min = st.sidebar.number_input("Pipe Roughness Min (m)", value=0.00005 if epsilon_dist in ['Uniform', 'Triangular'] else 0.0, format="%.6f")
epsilon_max = st.sidebar.number_input("Pipe Roughness Max (m)", value=0.00015 if epsilon_dist in ['Uniform', 'Triangular'] else 0.0, format="%.6f")

# Flow properties
st.sidebar.subheader("Flow Properties")
mass_flow_dist = st.sidebar.selectbox("Mass Flow Rate Distribution", ['Deterministic', 'Normal', 'Uniform', 'Triangular'], index=1)
mass_flow_mean = st.sidebar.number_input("Mass Flow Rate Mean (kg/s)", value=3.0, format="%.4f")
mass_flow_std = st.sidebar.number_input("Mass Flow Rate Std Dev (kg/s)", value=0.25 if mass_flow_dist == 'Normal' else 0.0, format="%.4f")
mass_flow_min = st.sidebar.number_input("Mass Flow Rate Min (kg/s)", value=2.0 if mass_flow_dist in ['Uniform', 'Triangular'] else 0.0, format="%.4f")
mass_flow_max = st.sidebar.number_input("Mass Flow Rate Max (kg/s)", value=4.0 if mass_flow_dist in ['Uniform', 'Triangular'] else 0.0, format="%.4f")

# Fittings and Valves
st.sidebar.subheader("Fittings and Valves")
num_fittings = st.sidebar.number_input("Number of Fittings", min_value=0, value=2, format="%d")
num_valves = st.sidebar.number_input("Number of Valves", min_value=0, value=1, format="%d")

# Fitting K values
st.sidebar.markdown("**Fitting K Values**")
default_fitting_k = 0.5  # Default K value for a standard elbow or tee
fitting_k = st.sidebar.number_input("K Value per Fitting", value=default_fitting_k, format="%.2f")

# Valve K values
st.sidebar.markdown("**Valve K Values**")
default_valve_k = 10.0   # Default K value for a globe valve
valve_k = st.sidebar.number_input("K Value per Valve", value=default_valve_k, format="%.2f")

# Elevation Change
st.sidebar.subheader("Elevation Change")
elevation_dist = st.sidebar.selectbox(
    "Elevation Change Distribution",
    ['Deterministic', 'Normal', 'Uniform', 'Triangular'],
    index=0
)
elevation_mean = st.sidebar.number_input("Elevation Change Mean (m)", value=0.0, format="%.2f")
elevation_std = st.sidebar.number_input("Elevation Change Std Dev (m)", value=0.0 if elevation_dist == 'Normal' else 0.0, format="%.2f")
elevation_min = st.sidebar.number_input("Elevation Change Min (m)", value=-3.0 if elevation_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")
elevation_max = st.sidebar.number_input("Elevation Change Max (m)", value=3.0 if elevation_dist in ['Uniform', 'Triangular'] else 0.0, format="%.2f")

# Gravity Acceleration
st.sidebar.subheader("Gravity Acceleration")
gravity = st.sidebar.number_input(
    "Acceleration due to Gravity (m/s²)",
    value=9.81,
    format="%.4f"
)

# Simulation parameters
st.sidebar.header("Simulation Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1000, max_value=50000, value=10000, step=1000, format="%d")
confidence_level = st.sidebar.slider("Confidence Interval (%)", min_value=90, max_value=99, value=95, step=1)

# Unit selection
st.sidebar.header("Display Units")
unit_selection = st.sidebar.selectbox("Select Pressure Unit", ['Pascals (Pa)', 'Pounds per Square Inch Absolute (psia)'], index=0)

# Button to run simulation
run_simulation = st.sidebar.button("Run Simulation")

# Function to convert Pa to psia
def pa_to_psia(pa_values):
    psi_values = pa_values * 0.000145038  # 1 Pa = 0.000145038 psi
    return psi_values

# Main section
if run_simulation:
    st.header("Simulation Results")

    # Generate random samples
    st.subheader("Generating Random Samples...")

    # Get samplers for each parameter
    rho_sampler = get_distribution(rho_dist, rho_mean, rho_std, rho_min, rho_max)
    mu_sampler = get_distribution(mu_dist, mu_mean, mu_std, mu_min, mu_max)
    D_sampler = get_distribution(D_dist, D_mean, D_std, D_min, D_max)
    L_sampler = get_distribution(L_dist, L_mean, L_std, L_min, L_max)
    epsilon_sampler = get_distribution(epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max)
    mass_flow_sampler = get_distribution(mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max)
    elevation_sampler = get_distribution(elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max)

    # Generate samples
    rho_samples = rho_sampler(num_simulations)
    mu_samples = mu_sampler(num_simulations)
    D_samples_inch = D_sampler(num_simulations)  # Diameter in inches
    L_samples = L_sampler(num_simulations)
    epsilon_samples = epsilon_sampler(num_simulations)
    mass_flow_samples = mass_flow_sampler(num_simulations)
    elevation_samples = elevation_sampler(num_simulations)

    # Convert viscosity from μPa·s to Pa·s
    mu_samples = mu_samples * 1e-6  # Convert μPa·s to Pa·s

    # Convert diameter from inches to meters
    D_samples = D_samples_inch * 0.0254  # Convert inches to meters

    # Ensure all samples are positive and within physical limits
    rho_samples = np.clip(rho_samples, a_min=1e-6, a_max=None)
    mu_samples = np.clip(mu_samples, a_min=1e-12, a_max=None)
    D_samples = np.clip(D_samples, a_min=1e-6, a_max=None)
    L_samples = np.clip(L_samples, a_min=1e-6, a_max=None)
    epsilon_samples = np.clip(epsilon_samples, a_min=0.0, a_max=None)
    mass_flow_samples = np.clip(mass_flow_samples, a_min=1e-6, a_max=None)
    # Elevation change can be positive or negative, no need to clip to positive values

    st.success("Random samples generated successfully.")

    # Calculate intermediate variables
    st.subheader("Performing Calculations...")

    # Volumetric flow rate Q = mass_flow / density
    Q_samples = mass_flow_samples / rho_samples

    # Cross-sectional area
    A_samples = np.pi * (D_samples / 2) ** 2

    # Fluid velocity
    v_samples = Q_samples / A_samples

    # Reynolds number
    Re_samples = (rho_samples * v_samples * D_samples) / mu_samples

    # Friction factor using Swamee-Jain equation (approximate explicit formula)
    with np.errstate(divide='ignore', invalid='ignore'):
        f_samples = 0.25 / (np.log10((epsilon_samples / (3.7 * D_samples)) + (5.74 / Re_samples ** 0.9))) ** 2

    # For laminar flow (Re <= 2000), use f = 64 / Re
    laminar_flow = Re_samples <= 2000
    f_samples[laminar_flow] = 64 / Re_samples[laminar_flow]

    # Calculate head loss due to pipe friction
    head_loss_pipe = f_samples * (L_samples / D_samples) * (v_samples ** 2) / (2 * gravity)

    # Calculate K_total from fittings and valves
    K_fittings = num_fittings * fitting_k
    K_valves = num_valves * valve_k
    K_total = K_fittings + K_valves

    # Calculate head loss due to fittings and valves
    head_loss_fittings = K_total * (v_samples ** 2) / (2 * gravity)

    # Head loss due to elevation change
    head_loss_elevation = elevation_samples

    # Total head loss
    head_loss_total = head_loss_pipe + head_loss_fittings + head_loss_elevation

    # Convert head loss to pressure drop (ΔP = ρ * g * h)
    deltaP_samples = rho_samples * gravity * head_loss_total  # Pressure drop in Pascals

    # Convert pressure drop to desired units
    if unit_selection == 'Pounds per Square Inch Absolute (psia)':
        deltaP_samples = pa_to_psia(deltaP_samples)
        pressure_unit = 'psia'
    else:
        pressure_unit = 'Pa'

    st.success("Calculations completed.")

    # Analyze results
    st.subheader("Results")

    # Calculate statistics
    mean_deltaP = np.mean(deltaP_samples)
    median_deltaP = np.median(deltaP_samples)
    std_deltaP = np.std(deltaP_samples)
    ci_lower = np.percentile(deltaP_samples, (100 - confidence_level) / 2)
    ci_upper = np.percentile(deltaP_samples, confidence_level + (100 - confidence_level) / 2)

    st.write(f"**Mean Pressure Drop:** {mean_deltaP:.4f} {pressure_unit}")
    st.write(f"**Median Pressure Drop:** {median_deltaP:.4f} {pressure_unit}")
    st.write(f"**Standard Deviation:** {std_deltaP:.4f} {pressure_unit}")
    st.write(f"**{confidence_level}% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}] {pressure_unit}")

    # Plotting results
    st.subheader("Pressure Drop Distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deltaP_samples, bins=50, color='skyblue', edgecolor='black')
    ax.set_xlabel(f'Pressure Drop ({pressure_unit})')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Pressure Drop from Monte Carlo Simulation')
    st.pyplot(fig)

    # Show cumulative distribution function (CDF)
    st.subheader("Cumulative Distribution Function (CDF)")

    sorted_deltaP = np.sort(deltaP_samples)
    cdf = np.arange(1, num_simulations + 1) / num_simulations

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(sorted_deltaP, cdf, color='darkblue')
    ax2.set_xlabel(f'Pressure Drop ({pressure_unit})')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('CDF of Pressure Drop')
    ax2.grid(True)
    st.pyplot(fig2)

    # Sensitivity Analysis
    st.subheader("Sensitivity Analysis")

    # Create a DataFrame of inputs and outputs
    data = pd.DataFrame({
        'Pressure Drop': deltaP_samples,
        'Density': rho_samples,
        'Viscosity': mu_samples,
        'Diameter (in)': D_samples_inch,
        'Length': L_samples,
        'Roughness': epsilon_samples,
        'Mass Flow Rate': mass_flow_samples,
        'Elevation Change': elevation_samples,
        # Gravity is constant in each simulation run
    })

    # Calculate correlation coefficients
    corr_matrix = data.corr()
    pressure_drop_corr = corr_matrix['Pressure Drop'].drop('Pressure Drop')

    st.write("Correlation with Pressure Drop:")
    st.dataframe(pressure_drop_corr)

    # Display bar chart of correlation coefficients
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    pressure_drop_corr.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_title('Sensitivity Analysis')
    st.pyplot(fig3)

    st.success("Simulation and analysis completed successfully.")

    # Additional Functional Upgrade: Download Option
    st.subheader("Download Simulation Results")

    # Allow user to download the simulation data as a CSV file
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(data)

    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='simulation_results.csv',
        mime='text/csv',
    )

else:
    st.write("Configure the parameters in the sidebar and click **Run Simulation** to start.")
