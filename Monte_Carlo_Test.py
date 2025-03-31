import streamlit as st
import numpy as np  # Fixed import statement
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import triang
import io
import matplotlib as mpl

# Import CoolProp
from CoolProp.CoolProp import PropsSI

# --------------------------------------------------------------------------------
# ENGINEERING THEME SETUP
# --------------------------------------------------------------------------------
# Set engineering-friendly theme
COLORS = {
    "primary": "#1f77b4",    # Primary blue for main elements
    "secondary": "#ff7f0e",  # Orange for highlights/secondary elements
    "tertiary": "#2ca02c",   # Green for success/completion
    "highlight": "#d62728",  # Red for warnings/critical values
    "grid": "#cccccc",       # Light gray grid
    "background": "#f5f5f5", # Light background
    "text": "#333333"        # Dark text for contrast
}

# Custom matplotlib theme for all plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['font.family'] = 'DejaVu Sans'

# --------------------------------------------------------------------------------
# PAGE CONFIG AND TITLE
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Pressure Drop Monte Carlo Simulation",
    layout="wide"
)

# Initialize session state for persistent selections and inputs
if 'unit_selection' not in st.session_state:
    st.session_state.unit_selection = 'Pascals (Pa)'
if 'units_selected' not in st.session_state:
    st.session_state.units_selected = 'Inches'
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "setup"
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {}
if 'simulation_inputs' not in st.session_state:
    st.session_state.simulation_inputs = {}

# Custom CSS for better engineering visuals
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stApp {max-width: 1200px; margin: 0 auto;}
    h1 {color: #1f77b4;}
    h2 {color: #1f77b4; border-bottom: 1px solid #cccccc; padding-bottom: 0.3rem;}
    h3 {color: #2c3e50;}
    .stButton>button {background-color: #1f77b4; color: white;}
    .stProgress .st-bo {background-color: #2ca02c;}
    .tooltip {position: relative; display: inline-block; border-bottom: 1px dotted black;}
    .tooltip .tooltiptext {visibility: hidden; width: 200px; background-color: #555;
                          color: #fff; text-align: center; border-radius: 6px;
                          padding: 5px; position: absolute; z-index: 1;
                          bottom: 125%; left: 50%; margin-left: -100px;
                          opacity: 0; transition: opacity 0.3s;}
    .tooltip:hover .tooltiptext {visibility: visible; opacity: 1;}
    .stExpander {border: 1px solid #ddd; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("Pressure Drop Monte Carlo Simulation")
st.markdown("""
<div style="background-color: #e3f2fd; padding: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;">
This application conducts Monte Carlo simulations to analyze the uncertainty of <b>pressure drop</b> in pipe flow scenarios
using the <b>Darcy-Weisbach</b> equation and <b>CoolProp</b> for accurate fluid properties.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def get_distribution(dist_name, mean, std_dev, min_val=None, max_val=None):
    """Return a function that generates random samples from the specified distribution"""
    if dist_name == 'Normal':
        return lambda size: np.random.normal(mean, std_dev, size)
    elif dist_name == 'Uniform':
        return lambda size: np.random.uniform(min_val, max_val, size)
    elif dist_name == 'Triangular':
        c = (mean - min_val) / (max_val - min_val)
        return lambda size: triang.rvs(c, loc=min_val, scale=(max_val - min_val), size=size)
    else:
        # Deterministic (constant value)
        return lambda size: np.full(size, mean)

def pa_to_psia(pa_values):
    """Convert Pascal to psia"""
    return pa_values * 0.000145038

def create_distribution_inputs(label, default_mean, unit, key_prefix, default_dist="Deterministic",
                              default_std=None, default_min=None, default_max=None, tooltip=None):
    """Create standardized distribution input fields"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        dist = st.selectbox(
            f"{label} Distribution", 
            ['Deterministic', 'Normal', 'Uniform', 'Triangular'], 
            index=['Deterministic', 'Normal', 'Uniform', 'Triangular'].index(default_dist),
            key=f"{key_prefix}_dist"
        )
        if tooltip:
            st.markdown(f"<div class='tooltip'>?<span class='tooltiptext'>{tooltip}</span></div>", unsafe_allow_html=True)
    
    with col2:
        mean = st.number_input(
            f"Mean ({unit})", 
            value=default_mean,
            format="%.4g", 
            key=f"{key_prefix}_mean"
        )
        
        if dist == 'Normal':
            std = st.number_input(
                f"Std Dev ({unit})",
                value=default_std if default_std is not None else default_mean * 0.05,
                format="%.4g",
                key=f"{key_prefix}_std"
            )
        else:
            std = 0.0
            
        if dist in ['Uniform', 'Triangular']:
            min_val = st.number_input(
                f"Min ({unit})",
                value=default_min if default_min is not None else default_mean * 0.9,
                format="%.4g",
                key=f"{key_prefix}_min"
            )
            max_val = st.number_input(
                f"Max ({unit})",
                value=default_max if default_max is not None else default_mean * 1.1,
                format="%.4g",
                key=f"{key_prefix}_max"
            )
        else:
            min_val = 0.0
            max_val = 0.0
            
    return dist, mean, std, min_val, max_val

# Set up tab switching function
def switch_to_results():
    st.session_state.active_tab = "results"
    
def switch_to_setup():
    st.session_state.active_tab = "setup"

# Function to store all inputs for access in the results tab
def save_inputs_to_session_state(
        rho_dist, rho_mean, rho_std, rho_min, rho_max,
        mu_dist, mu_mean, mu_std, mu_min, mu_max,
        D_dist, D_mean, D_std, D_min, D_max,
        L_dist, L_mean, L_std, L_min, L_max,
        epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max,
        mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max,
        elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max,
        num_simulations, confidence_level, num_fittings, fitting_k, num_valves, valve_k, gravity, selected_fluid
    ):
    """Store all simulation inputs in the session state for persistence between tabs"""
    st.session_state.simulation_inputs = {
        # Fluid properties
        'rho_dist': rho_dist,
        'rho_mean': rho_mean,
        'rho_std': rho_std,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'mu_dist': mu_dist, 
        'mu_mean': mu_mean,
        'mu_std': mu_std,
        'mu_min': mu_min,
        'mu_max': mu_max,
        # Pipe geometry
        'D_dist': D_dist,
        'D_mean': D_mean,
        'D_std': D_std,
        'D_min': D_min, 
        'D_max': D_max,
        'L_dist': L_dist,
        'L_mean': L_mean,
        'L_std': L_std,
        'L_min': L_min,
        'L_max': L_max,
        'epsilon_dist': epsilon_dist,
        'epsilon_mean': epsilon_mean,
        'epsilon_std': epsilon_std,
        'epsilon_min': epsilon_min,
        'epsilon_max': epsilon_max,
        # Flow properties
        'mass_flow_dist': mass_flow_dist,
        'mass_flow_mean': mass_flow_mean,
        'mass_flow_std': mass_flow_std,
        'mass_flow_min': mass_flow_min,
        'mass_flow_max': mass_flow_max,
        'elevation_dist': elevation_dist,
        'elevation_mean': elevation_mean,
        'elevation_std': elevation_std,
        'elevation_min': elevation_min,
        'elevation_max': elevation_max,
        # Other parameters
        'num_simulations': num_simulations,
        'confidence_level': confidence_level,
        'num_fittings': num_fittings,
        'fitting_k': fitting_k,
        'num_valves': num_valves,
        'valve_k': valve_k,
        'gravity': gravity,
        'selected_fluid': selected_fluid
    }
    st.session_state.simulation_run = True

# Create tab-like UI with buttons
col1, col2, col3 = st.columns([1, 1, 6])
with col1:
    setup_button = st.button("📊 Setup", 
                            key="setup_tab", 
                            on_click=switch_to_setup,
                            use_container_width=True,
                            type="primary" if st.session_state.active_tab == "setup" else "secondary")
with col2:
    results_button = st.button("📈 Results", 
                              key="results_tab", 
                              on_click=switch_to_results,
                              use_container_width=True,
                              type="primary" if st.session_state.active_tab == "results" else "secondary")

# Setup Tab Content
if st.session_state.active_tab == "setup":
    # Create two columns: one for fluid & simulation setup, one for pipe parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Fluid Selection & Properties")
        
        # Fluid selection with common options - Fix parahydrogen name for CoolProp compatibility
        fluid_options = ["Water", "Oxygen", "Nitrogen", "Hydrogen", "Air", "CarbonDioxide"]
        fluid_coolprop_names = {
            "Water": "Water", 
            "Oxygen": "Oxygen", 
            "Nitrogen": "Nitrogen", 
            "Hydrogen": "ParaHydrogen",  # Correct CoolProp name for parahydrogen
            "Air": "Air", 
            "CarbonDioxide": "CarbonDioxide"
        }
        fluid_icons = {
            "Water": "💧", "Oxygen": "🔵", "Nitrogen": "🔹", 
            "Hydrogen": "⚛️", "Air": "💨", "CarbonDioxide": "🌫️"
        }
        
        selected_fluid_idx = st.selectbox(
            "Select Fluid:",
            range(len(fluid_options)),
            format_func=lambda x: f"{fluid_icons.get(fluid_options[x], '•')} {fluid_options[x]}"
        )
        selected_fluid = fluid_options[selected_fluid_idx]
        
        # Fluid state parameters in collapsible section
        with st.expander("Fluid State Parameters", expanded=True):
            # Pressure and temperature in a 2-column layout
            p_col, t_col = st.columns(2)
            
            with p_col:
                pressure_kPa = st.number_input(
                    "Pressure (kPa)", 
                    value=101.325, 
                    min_value=10.0,
                    step=10.0,
                    format="%.3g"
                )
                
            with t_col:
                temperature_K = st.number_input(
                    "Temperature (K)", 
                    value=300.0, 
                    min_value=14.0,  # Lower minimum temperature to allow cryogenic fluids
                    step=10.0,
                    format="%.3g"
                )
                
            # Query CoolProp for properties - use the correct CoolProp fluid name
            try:
                coolprop_fluid = fluid_coolprop_names[selected_fluid]
                calc_density = PropsSI("D", "T", temperature_K, "P", pressure_kPa * 1000.0, coolprop_fluid)  # kg/m³
                calc_viscosity = PropsSI("V", "T", temperature_K, "P", pressure_kPa * 1000.0, coolprop_fluid)  # Pa·s
                calc_viscosity_mPas = calc_viscosity * 1e3
                
                # Visual feedback for successful property retrieval
                st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <b>Fluid Properties (CoolProp)</b><br>
                    Density: <b>{calc_density:.4g}</b> kg/m³<br>
                    Viscosity: <b>{calc_viscosity_mPas:.4g}</b> mPa·s
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error retrieving fluid properties: {e}")
                calc_density = 1.0
                calc_viscosity = 1e-3
                calc_viscosity_mPas = 1.0
        
        # Fluid properties distributions with better organization
        with st.expander("Fluid Property Distributions", expanded=False):
            st.info("Adjust these only if you want to override CoolProp values with custom distributions")
            
            rho_dist, rho_mean, rho_std, rho_min, rho_max = create_distribution_inputs(
                "Density", calc_density, "kg/m³", "rho",
                tooltip="Fluid density affects Reynolds number and pressure drop"
            )
            
            mu_dist, mu_mean, mu_std, mu_min, mu_max = create_distribution_inputs(
                "Viscosity", calc_viscosity_mPas, "mPa·s", "mu",
                tooltip="Fluid viscosity affects Reynolds number and friction factor"
            )
        
        # Simulation parameters
        st.header("Simulation Parameters")
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            num_simulations = st.number_input(
                "Number of Simulations", 
                min_value=1000, 
                max_value=100000, 
                value=5000, 
                step=1000,
                format="%d"
            )
            
        with sim_col2:
            confidence_level = st.slider(
                "Confidence Interval (%)", 
                min_value=90, 
                max_value=99, 
                value=95, 
                step=1
            )
            
        # Output units selection with visual indicators - Fix unit selection persistence
        st.subheader("Output Units")
        
        unit_options = ['Pascals (Pa)', 'kPa', 'bar', 'psi']
        unit_icons = ['⚖️', '🔄', '📊', '🇺🇸']
        
        unit_cols = st.columns(len(unit_options))
        for i, (col, unit, icon) in enumerate(zip(unit_cols, unit_options, unit_icons)):
            with col:
                if st.button(f"{icon} {unit}", key=f"unit_{i}"):
                    st.session_state.unit_selection = unit_options[i]
        
        st.markdown(f"Selected unit: **{st.session_state.unit_selection}**")
        
    with col2:
        st.header("Pipe Configuration")
        
        # Units selection with more visual distinction - Fix units persistence
        units_col1, units_col2 = st.columns(2)
        with units_col1:
            if st.button("🇺🇸 US Units", key="us_units"):
                st.session_state.units_selected = "Inches"
        with units_col2:
            if st.button("🌍 Metric Units", key="metric_units"):
                st.session_state.units_selected = "Meters"
        
        st.markdown(f"Using **{st.session_state.units_selected}** for dimensions")
        
        def length_label(base_label):
            return f"{base_label} ({'inches' if st.session_state.units_selected=='Inches' else 'meters'})"

        # Pipe geometry in an expander
        with st.expander("Pipe Geometry", expanded=True):
            # Pipe diameter with distribution
            D_dist, D_mean, D_std, D_min, D_max = create_distribution_inputs(
                length_label("Pipe Diameter"), 
                2.0 if st.session_state.units_selected=="Inches" else 0.05, 
                st.session_state.units_selected.lower(), 
                "D",
                tooltip="Inner diameter of the pipe - key parameter for pressure drop"
            )
            
            # Pipe length with distribution
            L_dist, L_mean, L_std, L_min, L_max = create_distribution_inputs(
                length_label("Pipe Length"), 
                18.0 if st.session_state.units_selected=="Inches" else 6.0, 
                st.session_state.units_selected.lower(), 
                "L",
                tooltip="Total straight pipe length"
            )
            
            # Pipe roughness with distribution (always in meters)
            epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max = create_distribution_inputs(
                "Roughness", 0.000015, "m", "epsilon",
                tooltip="Absolute roughness - affects friction factor calculation"
            )
            
        # Flow properties
        with st.expander("Flow Configuration", expanded=True):
            # Mass flow with distribution
            mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max = create_distribution_inputs(
                "Mass Flow Rate", 3.0, "kg/s", "mass_flow", "Normal", 0.25,
                tooltip="Mass flow rate through the pipe"
            )
            
            # Elevation change with distribution
            elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max = create_distribution_inputs(
                "Elevation Change", 0.0, "m", "elevation",
                tooltip="Positive values indicate upward flow (increases pressure drop)"
            )
            
        # Fittings and valves in a compact layout
        with st.expander("Fittings & Valves", expanded=False):
            fit_col1, fit_col2 = st.columns(2)
            with fit_col1:
                num_fittings = st.number_input("Number of Fittings", min_value=0, value=2, format="%d")
                fitting_k = st.number_input("K Value per Fitting", value=0.5, format="%.2f",
                                          help="Typical values: 0.5 for elbows, 1.0 for tees")
            with fit_col2:
                num_valves = st.number_input("Number of Valves", min_value=0, value=1, format="%d")
                valve_k = st.number_input("K Value per Valve", value=10.0, format="%.2f",
                                        help="Typical values: 3-10 depending on valve type")
        
        # Advanced physics parameters
        with st.expander("Advanced Parameters", expanded=False):
            gravity = st.number_input(
                "Acceleration due to Gravity (m/s²)", 
                value=9.81, 
                format="%.4f",
                help="Default value is for Earth (9.81 m/s²)"
            )
            
        # Large, prominent "Run Simulation" button
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Define a function to run the simulation and save inputs - This function is simplified
        def run_and_save():
            # Call the proper function with all the input parameters
            save_inputs_to_session_state(
                rho_dist, rho_mean, rho_std, rho_min, rho_max,
                mu_dist, mu_mean, mu_std, mu_min, mu_max,
                D_dist, D_mean, D_std, D_min, D_max,
                L_dist, L_mean, L_std, L_min, L_max,
                epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max,
                mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max,
                elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max,
                num_simulations, confidence_level, num_fittings, fitting_k, num_valves, valve_k, gravity, selected_fluid
            )
        
        run_simulation = st.button(
            "▶ Run Monte Carlo Simulation", 
            key="run_simulation",
            on_click=run_and_save,
            use_container_width=True
        )
        
        # Display pipe geometry diagram - Fix valve distribution
        st.markdown("### Pipe Configuration Diagram")
        fig_pipe = plt.figure(figsize=(8, 3))
        ax = fig_pipe.add_subplot(111)
        
        # Draw pipe
        pipe_length = 8
        pipe_diameter = 0.5
        rect = plt.Rectangle((0, 0), pipe_length, pipe_diameter, fc='lightgray', ec='black')
        ax.add_patch(rect)
        
        # Draw fittings - distribute evenly
        if num_fittings > 0:
            for i in range(num_fittings):
                pos = (i+1) * pipe_length / (num_fittings + 1)
                plt.scatter(pos, pipe_diameter/2, color=COLORS["secondary"], s=100, zorder=3, marker='o')
                plt.text(pos, pipe_diameter/2-0.15, "Fitting", ha='center', fontsize=8)
        
        # Draw valves - distribute evenly like fittings
        if num_valves > 0:
            for i in range(num_valves):
                valve_pos = (i+1) * pipe_length / (num_valves + 1)
                plt.scatter(valve_pos, pipe_diameter/2, color=COLORS["highlight"], s=150, zorder=3, marker='s')
                plt.text(valve_pos, pipe_diameter/2-0.15, "Valve", ha='center', fontsize=8)
        
        # Labels
        plt.text(-0.2, pipe_diameter/2, "In", ha='center')
        plt.text(pipe_length+0.2, pipe_diameter/2, "Out", ha='center')
        plt.text(pipe_length/2, pipe_diameter+0.1, f"L = {L_mean} {st.session_state.units_selected.lower()}", ha='center')
        plt.text(pipe_length/2, -0.15, f"D = {D_mean} {st.session_state.units_selected.lower()}", ha='center')
        
        ax.set_xlim(-0.5, pipe_length+0.5)
        ax.set_ylim(-0.5, pipe_diameter+0.5)
        ax.axis('off')
        fig_pipe.tight_layout()
        
        st.pyplot(fig_pipe)
        
    # Add a notification after running simulation to navigate to Results tab
    if run_simulation:
        # Store simulation flag in session state
        st.session_state.simulation_run = True
        
        # Display a success message
        st.success("""
        ✅ **Simulation Completed Successfully!**
        
        Your results are now available in the Results tab. 
        Click below to view pressure drop analysis and statistics.
        """)
        
        # Use direct tab switching
        if st.button("📊 View Results Now", 
                     on_click=switch_to_results, 
                     use_container_width=True):
            pass  # This doesn't need any code as the on_click handler does the work

# Results Tab Content
elif st.session_state.active_tab == "results":
    if st.session_state.simulation_run:
        # Get all inputs from session state
        inputs = st.session_state.simulation_inputs
        
        # Unpack inputs for easier reference
        rho_dist = inputs['rho_dist']
        rho_mean = inputs['rho_mean']
        rho_std = inputs['rho_std']
        rho_min = inputs['rho_min']
        rho_max = inputs['rho_max']
        mu_dist = inputs['mu_dist']
        mu_mean = inputs['mu_mean']
        mu_std = inputs['mu_std']
        mu_min = inputs['mu_min']
        mu_max = inputs['mu_max']
        D_dist = inputs['D_dist']
        D_mean = inputs['D_mean']
        D_std = inputs['D_std']
        D_min = inputs['D_min']
        D_max = inputs['D_max']
        L_dist = inputs['L_dist']
        L_mean = inputs['L_mean']
        L_std = inputs['L_std']
        L_min = inputs['L_min']
        L_max = inputs['L_max']
        epsilon_dist = inputs['epsilon_dist']
        epsilon_mean = inputs['epsilon_mean']
        epsilon_std = inputs['epsilon_std']
        epsilon_min = inputs['epsilon_min']
        epsilon_max = inputs['epsilon_max']
        mass_flow_dist = inputs['mass_flow_dist']
        mass_flow_mean = inputs['mass_flow_mean']
        mass_flow_std = inputs['mass_flow_std']
        mass_flow_min = inputs['mass_flow_min']
        mass_flow_max = inputs['mass_flow_max']
        elevation_dist = inputs['elevation_dist']
        elevation_mean = inputs['elevation_mean']
        elevation_std = inputs['elevation_std']
        elevation_min = inputs['elevation_min']
        elevation_max = inputs['elevation_max']  # Fixed reference
        num_simulations = inputs['num_simulations']
        confidence_level = inputs['confidence_level']
        num_fittings = inputs['num_fittings']
        fitting_k = inputs['fitting_k']
        num_valves = inputs['num_valves']
        valve_k = inputs['valve_k']
        gravity = inputs['gravity']
        selected_fluid = inputs['selected_fluid']
        
        st.header("Simulation Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Generate Random Samples
        status_text.text("Generating random samples...")
        progress_bar.progress(10)
        
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
    
        # Convert units as needed
        mu_samples = mu_samples_mPa * 1e-3  # Convert from mPa·s to Pa·s (kg/m·s)
    
        # Convert diameter, length if needed
        if st.session_state.units_selected == "Inches":
            D_samples = D_input_samples * 0.0254  # Convert from inches to meters
            L_samples = L_input_samples * 0.0254  # Convert from inches to meters
        else:
            D_samples = D_input_samples  # Already in meters
            L_samples = L_input_samples  # Already in meters
    
        # Clip to avoid zeros or negative values
        rho_samples = np.clip(rho_samples, a_min=1e-6, a_max=None)
        mu_samples  = np.clip(mu_samples, a_min=1e-12, a_max=None)
        D_samples   = np.clip(D_samples, a_min=1e-6, a_max=None)
        L_samples   = np.clip(L_samples, a_min=1e-6, a_max=None)
        epsilon_samples = np.clip(epsilon_samples, a_min=1e-12, a_max=None)
        mass_flow_samples = np.clip(mass_flow_samples, a_min=1e-6, a_max=None)
    
        progress_bar.progress(30)
        status_text.text("Calculating flow parameters...")
        
        # Core calculations
        # 1) Volumetric flow (m³/s)
        Q_samples = mass_flow_samples / rho_samples  # kg/s ÷ kg/m³ = m³/s
        
        # 2) Cross-sectional area (m²)
        A_samples = np.pi * (D_samples/2) ** 2  # m²
        
        # 3) Velocity (m/s)
        v_samples = Q_samples / A_samples  # m³/s ÷ m² = m/s
        
        # 4) Reynolds number (dimensionless)
        # Re = ρvD/μ where:
        # ρ = density (kg/m³)
        # v = velocity (m/s)
        # D = diameter (m)
        # μ = dynamic viscosity (Pa·s or kg/m·s)
        Re_samples = (rho_samples * v_samples * D_samples) / mu_samples  # (kg/m³ × m/s × m) ÷ kg/(m·s) = dimensionless
        
        progress_bar.progress(50)
        status_text.text("Computing friction factors...")
        
        # 5) Friction factor (Swamee-Jain) with laminar correction
        with np.errstate(divide='ignore', invalid='ignore'):
            f_turbulent = 0.25 / (np.log10((epsilon_samples/(3.7*D_samples)) + (5.74/(Re_samples**0.9))))**2
        laminar_flow = (Re_samples <= 2000)
        f_samples = np.where(laminar_flow, 64.0 / Re_samples, f_turbulent)
        
        progress_bar.progress(70)
        status_text.text("Calculating pressure drops...")
        
        # 6) Head loss (pipe) in meters of fluid
        head_loss_pipe = f_samples * (L_samples/D_samples) * (v_samples**2) / (2.0*gravity)  # m
        
        # 7) Fittings & Valves head loss in meters of fluid
        K_fittings = num_fittings * fitting_k  # dimensionless
        K_valves = num_valves * valve_k  # dimensionless
        K_total = K_fittings + K_valves  # dimensionless
        head_loss_fittings = K_total * (v_samples**2) / (2.0*gravity)  # m
        
        # 8) Elevation head in meters of fluid
        head_loss_elevation = elevation_samples  # m
        
        # 9) Total head loss in meters of fluid
        head_loss_total = head_loss_pipe + head_loss_fittings + head_loss_elevation  # m
        
        # 10) Pressure drop calculation
        # ΔP = ρgh where:
        # ρ = density (kg/m³)
        # g = gravitational acceleration (m/s²)
        # h = head loss (m)
        deltaP_samples = rho_samples * gravity * head_loss_total  # kg/m³ × m/s² × m = kg/(m·s²) = Pa
        
        progress_bar.progress(80)
        status_text.text("Finalizing calculations...")
        
        # Convert to selected pressure units - Fix unit conversion
        if st.session_state.unit_selection == 'psi':
            deltaP_samples = pa_to_psia(deltaP_samples)
            pressure_unit = 'psi'
        elif st.session_state.unit_selection == 'kPa':
            deltaP_samples = deltaP_samples / 1000
            pressure_unit = 'kPa'
        elif st.session_state.unit_selection == 'bar':
            deltaP_samples = deltaP_samples / 100000
            pressure_unit = 'bar'
        else:  # Default is Pascal
            pressure_unit = 'Pa'
            
        progress_bar.progress(100)
        status_text.text("Simulation completed successfully!")
        
        # --------------------------------------------------------------------------------
        # RESULTS DISPLAY
        # --------------------------------------------------------------------------------
        st.header("Simulation Results")
        
        # Create columns for key statistics
        stat_cols = st.columns(4)
        
        # Calculate key statistics
        mean_deltaP = np.mean(deltaP_samples)
        median_deltaP = np.median(deltaP_samples)
        std_deltaP = np.std(deltaP_samples)
        ci_lower = np.percentile(deltaP_samples, (100 - confidence_level)/2)
        ci_upper = np.percentile(deltaP_samples, confidence_level + (100 - confidence_level)/2)
        
        # Display in metric cards
        with stat_cols[0]:
            st.metric("Mean Pressure Drop", f"{mean_deltaP:.4g} {pressure_unit}")
            
        with stat_cols[1]:
            st.metric("Median Pressure Drop", f"{median_deltaP:.4g} {pressure_unit}")
            
        with stat_cols[2]:
            st.metric("Standard Deviation", f"{std_deltaP:.4g} {pressure_unit}")
            
        with stat_cols[3]:
            st.metric(f"{confidence_level}% CI Width", f"{ci_upper-ci_lower:.4g} {pressure_unit}")
        
        # Show confidence interval as a range
        st.markdown(f"""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <b>{confidence_level}% Confidence Interval:</b> [{ci_lower:.4g}, {ci_upper:.4g}] {pressure_unit}
        </div>
        """, unsafe_allow_html=True)
        
        # Engineering metrics - dimensionless numbers
        st.subheader("Engineering Parameters (Averages)")
        eng_cols = st.columns(3)
        
        with eng_cols[0]:
            mean_re = np.mean(Re_samples)
            st.metric("Reynolds Number", f"{mean_re:.3g}")
            
            # Show a more detailed tooltip about Reynolds number
            st.markdown("""
            <div style="font-size: 0.85em; color: #666;">
            Re = ρvD/μ (density × velocity × diameter ÷ viscosity)
            </div>
            """, unsafe_allow_html=True)
            
            # More accurate flow regime classification with clear breakpoints
            if mean_re < 2000:
                flow_regime = "Laminar"
                re_icon = "➡️"
                regime_detail = "Smooth, orderly flow with parallel streamlines"
            elif mean_re < 4000:
                flow_regime = "Transitional"
                re_icon = "↔️"
                regime_detail = "Mix of laminar and turbulent characteristics"
            else:
                flow_regime = "Turbulent"
                re_icon = "🌊"
                regime_detail = "Chaotic flow with eddies and vortices"
                
            # Also show percentage of samples in each regime for more insight
            pct_laminar = np.mean(Re_samples < 2000) * 100
            pct_transitional = np.mean((Re_samples >= 2000) & (Re_samples < 4000)) * 100
            pct_turbulent = np.mean(Re_samples >= 4000) * 100
            
            st.markdown(f"Flow Regime: {re_icon} **{flow_regime}** <br><small>{regime_detail}</small>", unsafe_allow_html=True)
            
            # Only show detailed breakdown if there's a mix of regimes or specific threshold
            if min(pct_laminar, pct_turbulent) > 5 or pct_transitional > 20:
                st.markdown(f"""<small>
                Distribution: {pct_laminar:.1f}% Laminar, 
                {pct_transitional:.1f}% Transitional, 
                {pct_turbulent:.1f}% Turbulent</small>""", unsafe_allow_html=True)
        
        with eng_cols[1]:
            st.metric("Friction Factor", f"{np.mean(f_samples):.5g}")
            # Add unit clarification
            st.markdown(f"""
            <div style="font-size: 0.85em; color: #666;">
            Dimensionless coefficient in Darcy-Weisbach equation
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"ε/D: **{np.mean(epsilon_samples/D_samples):.2g}** <small>(Relative roughness)</small>", unsafe_allow_html=True)
            
        with eng_cols[2]:
            st.metric("Flow Velocity", f"{np.mean(v_samples):.3g} m/s")
            velocity_level = "High" if np.mean(v_samples) > 3.0 else "Moderate" if np.mean(v_samples) > 1.5 else "Low"
            st.markdown(f"Velocity Level: **{velocity_level}**")
            
            # Add unit conversion for convenience
            fps_velocity = np.mean(v_samples) * 3.28084
            st.markdown(f"<small>({fps_velocity:.3g} ft/s)</small>", unsafe_allow_html=True)
        
        # Tabbed visualization of results
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Histogram", "📈 CDF", "🔄 Sensitivity", "📑 Data Table"
        ])
        
        # Tab 1: Improved histogram
        with viz_tab1:
            hist_fig, hist_ax = plt.subplots(figsize=(10, 6))
            
            # Create histogram with KDE
            hist_ax.hist(
                deltaP_samples, 
                bins=50, 
                color=COLORS["primary"], 
                edgecolor='white', 
                alpha=0.7,
                density=True
            )
            
            # Add a KDE line
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(deltaP_samples)
            x = np.linspace(min(deltaP_samples), max(deltaP_samples), 1000)
            hist_ax.plot(x, kde(x), color=COLORS["secondary"], linewidth=2)
            
            # Add mean and CI markers
            hist_ax.axvline(mean_deltaP, color=COLORS["highlight"], linestyle='-', linewidth=2, label=f'Mean: {mean_deltaP:.4g} {pressure_unit}')
            hist_ax.axvline(ci_lower, color=COLORS["highlight"], linestyle='--', linewidth=1.5, label=f'{confidence_level}% CI Lower: {ci_lower:.4g} {pressure_unit}')
            hist_ax.axvline(ci_upper, color=COLORS["highlight"], linestyle='--', linewidth=1.5, label=f'{confidence_level}% CI Upper: {ci_upper:.4g} {pressure_unit}')
            
            # Format axes and add title/labels
            hist_ax.set_xlabel(f'Pressure Drop ({pressure_unit})', fontsize=12)
            hist_ax.set_ylabel('Probability Density', fontsize=12)
            hist_ax.set_title('Pressure Drop Distribution', fontsize=14, fontweight='bold')
            hist_ax.grid(True, linestyle='--', alpha=0.7)
            hist_ax.legend(loc='best')
            
            # Add annotations
            hist_ax.annotate(
                f"n = {num_simulations}\nμ = {mean_deltaP:.4g} {pressure_unit}\nσ = {std_deltaP:.4g} {pressure_unit}", 
                xy=(0.03, 0.92), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
            )
            
            hist_fig.tight_layout()
            st.pyplot(hist_fig)
            
        # Tab 2: Improved CDF
        with viz_tab2:
            cdf_fig, cdf_ax = plt.subplots(figsize=(10, 6))
            
            # Sort data for CDF
            sorted_deltaP = np.sort(deltaP_samples)
            cdf = np.arange(1, num_simulations+1)/num_simulations
            
            # Plot CDF with engineering styling
            cdf_ax.plot(sorted_deltaP, cdf, color=COLORS["primary"], linewidth=2.5)
            
            # Add confidence interval
            cdf_ax.axvline(ci_lower, color=COLORS["highlight"], linestyle='--', linewidth=1.5)
            cdf_ax.axhline((100-confidence_level)/200, color=COLORS["highlight"], linestyle=':', linewidth=1)
            cdf_ax.axvline(ci_upper, color=COLORS["highlight"], linestyle='--', linewidth=1.5)
            cdf_ax.axhline(1-((100-confidence_level)/200), color=COLORS["highlight"], linestyle=':', linewidth=1)
            
            # Fill confidence interval region
            idx_lower = np.searchsorted(sorted_deltaP, ci_lower)
            idx_upper = np.searchsorted(sorted_deltaP, ci_upper)
            cdf_ax.fill_between(
                sorted_deltaP[idx_lower:idx_upper+1], 
                cdf[idx_lower:idx_upper+1], 
                color=COLORS["primary"], 
                alpha=0.2,
                label=f'{confidence_level}% Confidence Interval'
            )
            
            # Format axes and title/labels
            cdf_ax.set_xlabel(f'Pressure Drop ({pressure_unit})', fontsize=12)
            cdf_ax.set_ylabel('Cumulative Probability', fontsize=12)
            cdf_ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
            cdf_ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add percentile lines
            for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
                percentile_val = np.percentile(deltaP_samples, p*100)
                cdf_ax.plot([percentile_val, percentile_val], [0, p], 'k:', linewidth=0.8, alpha=0.6)
                cdf_ax.plot([min(deltaP_samples), percentile_val], [p, p], 'k:', linewidth=0.8, alpha=0.6)
                cdf_ax.annotate(
                    f"{int(p*100)}%", 
                    xy=(percentile_val, 0.02), 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    ha='center', 
                    fontsize=8
                )
            
            cdf_fig.tight_layout()
            st.pyplot(cdf_fig)
            
        # Tab 3: Enhanced sensitivity analysis
        with viz_tab3:
            # Create dataframe with all parameters and results
            data = pd.DataFrame({
                'Pressure Drop': deltaP_samples,
                'Density (kg/m³)': rho_samples,
                'Viscosity (Pa·s)': mu_samples,
                f'Diameter ({st.session_state.units_selected})': D_input_samples,
                f'Length ({st.session_state.units_selected})': L_input_samples,
                'Roughness (m)': epsilon_samples,
                'Mass Flow Rate (kg/s)': mass_flow_samples,
                'Elevation Change (m)': elevation_samples,
                'Reynolds Number': Re_samples,
                'Friction Factor': f_samples,
                'Velocity (m/s)': v_samples
            })
            
            # Calculate correlation matrix
            corr_matrix = data.corr()
            pressure_drop_corr = corr_matrix['Pressure Drop'].drop('Pressure Drop').sort_values(key=abs, ascending=False)
            
            # Create two-column layout
            sens_col1, sens_col2 = st.columns([2, 1])
            
            with sens_col1:
                # Enhanced correlation bar chart
                sens_fig, sens_ax = plt.subplots(figsize=(10, 6))
                
                # Plot bars with color based on value
                bars = pressure_drop_corr.plot(
                    kind='barh', 
                    ax=sens_ax,
                    color=[COLORS["primary"] if x > 0 else COLORS["highlight"] for x in pressure_drop_corr]
                )
                
                # Add value labels to bars
                for i, v in enumerate(pressure_drop_corr):
                    sens_ax.text(
                        v + (0.01 if v >= 0 else -0.01), 
                        i, 
                        f'{v:.3f}', 
                        va='center', 
                        ha='left' if v >= 0 else 'right',
                        fontweight='bold'
                    )
                
                # Format axes and title/labels
                sens_ax.set_xlabel('Correlation Coefficient', fontsize=12)
                sens_ax.set_title('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
                sens_ax.grid(True, linestyle='--', alpha=0.7, axis='x')
                sens_ax.set_axisbelow(True)
                
                # Add reference line at zero
                sens_ax.axvline(0, color='gray', linewidth=0.8)
                
                # Add interpretation zone labels - Fixed positioning to avoid overlap
                sens_ax.text(0.85, -0.15, 'Strong +', ha='center', transform=sens_ax.transAxes, fontsize=8)
                sens_ax.text(0.5, -0.15, 'Moderate +', ha='center', transform=sens_ax.transAxes, fontsize=8)
                sens_ax.text(-0.85, -0.15, 'Strong -', ha='center', transform=sens_ax.transAxes, fontsize=8)
                sens_ax.text(-0.5, -0.15, 'Moderate -', ha='center', transform=sens_ax.transAxes, fontsize=8)
                
                # Ensure plot has enough bottom margin to display these labels
                plt.subplots_adjust(bottom=0.15)
                
                sens_fig.tight_layout()
                st.pyplot(sens_fig)
                
            with sens_col2:
                # Correlation table with color formatting
                st.subheader("Correlation Coefficients")
                
                # Format table with styling
                pd_corr = pd.DataFrame(pressure_drop_corr).reset_index()
                pd_corr.columns = ['Parameter', 'Correlation']
                
                # Apply background gradient to corr values
                st.dataframe(pd_corr.style.background_gradient(
                    cmap='RdBu_r', subset=['Correlation'], vmin=-1, vmax=1
                ))
                
                # Add interpretation
                st.subheader("Interpretation")
                
                # Find strongest correlations
                strongest_pos = pd_corr.loc[pd_corr['Correlation'] == pd_corr['Correlation'].max()]
                strongest_neg = pd_corr.loc[pd_corr['Correlation'] == pd_corr['Correlation'].min()]
                
                st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <b>Key Insights:</b><br>
                    • <b>{strongest_pos['Parameter'].values[0]}</b> has the strongest <b>positive</b> effect on pressure drop
                    ({strongest_pos['Correlation'].values[0]:.3f})<br><br>
                    • <b>{strongest_neg['Parameter'].values[0]}</b> has the strongest <b>negative</b> effect on pressure drop 
                    ({strongest_neg['Correlation'].values[0]:.3f})
                </div>
                """, unsafe_allow_html=True)
                
                # Engineering guidance based on correlations
                if abs(pressure_drop_corr['Mass Flow Rate (kg/s)']) > 0.5:
                    st.info("💡 Mass flow rate is a key driver of pressure drop; consider flow control strategies.")
                    
                if abs(pressure_drop_corr[f'Diameter ({st.session_state.units_selected})']) > 0.5:
                    st.info("💡 Pipe diameter significantly affects pressure drop; a small increase in diameter can greatly reduce pressure losses.")
                
        # Tab 4: Data table with aggregated results
        with viz_tab4:
            # Create summary statistics of all parameters
            summary_stats = pd.DataFrame({
                'Parameter': data.columns,
                'Mean': data.mean(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                '5%': data.quantile(0.05),
                '95%': data.quantile(0.95)
            })
            
            st.subheader("Parameter Statistics")
            # Fix: Use pandas formatting options instead of style formatter
            # Round numeric columns to 4 significant digits
            formatted_stats = summary_stats.copy()
            numeric_cols = formatted_stats.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.4g}")
            st.dataframe(formatted_stats)
            
            # Provide a sample of the raw data
            st.subheader("Sample Data (First 10 Simulations)")
            # Also fix formatting for the sample data
            sample_data = data.head(10).copy()
            for col in sample_data.select_dtypes(include=['float64', 'int64']).columns:
                sample_data[col] = sample_data[col].apply(lambda x: f"{x:.4g}")
            st.dataframe(sample_data)
            
            # Excel export button
            st.subheader("Export Results")
            
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
                    
                    # Add some additional formatting
                    header_format = workbook.add_format({
                        'bold': True, 
                        'bg_color': '#4285F4', 
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    # Format headers in each sheet
                    for col_num, value in enumerate(summary_data.columns.values):
                        summary_ws.write(0, col_num, value, header_format)
                    
                    # Histogram
                    png_hist = io.BytesIO()
                    hist_fig.savefig(png_hist, format='png', bbox_inches='tight')
                    png_hist.seek(0)
                    summary_ws.insert_image('D10', 'Histogram', {'image_data': png_hist})
                    
                    # CDF
                    png_cdf = io.BytesIO()
                    cdf_fig.savefig(png_cdf, format='png', bbox_inches='tight')
                    png_cdf.seek(0)
                    summary_ws.insert_image('D30', 'CDF', {'image_data': png_cdf})
                    
                    # Sensitivity
                    png_sens = io.BytesIO()
                    sens_fig.savefig(png_sens, format='png', bbox_inches='tight')
                    png_sens.seek(0)  # Fixed syntax error here
                    summary_ws.insert_image('D50', 'Sensitivity', {'image_data': png_sens})
                    
                    # Add summary at the top
                    summary_ws.write('A1', 'Pressure Drop Simulation Results', workbook.add_format({
                        'bold': True, 
                        'font_size': 14
                    }))
                    summary_ws.write('A2', f'Fluid: {selected_fluid}, Simulations: {num_simulations}')
                    summary_ws.write('A4', 'Key Statistics:')
                    summary_ws.write('A5', f'Mean Pressure Drop: {mean_deltaP:.4g} {pressure_unit}')
                    summary_ws.write('A6', f'Standard Deviation: {std_deltaP:.4g} {pressure_unit}')
                    summary_ws.write('A7', f'{confidence_level}% Confidence Interval: [{ci_lower:.4g}, {ci_upper:.4g}] {pressure_unit}')
                
                return out_xlsx.getvalue()
            
            # Build Excel file
            excel_file = to_excel(
                data, 
                summary_stats, 
                pd.DataFrame(pressure_drop_corr).reset_index(), 
                hist_fig, 
                cdf_fig, 
                sens_fig
            )
            
            # Create a prominent button for download
            st.download_button(
                label="📊 Download Complete Results (Excel)",
                data=excel_file,
                file_name=f'pressure_drop_{selected_fluid}_{num_simulations}sims.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
            
    else:
        # Message for when no simulation has been run
        st.info("Run a simulation first to see results here.")
        if st.button("⚙️ Go to Simulation Setup", on_click=switch_to_setup):
            pass

# Replace the old else block
else:
    st.write("Adjust parameters and click **Run Simulation** to begin.")

# --------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>Engineering Monte Carlo Pressure Drop Calculator • v2.0 • Using CoolProp</small>
</div>
""", unsafe_allow_html=True)

# Display equations when expanded
with st.expander("📐 View Equations Used", expanded=False):
    st.markdown("""
    ### Governing Equations with Unit Analysis
    
    **Volumetric Flow Rate:**
    """)
    st.latex(r"Q = \frac{\dot{m}}{\rho} \quad [\frac{kg/s}{kg/m^3} = m^3/s]")
    
    st.markdown("**Reynolds Number:**")
    st.latex(r"Re = \frac{\rho \, v \, D}{\mu} \quad [\frac{kg/m^3 \times m/s \times m}{kg/(m \cdot s)} = \text{dimensionless}]")
    
    st.markdown("**Friction Factor (Swamee-Jain for turbulent flow):**")
    st.latex(r"f = 0.25 \Big/ \left[\log_{10}\!\Bigl(\frac{\epsilon}{3.7\,D} + \frac{5.74}{Re^{0.9}}\Bigr)\right]^2 \quad [\text{dimensionless}]")
    
    st.markdown("**Friction Factor (laminar flow):**")
    st.latex(r"f = \frac{64}{Re} \quad \text{for} \quad Re \leq 2000 \quad [\text{dimensionless}]")
    
    st.markdown("**Head Loss (Darcy-Weisbach):**")
    st.latex(r"h_f = f\,\frac{L}{D}\,\frac{v^2}{2g} \quad [\text{dimensionless} \times \frac{m}{m} \times \frac{(m/s)^2}{m/s^2} = m]")
    
    st.markdown("**Head Loss from Fittings and Valves:**")
    st.latex(r"h_\mathrm{fittings} = K_\mathrm{total}\,\frac{v^2}{2g} \quad [\text{dimensionless} \times \frac{(m/s)^2}{m/s^2} = m]")
    
    st.markdown("**Total Head Loss:**")
    st.latex(r"h_\mathrm{total} = h_f + h_\mathrm{fittings} + \Delta z \quad [m + m + m = m]")
    
    st.markdown("**Pressure Drop:**")
    st.latex(r"\Delta P = \rho\,g\,h_\mathrm{total} \quad [kg/m^3 \times m/s^2 \times m = kg/(m \cdot s^2) = Pa]")
    
    st.markdown("""
    ### Notes on Viscosity Units
    
    - Dynamic viscosity (μ) is used in these calculations with units of Pa·s or kg/(m·s)
    - The CoolProp library returns viscosity in Pa·s
    - User inputs are in mPa·s (1 mPa·s = 0.001 Pa·s) for convenience
    - For water at 20°C: μ ≈ 1.0 mPa·s = 0.001 Pa·s
    - For air at 20°C: μ ≈ 0.018 mPa·s = 0.000018 Pa·s
    """)