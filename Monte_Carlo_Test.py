import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import triang
import io
from datetime import datetime
import time

# Import CoolProp
from CoolProp.CoolProp import PropsSI

# --------------------------------------------------------------------------------
# COMMON K VALUES (Define globally or near top)
# --------------------------------------------------------------------------------
# K values based on industry standards:
# - Crane Technical Paper No. 410
# - ASHRAE Fundamentals Handbook
# - Cameron Hydraulic Data
# - Idelchik's "Handbook of Hydraulic Resistance"
COMMON_K_VALUES = {
    # 1. Elbows, bends, returns
    "90° LR Elbow (B16.9)": 0.25,  # Crane TP-410: Long radius elbow
    "90° SR Elbow": 0.75,  # Crane TP-410: Standard radius elbow
    "90° Mitered Elbow": 1.10,  # Crane TP-410: Single miter
    "45° LR Elbow": 0.15,  # Crane TP-410
    "180° LR Return Bend": 0.20,  # Crane TP-410
    "180° SR Return Bend": 0.60,  # Crane TP-410
    "Space-Saver Forged Elbow (Rc~0.5D)": 1.20,  # Adjusted: Typical forged elbow

    # 2. Tees, laterals, crosses
    "Tee, Run Through (Line -> Line)": 0.60,  # Crane TP-410: Flow through run
    "Tee, Side-Out Branch (Line -> Branch)": 1.50,  # Adjusted: Industry typical
    "Tee, Combining Branch (Branch -> Line)": 1.55,  # Crane TP-410
    "45° Lateral Wye, Main Flow": 0.40,  # Crane TP-410
    "Pipe Cross, Straight Run": 1.00,  # Crane TP-410

    # 3. Valves (fully open unless noted)
    "Gate Valve (Plain Wedge)": 0.08,  # Crane TP-410: Fully open
    "Ball Valve, Full Port": 0.05,  # Crane TP-410: Full bore
    "Ball Valve, Reduced Port (70% Area)": 0.40,  # Crane TP-410
    "Butterfly Valve, Fully Open": 0.40,  # Crane TP-410: 0° position
    "Butterfly Valve, 30° Open": 6.00,  # Adjusted: Industry range 5-10
    "Butterfly Valve, 60° Open": 15.0,  # Crane TP-410
    "Globe Valve, Z-Pattern": 10.0,  # Crane TP-410: Fully open
    "Globe Valve, Angle-Pattern": 5.0,  # Crane TP-410: Fully open
    "Plug Valve, Fully Open": 0.20,  # Crane TP-410: Straightway
    "Swing Check Valve, Forward Flow": 2.0,  # Crane TP-410
    "Lift Check Valve, Forward Flow": 10.0,  # Crane TP-410: Vertical
    "Cryogenic DBB Valve, Full Port": 0.11,  # Manufacturer data

    # 4. Reducers, diffusers, contractions
    "Sudden Enlargement (D2/D1 = 2)": 0.50,  # Borda-Carnot: (1-A1/A2)²
    "Tapered Diffuser (15° Half Angle, 3D Long)": 0.10,  # Idelchik
    "Sudden Contraction (D2/D1 = 0.5)": 0.40,  # Crane TP-410
    "Tapered Reducer (30° Total Angle)": 0.20,  # Idelchik

    # 5. Junctions, entrances, exits
    "Sharp-Edged Pipe Entrance": 0.50,  # Crane TP-410: Square edge
    "Rounded Entrance (r/D >= 0.15)": 0.04,  # Crane TP-410: Well rounded
    "Pipe Exit (to Large Tank)": 1.00,  # Crane TP-410: Exit loss
    "Re-Entry (Flush)": 2.00,  # Crane TP-410

    # 6. Miscellaneous inline devices
    "Orifice Plate (β = 0.60)": 2.43,  # Permanent loss K, typical; varies with Cd/Cc/Cv
    "Coriolis Mass-Flow Meter (Standard)": 2.0,  # Manufacturer typical
    "Wire-Mesh Strainer (40 Mesh, Clean)": 1.2,  # Cameron Hydraulic
    "Y-Strainer (Standard Mesh)": 4.0,  # Cameron Hydraulic: Typical
    "Basket Strainer": 1.3,  # Cameron Hydraulic
    "Plate-Type Heat Exchanger (Port Section, per pass)": 4.0,  # Manufacturer
    "Rupture Disk Holder (ASME Type)": 2.5,  # Manufacturer data
    "Venturi Meter": 0.30,  # Crane TP-410: Recovery cone included

    # Additional common components
    "Union, Threaded": 0.08,  # Crane TP-410
    "Water meter": 7.0,  # Manufacturer typical
    "Bellows (1 Convolution)": 0.1,  # Highly variable - consult manufacturer
}
# Regenerate component options based on the updated dictionary, sorted alphabetically
component_options = sorted(list(COMMON_K_VALUES.keys())) + ["Custom"]

# Typical absolute roughness presets (meters)
ROUGHNESS_PRESETS = {
    "Commercial Steel": 4.5e-5,
    "Stainless Steel (Smooth)": 1.0e-5,
    "Drawn Copper / Brass": 1.5e-5,
    "PVC / Smooth Plastic": 1.5e-5,
    "Galvanized Iron": 1.5e-4,
    "Cast Iron": 2.6e-4,
    "Concrete (New)": 3.0e-4,
    "Concrete (Old)": 9.0e-4
}

# Define fluid options globally for use throughout the application
# Common fluid options for CoolProp with display names
FLUID_OPTIONS = ["Water", "Oxygen", "Nitrogen", "Hydrogen", "Air", "CarbonDioxide"]
FLUID_COOLPROP_NAMES = {
    "Water": "Water", 
    "Oxygen": "Oxygen", 
    "Nitrogen": "Nitrogen", 
    "Hydrogen": "ParaHydrogen",  # Correct CoolProp name for parahydrogen
    "Air": "Air", 
    "CarbonDioxide": "CarbonDioxide"
}
FLUID_ICONS = {
    "Water": "💧", "Oxygen": "🔵", "Nitrogen": "🔹", 
    "Hydrogen": "⚛️", "Air": "💨", "CarbonDioxide": "🌫️"
}

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
# Initialize minor losses data structure
if 'minor_losses_data' not in st.session_state:
    # Initialize as empty DataFrame with required columns
    st.session_state.minor_losses_data = pd.DataFrame(
        columns=['component_type','quantity','k_value']
    )
# Preserve fluid selection between runs
if 'selected_fluid_idx' not in st.session_state:
    st.session_state.selected_fluid_idx = 0
# Preserve minor loss components between runs
if 'minor_loss_multiselect' not in st.session_state:
    st.session_state.minor_loss_multiselect = []
if 'minor_losses_state' not in st.session_state:
    st.session_state.minor_losses_state = {}
if 'roughness_preset' not in st.session_state:
    st.session_state.roughness_preset = 'Custom'
if 'simulation_cache' not in st.session_state:
    st.session_state.simulation_cache = {}
if 'simulation_metadata' not in st.session_state:
    st.session_state.simulation_metadata = {}
if 'custom_k_counter' not in st.session_state:
    st.session_state.custom_k_counter = 1
# Initialize pipe sections data structure
if 'pipe_sections' not in st.session_state:
    st.session_state.pipe_sections = []
if 'use_multiple_sections' not in st.session_state:
    st.session_state.use_multiple_sections = False
if 'section_counter' not in st.session_state:
    st.session_state.section_counter = 1

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
using <b>CoolProp</b> for accurate fluid properties.
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def get_distribution(dist_name, mean, std_dev, min_val=None, max_val=None):
    """Return a function that generates random samples from the specified distribution.
    Notes:
      - For Triangular, the 'mean' argument is treated as MODE (UI label = Mode).
    Handles invalid parameters by defaulting to deterministic value.
    """
    if dist_name == 'Normal':
        if std_dev is not None and std_dev > 1e-15:
            return lambda size: np.random.normal(mean, std_dev, size)
        st.warning(f"Std Dev for Normal distribution is zero or invalid. Using deterministic mean ({mean}).")
        return lambda size: np.full(size, mean)
    if dist_name == 'Uniform':
        if min_val is not None and max_val is not None and max_val > min_val:
            return lambda size: np.random.uniform(min_val, max_val, size)
        st.warning(f"Min/Max for Uniform distribution are invalid ({min_val}, {max_val}). Using deterministic mean ({mean}).")
        return lambda size: np.full(size, mean)
    if dist_name == 'Triangular':
        mode = mean
        if min_val is not None and max_val is not None and max_val > min_val:
            if min_val <= mode <= max_val:
                scale = max_val - min_val
                c = (mode - min_val) / scale if scale > 1e-15 else 0.5
                return lambda size: triang.rvs(c, loc=min_val, scale=scale, size=size)
            st.warning(f"Mode ({mode}) for Triangular distribution is outside range [{min_val}, {max_val}]. Using deterministic value.")
            return lambda size: np.full(size, mode)
        st.warning(f"Min/Max for Triangular distribution are invalid ({min_val}, {max_val}). Using deterministic value ({mode}).")
        return lambda size: np.full(size, mode)
    if dist_name != 'Deterministic':
        st.warning(f"Unrecognized distribution '{dist_name}'. Using deterministic mean ({mean}).")
    return lambda size: np.full(size, mean)

def pa_to_psi(pa_values):
    """Convert Pascal to psi"""
    return pa_values * 0.000145038

def create_distribution_inputs(label, default_value_for_mean, unit, key_prefix, default_dist="Deterministic",
                              default_std=None, default_min=None, default_max=None, tooltip=None):
    """Create standardized distribution input fields.

    Prioritizes existing session state values over initial defaults to preserve user input.
    """
    col1, col2 = st.columns([1, 2])

    with col1:
        # Get the currently selected distribution type for this parameter
        dist_key = f"{key_prefix}_dist"
        dist_options = ['Deterministic', 'Normal', 'Uniform', 'Triangular']
        
        # Initialize session state if not exists
        if dist_key not in st.session_state:
            st.session_state[dist_key] = default_dist

        dist = st.selectbox(
            f"{label} Distribution",
            dist_options,
            key=dist_key  # Let Streamlit manage the state entirely
        )

        if tooltip:
            st.markdown(f"<div class='tooltip'>?<span class='tooltiptext'>{tooltip}</span></div>", unsafe_allow_html=True)

    with col2:
        mean_key = f"{key_prefix}_mean"
        std_key = f"{key_prefix}_std"
        min_key = f"{key_prefix}_min"
        max_key = f"{key_prefix}_max"

        # Initialize mean value in session state if not exists
        if mean_key not in st.session_state:
            st.session_state[mean_key] = default_value_for_mean

        # Dynamic label: Mode for Triangular, Mean otherwise
        dynamic_label = "Mode" if dist == 'Triangular' else "Mean"
        mean = st.number_input(
            f"{dynamic_label} ({unit})",
            format="%.4g",
            key=mean_key  # Only use key, no value parameter
        )

        # Handle std, min, max based on distribution type
        std = 0.0
        min_val = 0.0
        max_val = 0.0

        if dist == 'Normal':
            # Initialize std in session state if not exists
            if std_key not in st.session_state:
                # Use the default_std passed in if available, otherwise calculate 5%
                current_mean = st.session_state[mean_key]
                st.session_state[std_key] = default_std if default_std is not None else current_mean * 0.05
            
            std = st.number_input(
                f"Std Dev ({unit})",
                format="%.4g",
                key=std_key
            )
        elif dist in ['Uniform', 'Triangular']:
            # Initialize min/max in session state if not exists
            if min_key not in st.session_state:
                current_mean = st.session_state[mean_key]
                st.session_state[min_key] = default_min if default_min is not None else current_mean * 0.9
            if max_key not in st.session_state:
                current_mean = st.session_state[mean_key]
                st.session_state[max_key] = default_max if default_max is not None else current_mean * 1.1
            
            min_val = st.number_input(
                f"Min ({unit})",
                format="%.4g",
                key=min_key
            )
            max_val = st.number_input(
                f"Max ({unit})",
                format="%.4g",
                key=max_key
            )

    # Return the current values from session state
    return dist, mean, std, min_val, max_val

# Set up tab switching function
def switch_to_results():
    st.session_state.active_tab = "results"
    
def switch_to_setup():
    # <- new: restore all the keys that were saved after the last run
    st.session_state.update(st.session_state.get('simulation_inputs', {}))
    st.session_state.active_tab = "setup"

# Sequential pressure drop calculation function (integrated locally)
def calculate_sequential_pressure_drop(
    pipe_sections,
    mass_flow_samples,
    temperature_K,
    upstream_pressure_Pa,
    back_pressure_Pa,
    gravity,
    friction_model,
    coolprop_fluid,
    num_simulations,
    calculate_friction_factor_func,
    per_section_minor_K=None,
    elevation_change_m=0.0,
    return_detailed=True,
    override_rho=None,
    override_mu=None,
    rho_bias=None,
    mu_bias=None
):
    """
    Calculate pressure drop sequentially through multiple pipe sections with detailed breakdown.
    
    For each section:
    1. Calculate fluid properties at the current pressure
    2. Calculate friction pressure drop for that section
    3. Apply minor losses using local section velocity
    4. Calculate transition losses at junctions
    5. Update pressure for next section
    
    Parameters:
    -----------
    override_rho, override_mu : array-like, optional
        If provided, use these constant values for all sections instead of CoolProp
    rho_bias, mu_bias : array-like, optional
        If provided, multiply CoolProp values by these biases
    """
    
    # Initialize arrays
    n_sections = len(pipe_sections)
    current_pressure = np.full(num_simulations, upstream_pressure_Pa)
    
    # Numerical safety: clamp minimum pressure
    current_pressure = np.clip(current_pressure, 500.0, None)  # Min 500 Pa
    
    # Phase boundary check for liquids
    phase_warning_issued = False
    coolprop_fallback_count = 0
    try:
        # Check saturation pressure at temperature
        Psat_T = PropsSI("P", "T", temperature_K, "Q", 0, coolprop_fluid)
        tolerance = 0.02  # 2% tolerance
        
        # Check if any initial pressure is near saturation
        if np.any(upstream_pressure_Pa <= Psat_T * (1 + tolerance)):
            st.warning(f"⚠️ Two-phase conditions likely: Upstream pressure ({upstream_pressure_Pa/1000:.1f} kPa) "
                      f"is near saturation pressure ({Psat_T/1000:.1f} kPa) at {temperature_K:.1f} K. "
                      f"Results may be invalid in flashing regime.")
            phase_warning_issued = True
    except:
        pass  # If saturation check fails, continue anyway
    
    # Arrays for detailed tracking
    dp_friction_by_section = np.zeros((num_simulations, n_sections))
    dp_transitions = np.zeros((num_simulations, max(0, n_sections - 1)))
    dp_minor_by_section = np.zeros((num_simulations, n_sections))
    v_by_section = np.zeros((num_simulations, n_sections))
    Re_by_section = np.zeros((num_simulations, n_sections))
    f_by_section = np.zeros((num_simulations, n_sections))
    rho_by_section = np.zeros((num_simulations, n_sections))
    mu_by_section = np.zeros((num_simulations, n_sections))
    
    pressure_profile = [current_pressure.copy()]
    
    # Calculate total pipe length for elevation distribution
    total_length = sum(section.get('length_m', section.get('length', 0)) for section in pipe_sections)
    
    # Process each section
    for i, section in enumerate(pipe_sections):
        # Get section dimensions (should already be in meters if properly normalized)
        D_section = section.get('diameter_m', section.get('diameter', 0))
        L_section = section.get('length_m', section.get('length', 0))
        epsilon_section = section.get('roughness', 4.5e-5)
        
        # Handle legacy format with unit conversion if needed
        if 'units' in section and section['units'] == 'Inches':
            if 'diameter_m' not in section:
                D_section = section['diameter'] * 0.0254
            if 'length_m' not in section:
                L_section = section['length'] * 0.0254
        
        # Calculate fluid properties at current pressure for this section
        for j in range(num_simulations):
            # Apply property overrides if provided
            if override_rho is not None and override_mu is not None:
                # Use constant override values for all sections
                rho_by_section[j, i] = override_rho[j]
                mu_by_section[j, i] = override_mu[j] * 1e-3  # Convert mPa·s to Pa·s if needed
            else:
                try:
                    # Clamp pressure for safety
                    safe_pressure = max(current_pressure[j], 500.0)
                    
                    # Get properties at current pressure for this simulation
                    rho_by_section[j, i] = PropsSI("D", "T", temperature_K, "P", safe_pressure, coolprop_fluid)
                    mu_by_section[j, i] = PropsSI("V", "T", temperature_K, "P", safe_pressure, coolprop_fluid)
                    
                    # Apply biases if provided
                    if rho_bias is not None:
                        rho_by_section[j, i] *= rho_bias[j]
                    if mu_bias is not None:
                        mu_by_section[j, i] *= mu_bias[j]
                        
                except Exception as e:
                    # If CoolProp fails, try with clamped pressure
                    coolprop_fallback_count += 1
                    try:
                        # Retry with upstream pressure
                        rho_by_section[j, i] = PropsSI("D", "T", temperature_K, "P", upstream_pressure_Pa, coolprop_fluid)
                        mu_by_section[j, i] = PropsSI("V", "T", temperature_K, "P", upstream_pressure_Pa, coolprop_fluid)
                    except:
                        # Final fallback to reasonable defaults
                        rho_by_section[j, i] = 1000.0 if "Water" in coolprop_fluid else 1.2  # Water vs gas
                        mu_by_section[j, i] = 1e-3 if "Water" in coolprop_fluid else 1.8e-5
        
        # Calculate flow parameters
        Q_section = mass_flow_samples / rho_by_section[:, i]
        A_section = np.pi * (D_section / 2) ** 2
        v_by_section[:, i] = Q_section / A_section
        Re_by_section[:, i] = (rho_by_section[:, i] * v_by_section[:, i] * D_section) / mu_by_section[:, i]
        
        # Calculate friction factor
        f_by_section[:, i] = calculate_friction_factor_func(
            Re_by_section[:, i], epsilon_section, D_section, friction_model
        )
        
        # Calculate friction pressure drop for this section
        # Using dp = 0.5 * rho * f * (L/D) * v^2
        dp_friction_by_section[:, i] = 0.5 * rho_by_section[:, i] * f_by_section[:, i] * (L_section / D_section) * (v_by_section[:, i] ** 2)
        
        # Apply friction loss
        current_pressure = current_pressure - dp_friction_by_section[:, i]
        
        # Add transition losses if not the first section (FIXED VERSION)
        if i > 0:
            prev_section = pipe_sections[i-1]
            D_prev = prev_section.get('diameter_m', prev_section.get('diameter', 0))
            if 'units' in prev_section and prev_section['units'] == 'Inches' and 'diameter_m' not in prev_section:
                D_prev = prev_section['diameter'] * 0.0254
            
            A_prev = np.pi * (D_prev / 2) ** 2
            A_curr = A_section
            
            if A_curr > A_prev:
                # Expansion: K = (1 - A1/A2)² (Borda-Carnot formula for sudden expansion)
                # Valid for sudden, fully turbulent expansion
                # Use upstream velocity and upstream density
                K_trans = (1 - A_prev / A_curr) ** 2
                v_trans = v_by_section[:, i-1]
                rho_trans = rho_by_section[:, i-1]
            else:
                # Contraction: K = 0.5 * (1 - A2/A1) (Ludwig formula for sudden contraction)
                # Valid for sharp, sudden contraction at high Re
                # For gradual contractions or low Re, use charts or 2-K/3-K methods
                # Use downstream velocity and downstream density
                K_trans = 0.5 * (1 - A_curr / A_prev)
                v_trans = v_by_section[:, i]
                rho_trans = rho_by_section[:, i]
            
            # dp = 0.5 * rho * K * v^2 (no gravity factor)
            dp_transitions[:, i-1] = 0.5 * rho_trans * K_trans * (v_trans ** 2)
            
            # Apply transition loss
            current_pressure = current_pressure - dp_transitions[:, i-1]
        
        # Add minor losses for this section using local velocity
        if per_section_minor_K and section.get('id') in per_section_minor_K:
            K_minor_section = per_section_minor_K[section.get('id')]
            # dp = 0.5 * rho * K * v^2
            dp_minor_by_section[:, i] = 0.5 * rho_by_section[:, i] * K_minor_section * (v_by_section[:, i] ** 2)
            
            # Apply minor loss
            current_pressure = current_pressure - dp_minor_by_section[:, i]
        
        # Store pressure at section outlet
        pressure_profile.append(current_pressure.copy())
    
    # Handle special upstream/downstream minor losses
    if per_section_minor_K:
        # Upstream minor losses (use first section properties)
        if 'upstream' in per_section_minor_K:
            K_upstream = per_section_minor_K['upstream']
            # dp = 0.5 * rho * K * v^2
            dp_upstream = 0.5 * rho_by_section[:, 0] * K_upstream * (v_by_section[:, 0] ** 2)
            dp_minor_by_section[:, 0] += dp_upstream
            current_pressure = current_pressure - dp_upstream
        
        # Downstream minor losses (use last section properties)
        if 'downstream' in per_section_minor_K:
            K_downstream = per_section_minor_K['downstream']
            # dp = 0.5 * rho * K * v^2
            dp_downstream = 0.5 * rho_by_section[:, -1] * K_downstream * (v_by_section[:, -1] ** 2)
            dp_minor_by_section[:, -1] += dp_downstream
            current_pressure = current_pressure - dp_downstream
    
    # Add elevation change (distributed proportionally by section length)
    dp_elevation = np.zeros(num_simulations)
    if elevation_change_m != 0 and total_length > 0:
        for i, section in enumerate(pipe_sections):
            L_section = section.get('length_m', section.get('length', 0))
            if 'units' in section and section['units'] == 'Inches' and 'length_m' not in section:
                L_section = section['length'] * 0.0254
            
            # Proportional elevation change for this section
            section_elevation = elevation_change_m * (L_section / total_length)
            dp_elevation_section = rho_by_section[:, i] * gravity * section_elevation
            dp_elevation += dp_elevation_section
            current_pressure = current_pressure - dp_elevation_section
    
    # Calculate total pressure drop
    total_deltaP = upstream_pressure_Pa - current_pressure
    
    # Check if back pressure is achieved
    back_pressure_achieved = current_pressure >= back_pressure_Pa
    
    # Compressible gas screening and warnings
    compressible_warning_issued = False
    choking_warning_issued = False
    
    # Check for compressible effects if gas (density < 10 kg/m³ typically indicates gas)
    if np.mean(rho_by_section) < 10:  # Likely a gas
        try:
            # Calculate speed of sound and Mach number for each section
            for i in range(n_sections):
                # Get average properties for this section
                avg_pressure = np.mean(pressure_profile[i] if i < len(pressure_profile) else current_pressure)
                
                # Calculate speed of sound
                try:
                    a = PropsSI("A", "T", temperature_K, "P", avg_pressure, coolprop_fluid)
                    
                    # Calculate Mach number
                    M = np.mean(v_by_section[:, i]) / a
                    
                    if M > 0.3 and not compressible_warning_issued:
                        st.warning(f"⚠️ Mach number > 0.3 detected (M={M:.2f} in section {i+1}). "
                                 f"Compressible flow effects may be significant. Consider using compressible flow equations.")
                        compressible_warning_issued = True
                except:
                    pass  # If speed of sound calc fails, skip warning
                
                # Check fractional pressure drop
                if i > 0:
                    dp_section = np.mean(dp_friction_by_section[:, i])
                    p_in_section = np.mean(pressure_profile[i-1])
                    if p_in_section > 0:
                        dp_fraction = dp_section / p_in_section
                        if dp_fraction > 0.2 and not compressible_warning_issued:
                            st.warning(f"⚠️ Large pressure drop detected (ΔP/P = {dp_fraction:.1%} in section {i+1}). "
                                     f"Incompressible Darcy equation may not be valid for gases.")
                            compressible_warning_issued = True
                
                # Check for choking (simplified - assumes gamma ~1.4 for air-like gases)
                if i < n_sections - 1:
                    P_up = np.mean(pressure_profile[i])
                    P_down = np.mean(pressure_profile[i+1])
                    gamma = 1.4  # Approximate for diatomic gases
                    P_critical_ratio = (2/(gamma+1))**(gamma/(gamma-1))  # ~0.528 for gamma=1.4
                    
                    if P_down/P_up < P_critical_ratio and not choking_warning_issued:
                        st.warning(f"⚠️ Possible choking condition detected between sections {i+1} and {i+2}. "
                                 f"Pressure ratio ({P_down/P_up:.3f}) below critical ratio ({P_critical_ratio:.3f}). "
                                 f"K-based incompressible loss calculations may be invalid.")
                        choking_warning_issued = True
        except:
            pass  # If any checks fail, continue without warnings
    
    # Issue consolidated CoolProp fallback warning if needed
    if coolprop_fallback_count > 0:
        st.info(f"ℹ️ CoolProp property calculation required {coolprop_fallback_count} fallback(s) to upstream pressure. "
                f"This typically occurs when pressure drops below valid range. Results may be less accurate in affected sections.")
    
    # Build return dictionary
    result = {
        'total_deltaP': total_deltaP,
        'outlet_pressure': current_pressure,
        'pressure_profile': pressure_profile,
        'back_pressure_achieved': back_pressure_achieved
    }
    
    # Add detailed breakdown if requested
    if return_detailed:
        result.update({
            'dp_friction_by_section': dp_friction_by_section,
            'dp_transitions': dp_transitions,
            'dp_minor_by_section': dp_minor_by_section,
            'dp_elevation': dp_elevation,
            'v_by_section': v_by_section,
            'Re_by_section': Re_by_section,
            'f_by_section': f_by_section,
            'rho_by_section': rho_by_section,
            'mu_by_section': mu_by_section
        })
    
    return result

# Function to calculate friction factor
def calculate_friction_factor(Re, epsilon, D, friction_model):
    """Calculate friction factor based on Reynolds number and pipe properties"""
    # Numerical safety: clip Reynolds number to avoid division by zero
    Re = np.clip(Re, 1e-6, None)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        if friction_model == "Churchill (All Regimes)":
            Re_eff = np.clip(Re, 1e-6, None)
            rr = epsilon / D
            # Guard log arguments
            log_arg = 1.0 / ((7.0 / Re_eff)**0.9 + 0.27 * rr)
            log_arg = np.clip(log_arg, 1e-12, None)
            A = (2.457 * np.log(log_arg))**16
            B = (37530.0 / Re_eff)**16
            f = 8.0 * (((8.0 / Re_eff)**12) + 1.0 / ((A + B)**1.5))**(1/12)
        elif friction_model == "Blended (Transitional)":
            # Guard log arguments
            log_arg = (epsilon/(3.7*D)) + (5.74/(Re**0.9))
            log_arg = np.clip(log_arg, 1e-12, None)
            f_turbulent = 0.25 / (np.log10(log_arg))**2
            f_lam = 64.0 / Re
            alpha = (Re - 2000)/2000
            f = np.where(Re <= 2000, f_lam,
                        np.where(Re >= 4000, f_turbulent,
                                (1-alpha)*f_lam + alpha*f_turbulent))
        else:  # Standard (Laminar + Swamee-Jain)
            # Guard log arguments
            log_arg = (epsilon/(3.7*D)) + (5.74/(Re**0.9))
            log_arg = np.clip(log_arg, 1e-12, None)
            f_turbulent = 0.25 / (np.log10(log_arg))**2
            laminar_flow = (Re <= 2000)
            f = np.where(laminar_flow, 64.0/Re, f_turbulent)
    return f

# Function to handle fluid selection and fluid properties
def render_fluid_section():
    st.header("Fluid Selection & Properties")
    
    # Use the global constants for fluid options for consistency
    # Fluid selection with common options
    selected_fluid_idx = st.selectbox(
        "Select Fluid:",
        range(len(FLUID_OPTIONS)),
        format_func=lambda x: f"{FLUID_ICONS.get(FLUID_OPTIONS[x], '•')} {FLUID_OPTIONS[x]}",
        key="selected_fluid_idx"
    )
    selected_fluid = FLUID_OPTIONS[selected_fluid_idx]
    
    # Fluid state parameters in collapsible section
    with st.expander("Fluid State Parameters", expanded=True):
        # Pressure and temperature in a 2-column layout
        p_col, t_col = st.columns(2)
        
        with p_col:
            # Unified pressure input supporting kPa and psia with internal storage in kPa
            if 'pressure_unit' not in st.session_state:
                st.session_state.pressure_unit = 'kPa'
            if 'pressure_kPa' not in st.session_state:
                st.session_state.pressure_kPa = 101.325  # 1 atm
            if 'back_pressure_kPa' not in st.session_state:
                st.session_state.back_pressure_kPa = 101.325  # 1 atm default
            pressure_unit = st.selectbox("Pressure Unit", ['kPa', 'psia'], key='pressure_unit')
            
            # Upstream pressure input
            display_pressure = st.session_state.pressure_kPa if pressure_unit == 'kPa' else st.session_state.pressure_kPa / 6.894757
            pressure_input = st.number_input(
                f"Upstream Pressure ({pressure_unit})",
                min_value=0.1,
                step=1.0,
                format="%.4g",
                value=display_pressure,
                key='pressure_input',
                help="Inlet pressure at the beginning of the pipe system"
            )
            # Convert back to kPa for internal use
            st.session_state.pressure_kPa = pressure_input if pressure_unit == 'kPa' else pressure_input * 6.894757
            pressure_kPa = st.session_state.pressure_kPa
            
            # Back pressure input
            display_back_pressure = st.session_state.back_pressure_kPa if pressure_unit == 'kPa' else st.session_state.back_pressure_kPa / 6.894757
            back_pressure_input = st.number_input(
                f"Downstream/Back Pressure ({pressure_unit})",
                min_value=0.1,
                step=1.0,
                format="%.4g",
                value=display_back_pressure,
                key='back_pressure_input',
                help="Outlet pressure at the end of the pipe system"
            )
            # Convert back to kPa for internal use
            st.session_state.back_pressure_kPa = back_pressure_input if pressure_unit == 'kPa' else back_pressure_input * 6.894757
            back_pressure_kPa = st.session_state.back_pressure_kPa
            
        with t_col:
            # Ensure session state exists before creating widget
            if 'temperature_K' not in st.session_state:
                st.session_state.temperature_K = 300.0 # Default value
            temperature_K = st.number_input(
                "Temperature (K)", 
                # value=st.session_state.get('temperature_K', 300.0), # REMOVED
                min_value=14.0,  # Lower minimum temperature to allow cryogenic fluids
                step=10.0,
                format="%.3g",
                key="temperature_K"
            )
            
        # Query CoolProp for properties - use the correct CoolProp fluid name
        try:
            coolprop_fluid = FLUID_COOLPROP_NAMES[selected_fluid]
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
        
        # Add property override mode selector for multi-section runs
        if st.session_state.get('use_multiple_sections', False):
            st.markdown("### Property Override Mode (Multi-Section)")
            property_mode = st.selectbox(
                "How to handle fluid properties in multi-section calculations:",
                ["CoolProp Only (Default)", "Constant Override", "Bias vs CoolProp"],
                key="property_override_mode",
                help="""
                • CoolProp Only: Use CoolProp at each section's local pressure
                • Constant Override: Use the sampled ρ/μ values for all sections
                • Bias vs CoolProp: Multiply CoolProp values by sampled bias factors
                """
            )
        else:
            property_mode = "CoolProp Only (Default)"
        
        # Calculate default std dev based on CoolProp mean
        default_rho_std = calc_density * 0.05
        default_mu_std = calc_viscosity_mPas * 0.05

        rho_dist, rho_mean, rho_std, rho_min, rho_max = create_distribution_inputs(
            "Density", calc_density, "kg/m³", "rho",
            default_dist=st.session_state.get('rho_dist', 'Deterministic'),
            default_std=st.session_state.get('rho_std', default_rho_std), 
            default_min=st.session_state.get('rho_min', None),
            default_max=st.session_state.get('rho_max', None),
            tooltip="Fluid density affects Reynolds number and pressure drop"
        )
        
        mu_dist, mu_mean, mu_std, mu_min, mu_max = create_distribution_inputs(
            "Viscosity", calc_viscosity_mPas, "mPa·s", "mu",
            default_dist=st.session_state.get('mu_dist', 'Deterministic'),
            default_std=st.session_state.get('mu_std', default_mu_std), 
            default_min=st.session_state.get('mu_min', None),
            default_max=st.session_state.get('mu_max', None),
            tooltip="Fluid viscosity affects Reynolds number and friction factor"
        )
        
        # Store the property mode in session state
        st.session_state['property_mode'] = property_mode
    
    # Return all needed values from fluid section
    return selected_fluid, calc_density, calc_viscosity_mPas, rho_dist, rho_mean, rho_std, rho_min, rho_max, mu_dist, mu_mean, mu_std, mu_min, mu_max

# Function to render multiple pipe sections UI
def render_pipe_sections():
    """Render the UI for managing multiple pipe sections"""
    st.markdown("### 🔧 Pipe Section Configuration")
    
    # Toggle for single vs multiple sections
    use_multiple = st.checkbox(
        "Use Multiple Pipe Sections",
        key="use_multiple_sections",
        help="Enable to model pipes with varying diameters (e.g., 2\" to 3\" transition)"
    )
    
    if not use_multiple:
        # Return None to indicate single section mode
        return None
    
    # Multiple sections mode
    st.info("📏 Define each pipe section with its diameter, length, and roughness. Transition losses are calculated automatically.")
    
    # Add new section interface
    with st.expander("➕ Add New Section", expanded=False):
        # Add name field in first row
        name_col, placeholder = st.columns([3, 1])
        with name_col:
            new_section_name = st.text_input(
                "Section Name (optional)",
                placeholder="e.g., Inlet Pipe, Heat Exchanger, Outlet",
                key="new_section_name",
                help="Give this section a descriptive name for easy identification"
            )
        
        # Dimension inputs in second row
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            new_diameter = st.number_input(
                f"Diameter ({st.session_state.units_selected.lower()})",
                min_value=0.001,
                value=2.0 if st.session_state.units_selected == "Inches" else 0.05,
                format="%.4g",
                key="new_section_diameter"
            )
        
        with col2:
            new_length = st.number_input(
                f"Length ({st.session_state.units_selected.lower()})",
                min_value=0.001,
                value=10.0 if st.session_state.units_selected == "Inches" else 0.254,
                format="%.4g",
                key="new_section_length"
            )
        
        with col3:
            new_roughness = st.number_input(
                "Roughness (m)",
                min_value=0.0,
                value=4.5e-5,
                format="%.2e",
                key="new_section_roughness"
            )
        
        with col4:
            if st.button("Add Section", key="add_section_btn"):
                # Store dimensions in meters internally for consistency
                diameter_m = new_diameter * 0.0254 if st.session_state.units_selected == "Inches" else new_diameter
                length_m = new_length * 0.0254 if st.session_state.units_selected == "Inches" else new_length
                
                # Use custom name if provided, otherwise auto-generate
                section_id = f"Section_{st.session_state.section_counter}"
                section_name = new_section_name.strip() if new_section_name.strip() else section_id
                
                new_section = {
                    'id': section_id,
                    'name': section_name,  # Custom name for display
                    'diameter': new_diameter,  # Original value for display
                    'length': new_length,      # Original value for display
                    'diameter_m': diameter_m,  # Normalized to meters
                    'length_m': length_m,      # Normalized to meters
                    'roughness': new_roughness,
                    'units': st.session_state.units_selected
                }
                st.session_state.pipe_sections.append(new_section)
                st.session_state.section_counter += 1
                st.success(f"Added: {section_name}")
    
    # Display existing sections
    if st.session_state.pipe_sections:
        st.markdown("#### Current Pipe Sections")
        
        # Add section editing interface
        edit_mode = st.checkbox("✏️ Edit Section Names", key="edit_sections_mode")
        
        if edit_mode:
            st.info("📝 Edit section names below and click 'Save Changes' when done")
            
            # Create editable fields for each section
            edited_sections = []
            for i, section in enumerate(st.session_state.pipe_sections):
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_name = st.text_input(
                        f"Section {i+1} Name",
                        value=section['name'],
                        key=f"edit_name_{section['id']}"
                    )
                    edited_sections.append({**section, 'name': new_name})
                with col2:
                    st.text(f"ID: {section['id']}")
            
            # Save button
            if st.button("💾 Save Changes", key="save_section_names"):
                st.session_state.pipe_sections = edited_sections
                st.success("Section names updated!")
                st.rerun()
        
        # Create DataFrame for display
        sections_df = pd.DataFrame(st.session_state.pipe_sections)
        
        # Add calculated fields
        for i, section in enumerate(st.session_state.pipe_sections):
            # Convert to meters if needed
            if section['units'] == 'Inches':
                sections_df.loc[i, 'Diameter (m)'] = section['diameter'] * 0.0254
                sections_df.loc[i, 'Length (m)'] = section['length'] * 0.0254
            else:
                sections_df.loc[i, 'Diameter (m)'] = section['diameter']
                sections_df.loc[i, 'Length (m)'] = section['length']
            
            # Calculate area
            sections_df.loc[i, 'Area (m²)'] = np.pi * (sections_df.loc[i, 'Diameter (m)'] / 2) ** 2
            
            # Calculate transition K if not first section
            if i > 0:
                A1 = sections_df.loc[i-1, 'Area (m²)']
                A2 = sections_df.loc[i, 'Area (m²)']
                if A2 > A1:  # Expansion
                    K_trans = (1 - A1/A2) ** 2
                    sections_df.loc[i, 'Transition'] = f"Expansion (K={K_trans:.3f})"
                else:  # Contraction
                    K_trans = 0.5 * (1 - A2/A1)
                    sections_df.loc[i, 'Transition'] = f"Contraction (K={K_trans:.3f})"
            else:
                sections_df.loc[i, 'Transition'] = "Inlet"
        
        # Display table with custom names
        display_cols = ['name', 'diameter', 'length', 'roughness', 'units', 'Transition']
        st.dataframe(sections_df[display_cols], use_container_width=True)
        
        # Delete sections
        if len(st.session_state.pipe_sections) > 0:
            # Create a mapping of display names to IDs for deletion
            section_options = {s['name']: s['id'] for s in st.session_state.pipe_sections}
            section_to_delete_name = st.selectbox(
                "Select section to delete:",
                options=list(section_options.keys()),
                key="delete_section_select"
            )
            if st.button("🗑️ Delete Selected Section", key="delete_section_btn"):
                section_id_to_delete = section_options[section_to_delete_name]
                st.session_state.pipe_sections = [
                    s for s in st.session_state.pipe_sections if s['id'] != section_id_to_delete
                ]
                st.rerun()
        
        # Summary
        total_length_m = sum([
            s['length'] * 0.0254 if s['units'] == 'Inches' else s['length']
            for s in st.session_state.pipe_sections
        ])
        st.info(f"**Total System Length:** {total_length_m:.3f} m ({len(st.session_state.pipe_sections)} sections)")
    else:
        st.warning("No sections added yet. Add at least one section to continue.")
    
    return st.session_state.pipe_sections

# Function to render the minor losses section with improved interface
def render_minor_losses_section():
    with st.expander("Minor Losses (Fittings, Valves, etc.)", expanded=True):
        st.markdown("""
            ### Component Minor Losses
            Add components contributing to minor pressure losses. Specify location for multi-section pipes.
        """)
        
        # Initialize minor losses list if not exists
        if 'minor_losses_list' not in st.session_state:
            st.session_state.minor_losses_list = []
        if 'minor_loss_counter' not in st.session_state:
            st.session_state.minor_loss_counter = 1
            
        # Check if we're in multi-section mode
        use_multiple_sections = st.session_state.get('use_multiple_sections', False)
        pipe_sections = st.session_state.get('pipe_sections', [])
        
        # Location explanation (use info box instead of nested expander)
        if use_multiple_sections and pipe_sections:
            st.info("""
            📍 **Location Guide:**
            - **Upstream**: Applied at pipe inlet (before Section 1)
            - **Downstream**: Applied at pipe outlet (after last section)
            - **All Sections**: Apply to all sections (see option below)
            - **Section_X**: Applied at specific section using local velocity
            """)
            
            # Add replicate vs distribute option for "All Sections"
            st.markdown("**'All Sections' Behavior:**")
            all_sections_mode = st.radio(
                "When applying K values to 'All Sections':",
                ["Distribute evenly (divide K by number of sections)", 
                 "Replicate in every section (apply full K to each)"],
                key="all_sections_mode",
                help="Choose how K values are applied when 'All Sections' is selected"
            )
        
        # Add new component interface
        st.markdown("#### ➕ Add New Component")
        add_cols = st.columns([2, 1, 1, 1.5, 1] if use_multiple_sections else [2, 1, 1, 1])
        
        with add_cols[0]:
            # Component type selector
            comp_type = st.selectbox(
                "Component Type",
                options=['Select...'] + component_options,
                key="new_comp_type",
                help="Select a standard component or 'Custom' for user-defined"
            )
        
        with add_cols[1]:
            # Quantity
            qty = st.number_input(
                "Quantity",
                min_value=1,
                value=1,
                step=1,
                key="new_comp_qty"
            )
        
        with add_cols[2]:
            # K value - auto-fill for standard components
            if comp_type != 'Select...' and comp_type != 'Custom':
                default_k = COMMON_K_VALUES.get(comp_type, 0.0)
            else:
                default_k = 0.0
            
            k_value = st.number_input(
                "K Value",
                min_value=0.0,
                value=default_k,
                format="%.3f",
                key="new_comp_k",
                help="Loss coefficient (auto-filled for standard components)"
            )
        
        # Location selector (only for multi-section)
        if use_multiple_sections and pipe_sections:
            with add_cols[3]:
                # Use custom names for sections in dropdown
                location_options = ['All Sections', 'Upstream', 'Downstream'] + [s['name'] for s in pipe_sections]
                location = st.selectbox(
                    "Location",
                    options=location_options,
                    key="new_comp_location"
                )
        else:
            location = 'Single Pipe'
        
        # Add button
        with add_cols[-1]:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("Add", key="add_comp_btn", type="primary", disabled=(comp_type == 'Select...')):
                # Handle custom component naming
                if comp_type == 'Custom':
                    # Prompt for custom name
                    comp_name = f"Custom Component #{st.session_state.minor_loss_counter}"
                    st.session_state.minor_loss_counter += 1
                else:
                    # For standard components, add instance number if duplicates exist
                    existing_count = sum(1 for item in st.session_state.minor_losses_list 
                                       if item['component_type'].startswith(comp_type))
                    if existing_count > 0:
                        comp_name = f"{comp_type} #{existing_count + 1}"
                    else:
                        comp_name = comp_type
                
                # Add to list
                new_component = {
                    'id': len(st.session_state.minor_losses_list),
                    'component_type': comp_name,
                    'base_type': comp_type,  # Store original type for reference
                    'quantity': qty,
                    'k_value': k_value,
                    'location': location,
                    'total_k': qty * k_value
                }
                st.session_state.minor_losses_list.append(new_component)
                st.success(f"Added: {comp_name}")
                st.rerun()
        
        # Custom component name input (shown when Custom is selected)
        if comp_type == 'Custom':
            st.text_input(
                "Custom Component Name (optional)",
                key="custom_comp_name",
                placeholder="e.g., Special Orifice, Flow Meter, etc."
            )
        
        # Display existing components
        if st.session_state.minor_losses_list:
            st.markdown("#### 📋 Current Components")
            
            # Create display DataFrame
            display_df = pd.DataFrame(st.session_state.minor_losses_list)
            
            # Reorder columns for better display
            if use_multiple_sections:
                display_cols = ['component_type', 'quantity', 'k_value', 'location', 'total_k']
            else:
                display_cols = ['component_type', 'quantity', 'k_value', 'total_k']
                # Remove location column for single pipe
                if 'location' in display_df.columns:
                    display_df = display_df.drop(columns=['location'])
            
            # Display with actions
            for idx, row in display_df.iterrows():
                col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1.5, 1, 1] if use_multiple_sections else [3, 1, 1, 0, 1, 1])
                
                with col1:
                    st.text(row['component_type'])
                with col2:
                    st.text(f"{row['quantity']}")
                with col3:
                    st.text(f"{row['k_value']:.3f}")
                if use_multiple_sections:
                    with col4:
                        st.text(row.get('location', 'All Sections'))
                with col5:
                    st.text(f"{row['total_k']:.3f}")
                with col6:
                    if st.button("🗑️", key=f"delete_{idx}", help="Delete this component"):
                        st.session_state.minor_losses_list = [
                            item for item in st.session_state.minor_losses_list 
                            if item['id'] != row['id']
                        ]
                        st.rerun()
            
            # Summary statistics
            st.markdown("---")
            total_k = sum(item['total_k'] for item in st.session_state.minor_losses_list)
            
            # Group by location if multi-section
            if use_multiple_sections and pipe_sections:
                location_summary = {}
                for item in st.session_state.minor_losses_list:
                    loc = item['location']
                    if loc not in location_summary:
                        location_summary[loc] = 0
                    if loc == 'All Sections':
                        # Distributed among all sections
                        location_summary[loc] += item['total_k'] / len(pipe_sections)
                    else:
                        location_summary[loc] += item['total_k']
                
                summary_cols = st.columns(2)
                with summary_cols[0]:
                    st.info(f"**Total System K Value: {total_k:.3f}**")
                with summary_cols[1]:
                    st.info(f"**Components: {len(st.session_state.minor_losses_list)}**")
                
                # Show per-location summary
                if location_summary:
                    st.markdown("**K Values by Location:**")
                    for loc, k_val in location_summary.items():
                        st.text(f"  • {loc}: {k_val:.3f}")
            else:
                st.info(f"**Total K Value: {total_k:.3f}** | **Components: {len(st.session_state.minor_losses_list)}**")
        else:
            st.info("No components added yet. Use the form above to add minor loss components.")
        
        # Build DataFrame for compatibility with rest of code
        if st.session_state.minor_losses_list:
            minor_losses_data = pd.DataFrame([
                {
                    'component_type': item['component_type'],
                    'quantity': item['quantity'],
                    'k_value': item['k_value'],
                    'location': item.get('location', 'All Sections')
                }
                for item in st.session_state.minor_losses_list
            ])
        else:
            columns = ['component_type', 'quantity', 'k_value']
            if use_multiple_sections:
                columns.append('location')
            minor_losses_data = pd.DataFrame(columns=columns)
        
        st.session_state.minor_losses_data = minor_losses_data
    
    return st.session_state.minor_losses_data

# Function to store all inputs for access in the results tab
def save_inputs_to_session_state(
        rho_dist, rho_mean, rho_std, rho_min, rho_max,
        mu_dist, mu_mean, mu_std, mu_min, mu_max,
        D_dist, D_mean, D_std, D_min, D_max,
        L_dist, L_mean, L_std, L_min, L_max,
        epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max,
        mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max,
        elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max,
        num_simulations, confidence_level, minor_losses_data, gravity, selected_fluid,
        temperature_K, pressure_kPa, back_pressure_kPa, pressure_unit, minor_loss_multiselect, selected_fluid_idx,
        units_selected, friction_model, roughness_preset, pipe_sections=None, use_multiple_sections=False
    ):
    """Store all simulation inputs in the session state for persistence between tabs"""
    st.session_state.simulation_inputs = {
        # Fluid properties
        'selected_fluid': selected_fluid,
        'selected_fluid_idx': selected_fluid_idx,
        'temperature_K': temperature_K,
        'pressure_kPa': pressure_kPa,
        'back_pressure_kPa': back_pressure_kPa,
        'pressure_unit': pressure_unit,
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
        # Multi-section pipe data
        'pipe_sections': pipe_sections,
        'use_multiple_sections': use_multiple_sections,
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
        'minor_losses_data': minor_losses_data,
        'minor_loss_multiselect': minor_loss_multiselect,
        'gravity': gravity,
        'units_selected': units_selected,
        'friction_model': friction_model,
        'roughness_preset': roughness_preset,
    }
    st.session_state.simulation_run = True

# Create tab-like UI with buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 5])
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
with col3:
    # Change the button to have no on_click handler
    refresh_button = st.button("🔄 Refresh", 
                              key="refresh_tab", 
                              use_container_width=True,
                              help="Rerun the application script")

if st.session_state.active_tab == "setup":
    # Single column layout for better visibility
    st.markdown("---")
    
    # Fluid Selection and Properties Section
    selected_fluid, calc_density, calc_viscosity_mPas, rho_dist, rho_mean, rho_std, rho_min, rho_max, mu_dist, mu_mean, mu_std, mu_min, mu_max = render_fluid_section()
    
    st.markdown("---")
    
    # Simulation Parameters Section
    st.header("Simulation Parameters")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        if 'num_simulations' not in st.session_state:
            st.session_state.num_simulations = 5000
        num_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=200000,
            step=100,
            format="%d",
            key='num_simulations'
        )
    with sim_col2:
        if 'confidence_level' not in st.session_state:
            st.session_state.confidence_level = 95
        confidence_level = st.slider(
            "Confidence Interval (%)",
            min_value=90,
            max_value=99,
            step=1,
            key='confidence_level'
        )
    
    # Output units selection
    st.subheader("Output Units")
    unit_options = ['Pascals (Pa)', 'kPa', 'bar', 'psi']
    unit_icons = ['⚖️', '🔄', '📊', '🇺🇸']
    
    unit_cols = st.columns(len(unit_options))
    for i, (col, unit, icon) in enumerate(zip(unit_cols, unit_options, unit_icons)):
        with col:
            if st.button(f"{icon} {unit}", key=f"unit_{i}"):
                st.session_state.unit_selection = unit_options[i]
    
    st.markdown(f"Selected unit: **{st.session_state.unit_selection}**")
    
    st.markdown("---")
    
    # Pipe Configuration Section
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

    # Call the new multi-section pipe configuration function
    pipe_sections = render_pipe_sections()
    
    # If not using multiple sections, show the single pipe configuration
    if pipe_sections is None:
        with st.expander("Single Pipe Geometry", expanded=True):
            # Pipe diameter with distribution
            D_dist, D_mean, D_std, D_min, D_max = create_distribution_inputs(
                length_label("Pipe Diameter"), 
                2.0 if st.session_state.units_selected=="Inches" else 0.05, 
                st.session_state.units_selected.lower(), 
                "D",
                default_dist=st.session_state.get('D_dist', 'Deterministic'),
                default_std=st.session_state.get('D_std', None),
                default_min=st.session_state.get('D_min', None),
                default_max=st.session_state.get('D_max', None),
                tooltip="Inner diameter of the pipe - key parameter for pressure drop"
            )
            
            # Pipe length with distribution
            L_dist, L_mean, L_std, L_min, L_max = create_distribution_inputs(
                length_label("Pipe Length"), 
                18.0 if st.session_state.units_selected=="Inches" else 0.4572, 
                st.session_state.units_selected.lower(), 
                "L",
                default_dist=st.session_state.get('L_dist', 'Deterministic'),
                default_std=st.session_state.get('L_std', None),
                default_min=st.session_state.get('L_min', None),
                default_max=st.session_state.get('L_max', None),
                tooltip="Total straight pipe length"
            )
            
            # Roughness presets selector
            preset_options = ['Custom'] + [f"{k} (ε={v:.2g} m)" for k,v in ROUGHNESS_PRESETS.items()]
            display_to_key = {f"{k} (ε={v:.2g} m)": k for k,v in ROUGHNESS_PRESETS.items()}
            selected_display = st.selectbox(
                "Roughness Material Preset",
                preset_options,
                key='roughness_preset',
                help="Select a material to auto-fill roughness (absolute ε). Choose 'Custom' to input manually."
            )
            selected_preset = display_to_key.get(selected_display, 'Custom')
            if selected_preset != 'Custom':
                st.session_state.epsilon_dist = 'Deterministic'
                st.session_state.epsilon_mean = ROUGHNESS_PRESETS[selected_preset]
                var_col, pct_col = st.columns([1,1])
                with var_col:
                    apply_var = st.checkbox("±% Var", key="epsilon_var_toggle", help="Apply symmetric percentage variability as Uniform bounds")
                with pct_col:
                    pct_var = st.number_input("Percent", min_value=1, max_value=50, value=10, step=1, key="epsilon_var_pct") if apply_var else 0
                if apply_var:
                    mean_val = ROUGHNESS_PRESETS[selected_preset]
                    span = mean_val * (pct_var/100)
                    st.session_state.epsilon_dist = 'Uniform'
                    st.session_state.epsilon_min = mean_val - span
                    st.session_state.epsilon_max = mean_val + span

            # Pipe roughness with distribution (always in meters); default value uses session state epsilon_mean if set
            epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max = create_distribution_inputs(
                    "Roughness", 0.000015, "m", "epsilon",
                    default_dist=st.session_state.get('epsilon_dist', 'Deterministic'),
                    default_std=st.session_state.get('epsilon_std', None),
                    default_min=st.session_state.get('epsilon_min', None),
                    default_max=st.session_state.get('epsilon_max', None),
                    tooltip="Absolute roughness - affects friction factor calculation"
                )
    else:
        # For multiple sections, set dummy values for single pipe parameters
        D_dist = D_mean = D_std = D_min = D_max = None
        L_dist = L_mean = L_std = L_min = L_max = None
        epsilon_dist = epsilon_mean = epsilon_std = epsilon_min = epsilon_max = None
    
    # Flow properties
    with st.expander("Flow Configuration", expanded=True):
        # Mass flow with distribution
        mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max = create_distribution_inputs(
            "Mass Flow Rate", 3.0, "kg/s", "mass_flow",
            default_dist=st.session_state.get('mass_flow_dist', 'Normal'),
            default_std=st.session_state.get('mass_flow_std', 0.25),
            default_min=st.session_state.get('mass_flow_min', None),
            default_max=st.session_state.get('mass_flow_max', None),
            tooltip="Mass flow rate through the pipe"
        )
        
        # Elevation change with distribution
        elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max = create_distribution_inputs(
            "Elevation Change", 0.0, "m", "elevation",
            default_dist=st.session_state.get('elevation_dist', 'Deterministic'),
            default_std=st.session_state.get('elevation_std', None),
            default_min=st.session_state.get('elevation_min', None),
            default_max=st.session_state.get('elevation_max', None),
            tooltip="Positive values indicate upward flow (increases pressure drop)"
        )
    
    # --- Minor Losses Section (Multiselect UI) ---
    minor_losses_data = render_minor_losses_section()
    # --- End Minor Losses Multiselect UI ---
    
    # Advanced physics parameters
    with st.expander("Advanced Parameters", expanded=False):
        # Ensure session state exists before creating widget
        if 'gravity' not in st.session_state:
            st.session_state.gravity = 9.81 # Default value
        gravity = st.number_input(
            "Acceleration due to Gravity (m/s²)", 
            # value=st.session_state.get('gravity', 9.81), # REMOVED
            format="%.4f",
            help="Default value is for Earth (9.81 m/s²)",
            key="gravity"
        )
        # Friction factor model selection
        if 'friction_model' not in st.session_state:
            st.session_state.friction_model = "Standard (Laminar + Swamee-Jain)"
        friction_model = st.selectbox(
                "Friction Factor Model",
                ["Standard (Laminar + Swamee-Jain)", "Churchill (All Regimes)", "Blended (Transitional)"],
                key="friction_model",
                help="Blended: linear blend 2000<Re<4000 between laminar and turbulent; Churchill: universal equation."
            )
        
        # Reproducibility controls
        st.subheader("Reproducibility")
        seed_col1, seed_col2 = st.columns([2,1])
        with seed_col1:
            rng_seed = st.text_input("Random Seed (optional integer)", key="rng_seed")
        with seed_col2:
            if rng_seed and not rng_seed.isdigit():
                st.warning("Seed must be an integer.")
    
    # Large, prominent "Run Simulation" button - MOVED OUTSIDE OF EXPANDER
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Define the function to run the simulation and save inputs
    def validate_inputs():
        errs = []
        
        # Check if using multiple sections
        if st.session_state.use_multiple_sections:
            # Validate pipe sections
            if not st.session_state.pipe_sections:
                errs.append("At least one pipe section must be defined when using multiple sections.")
            else:
                for section in st.session_state.pipe_sections:
                    if section['diameter'] <= 0:
                        errs.append(f"{section['id']}: Diameter must be > 0.")
                    if section['length'] <= 0:
                        errs.append(f"{section['id']}: Length must be > 0.")
                    if section['roughness'] < 0:
                        errs.append(f"{section['id']}: Roughness must be >= 0.")
        else:
            # Single pipe validation
            for prefix, label in [("D","Diameter"),("L","Length"),("epsilon","Roughness"),("mass_flow","Mass Flow"),("elevation","Elevation"),("rho","Density"),("mu","Viscosity")]:
                dist = st.session_state.get(f"{prefix}_dist")
                if dist in ['Uniform','Triangular']:
                    mn = st.session_state.get(f"{prefix}_min")
                    mx = st.session_state.get(f"{prefix}_max")
                    if (mn is None) or (mx is None) or not (mx > mn):
                        errs.append(f"{label}: Min must be < Max for {dist} distribution.")
                if dist == 'Normal':
                    std = st.session_state.get(f"{prefix}_std")
                    if std is None or std <= 0:
                        errs.append(f"{label}: Std Dev must be > 0 for Normal distribution.")
            if st.session_state.get('D_mean',0) <= 0: errs.append("Diameter mean must be > 0.")
            if st.session_state.get('L_mean',0) <= 0: errs.append("Length mean must be > 0.")
        
        # Always validate mass flow
        if st.session_state.get('mass_flow_mean',0) <= 0: errs.append("Mass flow mean must be > 0.")
        return errs

    def run_and_save():
        # Gather all input values from the widgets
            # Note: We retrieve the values directly from the widgets/session state keys
            # Fluid Properties
            selected_fluid_idx_val = st.session_state.selected_fluid_idx # <- Get index
            selected_fluid_val = FLUID_OPTIONS[selected_fluid_idx_val]
            temperature_K_val = st.session_state.temperature_K # <- Get value
            pressure_kPa_val = st.session_state.pressure_kPa   # <- Get value
            rho_dist_val = st.session_state.rho_dist
            rho_mean_val = st.session_state.rho_mean
            rho_std_val = st.session_state.get('rho_std', 0.0) # Use .get for optional keys
            rho_min_val = st.session_state.get('rho_min', 0.0)
            rho_max_val = st.session_state.get('rho_max', 0.0)
            mu_dist_val = st.session_state.mu_dist
            mu_mean_val = st.session_state.mu_mean
            mu_std_val = st.session_state.get('mu_std', 0.0)
            mu_min_val = st.session_state.get('mu_min', 0.0)
            mu_max_val = st.session_state.get('mu_max', 0.0)
            
            # Pipe Geometry - handle both single and multiple sections
            if st.session_state.use_multiple_sections:
                # For multiple sections, use None for single pipe parameters
                D_dist_val = D_mean_val = D_std_val = D_min_val = D_max_val = None
                L_dist_val = L_mean_val = L_std_val = L_min_val = L_max_val = None
                epsilon_dist_val = epsilon_mean_val = epsilon_std_val = epsilon_min_val = epsilon_max_val = None
            else:
                # Single pipe geometry
                D_dist_val = st.session_state.get('D_dist', 'Deterministic')
                D_mean_val = st.session_state.get('D_mean', 0.05)
                D_std_val = st.session_state.get('D_std', 0.0)
                D_min_val = st.session_state.get('D_min', 0.0)
                D_max_val = st.session_state.get('D_max', 0.0)
                L_dist_val = st.session_state.get('L_dist', 'Deterministic')
                L_mean_val = st.session_state.get('L_mean', 0.5)
                L_std_val = st.session_state.get('L_std', 0.0)
                L_min_val = st.session_state.get('L_min', 0.0)
                L_max_val = st.session_state.get('L_max', 0.0)
                epsilon_dist_val = st.session_state.get('epsilon_dist', 'Deterministic')
                epsilon_mean_val = st.session_state.get('epsilon_mean', 4.5e-5)
                epsilon_std_val = st.session_state.get('epsilon_std', 0.0)
                epsilon_min_val = st.session_state.get('epsilon_min', 0.0)
                epsilon_max_val = st.session_state.get('epsilon_max', 0.0)
            # Flow Properties
            mass_flow_dist_val = st.session_state.mass_flow_dist
            mass_flow_mean_val = st.session_state.mass_flow_mean
            mass_flow_std_val = st.session_state.get('mass_flow_std', 0.0)
            mass_flow_min_val = st.session_state.get('mass_flow_min', 0.0)
            mass_flow_max_val = st.session_state.get('mass_flow_max', 0.0)
            elevation_dist_val = st.session_state.elevation_dist
            elevation_mean_val = st.session_state.elevation_mean
            elevation_std_val = st.session_state.get('elevation_std', 0.0)
            elevation_min_val = st.session_state.get('elevation_min', 0.0)
            elevation_max_val = st.session_state.get('elevation_max', 0.0)
            # Other Parameters
            num_simulations_val = st.session_state.num_simulations
            confidence_level_val = st.session_state.confidence_level
            minor_losses_data_val = st.session_state.minor_losses_data # Already a DataFrame
            # For the new interface, we don't have minor_loss_multiselect anymore
            minor_loss_multiselect_val = []  # Empty list for compatibility
            gravity_val = st.session_state.gravity
            back_pressure_kPa_val = st.session_state.back_pressure_kPa  # Get back pressure

            # Call the function to save these inputs to session state
            save_inputs_to_session_state(
                rho_dist_val, rho_mean_val, rho_std_val, rho_min_val, rho_max_val,
                mu_dist_val, mu_mean_val, mu_std_val, mu_min_val, mu_max_val,
                D_dist_val, D_mean_val, D_std_val, D_min_val, D_max_val,
                L_dist_val, L_mean_val, L_std_val, L_min_val, L_max_val,
                epsilon_dist_val, epsilon_mean_val, epsilon_std_val, epsilon_min_val, epsilon_max_val,
                mass_flow_dist_val, mass_flow_mean_val, mass_flow_std_val, mass_flow_min_val, mass_flow_max_val,
                elevation_dist_val, elevation_mean_val, elevation_std_val, elevation_min_val, elevation_max_val,
                num_simulations_val, confidence_level_val, minor_losses_data_val, gravity_val, selected_fluid_val,
                temperature_K_val, pressure_kPa_val, back_pressure_kPa_val, st.session_state.pressure_unit, minor_loss_multiselect_val, selected_fluid_idx_val,
                st.session_state.units_selected, st.session_state.friction_model, st.session_state.roughness_preset,
                pipe_sections=st.session_state.pipe_sections, use_multiple_sections=st.session_state.use_multiple_sections
            )
            # Removed bulk st.session_state.update to avoid StreamlitAPIException when setting widget keys (e.g., selected_fluid_idx) post-instantiation.
            # Switch to the results tab after saving
            switch_to_results()

    errors = validate_inputs()
    if errors:
        st.error("Resolve before running:\n- " + "\n- ".join(errors))
    
    # Get the seed value from the Advanced Parameters section
    rng_seed = st.session_state.get('rng_seed', '')
    
    run_clicked = st.button("▶ Run Monte Carlo Simulation", key="run_sim_btn", use_container_width=True, disabled=bool(errors))
    if run_clicked and not errors:
        if rng_seed and rng_seed.isdigit():
            np.random.seed(int(rng_seed))
        run_and_save()
        st.session_state.simulation_metadata = {
            'seed': rng_seed if (rng_seed and rng_seed.isdigit()) else None,
            'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'version': 'v2.7'
        }
        st.success("Simulation configured. View Results tab.")

# Results Tab Content
elif st.session_state.active_tab == "results":
    if st.session_state.simulation_run:
        inputs = st.session_state.simulation_inputs
        # Unpack common parameters
        rho_dist = inputs['rho_dist']; rho_mean = inputs['rho_mean']; rho_std = inputs['rho_std']; rho_min = inputs['rho_min']; rho_max = inputs['rho_max']
        mu_dist = inputs['mu_dist']; mu_mean = inputs['mu_mean']; mu_std = inputs['mu_std']; mu_min = inputs['mu_min']; mu_max = inputs['mu_max']
        mass_flow_dist = inputs['mass_flow_dist']; mass_flow_mean = inputs['mass_flow_mean']; mass_flow_std = inputs['mass_flow_std']; mass_flow_min = inputs['mass_flow_min']; mass_flow_max = inputs['mass_flow_max']
        elevation_dist = inputs['elevation_dist']; elevation_mean = inputs['elevation_mean']; elevation_std = inputs['elevation_std']; elevation_min = inputs['elevation_min']; elevation_max = inputs['elevation_max']
        num_simulations = inputs['num_simulations']; confidence_level = inputs['confidence_level']
        minor_losses_data = inputs['minor_losses_data']; gravity = inputs['gravity']
        units_selected_run = inputs.get('units_selected', st.session_state.units_selected)
        friction_model = inputs.get('friction_model', "Standard (Laminar + Swamee-Jain)")
        selected_fluid = inputs['selected_fluid']
        use_multiple_sections = inputs.get('use_multiple_sections', False)
        pipe_sections = inputs.get('pipe_sections', [])

        st.header("Simulation Progress")
        progress_bar = st.progress(0); status_text = st.empty()
        status_text.text("Generating random samples..."); progress_bar.progress(10)
        
        # Generate fluid property samples
        rho_sampler = get_distribution(rho_dist, rho_mean, rho_std, rho_min, rho_max)
        mu_sampler = get_distribution(mu_dist, mu_mean, mu_std, mu_min, mu_max)
        mf_sampler = get_distribution(mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max)
        elev_sampler = get_distribution(elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max)
        
        rho_samples = rho_sampler(num_simulations)
        mu_samples_mPa = mu_sampler(num_simulations)
        mass_flow_samples = mf_sampler(num_simulations)
        elevation_samples = elev_sampler(num_simulations)
        mu_samples = mu_samples_mPa * 1e-3
        
        # Handle single vs multiple pipe sections
        if use_multiple_sections and pipe_sections:
            # Use the local sequential calculation function
            
            # Get pressure values
            upstream_pressure_Pa = inputs['pressure_kPa'] * 1000  # Convert kPa to Pa
            back_pressure_Pa = inputs['back_pressure_kPa'] * 1000  # Convert kPa to Pa
            
            # Get CoolProp fluid name
            coolprop_fluid = FLUID_COOLPROP_NAMES[selected_fluid]
            
            # Build per-section minor K dictionary from minor losses data
            per_section_minor_K = {}
            if not minor_losses_data.empty:
                minor_losses_df = pd.DataFrame(minor_losses_data)
                minor_losses_df['quantity'] = pd.to_numeric(minor_losses_df['quantity'], errors='coerce').fillna(0)
                minor_losses_df['k_value'] = pd.to_numeric(minor_losses_df['k_value'], errors='coerce').fillna(0)
                
                # Check if location column exists (multi-section mode)
                if 'location' in minor_losses_df.columns:
                    # Get the "All Sections" mode preference
                    all_sections_mode = st.session_state.get('all_sections_mode', 
                                                            "Distribute evenly (divide K by number of sections)")
                    
                    for _, row in minor_losses_df.iterrows():
                        location = row.get('location', 'All Sections')
                        k_contribution = row['quantity'] * row['k_value']
                        
                        if location == 'All Sections':
                            # Check the mode for "All Sections" behavior
                            if "Replicate" in all_sections_mode:
                                # Replicate: Apply full K to each section
                                for section in pipe_sections:
                                    section_id = section['id']
                                    if section_id not in per_section_minor_K:
                                        per_section_minor_K[section_id] = 0
                                    per_section_minor_K[section_id] += k_contribution  # Full K value
                            else:
                                # Distribute: Divide K equally among all sections (default)
                                for section in pipe_sections:
                                    section_id = section['id']
                                    if section_id not in per_section_minor_K:
                                        per_section_minor_K[section_id] = 0
                                    per_section_minor_K[section_id] += k_contribution / len(pipe_sections)
                        elif location == 'Upstream':
                            if 'upstream' not in per_section_minor_K:
                                per_section_minor_K['upstream'] = 0
                            per_section_minor_K['upstream'] += k_contribution
                        elif location == 'Downstream':
                            if 'downstream' not in per_section_minor_K:
                                per_section_minor_K['downstream'] = 0
                            per_section_minor_K['downstream'] += k_contribution
                        else:
                            # Specific section - FIXED: Map by section name, not location string
                            # Find the section with matching name
                            section_found = False
                            for section in pipe_sections:
                                if section['name'] == location:
                                    section_id = section['id']
                                    if section_id not in per_section_minor_K:
                                        per_section_minor_K[section_id] = 0
                                    per_section_minor_K[section_id] += k_contribution
                                    section_found = True
                                    break
                            
                            if not section_found:
                                st.warning(f"Warning: Could not find section named '{location}' for minor loss assignment")
                else:
                    # Single section mode - put all K at upstream
                    total_K = (minor_losses_df['quantity'] * minor_losses_df['k_value']).sum()
                    if total_K > 0:
                        per_section_minor_K['upstream'] = total_K
            
            # Get elevation mean
            elevation_mean = np.mean(elevation_samples)
            
            # Prepare property override parameters based on mode
            property_mode = st.session_state.get('property_mode', 'CoolProp Only (Default)')
            override_rho = None
            override_mu = None
            rho_bias = None
            mu_bias = None
            
            if property_mode == "Constant Override":
                # Use sampled values as constant overrides
                override_rho = rho_samples
                override_mu = mu_samples_mPa  # Will be converted to Pa·s in the function
            elif property_mode == "Bias vs CoolProp":
                # Calculate bias factors relative to upstream properties
                try:
                    rho_at_upstream = PropsSI("D", "T", inputs['temperature_K'], "P", upstream_pressure_Pa, coolprop_fluid)
                    mu_at_upstream = PropsSI("V", "T", inputs['temperature_K'], "P", upstream_pressure_Pa, coolprop_fluid) * 1e3  # Convert to mPa·s
                    rho_bias = rho_samples / rho_at_upstream
                    mu_bias = mu_samples_mPa / mu_at_upstream
                except:
                    # Fallback to no bias if CoolProp fails
                    pass
            
            # Perform enhanced sequential pressure calculation with per-section minor losses
            seq_results = calculate_sequential_pressure_drop(
                pipe_sections=pipe_sections,
                mass_flow_samples=mass_flow_samples,
                temperature_K=inputs['temperature_K'],
                upstream_pressure_Pa=upstream_pressure_Pa,
                back_pressure_Pa=back_pressure_Pa,
                gravity=gravity,
                friction_model=friction_model,
                coolprop_fluid=coolprop_fluid,
                num_simulations=num_simulations,
                calculate_friction_factor_func=calculate_friction_factor,
                per_section_minor_K=per_section_minor_K,
                elevation_change_m=elevation_mean,
                return_detailed=True,
                override_rho=override_rho,
                override_mu=override_mu,
                rho_bias=rho_bias,
                mu_bias=mu_bias
            )
            
            # Extract results
            deltaP_samples = seq_results['total_deltaP']
            outlet_pressure_samples = seq_results['outlet_pressure']
            pressure_profile = seq_results['pressure_profile']
            back_pressure_achieved = seq_results['back_pressure_achieved']
            
            # Store additional results for display
            st.session_state.simulation_results['pressure_profile'] = pressure_profile
            st.session_state.simulation_results['outlet_pressure'] = outlet_pressure_samples
            st.session_state.simulation_results['back_pressure_achieved'] = back_pressure_achieved
            # Store detailed section data if available
            if 'rho_by_section' in seq_results:
                st.session_state.simulation_results['rho_by_section'] = seq_results.get('rho_by_section')
                st.session_state.simulation_results['mu_by_section'] = seq_results.get('mu_by_section')
                st.session_state.simulation_results['v_by_section'] = seq_results.get('v_by_section')
            
            # For display purposes, create descriptive values
            diameter_str = " → ".join([f"{s['diameter']:.2f}" for s in pipe_sections])
            total_length_m = sum(s['length'] * (0.0254 if s['units'] == 'Inches' else 1.0) for s in pipe_sections)
            
            # Use weighted average values for display
            weighted_D = sum(s['diameter'] * s['length'] for s in pipe_sections) / sum(s['length'] for s in pipe_sections)
            D_input_samples = np.full(num_simulations, weighted_D)
            
            if pipe_sections[0]['units'] == 'Inches':
                L_input_samples = np.full(num_simulations, total_length_m / 0.0254)
            else:
                L_input_samples = np.full(num_simulations, total_length_m)
            
            weighted_roughness = sum(s['roughness'] * s['length'] for s in pipe_sections) / sum(s['length'] for s in pipe_sections)
            epsilon_samples = np.full(num_simulations, weighted_roughness)
            
            # For display, use initial fluid properties
            D_samples = D_input_samples * (0.0254 if units_selected_run == 'Inches' else 1.0)
            L_samples = L_input_samples
            v_samples = mass_flow_samples / (rho_samples * np.pi * (D_samples/2)**2)
            Re_samples = (rho_samples * v_samples * D_samples) / mu_samples
            f_samples = calculate_friction_factor(Re_samples, epsilon_samples, D_samples, friction_model)
            head_loss_pipe = np.zeros(num_simulations)  # Already calculated in deltaP_samples
            
        else:
            # Single pipe section - original code
            D_dist = inputs.get('D_dist', 'Deterministic')
            D_mean = inputs.get('D_mean', 0.05)
            D_std = inputs.get('D_std', 0.0)
            D_min = inputs.get('D_min', 0.0)
            D_max = inputs.get('D_max', 0.0)
            L_dist = inputs.get('L_dist', 'Deterministic')
            L_mean = inputs.get('L_mean', 0.5)
            L_std = inputs.get('L_std', 0.0)
            L_min = inputs.get('L_min', 0.0)
            L_max = inputs.get('L_max', 0.0)
            epsilon_dist = inputs.get('epsilon_dist', 'Deterministic')
            epsilon_mean = inputs.get('epsilon_mean', 4.5e-5)
            epsilon_std = inputs.get('epsilon_std', 0.0)
            epsilon_min = inputs.get('epsilon_min', 0.0)
            epsilon_max = inputs.get('epsilon_max', 0.0)
            
            D_sampler = get_distribution(D_dist, D_mean, D_std, D_min, D_max)
            L_sampler = get_distribution(L_dist, L_mean, L_std, L_min, L_max)
            e_sampler = get_distribution(epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max)
            
            D_input_samples = D_sampler(num_simulations)
            L_input_samples = L_sampler(num_simulations)
            epsilon_samples = e_sampler(num_simulations)
            
            if units_selected_run == "Inches":
                D_samples = D_input_samples * 0.0254
                L_samples = L_input_samples * 0.0254
            else:
                D_samples = D_input_samples
                L_samples = L_input_samples
            
            # Clip values
            D_samples = np.clip(D_samples, 1e-6, None)
            L_samples = np.clip(L_samples, 1e-6, None)
            epsilon_samples = np.clip(epsilon_samples, 1e-12, None)
            
            # Calculate flow parameters
            Q_samples = mass_flow_samples / rho_samples
            A_samples = np.pi * (D_samples/2)**2
            v_samples = Q_samples / A_samples
            Re_samples = (rho_samples * v_samples * D_samples) / mu_samples
            
            # Calculate friction factor
            f_samples = calculate_friction_factor(Re_samples, epsilon_samples, D_samples, friction_model)
            
            # Calculate pressure drop
            head_loss_pipe = f_samples * (L_samples/D_samples) * (v_samples**2) / (2.0*gravity)
            deltaP_pipe = rho_samples * gravity * head_loss_pipe
        
        # Minor losses and elevation - only compute for single-pipe runs
        progress_bar.progress(70); status_text.text("Finalizing pressure drop calculations...")
        
        if not (use_multiple_sections and pipe_sections):
            # Single-pipe path: compute and apply minor + elevation here
            total_K = 0
            if not minor_losses_data.empty:
                minor_losses_df = pd.DataFrame(minor_losses_data)
                minor_losses_df['quantity'] = pd.to_numeric(minor_losses_df['quantity'], errors='coerce').fillna(0)
                minor_losses_df['k_value'] = pd.to_numeric(minor_losses_df['k_value'], errors='coerce').fillna(0)
                total_K = (minor_losses_df['quantity'] * minor_losses_df['k_value']).sum()
            
            head_loss_minor = total_K * (v_samples**2) / (2.0*gravity)
            head_loss_elevation = elevation_samples
            head_loss_total = head_loss_pipe + head_loss_minor + head_loss_elevation
            deltaP_samples = rho_samples * gravity * head_loss_total
        else:
            # Multi-section path: already fully handled inside the sequential solver
            total_K = 0
            head_loss_minor = 0.0
            head_loss_elevation = 0.0
        
        progress_bar.progress(80); status_text.text("Finalizing calculations...")
        if st.session_state.unit_selection == 'psi':
            deltaP_samples = pa_to_psi(deltaP_samples); pressure_unit = 'psi'
        elif st.session_state.unit_selection == 'kPa':
            deltaP_samples = deltaP_samples / 1000; pressure_unit = 'kPa'
        elif st.session_state.unit_selection == 'bar':
            deltaP_samples = deltaP_samples / 100000; pressure_unit = 'bar'
        else:
            pressure_unit = 'Pa'
        progress_bar.progress(100); status_text.text("Simulation completed successfully!")
        st.header("Simulation Results")
        
        # Display multi-section configuration if applicable
        if use_multiple_sections and pipe_sections:
            st.info(f"**Multi-Section Pipe System:** {len(pipe_sections)} sections with diameters: {diameter_str} {units_selected_run}")
            
            # Show pressure profile if available
            if 'pressure_profile' in st.session_state.simulation_results:
                with st.expander("Pressure Profile Through System", expanded=True):
                    pressure_profile = st.session_state.simulation_results['pressure_profile']
                    outlet_pressure = st.session_state.simulation_results['outlet_pressure']
                    back_pressure_achieved = st.session_state.simulation_results['back_pressure_achieved']
                    
                    # Convert pressures to user's selected units
                    if st.session_state.unit_selection == 'psi':
                        pressure_converter = lambda p: pa_to_psi(p)
                        pressure_unit_label = 'psi'
                    elif st.session_state.unit_selection == 'kPa':
                        pressure_converter = lambda p: p / 1000
                        pressure_unit_label = 'kPa'
                    elif st.session_state.unit_selection == 'bar':
                        pressure_converter = lambda p: p / 100000
                        pressure_unit_label = 'bar'
                    else:  # Pascals
                        pressure_converter = lambda p: p
                        pressure_unit_label = 'Pa'
                    
                    # Create pressure profile plot
                    fig_profile, ax_profile = plt.subplots(figsize=(10, 6))
                    
                    # Calculate positions for each section boundary (use normalized lengths in meters)
                    positions = [0.0]
                    for section in pipe_sections:
                        length_m = section.get('length_m', section['length'] * (0.0254 if section['units'] == 'Inches' else 1.0))
                        positions.append(positions[-1] + length_m)
                    
                    # Plot mean pressure profile (converted to selected units)
                    mean_pressures = [pressure_converter(np.mean(p)) for p in pressure_profile]
                    ax_profile.plot(positions, mean_pressures, 'b-', linewidth=2, label='Mean Pressure')
                    
                    # Add confidence bands (converted to selected units)
                    lower_pressures = [pressure_converter(np.percentile(p, (100-confidence_level)/2)) for p in pressure_profile]
                    upper_pressures = [pressure_converter(np.percentile(p, confidence_level + (100-confidence_level)/2)) for p in pressure_profile]
                    ax_profile.fill_between(positions, lower_pressures, upper_pressures, alpha=0.3, color='blue', 
                                           label=f'{confidence_level}% CI')
                    
                    # Add back pressure line (converted to selected units)
                    back_pressure_Pa = inputs['back_pressure_kPa'] * 1000
                    back_pressure_display = pressure_converter(back_pressure_Pa)
                    ax_profile.axhline(y=back_pressure_display, color='r', linestyle='--', label='Back Pressure')
                    
                    # Mark section boundaries with custom names
                    for i, pos in enumerate(positions[1:-1], 1):
                        ax_profile.axvline(x=pos, color='gray', linestyle=':', alpha=0.5)
                        # Use custom name if available
                        section_name = pipe_sections[i].get('name', f'Section {i+1}')
                        ax_profile.text(pos, ax_profile.get_ylim()[1]*0.95, section_name, 
                                      rotation=90, ha='right', va='top', fontsize=8)
                    
                    ax_profile.set_xlabel('Distance Along Pipe (m)')
                    ax_profile.set_ylabel(f'Pressure ({pressure_unit_label})')
                    ax_profile.set_title('Pressure Profile Through Multi-Section Pipe')
                    ax_profile.legend()
                    ax_profile.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_profile)
                    
                    # Check if back pressure is achieved
                    pct_achieved = np.mean(back_pressure_achieved) * 100
                    if pct_achieved < 100:
                        st.warning(f"⚠️ Only {pct_achieved:.1f}% of simulations achieved the specified back pressure. "
                                 f"The system may be experiencing choking or excessive pressure drop.")
                    else:
                        st.success(f"✅ All simulations achieved the specified back pressure of {back_pressure_display:.1f} {pressure_unit_label}")
                    
                    # Display outlet pressure statistics (in selected units)
                    outlet_p_mean = pressure_converter(np.mean(outlet_pressure))
                    outlet_p_std = pressure_converter(np.std(outlet_pressure))
                    st.metric("Mean Outlet Pressure", f"{outlet_p_mean:.2f} ± {outlet_p_std:.2f} {pressure_unit_label}")
            
            # Show section details
            with st.expander("Section Details", expanded=False):
                section_info = []
                for i, section in enumerate(pipe_sections):
                    # Create sections_df for transition info
                    sections_df = pd.DataFrame(pipe_sections)
                    for j in range(len(pipe_sections)):
                        if pipe_sections[j]['units'] == 'Inches':
                            D_m = pipe_sections[j]['diameter'] * 0.0254
                        else:
                            D_m = pipe_sections[j]['diameter']
                        A_m2 = np.pi * (D_m / 2) ** 2
                        
                        if j > 0:
                            if pipe_sections[j-1]['units'] == 'Inches':
                                D_prev = pipe_sections[j-1]['diameter'] * 0.0254
                            else:
                                D_prev = pipe_sections[j-1]['diameter']
                            A_prev = np.pi * (D_prev / 2) ** 2
                            
                            if A_m2 > A_prev:
                                K_trans = (1 - A_prev/A_m2) ** 2
                                sections_df.loc[j, 'Transition'] = f"Expansion (K={K_trans:.3f})"
                            else:
                                K_trans = 0.5 * (1 - A_m2/A_prev)
                                sections_df.loc[j, 'Transition'] = f"Contraction (K={K_trans:.3f})"
                        else:
                            sections_df.loc[j, 'Transition'] = "Inlet"
                    
                    section_info.append({
                        'Section': section['id'],
                        f'Diameter ({section["units"]})': section['diameter'],
                        f'Length ({section["units"]})': section['length'],
                        'Roughness (m)': f"{section['roughness']:.2e}",
                        'Transition': sections_df.loc[i, 'Transition']
                    })
                st.dataframe(pd.DataFrame(section_info), use_container_width=True)
                st.caption(f"**Note:** Results show weighted average diameter ({weighted_D:.3f} {units_selected_run}) for correlation analysis")
        stat_cols = st.columns(4)
        mean_deltaP = np.mean(deltaP_samples); median_deltaP = np.median(deltaP_samples); std_deltaP = np.std(deltaP_samples)
        ci_lower = np.percentile(deltaP_samples, (100 - confidence_level)/2); ci_upper = np.percentile(deltaP_samples, confidence_level + (100 - confidence_level)/2)
        stat_cols[0].metric("Mean Pressure Drop", f"{mean_deltaP:.4g} {pressure_unit}")
        stat_cols[1].metric("Median Pressure Drop", f"{median_deltaP:.4g} {pressure_unit}")
        stat_cols[2].metric("Standard Deviation", f"{std_deltaP:.4g} {pressure_unit}")
        stat_cols[3].metric(f"{confidence_level}% CI Width", f"{(ci_upper-ci_lower):.4g} {pressure_unit}")
        st.markdown(f"<div style='background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0;'><b>{confidence_level}% Confidence Interval:</b> [{ci_lower:.4g}, {ci_upper:.4g}] {pressure_unit}</div>", unsafe_allow_html=True)
        st.subheader("Distributions & Sensitivity")
        # Build DataFrame of simulation outputs and key inputs
        if use_multiple_sections and pipe_sections:
            # For multi-section pipes, don't include misleading averaged diameter/length
            # Don't include string columns that would break correlation analysis
            data = pd.DataFrame({
                'Pressure Drop': deltaP_samples,
                'Mass Flow Rate (kg/s)': mass_flow_samples,
                'Elevation Change (m)': elevation_samples,
                'Reynolds Number (avg)': Re_samples,  # Mark as average
                'Friction Factor (avg)': f_samples,   # Mark as average
                'Velocity (avg)': v_samples           # Mark as average
            })
        else:
            # For single pipes, include actual diameter/length
            data = pd.DataFrame({
                'Pressure Drop': deltaP_samples,
                f'Diameter ({units_selected_run})': D_input_samples,
                f'Length ({units_selected_run})': L_input_samples,
                'Roughness (m)': epsilon_samples,
                'Mass Flow Rate (kg/s)': mass_flow_samples,
                'Elevation Change (m)': elevation_samples,
                'Reynolds Number': Re_samples,
                'Friction Factor': f_samples,
                'Velocity (m/s)': v_samples
            })

        # Histogram
        hist_fig, hist_ax = plt.subplots()
        hist_ax.hist(deltaP_samples, bins=50, color=COLORS['primary'], alpha=0.75, edgecolor='black')
        hist_ax.axvline(mean_deltaP, color='black', linestyle='--', label=f"Mean {mean_deltaP:.3g}")
        hist_ax.axvspan(ci_lower, ci_upper, color=COLORS['secondary'], alpha=0.2,
                        label=f"{confidence_level}% CI")
        hist_ax.set_title("Pressure Drop Distribution")
        hist_ax.set_xlabel(f"Pressure Drop ({pressure_unit})")
        hist_ax.set_ylabel("Frequency")
        hist_ax.legend()
        st.pyplot(hist_fig)

        # CDF plot
        cdf_fig, cdf_ax = plt.subplots()
        sorted_dp = np.sort(deltaP_samples)
        cdf = np.linspace(0, 1, len(sorted_dp))
        cdf_ax.plot(sorted_dp, cdf, color=COLORS['primary'])
        cdf_ax.set_title("Pressure Drop Empirical CDF")
        cdf_ax.set_xlabel(f"Pressure Drop ({pressure_unit})")
        cdf_ax.set_ylabel("Cumulative Probability")
        st.pyplot(cdf_fig)

        # Correlation / sensitivity
        corr_matrix = data.corr(numeric_only=True)
        pressure_drop_corr = corr_matrix['Pressure Drop'].drop('Pressure Drop').sort_values(key=lambda s: s.abs(), ascending=False)
        sens_col1, sens_col2 = st.columns([2,1])
        with sens_col1:
            sens_fig, sens_ax = plt.subplots(figsize=(8,6))
            pressure_drop_corr.plot(kind='barh', ax=sens_ax, color=[COLORS['primary'] if x>0 else COLORS['highlight'] for x in pressure_drop_corr])
            sens_ax.set_xlabel('Correlation Coefficient')
            sens_ax.set_title('Parameter Sensitivity (Linear Corr)')
            sens_ax.axvline(0, color='gray', linewidth=0.8)
            sens_fig.tight_layout()
            st.pyplot(sens_fig)
        with sens_col2:
            st.subheader("Correlations")
            pd_corr = pd.DataFrame(pressure_drop_corr).reset_index()
            pd_corr.columns = ['Parameter','Correlation']
            st.dataframe(pd_corr.style.background_gradient(cmap='RdBu_r', subset=['Correlation'], vmin=-1, vmax=1), use_container_width=True)
            strongest_pos = pd_corr.loc[pd_corr['Correlation'] == pd_corr['Correlation'].max()]
            strongest_neg = pd_corr.loc[pd_corr['Correlation'] == pd_corr['Correlation'].min()]
            pos_insight = "No significant positive correlation found." if strongest_pos.empty else f"• <b>{strongest_pos['Parameter'].values[0]}</b> strongest positive ({strongest_pos['Correlation'].values[0]:.3f})"
            neg_insight = "No significant negative correlation found." if strongest_neg.empty else f"• <b>{strongest_neg['Parameter'].values[0]}</b> strongest negative ({strongest_neg['Correlation'].values[0]:.3f})"
            st.markdown(f"<div style='background-color:#e8f5e9;padding:10px;border-radius:5px;margin:10px 0;'><b>Key Insights:</b><br>{pos_insight}<br><br>{neg_insight}</div>", unsafe_allow_html=True)

        # Summary statistics
        st.subheader("Parameter Statistics")
        if use_multiple_sections and pipe_sections:
            st.info("📊 **Note:** Diameter and Length values shown below are weighted averages used for correlation analysis. "
                   "The actual sequential pressure calculation uses individual section values as shown in the Section Details above.")
        summary_stats = pd.DataFrame({
            'Parameter': data.columns,
            'Mean': data.mean(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            '5%': data.quantile(0.05),
            '95%': data.quantile(0.95)
        })
        formatted_stats = summary_stats.copy()
        numeric_cols = formatted_stats.select_dtypes(include=['float64','int64']).columns
        for col in numeric_cols:
            if col != 'Parameter':
                formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.4g}")
        st.dataframe(formatted_stats, use_container_width=True)

        # Sample data
        st.subheader("Sample Data (First 10 Simulations)")
        if use_multiple_sections and pipe_sections:
            # Show the actual pipe configuration used
            st.info("📏 **Actual Pipe Configuration Used in Calculations:**")
            config_cols = st.columns(len(pipe_sections))
            for i, (col, section) in enumerate(zip(config_cols, pipe_sections)):
                with col:
                    st.metric(
                        f"Section {i+1}",
                        f"{section['diameter']:.2f} {section['units'].lower()}",
                        f"L: {section['length']:.1f} {section['units'].lower()}"
                    )
            
            # Create modified sample data that shows section-specific info
            sample_data_display = data.head(10).copy()
            
            # Add descriptive columns for multi-section configuration
            sample_data_display[f'Pipe Config'] = f"{len(pipe_sections)} sections"
            sample_data_display[f'Diameters ({units_selected_run})'] = diameter_str
            sample_data_display[f'Total Length ({units_selected_run})'] = f"{total_length_m:.3f} m"
            
            # Reorder columns for better display - use actual column names for multi-section
            cols_order = ['Pressure Drop', 'Pipe Config', f'Diameters ({units_selected_run})', 
                         f'Total Length ({units_selected_run})', 'Mass Flow Rate (kg/s)', 
                         'Elevation Change (m)', 'Reynolds Number (avg)', 'Friction Factor (avg)', 
                         'Velocity (avg)']
            # Only include columns that exist
            cols_order = [col for col in cols_order if col in sample_data_display.columns]
            sample_data_display = sample_data_display[cols_order]
            
            for col in sample_data_display.select_dtypes(include=['float64','int64']).columns:
                sample_data_display[col] = sample_data_display[col].apply(lambda x: f"{x:.4g}")
            
            st.caption("💡 Reynolds Number, Friction Factor, and Velocity shown are based on weighted average diameter for correlation analysis.")
            st.dataframe(sample_data_display, use_container_width=True)
        else:
            # Single pipe - show normal data
            sample_data = data.head(10).copy()
            for col in sample_data.select_dtypes(include=['float64','int64']).columns:
                sample_data[col] = sample_data[col].apply(lambda x: f"{x:.4g}")
            st.dataframe(sample_data, use_container_width=True)
        
        # Export to Excel section
        st.subheader("📥 Export Results")
        
        # Prepare Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Summary Statistics
            summary_export = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 
                          f'{confidence_level}% CI Lower', f'{confidence_level}% CI Upper'],
                f'Pressure Drop ({pressure_unit})': [
                    mean_deltaP, median_deltaP, std_deltaP,
                    np.min(deltaP_samples), np.max(deltaP_samples),
                    ci_lower, ci_upper
                ]
            })
            summary_export.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Full simulation data with configuration info for multi-section
            if use_multiple_sections and pipe_sections:
                # Create a modified DataFrame for export that includes configuration details
                export_data = data.copy()
                # Add configuration details columns
                export_data['Total Sections'] = len(pipe_sections)
                export_data['Diameters'] = diameter_str + f" {units_selected_run}"
                export_data['Total Length'] = f"{total_length_m:.3f} m"
                # Reorder columns for clarity
                cols_order = ['Pressure Drop', 'Total Sections', 'Diameters', 'Total Length', 
                             'Mass Flow Rate (kg/s)', 'Elevation Change (m)',
                             'Reynolds Number (avg)', 'Friction Factor (avg)', 'Velocity (avg)']
                export_data = export_data[cols_order]
                export_data.to_excel(writer, sheet_name='Full Results', index=False)
            else:
                # Single pipe - export as is
                data.to_excel(writer, sheet_name='Full Results', index=False)
            
            # Sheet 3: Input parameters
            # Calculate display pressures based on the unit
            pressure_unit_export = inputs["pressure_unit"]
            if pressure_unit_export == 'kPa':
                display_pressure_export = inputs['pressure_kPa']
                display_back_pressure_export = inputs['back_pressure_kPa']
            else:  # psia
                display_pressure_export = inputs['pressure_kPa'] / 6.894757
                display_back_pressure_export = inputs['back_pressure_kPa'] / 6.894757
            
            input_params = pd.DataFrame({
                'Parameter': [
                    'Fluid', 'Temperature (K)', f'Upstream Pressure ({pressure_unit_export})',
                    f'Back Pressure ({pressure_unit_export})', 'Number of Simulations',
                    'Confidence Level (%)', 'Friction Model', 'Gravity (m/s²)'
                ],
                'Value': [
                    selected_fluid, inputs['temperature_K'],
                    display_pressure_export,
                    display_back_pressure_export,
                    num_simulations, confidence_level, friction_model, gravity
                ]
            })
            input_params.to_excel(writer, sheet_name='Input Parameters', index=False)
            
            # Sheet 4: Pipe configuration
            if use_multiple_sections and pipe_sections:
                pipe_config_df = pd.DataFrame(pipe_sections)
                pipe_config_df.to_excel(writer, sheet_name='Pipe Configuration', index=False)
            else:
                pipe_config_single = pd.DataFrame({
                    'Parameter': ['Diameter', 'Length', 'Roughness', 'Units'],
                    'Value': [
                        inputs.get('D_mean', 'N/A'),
                        inputs.get('L_mean', 'N/A'),
                        inputs.get('epsilon_mean', 'N/A'),
                        units_selected_run
                    ]
                })
                pipe_config_single.to_excel(writer, sheet_name='Pipe Configuration', index=False)
            
            # Sheet 5: Minor losses
            if not minor_losses_data.empty:
                minor_losses_data.to_excel(writer, sheet_name='Minor Losses', index=False)
            
            # Sheet 6: Correlation analysis
            correlation_export = pd.DataFrame(pressure_drop_corr).reset_index()
            correlation_export.columns = ['Parameter', 'Correlation with Pressure Drop']
            correlation_export.to_excel(writer, sheet_name='Correlations', index=False)
        
        # Create download button
        excel_data = output.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pressure_drop_simulation_{timestamp}.xlsx"
        
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download complete simulation results as Excel file with multiple sheets"
        )
        
        st.success(f"✅ Excel file ready for download: {filename}")
# --------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>Engineering Monte Carlo Pressure Drop Calculator • v2.6 • Using CoolProp</small>
</div>
""", unsafe_allow_html=True)

# Display equations when expanded
with st.expander("📐 View Equations Used", expanded=False):
    st.markdown("""
    ## Governing Equations with Unit Analysis
    
    ### Basic Flow Equations
    
    **Volumetric Flow Rate:**
    """)
    st.latex(r"Q = \frac{\dot{m}}{\rho} \quad [\frac{kg/s}{kg/m^3} = m^3/s]")
    
    st.markdown("**Flow Velocity:**")
    st.latex(r"v = \frac{Q}{A} = \frac{4Q}{\pi D^2} \quad [\frac{m^3/s}{m^2} = m/s]")
    
    st.markdown("**Reynolds Number:**")
    st.latex(r"Re = \frac{\rho \, v \, D}{\mu} \quad [\frac{kg/m^3 \times m/s \times m}{kg/(m \cdot s)} = \text{dimensionless}]")
    
    st.markdown("---")
    st.markdown("### Friction Factor Models")
    
    st.markdown("**1. Standard Model (Laminar + Swamee-Jain):**")
    st.markdown("For laminar flow (Re ≤ 2000):")
    st.latex(r"f = \frac{64}{Re}")
    st.markdown("For turbulent flow (Re > 2000) - Swamee-Jain equation:")
    st.latex(r"f = 0.25 \Big/ \left[\log_{10}\!\Bigl(\frac{\epsilon}{3.7\,D} + \frac{5.74}{Re^{0.9}}\Bigr)\right]^2")
    
    st.markdown("**2. Churchill Universal Model (All Flow Regimes):**")
    st.latex(r"f = 8 \left[\left(\frac{8}{Re}\right)^{12} + \frac{1}{\left(A + B\right)^{1.5}}\right]^{1/12}")
    st.markdown("Where:")
    st.latex(r"A = \left[2.457 \ln \left( \frac{1}{ (7/Re)^{0.9} + 0.27\, (\epsilon/D) } \right) \right]^{16}")
    st.latex(r"B = \left(\frac{37530}{Re}\right)^{16}")
    
    st.markdown("**3. Blended Model (Smooth Transition):**")
    st.markdown("For laminar flow (Re ≤ 2000):")
    st.latex(r"f_{lam} = \frac{64}{Re}")
    st.markdown("For turbulent flow (Re ≥ 4000) - Swamee-Jain:")
    st.latex(r"f_{turb} = 0.25 \Big/ \left[\log_{10}\!\Bigl(\frac{\epsilon}{3.7\,D} + \frac{5.74}{Re^{0.9}}\Bigr)\right]^2")
    st.markdown("For transitional flow (2000 < Re < 4000) - Linear interpolation:")
    st.latex(r"f = (1-\alpha) \cdot f_{lam} + \alpha \cdot f_{turb}")
    st.latex(r"\alpha = \frac{Re - 2000}{2000}")
    
    st.markdown("---")
    st.markdown("### Pressure Drop Calculations")
    
    st.markdown("**Friction Pressure Drop (Darcy-Weisbach):**")
    st.latex(r"\Delta P_f = \frac{1}{2} \rho f \frac{L}{D} v^2")
    
    st.markdown("**Minor Loss Pressure Drop:**")
    st.latex(r"\Delta P_{minor} = \frac{1}{2} \rho K v^2")
    
    st.markdown("**Elevation Pressure Drop:**")
    st.latex(r"\Delta P_{elev} = \rho g \Delta z")
    
    st.markdown("**Total Pressure Drop:**")
    st.latex(r"\Delta P_{total} = \Delta P_f + \Delta P_{minor} + \Delta P_{elev}")
    
    st.markdown("---")
    st.markdown("### Transition Losses (Multi-Section Pipes)")
    
    st.markdown("**Sudden Expansion (Borda-Carnot Formula):**")
    st.latex(r"K_{exp} = \left(1 - \frac{A_1}{A_2}\right)^2 = \left(1 - \frac{D_1^2}{D_2^2}\right)^2")
    st.markdown("*Uses upstream velocity and density*")
    
    st.markdown("**Sudden Contraction (Ludwig Formula):**")
    st.latex(r"K_{cont} = 0.5 \left(1 - \frac{A_2}{A_1}\right) = 0.5 \left(1 - \frac{D_2^2}{D_1^2}\right)")
    st.markdown("*Uses downstream velocity and density*")
    
    st.markdown("---")
    st.markdown("### Sequential Pressure Drop Algorithm")
    
    st.markdown("""
    For multi-section pipes, the calculation proceeds sequentially:
    
    1. **Initialize:** P₀ = Upstream Pressure
    2. **For each section i:**
       - Calculate properties at current pressure: ρᵢ(Pᵢ), μᵢ(Pᵢ)
       - Calculate flow parameters: vᵢ, Reᵢ, fᵢ
       - Calculate friction loss: ΔPf,ᵢ = ½ρᵢfᵢ(Lᵢ/Dᵢ)vᵢ²
       - Calculate transition loss (if not first section)
       - Calculate minor losses for section: ΔPm,ᵢ = ½ρᵢKᵢvᵢ²
       - Update pressure: Pᵢ₊₁ = Pᵢ - ΔPf,ᵢ - ΔPtrans,ᵢ - ΔPm,ᵢ
    3. **Add elevation loss:** ΔPelev = ρg·Δz (distributed proportionally)
    4. **Check back pressure:** Verify if Pfinal ≥ Pback
    """)
    
    st.markdown("---")
    st.markdown("### Notes on Units and Properties")
    
    st.markdown("""
    **Pressure Units:**
    - Pa (Pascal) = kg/(m·s²) [SI base unit]
    - 1 kPa = 1,000 Pa
    - 1 bar = 100,000 Pa
    - 1 psi = 6,894.757 Pa
    
    **Viscosity Units:**
    - Dynamic viscosity (μ): Pa·s or kg/(m·s)
    - 1 mPa·s = 0.001 Pa·s = 1 cP (centipoise)
    - Water at 20°C: μ ≈ 1.0 mPa·s
    - Air at 20°C: μ ≈ 0.018 mPa·s
    
    **Roughness Values (ε):**
    - Commercial Steel: 4.5×10⁻⁵ m
    - Stainless Steel: 1.0×10⁻⁵ m
    - PVC/Plastic: 1.5×10⁻⁶ m
    - Cast Iron: 2.6×10⁻⁴ m
    """)
