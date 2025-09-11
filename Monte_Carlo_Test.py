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
COMMON_K_VALUES = {
    # 1. Elbows, bends, returns
    "90° LR Elbow (B16.9)": 0.25,
    "90° SR Elbow": 0.75,
    "45° LR Elbow": 0.15,
    "180° LR Return Bend": 0.20,
    "180° SR Return Bend": 0.60,
    # "Smooth pipe bend, m-miter (2<=m<=5)": 0.10 * m, # Cannot represent 'm' directly, user must use Custom
    "Space-Saver Forged Elbow (Rc~0.5D)": 1.50,

    # 2. Tees, laterals, crosses
    "Tee, Run Through (Line -> Line)": 0.60,
    "Tee, Side-Out Branch (Line -> Branch)": 1.80,
    "Tee, Combining Branch (Branch -> Line)": 1.55,
    "45° Lateral Wye, Main Flow": 0.40,
    # "45° Lateral Wye, Branch Flow": ?, # K value not provided for branch flow in table
    "Pipe Cross, Straight Run": 1.00,
    # "Pipe Cross, Branch Flow": ?, # K value not provided for branch flow

    # 3. Valves (fully open unless noted)
    "Gate Valve (Plain Wedge)": 0.08,
    "Ball Valve, Full Port": 0.05,
    "Ball Valve, Reduced Port (70% Area)": 0.40,
    "Globe Valve, Z-Pattern": 10.0,
    "Globe Valve, Angle-Pattern": 5.0,
    "Butterfly Valve, 30° Open": 2.0,
    "Butterfly Valve, 60° Open": 15.0,
    "Swing Check Valve, Forward Flow": 2.0,
    "Lift Check Valve, Forward Flow": 10.0,
    "Cryogenic DBB Valve, Full Port": 0.11,

    # 4. Reducers, diffusers, contractions (Specific examples, use Custom for others)
    "Sudden Enlargement (D2/D1 = 2)": 0.50,
    "Tapered Diffuser (15° Half Angle, 3D Long)": 0.10,
    "Sudden Contraction (D2/D1 = 0.5)": 0.40,
    "Tapered Reducer (30° Total Angle)": 0.20,

    # 5. Junctions, entrances, exits
    "Sharp-Edged Pipe Entrance": 0.50,
    "Rounded Entrance (r/D >= 0.15)": 0.04,
    "Pipe Exit (to Large Tank)": 1.00,
    "Re-Entry (Flush)": 2.00,

    # 6. Miscellaneous inline devices
    "Orifice Plate (β = 0.60)": 4.3, # Calculated from 0.56 / 0.60^4
    "Coriolis Mass-Flow Meter (Standard)": 2.0,
    "Wire-Mesh Strainer (40 Mesh, Clean)": 1.2,
    "Plate-Type Heat Exchanger (Port Section, per pass)": 4.0,
    "Rupture Disk Holder (ASME Type)": 2.5,

    # Keep previous useful entries if not directly replaced
    "Union, Threaded": 0.08, # From previous list
    "Water meter": 7.0, # From previous list
    "Bellows (1 Convolution)": 0.1, # Example, highly variable
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
            pressure_unit = st.selectbox("Pressure Unit", ['kPa', 'psia'], key='pressure_unit')
            # Derive display value
            display_pressure = st.session_state.pressure_kPa if pressure_unit == 'kPa' else st.session_state.pressure_kPa / 6.894757
            pressure_input = st.number_input(
                f"Pressure ({pressure_unit})",
                min_value=0.1,
                step=1.0,
                format="%.4g",
                value=display_pressure,
                key='pressure_input'
            )
            # Convert back to kPa for internal use
            st.session_state.pressure_kPa = pressure_input if pressure_unit == 'kPa' else pressure_input * 6.894757
            pressure_kPa = st.session_state.pressure_kPa
            
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
    
    # Return all needed values from fluid section
    return selected_fluid, calc_density, calc_viscosity_mPas, rho_dist, rho_mean, rho_std, rho_min, rho_max, mu_dist, mu_mean, mu_std, mu_min, mu_max

# Function to render the minor losses section and handle component selection
def render_minor_losses_section():
    with st.expander("Minor Losses (Fittings, Valves, etc.)", expanded=True):
        st.markdown("""
            ### Component Minor Losses
            Select components contributing to minor pressure losses. Quantity defaults to 1 and K auto-fills (editable).
        """)
        
        # Initialize state for individual component entries if needed
        if 'minor_losses_state' not in st.session_state:
            st.session_state.minor_losses_state = {}
            
        # Ensure multiselect key exists to avoid widget key conflict
        if 'minor_loss_multiselect' not in st.session_state:
            st.session_state.minor_loss_multiselect = []
            
        # Use the session state default value for the multiselect
        # This is the key fix to preserve component selections between reruns
        selected_comps = st.multiselect(
            "Select Components:",
            options=component_options,
            key="minor_loss_multiselect",
            help="Select standard components or choose 'Custom' to add a user-defined K value."
        )

        # Custom component creation UI
        if 'Custom' in selected_comps:
            st.markdown("**Custom Component Entry**")
            cc1, cc2, cc3 = st.columns([2,1,1])
            with cc1:
                custom_name = st.text_input("Name", key="custom_k_name", placeholder="e.g., Special Insert")
            with cc2:
                custom_qty = st.number_input("Qty", min_value=1, step=1, key="custom_k_qty")
            with cc3:
                custom_k = st.number_input("K", min_value=0.0, format="%.5g", key="custom_k_val")
            add_custom = st.button("➕ Add Custom Component")
            if add_custom:
                base = custom_name.strip() or 'Custom Component'
                unique = f"{base} #{st.session_state.custom_k_counter}"
                st.session_state.custom_k_counter += 1
                st.session_state.minor_losses_state[unique] = {"quantity": custom_qty, "k_value": custom_k}
                st.session_state[f"qty_{unique}"] = custom_qty
                st.session_state[f"k_{unique}"] = custom_k
                st.success(f"Added {unique}")
        
        # Remove the line that tries to modify session state after widget creation
        # st.session_state.minor_loss_multiselect = selected_comps

        # Remove deselected components from state
        for comp in list(st.session_state.minor_losses_state.keys()):
            if comp not in selected_comps:
                st.session_state.minor_losses_state.pop(comp)

        # For each selected component, show inputs
        for comp in selected_comps:
            comp_qty_key = f"qty_{comp}"
            comp_k_key = f"k_{comp}"
            
            # Initialize state for individual component entries if needed
            if comp not in st.session_state.minor_losses_state:
                default_k = COMMON_K_VALUES.get(comp, 0.0)
                st.session_state.minor_losses_state[comp] = {'quantity': 1, 'k_value': default_k}
                # Also initialize widget keys if component is new
                st.session_state[comp_qty_key] = 1
                st.session_state[comp_k_key] = default_k
            else:
                # Ensure widget keys exist even if component state exists (e.g., after script rerun)
                if comp_qty_key not in st.session_state:
                     st.session_state[comp_qty_key] = st.session_state.minor_losses_state[comp]['quantity']
                if comp_k_key not in st.session_state:
                     st.session_state[comp_k_key] = st.session_state.minor_losses_state[comp]['k_value']

            q_col, k_col = st.columns([1, 1])
            with q_col:
                # Just create the widget, don't assign the return value
                st.number_input(
                    f"Quantity for {comp}",
                    min_value=1,
                    step=1,
                    key=comp_qty_key
                )
            with k_col:
                # Just create the widget, don't assign the return value
                st.number_input(
                    f"K Value for {comp}",
                    min_value=0.0,
                    format="%.3f",
                    key=comp_k_key
                )
            
            # Update the state dictionary from the widget keys after creation
            st.session_state.minor_losses_state[comp]['quantity'] = st.session_state[comp_qty_key]
            st.session_state.minor_losses_state[comp]['k_value'] = st.session_state[comp_k_key]

        # Build DataFrame for storage and later use
        minor_list = []
        for comp, vals in st.session_state.minor_losses_state.items():
            minor_list.append({'component_type': comp, 'quantity': vals['quantity'], 'k_value': vals['k_value']})
        st.session_state.minor_losses_data = pd.DataFrame(minor_list)
        
        # If no components selected, ensure DataFrame has required columns to avoid KeyError
        if st.session_state.minor_losses_data.empty:
            st.session_state.minor_losses_data = pd.DataFrame(columns=['component_type','quantity','k_value'])

        # Display table
        st.dataframe(st.session_state.minor_losses_data, use_container_width=True)
        
        # Safely compute total K only if quantity and k_value columns exist
        df_ml = st.session_state.minor_losses_data
        if isinstance(df_ml, pd.DataFrame) and {'quantity', 'k_value'}.issubset(df_ml.columns):
            total_k = (df_ml['quantity'] * df_ml['k_value']).sum()
        else:
            total_k = 0.0
        st.info(f"Total K Value: **{total_k:.3f}**")
    
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
        temperature_K, pressure_kPa, pressure_unit, minor_loss_multiselect, selected_fluid_idx,
        units_selected, friction_model, roughness_preset
    ):
    """Store all simulation inputs in the session state for persistence between tabs"""
    st.session_state.simulation_inputs = {
        # Fluid properties
        'selected_fluid': selected_fluid,
        'selected_fluid_idx': selected_fluid_idx, # <- Added
        'temperature_K': temperature_K,
    'pressure_kPa': pressure_kPa,
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
        'minor_losses_data': minor_losses_data, # Store the minor losses data
        'minor_loss_multiselect': minor_loss_multiselect, # <- Added selected components list
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
    # Create two columns: one for fluid & simulation setup, one for pipe parameters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_fluid, calc_density, calc_viscosity_mPas, rho_dist, rho_mean, rho_std, rho_min, rho_max, mu_dist, mu_mean, mu_std, mu_min, mu_max = render_fluid_section()
        
        # Simulation parameters
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
            
        # Output units selection with visual indicators - Fix unit selection persistence
        st.subheader("Output Units")
        
        unit_options = ['Pascals (Pa)', 'kPa', 'bar', 'psi']
        unit_icons = ['⚖️', '🔄', '📊', '🇺🇸']
        
        unit_cols = st.columns(len(unit_options))
        # Fix: Include unit_options in the zip function
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
            
        # Large, prominent "Run Simulation" button
        st.markdown("<br>", unsafe_allow_html=True)

        # Define the function to run the simulation and save inputs
        def validate_inputs():
            errs = []
            # Min/max checks for Uniform / Triangular
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
            # Pipe Geometry
            D_dist_val = st.session_state.D_dist
            D_mean_val = st.session_state.D_mean
            D_std_val = st.session_state.get('D_std', 0.0)
            D_min_val = st.session_state.get('D_min', 0.0)
            D_max_val = st.session_state.get('D_max', 0.0)
            L_dist_val = st.session_state.L_dist
            L_mean_val = st.session_state.L_mean
            L_std_val = st.session_state.get('L_std', 0.0)
            L_min_val = st.session_state.get('L_min', 0.0)
            L_max_val = st.session_state.get('L_max', 0.0)
            epsilon_dist_val = st.session_state.epsilon_dist
            epsilon_mean_val = st.session_state.epsilon_mean
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
            minor_loss_multiselect_val = st.session_state.minor_loss_multiselect # <- Get value
            gravity_val = st.session_state.gravity

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
                temperature_K_val, pressure_kPa_val, st.session_state.pressure_unit, minor_loss_multiselect_val, selected_fluid_idx_val,
                st.session_state.units_selected, st.session_state.friction_model, st.session_state.roughness_preset
            )
            # Removed bulk st.session_state.update to avoid StreamlitAPIException when setting widget keys (e.g., selected_fluid_idx) post-instantiation.
            # Switch to the results tab after saving
            switch_to_results()

        # Reproducibility controls
        st.subheader("Reproducibility")
        seed_col1, seed_col2 = st.columns([2,1])
        with seed_col1:
            rng_seed = st.text_input("Random Seed (optional integer)", key="rng_seed")
        with seed_col2:
            if rng_seed and not rng_seed.isdigit():
                st.warning("Seed must be an integer.")

        errors = validate_inputs()
        if errors:
            st.error("Resolve before running:\n- " + "\n- ".join(errors))
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
        # Unpack
        rho_dist = inputs['rho_dist']; rho_mean = inputs['rho_mean']; rho_std = inputs['rho_std']; rho_min = inputs['rho_min']; rho_max = inputs['rho_max']
        mu_dist = inputs['mu_dist']; mu_mean = inputs['mu_mean']; mu_std = inputs['mu_std']; mu_min = inputs['mu_min']; mu_max = inputs['mu_max']
        D_dist = inputs['D_dist']; D_mean = inputs['D_mean']; D_std = inputs['D_std']; D_min = inputs['D_min']; D_max = inputs['D_max']
        L_dist = inputs['L_dist']; L_mean = inputs['L_mean']; L_std = inputs['L_std']; L_min = inputs['L_min']; L_max = inputs['L_max']
        epsilon_dist = inputs['epsilon_dist']; epsilon_mean = inputs['epsilon_mean']; epsilon_std = inputs['epsilon_std']; epsilon_min = inputs['epsilon_min']; epsilon_max = inputs['epsilon_max']
        mass_flow_dist = inputs['mass_flow_dist']; mass_flow_mean = inputs['mass_flow_mean']; mass_flow_std = inputs['mass_flow_std']; mass_flow_min = inputs['mass_flow_min']; mass_flow_max = inputs['mass_flow_max']
        elevation_dist = inputs['elevation_dist']; elevation_mean = inputs['elevation_mean']; elevation_std = inputs['elevation_std']; elevation_min = inputs['elevation_min']; elevation_max = inputs['elevation_max']
        num_simulations = inputs['num_simulations']; confidence_level = inputs['confidence_level']
        minor_losses_data = inputs['minor_losses_data']; gravity = inputs['gravity']
        units_selected_run = inputs.get('units_selected', st.session_state.units_selected)
        friction_model = inputs.get('friction_model', "Standard (Laminar + Swamee-Jain)")
        selected_fluid = inputs['selected_fluid']

        st.header("Simulation Progress")
        progress_bar = st.progress(0); status_text = st.empty()
        status_text.text("Generating random samples..."); progress_bar.progress(10)
        rho_sampler = get_distribution(rho_dist, rho_mean, rho_std, rho_min, rho_max)
        mu_sampler = get_distribution(mu_dist, mu_mean, mu_std, mu_min, mu_max)
        D_sampler = get_distribution(D_dist, D_mean, D_std, D_min, D_max)
        L_sampler = get_distribution(L_dist, L_mean, L_std, L_min, L_max)
        e_sampler = get_distribution(epsilon_dist, epsilon_mean, epsilon_std, epsilon_min, epsilon_max)
        mf_sampler = get_distribution(mass_flow_dist, mass_flow_mean, mass_flow_std, mass_flow_min, mass_flow_max)
        elev_sampler = get_distribution(elevation_dist, elevation_mean, elevation_std, elevation_min, elevation_max)
        rho_samples = rho_sampler(num_simulations); mu_samples_mPa = mu_sampler(num_simulations)
        D_input_samples = D_sampler(num_simulations); L_input_samples = L_sampler(num_simulations)
        epsilon_samples = e_sampler(num_simulations); mass_flow_samples = mf_sampler(num_simulations)
        elevation_samples = elev_sampler(num_simulations)
        mu_samples = mu_samples_mPa * 1e-3
        if units_selected_run == "Inches":
            D_samples = D_input_samples * 0.0254; L_samples = L_input_samples * 0.0254
        else:
            D_samples = D_input_samples; L_samples = L_input_samples
        rho_samples = np.clip(rho_samples, 1e-6, None); mu_samples = np.clip(mu_samples, 1e-12, None)
        D_samples = np.clip(D_samples, 1e-6, None); L_samples = np.clip(L_samples, 1e-6, None)
        epsilon_samples = np.clip(epsilon_samples, 1e-12, None); mass_flow_samples = np.clip(mass_flow_samples, 1e-6, None)
        progress_bar.progress(30); status_text.text("Calculating flow parameters...")
        Q_samples = mass_flow_samples / rho_samples; A_samples = np.pi * (D_samples/2)**2; v_samples = Q_samples / A_samples
        Re_samples = (rho_samples * v_samples * D_samples) / mu_samples
        progress_bar.progress(50); status_text.text("Computing friction factors...")
        with np.errstate(divide='ignore', invalid='ignore'):
            if friction_model == "Churchill (All Regimes)":
                Re_eff = np.clip(Re_samples, 1e-12, None); rr = epsilon_samples / D_samples
                A = (2.457 * np.log(1.0 / ((7.0 / Re_eff)**0.9 + 0.27 * rr)))**16; B = (37530.0 / Re_eff)**16
                f_samples = 8.0 * (((8.0 / Re_eff)**12) + 1.0 / ((A + B)**1.5))**(1/12)
            elif friction_model == "Blended (Transitional)":
                f_turbulent = 0.25 / (np.log10((epsilon_samples/(3.7*D_samples)) + (5.74/(Re_samples**0.9))))**2
                f_lam = 64.0 / Re_samples
                alpha = (Re_samples - 2000)/2000
                f_samples = np.where(Re_samples <= 2000, f_lam,
                               np.where(Re_samples >= 4000, f_turbulent,
                                        (1-alpha)*f_lam + alpha*f_turbulent))
            else:
                f_turbulent = 0.25 / (np.log10((epsilon_samples/(3.7*D_samples)) + (5.74/(Re_samples**0.9))))**2
                laminar_flow = (Re_samples <= 2000); f_samples = np.where(laminar_flow, 64.0/Re_samples, f_turbulent)
        progress_bar.progress(70); status_text.text("Calculating pressure drops...")
        head_loss_pipe = f_samples * (L_samples/D_samples) * (v_samples**2) / (2.0*gravity)
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
        sample_data = data.head(10).copy()
        for col in sample_data.select_dtypes(include=['float64','int64']).columns:
            sample_data[col] = sample_data[col].apply(lambda x: f"{x:.4g}")
        st.dataframe(sample_data, use_container_width=True)

        # Export
        st.subheader("Export Results")
        def to_excel(sim_data, summary_data, sensitivity_data, hist_fig, cdf_fig, sens_fig, minor_losses_df):
            out_xlsx = io.BytesIO()
            with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
                minor_losses_df.to_excel(writer, sheet_name='Minor Losses', index=False)
                sim_data.to_excel(writer, sheet_name='Simulation Data', index=False)
                sensitivity_data.to_excel(writer, sheet_name='Sensitivity', index=False)
                summary_data.to_excel(writer, sheet_name='Summary', index=False, startrow=9)
                workbook = writer.book; summary_ws = writer.sheets['Summary']
                summary_ws.write('A1', 'Pressure Drop Simulation Results', workbook.add_format({'bold': True, 'font_size': 14}))
                summary_ws.write('A2', f'Fluid: {selected_fluid}, Simulations: {num_simulations}')
                summary_ws.write('A4', 'Key Statistics:')
                summary_ws.write('A5', f'Mean Pressure Drop: {mean_deltaP:.4g} {pressure_unit}')
                summary_ws.write('A6', f'Standard Deviation: {std_deltaP:.4g} {pressure_unit}')
                summary_ws.write('A7', f'{confidence_level}% Confidence Interval: [{ci_lower:.4g}, {ci_upper:.4g}] {pressure_unit}')
                png_hist = io.BytesIO(); hist_fig.savefig(png_hist, format='png', bbox_inches='tight'); png_hist.seek(0); summary_ws.insert_image('D2','Histogram',{'image_data': png_hist})
                png_cdf = io.BytesIO(); cdf_fig.savefig(png_cdf, format='png', bbox_inches='tight'); png_cdf.seek(0); summary_ws.insert_image('D22','CDF',{'image_data': png_cdf})
                png_sens = io.BytesIO(); sens_fig.savefig(png_sens, format='png', bbox_inches='tight'); png_sens.seek(0); summary_ws.insert_image('D42','Sensitivity',{'image_data': png_sens})
            return out_xlsx.getvalue()
        minor_losses_df_export = pd.DataFrame(minor_losses_data) if not minor_losses_data.empty else pd.DataFrame(columns=['component_type','quantity','k_value'])
        excel_file = to_excel(data, summary_stats, pd.DataFrame(pressure_drop_corr).reset_index(), hist_fig, cdf_fig, sens_fig, minor_losses_df_export)
        st.download_button(label="📊 Download Complete Results (Excel)", data=excel_file, file_name=f'pressure_drop_{selected_fluid}_{num_simulations}sims.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
    else:
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
    <small>Engineering Monte Carlo Pressure Drop Calculator • v2.6 • Using CoolProp</small>
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
    st.latex(r"f = 0.25 \Big/ \left[\log_{10}\!\Bigl(\frac{\epsilon}{3.7\,D} + \frac{5.74}{Re^{0.9}}\Bigr)\right]^2")

    st.markdown("**Friction Factor (laminar flow):**")
    st.latex(r"f = \frac{64}{Re} \quad (Re \leq 2000)")

    st.markdown("**Churchill Universal Friction Factor (All Regimes):**")
    st.latex(r"f = 8 \left[\left(\frac{8}{Re}\right)^{12} + \frac{1}{\left(A + B\right)^{1.5}}\right]^{1/12}")
    st.latex(r"A = \left[2.457 \ln \left( \frac{1}{ (7/Re)^{0.9} + 0.27\, (\epsilon/D) } \right) \right]^{16} \quad B = \left(\frac{37530}{Re}\right)^{16}")
    
    st.markdown("**Head Loss (Darcy-Weisbach):**")
    st.latex(r"h_f = f\,\frac{L}{D}\,\frac{v^2}{2g} \quad [\text{dimensionless} \times \frac{m}{m} \times \frac{(m/s)^2}{m/s^2} = m]")
    
    st.markdown("**Head Loss from Fittings and Valves:**")
    st.latex(r"h_\mathrm{fittings} = K_\mathrm{total}\,\frac{v^2}{2g} \quad [\text{dimensionless} \times \frac{(m/s)^2}{m/s^2} = m]")
    
    st.markdown("**Total Head Loss:**")
    st.latex(r"h_\mathrm{total} = h_f + h_\mathrm{fittings} + \Delta z \quad [m + m + m = m]")
    
    st.markdown("""
    **Pressure Drop:**""")
    st.latex(r"\Delta P = \rho\,g\,h_\mathrm{total} \quad [kg/m^3 \times m/s^2 \times m = kg/(m \cdot s^2) = Pa]")
    
    st.markdown("""
    ### Notes on Viscosity Units
    
    - Dynamic viscosity (μ) is used in these calculations with units of Pa·s or kg/(m·s)
    - The CoolProp library returns viscosity in Pa·s
    - User inputs are in mPa·s (1 mPa·s = 0.001 Pa·s) for convenience
    - For water at 20°C: μ ≈ 1.0 mPa·s = 0.001 Pa·s
    - For air at 20°C: μ ≈ 0.018 mPa·s = 0.000018 Pa·s
    """)
