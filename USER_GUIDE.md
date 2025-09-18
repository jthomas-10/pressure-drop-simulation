# Monte Carlo Pressure Drop Simulation - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [User Interface Overview](#user-interface-overview)
5. [How to Use](#how-to-use)
6. [Understanding the Calculations](#understanding-the-calculations)
7. [Interpreting Results](#interpreting-results)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

---

## Introduction

The **Monte Carlo Pressure Drop Simulation** is a powerful engineering tool that calculates pressure drop in pipe systems using probabilistic methods. Unlike traditional calculators that provide single-point estimates, this tool accounts for uncertainties in input parameters to provide a statistical distribution of possible outcomes.

### Key Features
- **Monte Carlo simulation** with up to 200,000 iterations
- **Multi-section pipe support** with automatic transition loss calculations
- **CoolProp integration** for accurate fluid properties
- **Three friction factor models** (Standard, Churchill, Blended)
- **Comprehensive minor loss library** based on industry standards
- **Statistical analysis** with confidence intervals and sensitivity analysis
- **Excel export** for detailed reporting

### When to Use This Tool
- Designing pipe systems with uncertain operating conditions
- Analyzing existing systems with measurement uncertainties
- Risk assessment for pressure drop calculations
- Sensitivity analysis to identify critical parameters
- Multi-diameter pipe systems with transitions

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Internet connection (for CoolProp property calculations)

### Required Libraries
Install the required Python packages:

```bash
pip install streamlit numpy pandas matplotlib scipy CoolProp openpyxl
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

### Running the Application

1. Open a terminal/command prompt
2. Navigate to the application directory
3. Run the following command:

```bash
streamlit run Monte_Carlo_Test.py
```

4. The application will open in your default web browser at `http://localhost:8501`

---

## Quick Start

### Basic Single-Pipe Calculation

1. **Select Fluid**: Choose from Water, Air, Nitrogen, Oxygen, Hydrogen, or CO₂
2. **Set Operating Conditions**: 
   - Enter upstream pressure (kPa or psia)
   - Enter temperature (K)
   - Enter back pressure (downstream pressure)
3. **Define Pipe Geometry**:
   - Enter pipe diameter (inches or meters)
   - Enter pipe length
   - Select roughness material or enter custom value
4. **Set Flow Rate**: Enter mass flow rate (kg/s)
5. **Add Minor Losses** (optional): Select fittings and valves
6. **Run Simulation**: Click "Run Monte Carlo Simulation"
7. **View Results**: Switch to Results tab

---

## User Interface Overview

### Setup Tab

The Setup tab is divided into several sections:

#### 1. Fluid Selection & Properties
- **Fluid dropdown**: Select working fluid
- **Fluid State Parameters**: Set pressure and temperature
- **Fluid Property Distributions**: Override CoolProp values if needed

#### 2. Simulation Parameters
- **Number of Simulations**: 100 to 200,000 iterations
- **Confidence Interval**: Statistical confidence level (90-99%)
- **Output Units**: Choose Pa, kPa, bar, or psi

#### 3. Pipe Configuration
- **Unit System**: US (inches) or Metric (meters)
- **Single vs Multiple Sections**: Toggle for multi-diameter pipes
- **Pipe Geometry**: Diameter, length, and roughness inputs

#### 4. Flow Configuration
- **Mass Flow Rate**: Set flow rate and distribution
- **Elevation Change**: Positive for upward flow

#### 5. Minor Losses
- **Component Library**: Pre-defined K values for common fittings
- **Custom Components**: Add user-defined K values
- **Location Assignment**: For multi-section pipes

#### 6. Advanced Parameters
- **Gravity**: Adjust for non-Earth applications
- **Friction Model**: Choose calculation method
- **Random Seed**: For reproducible results

### Results Tab

The Results tab displays:
- **Statistical Summary**: Mean, median, standard deviation, confidence intervals
- **Pressure Profile**: For multi-section pipes
- **Distribution Plots**: Histogram and CDF
- **Sensitivity Analysis**: Parameter correlations
- **Data Tables**: Detailed simulation results
- **Export Options**: Download Excel report

---

## How to Use

### Step-by-Step Instructions

#### Single Pipe System

1. **Configure Fluid Properties**
   - Select your working fluid
   - Enter operating pressure and temperature
   - CoolProp will automatically calculate density and viscosity

2. **Set Distribution Types**
   - **Deterministic**: Fixed value (no uncertainty)
   - **Normal**: Bell curve distribution (specify mean and std dev)
   - **Uniform**: Equal probability between min and max
   - **Triangular**: Most likely value with min/max bounds

3. **Define Pipe Geometry**
   - Choose unit system (US or Metric)
   - Enter diameter, length, and roughness
   - Use material presets for roughness or enter custom value

4. **Add Minor Losses**
   - Click "Add New Component"
   - Select from library or choose "Custom"
   - Enter quantity and K value
   - Click "Add" to include in calculation

5. **Run Simulation**
   - Review all inputs
   - Click "Run Monte Carlo Simulation"
   - Wait for progress bar to complete

6. **Analyze Results**
   - Review statistical summary
   - Examine distribution plots
   - Check sensitivity analysis
   - Export to Excel if needed

#### Multi-Section Pipe System

1. **Enable Multiple Sections**
   - Check "Use Multiple Pipe Sections"
   - Add sections one by one

2. **Define Each Section**
   - Enter section name (optional)
   - Specify diameter and length
   - Set roughness value
   - Click "Add Section"

3. **Assign Minor Losses to Locations**
   - **Upstream**: Before first section
   - **Downstream**: After last section
   - **All Sections**: Applied to each section
   - **Specific Section**: By name

4. **Configure "All Sections" Behavior**
   - **Distribute**: Divide K equally among sections
   - **Replicate**: Apply full K to each section

5. **Review Transition Losses**
   - Automatic calculation for diameter changes
   - Expansion: Borda-Carnot formula
   - Contraction: Ludwig formula

---

## Understanding the Calculations

### Sequential Pressure Drop Algorithm

For multi-section pipes, the calculation proceeds sequentially through each section:

1. **Initialize** with upstream pressure P₀
2. **For each section i**:
   - Calculate fluid properties at current pressure: ρᵢ(Pᵢ), μᵢ(Pᵢ)
   - Calculate flow parameters: vᵢ, Reᵢ, fᵢ
   - Calculate friction loss: ΔPf,ᵢ = ½ρᵢfᵢ(Lᵢ/Dᵢ)vᵢ²
   - Calculate transition loss (if not first section)
   - Apply minor losses: ΔPm,ᵢ = ½ρᵢKᵢvᵢ²
   - Update pressure: Pᵢ₊₁ = Pᵢ - ΔPf,ᵢ - ΔPtrans,ᵢ - ΔPm,ᵢ
3. **Add elevation loss** distributed proportionally
4. **Check if back pressure is achieved**

### Friction Factor Models

#### 1. Standard Model (Default)
- **Laminar** (Re ≤ 2000): f = 64/Re
- **Turbulent** (Re > 2000): Swamee-Jain equation

#### 2. Churchill Model
- Universal equation valid for all Reynolds numbers
- Smooth transition between flow regimes
- More computationally intensive

#### 3. Blended Model
- Linear interpolation in transition zone (2000 < Re < 4000)
- Combines laminar and turbulent equations
- Good for systems operating near transition

### Transition Loss Calculations

#### Sudden Expansion
- **Formula**: K = (1 - A₁/A₂)² (Borda-Carnot)
- **Reference**: Upstream velocity and density
- **Valid for**: Sudden, fully turbulent expansion

#### Sudden Contraction
- **Formula**: K = 0.5(1 - A₂/A₁) (Ludwig)
- **Reference**: Downstream velocity and density
- **Valid for**: Sharp-edged, high Reynolds number

### Minor Loss Library

The K values are based on industry standards:
- Crane Technical Paper No. 410
- ASHRAE Fundamentals Handbook
- Cameron Hydraulic Data
- Idelchik's Handbook of Hydraulic Resistance

Common components include:
- **Elbows**: 90° LR (K=0.25), 90° SR (K=0.75)
- **Valves**: Gate (K=0.08), Ball (K=0.05), Globe (K=10.0)
- **Tees**: Run-through (K=0.60), Branch (K=1.50)
- **Entrances/Exits**: Sharp (K=0.50), Rounded (K=0.04)

### Property Override Modes (Multi-Section)

1. **CoolProp Only (Default)**
   - Recalculates properties at each section's local pressure
   - Most accurate for compressible fluids

2. **Constant Override**
   - Uses sampled ρ and μ values throughout
   - Useful for incompressible flow assumptions

3. **Bias vs CoolProp**
   - Applies multiplicative factors to CoolProp values
   - Accounts for systematic measurement biases

---

## Interpreting Results

### Statistical Metrics

- **Mean**: Average pressure drop across all simulations
- **Median**: Middle value (50th percentile)
- **Standard Deviation**: Measure of variability
- **Confidence Interval**: Range containing true value with specified probability

### Distribution Plots

#### Histogram
- Shows frequency distribution of results
- Check for normal distribution or skewness
- Identify outliers or multimodal behavior

#### Cumulative Distribution Function (CDF)
- Probability of pressure drop being less than a given value
- Useful for risk assessment
- Read percentiles directly from plot

### Sensitivity Analysis

The correlation chart shows which parameters most affect pressure drop:
- **Positive correlation**: Parameter increase → pressure drop increase
- **Negative correlation**: Parameter increase → pressure drop decrease
- **Near zero**: Parameter has minimal effect

Typical correlations:
- Mass flow rate: Strong positive
- Diameter: Strong negative
- Length: Moderate positive
- Roughness: Weak to moderate positive

### Pressure Profile (Multi-Section)

The pressure profile plot shows:
- Pressure evolution through the system
- Confidence bands for uncertainty
- Back pressure achievement
- Section boundaries and transitions

### Warning Messages

#### Phase Boundary Warning
- Appears when liquid is near saturation
- Indicates potential two-phase flow
- Results may be invalid in flashing regime

#### Compressible Flow Warning
- Mach number > 0.3 detected
- Large pressure drop ratio (ΔP/P > 0.2)
- Consider using compressible flow equations

#### Choking Warning
- Pressure ratio below critical value
- Flow may be choked
- Maximum flow rate reached

---

## Advanced Features

### Multi-Section Pipe Systems

Create complex piping systems with:
- Variable diameters (expansions/contractions)
- Different roughness values per section
- Automatic transition loss calculations
- Section-specific minor losses

Example applications:
- Heat exchanger piping
- Manifold systems
- Pipeline with diameter changes
- Process piping with multiple components

### Distribution Types for Uncertainty

#### When to Use Each Distribution

**Deterministic**
- Known, fixed values
- No uncertainty in parameter
- Baseline calculations

**Normal Distribution**
- Natural variability (manufacturing tolerances)
- Measurement uncertainties
- When many factors contribute to variation

**Uniform Distribution**
- Equal probability within range
- Limited knowledge about distribution
- Regulatory limits (min/max)

**Triangular Distribution**
- Most likely value known
- Minimum and maximum bounds
- Expert estimates

### Reproducible Results

Use the random seed feature to:
- Generate identical results across runs
- Share reproducible analyses
- Debug unexpected behavior
- Validate calculations

### Excel Export Features

The exported Excel file contains:
- **Summary Sheet**: Key statistics
- **Full Results**: All simulation data
- **Input Parameters**: Complete input record
- **Pipe Configuration**: Geometry details
- **Minor Losses**: Component list
- **Correlations**: Sensitivity analysis

---

## Troubleshooting

### Common Issues and Solutions

#### Application Won't Start
- Check Python version (≥ 3.8 required)
- Verify all packages are installed
- Try: `pip install --upgrade streamlit`

#### CoolProp Errors
- Check fluid name is valid
- Verify temperature/pressure in valid range
- Ensure internet connection for property lookup

#### Slow Performance
- Reduce number of simulations
- Close other applications
- Use Churchill model (faster than Blended)

#### Unexpected Results
- Verify unit consistency
- Check distribution parameters
- Review minor loss assignments
- Confirm elevation sign convention

#### Validation Errors
- Diameter and length must be positive
- Min < Max for uniform/triangular distributions
- Standard deviation > 0 for normal distribution
- At least one pipe section required

### Performance Tips

1. **Start with fewer simulations** (1000-5000) for initial runs
2. **Use deterministic values** first to verify setup
3. **Add uncertainty gradually** to understand effects
4. **Save configurations** using Excel export
5. **Use random seed** for reproducible debugging

---

## Examples

### Example 1: Simple Water Pipeline

**Scenario**: 100m horizontal water pipeline

**Inputs**:
- Fluid: Water at 20°C, 101.325 kPa
- Pipe: 2" diameter, 100m length, commercial steel
- Flow: 5 kg/s (deterministic)
- Minor losses: 2 gate valves, 4 elbows

**Expected Result**: ~50-60 kPa pressure drop

### Example 2: Compressed Air System

**Scenario**: Air distribution with pressure reduction

**Inputs**:
- Fluid: Air at 25°C, 700 kPa upstream, 600 kPa downstream
- Sections: 
  - Section 1: 3" × 50m
  - Section 2: 2" × 30m (contraction)
- Flow: 0.5 ± 0.05 kg/s (normal distribution)
- Check for compressible flow warnings

### Example 3: Cryogenic System

**Scenario**: Liquid nitrogen transfer line

**Inputs**:
- Fluid: Nitrogen at 77K, 200 kPa
- Pipe: 1" diameter, 20m length, stainless steel
- Flow: 0.1 kg/s
- Elevation: 5m rise
- Monitor for two-phase warnings

### Example 4: Heat Exchanger Piping

**Scenario**: Multi-section with expansions

**Inputs**:
- Fluid: Water at 80°C
- Sections:
  - Inlet: 2" × 5m
  - Expansion: 3" × 2m
  - Heat exchanger (minor K=4.0)
  - Reduction: 2" × 5m
- Use "replicate" mode for minor losses

---

## Appendix: Calculation Details

### Pressure Drop Equations

**Darcy-Weisbach Equation**:
```
ΔP_friction = (f × L/D × ρ × v²)/2
```

**Minor Loss Equation**:
```
ΔP_minor = (K × ρ × v²)/2
```

**Elevation Change**:
```
ΔP_elevation = ρ × g × Δz
```

### Reynolds Number
```
Re = (ρ × v × D)/μ
```

### Friction Factor Correlations

**Swamee-Jain (Turbulent)**:
```
f = 0.25 / [log₁₀(ε/(3.7D) + 5.74/Re^0.9)]²
```

**Churchill (Universal)**:
```
f = 8[(8/Re)¹² + 1/(A+B)^1.5]^(1/12)
where:
A = [2.457 ln(1/((7/Re)^0.9 + 0.27ε/D))]¹⁶
B = (37530/Re)¹⁶
```

### Unit Conversions

**Pressure**:
- 1 kPa = 1000 Pa
- 1 bar = 100,000 Pa  
- 1 psi = 6,894.757 Pa

**Viscosity**:
- 1 mPa·s = 0.001 Pa·s
- 1 cP = 1 mPa·s

**Length**:
- 1 inch = 0.0254 m
- 1 ft = 0.3048 m

---

## Support and Feedback

For questions, bug reports, or feature requests:
1. Check this user guide first
2. Review the troubleshooting section
3. Verify your inputs and calculations
4. Document the issue with screenshots/data
5. Contact your system administrator or developer

---

*Last Updated: September 2025*
*Version: 2.7*
