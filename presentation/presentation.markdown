# CPMG Fitting Tool Presentation: Python Implementation and Code Analysis

_Presented by Manas Mishra, Internship at IIT Guwahati, July 9, 2025_

## Introduction

This presentation elucidates the development and functionality of a Python-based CPMG Fitting Tool designed to analyze Nuclear Magnetic Resonance (NMR) data from Carr-Purcell-Meiboom-Gill (CPMG) experiments. The tool automates the fitting of three mathematical models to NMR data, providing researchers with an intuitive interface to upload data, visualize results, and extract parameters without requiring programming expertise. This document explains the tool’s purpose, the science behind it, and its Python implementation, with code blocks broken down for clarity, tailored for scientists unfamiliar with coding.

---

## 1. NMR and the Role of Python in CPMG Analysis

**Nuclear Magnetic Resonance (NMR)** is a powerful spectroscopic technique used to investigate the structural and dynamic properties of proteins by observing nuclear spin behavior in a magnetic field. The **Carr-Purcell-Meiboom-Gill (CPMG)** pulse sequence measures the transverse relaxation rate (R₂,eff) of nuclear spins, revealing protein conformational dynamics critical for understanding biological processes and drug design.

**Challenges in CPMG Analysis**: CPMG data analysis involves fitting complex mathematical models to experimental data, a process that is computationally intensive and prone to errors when performed manually. The CPMG Fitting Tool addresses this by automating model fitting, data visualization, and result exportation.

**Why Python?**: Python is a versatile, high-level programming language ideal for scientific computing due to its robust libraries for numerical analysis, data handling, and visualization. Its clear syntax facilitates the development of user-friendly applications, making it an excellent choice for automating NMR data analysis.

**Tool Overview**: The CPMG Fitting Tool is a web-based application that allows users to upload an Excel file containing CPMG data, input the magnetic field strength (B0), and obtain fitted model parameters, visualizations, and downloadable results. It leverages Python’s ecosystem to streamline the analysis process, enhancing efficiency and accessibility for researchers.

---

## 2. Python Fundamentals for Scientists

Python is a programming language that executes user-defined instructions to perform tasks such as data analysis and visualization. Below are key concepts explained for non-coders:

- **Variables**: Containers for storing data, such as numbers or text. For example, `rate = 10.5` assigns the value 10.5 to a variable named `rate`.
- **Functions**: Reusable blocks of code that perform specific tasks, e.g., a function to calculate R₂,eff from input parameters.
- **Lists**: Ordered collections of items, such as `[100, 200, 300]` for CPMG frequencies.
- **Libraries**: Pre-built modules that extend Python’s capabilities, such as tools for data processing or plotting.
- **Loops**: Instructions to repeat tasks, e.g., analyzing data for each amino acid in a dataset.
- **Conditionals**: Decision-making structures, e.g., checking if a dataset meets required formats.

These concepts form the foundation of the CPMG Fitting Tool, enabling it to process NMR data efficiently.
Example:

```python
box = 5  # Store 5 in a box
print(box)  # Show 5
numbers = [1, 2, 3]  # A list
```

---

## 3. Python Libraries Utilized

The tool integrates several Python libraries, each serving a specific function in the analysis pipeline:

- **Streamlit**: Creates an interactive web interface for data upload and result visualization.
- **Pandas**: Manages tabular data, enabling efficient reading and manipulation of Excel files.
- **NumPy**: Performs high-performance numerical computations, essential for model calculations.
- **Matplotlib**: Generates publication-quality plots of data and fitted models.
- **SciPy**: Implements curve-fitting algorithms to optimize model parameters.
- **XlsxWriter**: Exports results to Excel files for easy sharing.
- **io and base64**: Facilitate the handling and encoding of files for download.

These libraries collectively enable the tool to process data, perform calculations, and present results in an accessible format.

---

## 4. Functionality of the CPMG Fitting Tool

**Purpose**: The tool analyzes CPMG data to quantify protein conformational dynamics by fitting three mathematical models: No-Exchange, Luz–Meiboom, and Carver–Richards. It extracts parameters that describe protein motion, aiding in biological and pharmaceutical research.

**Workflow**:

1. **Input**: Users upload an Excel file (.xlsx) with columns: `AminoAcid` (e.g., ALA, GLY), `Frequency` (CPMG frequency in Hz), `R2eff` (effective relaxation rate in s⁻¹), and `R2eff_error` (uncertainty in R2eff).
2. **Parameter Input**: Users specify the magnetic field strength (B0, in MHz).
3. **Output**: The tool generates:
   - Plots comparing experimental data to model fits.
   - Tables of fitted parameters with uncertainties.
   - A summary of phi (φ) values from the Luz–Meiboom model.
   - Downloadable results in PNG and Excel formats.

**Advantages**: The tool eliminates the need for manual coding, offering a user-friendly interface that accelerates analysis and enhances reproducibility.

---

## 5. Scientific Basis: CPMG Models and Parameters

The tool fits three models to describe protein dynamics, each suitable for different exchange regimes. Below, we outline each model and its parameters.

### No-Exchange Model

- **Description**: Assumes no conformational exchange, modeling R₂,eff as a constant independent of CPMG frequency (ν_CPMG).
- **Parameters**:
  - **R₂**: Intrinsic transverse relaxation rate (s⁻¹).
  - **ν_CPMG**: CPMG pulse frequency (Hz).
- **Equation**:
  \[
  R*{2,\mathrm{eff}}(\nu*\mathrm{CPMG}) = R_2
  \]

### Luz–Meiboom Model

- **Description**: Models fast conformational exchange between two protein states, capturing dynamic effects on relaxation.
- **Parameters**:
  - **R₂**: Intrinsic relaxation rate (s⁻¹).
  - **kₑₓ**: Exchange rate between states (s⁻¹).
  - **φ**: Parameter reflecting the chemical shift difference and population of states, scaled by B0².
  - **B0**: Magnetic field strength (MHz).
  - **ν_CPMG**: CPMG frequency (Hz).
- **Equation**:
  \[
  R*{2,\mathrm{eff}}(\nu*\mathrm{CPMG}) = R*2 + \frac{\Phi}{k*\mathrm{ex}} \left(1 - \frac{4\nu*\mathrm{CPMG}}{k*\mathrm{ex}} \tanh\left(\frac{k*\mathrm{ex}}{4\nu*\mathrm{CPMG}}\right)\right)
  \]
  where \(\Phi = 4\pi^2 B_0^2 \phi\).

### Carver–Richards Model

- **Description**: Models general two-site exchange, accommodating both fast and slow regimes with complex dynamics.
- **Parameters**:
  - **R₂**: Intrinsic relaxation rate (s⁻¹).
  - **k_AB, k_BA**: Forward and backward exchange rates (s⁻¹).
  - **Δδ**: Chemical shift difference between states (ppm).
  - **kₑₓ**: Total exchange rate (k_AB + k_BA, s⁻¹).
  - **B0**: Magnetic field strength (MHz).
  - **ν_CPMG**: CPMG frequency (Hz).
  - **ψ, ζ, η, ξ, D_±, λ**: Intermediate terms for computing dynamics.
- **Equations**:
  \[
  R*{2,\mathrm{eff}}^{\mathrm{rel}}(\nu*\mathrm{CPMG}) = R*2 + \frac{k*{AB} + k*{BA}}{2} - 2\nu*\mathrm{CPMG} \cdot \ln(\lambda)
  \]
  \[
  \lambda = \sqrt{D*+ \cosh^2(\xi) - D*- \cos^2(\eta)} + \sqrt{D*+ \sinh^2(\xi) + D*- \sin^2(\eta)}
  \]
  \[
  D*\pm = \frac{1}{2} \left(1 \pm \frac{\psi + 2(2\pi \Delta\delta B_0)^2}{\sqrt{\psi^2 + \zeta^2}}\right)
  \]
  \[
  \eta = \frac{1}{2\nu*\mathrm{CPMG} \sqrt{8}} \sqrt{-\psi + \sqrt{\psi^2 + \zeta^2}}, \quad
  \xi = \frac{1}{2\nu*\mathrm{CPMG} \sqrt{8}} \sqrt{\psi + \sqrt{\psi^2 + \zeta^2}}
  \]
  \[
  \psi = (k*{AB} - k*{BA})^2 - (2\pi \Delta\delta B_0)^2 + 4k*{AB}k*{BA}, \quad
  \zeta = 4\pi \Delta\delta B_0 (k*{AB} - k\_{BA})
  \]

**Rationale**: These models cover a range of protein dynamics, from static (No-Exchange) to fast (Luz–Meiboom) and general (Carver–Richards) exchange, enabling comprehensive analysis of CPMG data.

---

## 6. Code Analysis: Application Initialization

**Code Block: Importing Libraries and Configuring Settings**

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
import base64
import xlsxwriter
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 12, 'legend.fontsize': 12})
st.set_page_config(layout="centered")
```

- **Functionality**: Initializes the application by importing required libraries and configuring visualization settings.
- **Details**:
  - Imports libraries for web interface (Streamlit), data handling (Pandas), numerical operations (NumPy), plotting (Matplotlib), curve fitting (SciPy), and file export (XlsxWriter, io, base64).
  - Sets plot fonts to Times New Roman (12-point) for professional presentation.
  - Configures the Streamlit app to use a centered layout.
- **Purpose**: Establishes the foundation for data processing and visualization, ensuring consistent and professional output.

---

## 7. Code Analysis: User Interface Setup

**Code Block: Sidebar and Input Controls**

```python
with st.sidebar:
    st.markdown("""
    <h3 style='font-family: Times New Roman; font-size: 18px;'>📄 Input File Specifications</h3>
    <p style='font-family: Times New Roman; font-size: 14px;'>
        Upload an Excel file (.xlsx) with the following columns:
        <ul>
            <li><b>AminoAcid</b>: Amino acid identifier (e.g., ALA, GLY).</li>
            <li><b>Frequency</b>: CPMG frequency (Hz, numeric).</li>
            <li><b>R2eff</b>: Effective relaxation rate (s⁻¹, numeric).</li>
            <li><b>R2eff_error</b>: Uncertainty in R2eff (s⁻¹, numeric).</li>
        </ul>
        The file must include a header row and no missing values.
    </p>
    """, unsafe_allow_html=True)
    st.number_input("Enter B0 value (MHz):", min_value=0.0, value=81.0, step=0.1, key="B0_input")
uploaded_file = st.file_uploader("📁 Upload Excel file (.xlsx)", type="xlsx")
```

- **Functionality**: Creates a sidebar for user guidance and input collection.
- **Details**:
  - Displays formatted instructions for the Excel file format using HTML for styling.
  - Provides a numeric input field for B0 (default: 81 MHz).
  - Offers a file uploader for Excel files containing CPMG data.
- **Purpose**: Ensures users provide correctly formatted data and parameters, enhancing usability.

---

## 8. Code Analysis: Displaying Application Content

**Code Block: Titles and Model Equations**

```python
st.markdown("""
    <h2 style="font-family: 'Times New Roman'; font-size: 20px;">
        CPMG Fitting Tool
    </h2>
""", unsafe_allow_html=True)
st.markdown("""
    <h2 style="font-family: 'Times New Roman'; font-size: 20px;">
        No-Exchange Model
    </h2>
""", unsafe_allow_html=True)
st.latex(r"""
R_{2,\mathrm{eff}}(\nu_\mathrm{CPMG}) = R_2
""")
st.markdown("""
    <h2 style="font-family: 'Times New Roman'; font-size: 20px;">
        Luz–Meiboom Model
    </h2>
""", unsafe_allow_html=True)
st.latex(r"""
R_{2,\mathrm{eff}}(\nu_\mathrm{CPMG}) = R_2 + \frac{\Phi}{k_\mathrm{ex}} \left(1 - \frac{4\nu_\mathrm{CPMG}}{k_\mathrm{ex}} \tanh\left(\frac{k_\mathrm{ex}}{4\nu_\mathrm{CPMG}}\right)\right)
""")
st.markdown("""
    <h2 style="font-family: 'Times New Roman'; font-size: 20px;">
        Carver–Richards Model
    </h2>
""", unsafe_allow_html=True)
st.latex(r"""
R_{2,\mathrm{eff}}^{\mathrm{rel}}(\nu_\mathrm{CPMG}) = R_2 + \frac{k_{AB} + k_{BA}}{2} - 2\nu_\mathrm{CPMG} \cdot \ln(\lambda)
""")
st.latex(r"""
\lambda = \sqrt{D_+ \cosh^2(\xi) - D_- \cos^2(\eta)} + \sqrt{D_+ \sinh^2(\xi) + D_- \sin^2(\eta)}
""")
st.latex(r"""
D_\pm = \frac{1}{2} \left(1 \pm \frac{\psi + 2(2\pi \Delta\delta B_0)^2}{\sqrt{\psi^2 + \zeta^2}}\right)
""")
st.latex(r"""
\eta = \frac{1}{2\nu_\mathrm{CPMG} \sqrt{8}} \sqrt{-\psi + \sqrt{\psi^2 + \zeta^2}}, \quad
\xi = \frac{1}{2\nu_\mathrm{CPMG} \sqrt{8}} \sqrt{\psi + \sqrt{\psi^2 + \zeta^2}}
""")
st.latex(r"""
\psi = (k_{AB} - k_{BA})^2 - (2\pi \Delta\delta B_0)^2 + 4k_{AB}k_{BA}, \quad
\zeta = 4\pi \Delta\delta B_0 (k_{AB} - k_{BA})
""")
st.markdown(
    "<i>The Carver–Richards model uses a four-parameter expression (R₂, k_AB, k_BA, Δδ). The No-Exchange model assumes a constant R₂.</i>", unsafe_allow_html=True)
st.markdown("""
    <h3 style='font-family: Times New Roman; font-size: 16px;'>References</h3>
    <p style='font-family: Times New Roman; font-size: 14px;'>
        1. Luz, Z.; Meiboom, S. (1963). Nuclear Magnetic Resonance study of the protolysis of trimethylammonium ion in aqueous solution—order of the reaction with respect to solvent. <i>J. Chem. Phys.</i>, 39, 366–370.<br>
        2. Carver, J. P.; Richards, R. E. (1972). General two-site solution for chemical exchange produced dependence of T2 upon Carr-Purcell pulse separation. <i>J. Magn. Reson.</i>, 6, 89–96.
    </p>
""", unsafe_allow_html=True)
```

- **Functionality**: Displays the application title, model equations, and references.
- **Details**:
  - Renders the tool’s title and model headings in Times New Roman.
  - Uses LaTeX to display mathematical equations for each model.
  - Provides references to foundational papers.
- **Purpose**: Introduces the scientific framework, ensuring users understand the models being applied.

---

## 9. Code Analysis: Model Definitions

**Code Block: Model Functions**

```python
B0 = st.session_state.B0_input
def no_exchange(v_cpmg, R2):
    return np.full_like(v_cpmg, R2)
def luz_meiboom(v_cpmg, R2, k_ex, phi, B0=st.session_state.B0_input):
    Phi = 4 * np.pi**2 * B0**2 * phi
    return R2 + (Phi / k_ex) * (1 - (4 * v_cpmg / k_ex) * np.tanh(k_ex / (4 * v_cpmg)))
def carver_richards(v_cpmg, R2, k_AB, k_BA, delta_ppm, B0=st.session_state.B0_input):
    delta = 2 * np.pi * delta_ppm * B0
    kex = k_AB + k_BA
    psi = (k_AB - k_BA)**2 - delta**2 + 4 * k_AB * k_BA
    zeta = 4 * np.pi * delta * (k_AB - k_BA)
    sqrt_term = np.sqrt(psi**2 + zeta**2)
    eta = np.sqrt(-psi + sqrt_term) / (2 * v_cpmg * np.sqrt(8))
    xi = np.sqrt(psi + sqrt_term) / (2 * v_cpmg * np.sqrt(8))
    D_plus = 0.5 * (1 + (psi + 2 * delta**2) / sqrt_term)
    D_minus = 0.5 * (-1 + (psi + 2 * delta**2) / sqrt_term)
    lambda_val = np.sqrt(D_plus * np.cosh(xi)**2 - D_minus * np.cos(eta)**2) + \
        np.sqrt(D_plus * np.sinh(xi)**2 + D_minus * np.sin(eta)**2)
    return R2 + kex / 2 - 2 * v_cpmg * np.log(lambda_val)
```

- **Functionality**: Defines mathematical functions for the three CPMG models.
- **Details**:
  - Stores the user-provided B0 value.
  - `no_exchange`: Returns a constant R₂ array for all ν_CPMG values.
  - `luz_meiboom`: Computes R₂,eff for fast exchange using R₂, kₑₓ, φ, and B0.
  - `carver_richards`: Calculates R₂,eff for general exchange with intermediate terms (ψ, ζ, etc.).
- **Purpose**: Implements the mathematical models for curve fitting, enabling accurate parameter estimation.

---

## 10. Code Analysis: Data Loading and Validation

**Code Block: Data Processing**

```python
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    required_cols = {"AminoAcid", "Frequency", "R2eff", "R2eff_error"}
    if not required_cols.issubset(df.columns):
        st.error("❌ Excel file must contain columns: AminoAcid, Frequency, R2eff, R2eff_error")
        st.stop()
    amino_acids = df['AminoAcid'].unique()
    st.markdown(f"<div style='font-family: Times New Roman; font-size: 14px;'>✅ Loaded data for <b>{len(amino_acids)}</b> amino acid(s).</div>", unsafe_allow_html=True)
    fit_summary = []
    phi_summary = []
    all_plots = []
    all_tables = []
```

- **Functionality**: Loads and validates the uploaded Excel file.
- **Details**:
  - Reads the Excel file into a Pandas DataFrame.
  - Verifies the presence of required columns.
  - Extracts unique amino acids and displays a confirmation message.
  - Initializes lists to store fitting results, phi values, plots, and tables.
- **Purpose**: Ensures data integrity before analysis, preventing errors in downstream processing.

---

## 11. Code Analysis: Model Fitting and Visualization

**Code Block: Processing Amino Acids and Plotting**

```python
    for aa in amino_acids:
        sub_df = df[df['AminoAcid'] == aa]
        x, y, yerr = sub_df['Frequency'].values, sub_df['R2eff'].values, sub_df['R2eff_error'].values
        st.markdown(f"<h4 style='font-family: Times New Roman; font-size: 18px;'>📈 Amino Acid: {aa}</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=yerr, fmt='o', color='black', capsize=4, label="Data")
```

- **Functionality**: Processes data for each amino acid and initializes a plot.
- **Details**:
  - Filters data for a specific amino acid.
  - Extracts frequency, R₂,eff, and error values.
  - Displays a heading for the amino acid.
  - Creates a plot with data points and error bars.
- **Purpose**: Prepares data for model fitting and visualization.

**Code Block: No-Exchange Model Fitting**

```python
        chi_noex = chi_luz = chi_cr = np.inf
        popt_noex = perr_noex = [np.nan]
        popt_luz = perr_luz = [np.nan] * 3
        popt_cr = perr_cr = [np.nan] * 4
        try:
            popt_noex, pcov_noex = curve_fit(no_exchange, x, y, sigma=yerr, absolute_sigma=True, p0=[10.0], bounds=([0], [np.inf]))
            perr_noex = np.sqrt(np.diag(pcov_noex))
            yfit_noex = no_exchange(x, *popt_noex)
            chi_noex = np.sum(((y - yfit_noex) / yerr)**2)
            ax.plot(np.linspace(min(x), max(x), 300), no_exchange(np.linspace(min(x), max(x), 300), *popt_noex), 'g-', label="No-Exchange Fit")
        except Exception as e:
            st.error(f"❌ No-Exchange fit failed: {e}")
```

- **Functionality**: Fits the No-Exchange model and plots the result.
- **Details**:
  - Initializes fit parameters and chi-squared values.
  - Uses SciPy’s `curve_fit` to optimize R₂, with bounds to ensure non-negative values.
  - Calculates fit errors and chi-squared for fit quality.
  - Plots the fitted curve in green.
- **Purpose**: Evaluates the simplest model for baseline comparison.

**Code Block: Luz–Meiboom and Carver–Richards Fitting**

```python
        try:
            popt_luz, pcov_luz = curve_fit(luz_meiboom, x, y, sigma=yerr, absolute_sigma=True, p0=[10.0, 5000.0, 0.001], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            perr_luz = np.sqrt(np.diag(pcov_luz))
            yfit_luz = luz_meiboom(x, *popt_luz)
            chi_luz = np.sum(((y - yfit_luz) / yerr)**2)
            ax.plot(np.linspace(min(x), max(x), 300), luz_meiboom(np.linspace(min(x), max(x), 300), *popt_luz), 'r-', label="Luz–Meiboom Fit")
            phi_summary.append({"Residue": aa, "phi": popt_luz[2], "phi_err": perr_luz[2]})
        except Exception as e:
            st.error(f"❌ Luz–Meiboom fit failed: {e}")
        try:
            popt_cr, pcov_cr = curve_fit(carver_richards, x, y, sigma=yerr, p0=[20.0, 500.0, 500.0, 1.0], absolute_sigma=True, maxfev=10000)
            perr_cr = np.sqrt(np.diag(pcov_cr))
            yfit_cr = carver_richards(x, *popt_cr)
            chi_cr = np.sum(((y - yfit_cr) / yerr)**2)
            ax.plot(np.linspace(min(x), max(x), 300), carver_richards(np.linspace(min(x), max(x), 300), *popt_cr), 'b--', label="Carver–Richards Fit")
        except Exception as e:
            st.error(f"❌ Carver–Richards fit failed: {e}")
```

- **Functionality**: Fits the Luz–Meiboom and Carver–Richards models and plots results.
- **Details**:
  - Fits Luz–Meiboom with initial guesses for R₂, kₑₓ, and φ, plotting in red.
  - Stores phi values for summary analysis.
  - Fits Carver–Richards with guesses for four parameters, allowing extended iterations, plotting in blue (dashed).
  - Handles fitting errors gracefully.
- **Purpose**: Tests complex models to capture dynamic behavior.

---

## 12. Code Analysis: Visualizing and Tabulating Results

**Code Block: Finalizing Plots**

```python
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)")
        ax.set_title(f"Model Fits for {aa}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        all_plots.append((f"{aa}_fit_plot.png", buf.getvalue()))
        plt.close(fig)
```

- **Functionality**: Completes and displays plots for each amino acid.
- **Details**:
  - Labels axes and adds a title and legend.
  - Displays the plot in the app and saves it as a PNG.
  - Closes the plot to free memory.
- **Purpose**: Provides visual comparison of data and model fits.

**Code Block: Parameter Tables**

```python
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🧾 No-Exchange Parameters (± 1σ):**")
            try:
                noex_table = pd.DataFrame({"Parameter": ["R₂"], "Value": [f"{val:.6g}" for val in popt_noex], "±1σ": [f"{err:.6g}" for err in perr_noex]})
                st.table(noex_table)
                all_tables.append((f"{aa}_no_exchange_params.txt", noex_table.to_csv(index=False)))
            except:
                st.markdown("_Fit failed._")
        with col2:
            st.markdown("**🧾 Luz–Meiboom Parameters (± 1σ):**")
            try:
                luz_table = pd.DataFrame({"Parameter": ["R₂", "kₑₓ", "φ"], "Value": [f"{val:.6g}" for val in popt_luz], "±1σ": [f"{err:.6g}" for err in perr_luz]})
                st.table(luz_table)
                all_tables.append((f"{aa}_luz_meiboom_params.txt", luz_table.to_csv(index=False)))
            except:
                st.markdown("_Fit failed._")
        with col3:
            st.markdown("**🧾 Carver–Richards Parameters (± 1σ):**")
            try:
                keff = popt_cr[1] + popt_cr[2]
                keff_err = np.sqrt(perr_cr[1]**2 + perr_cr[2]**2)
                cr_table = pd.DataFrame({"Parameter": ["R₂", "k_AB", "k_BA", "Δδ", "kₑₓ"], "Value": [f"{popt_cr[0]:.6g}", f"{popt_cr[1]:.6g}", f"{popt_cr[2]:.6g}", f"{popt_cr[3]:.6g}", f"{keff:.6g}"], "±1σ": [f"{perr_cr[0]:.6g}", f"{perr_cr[1]:.6g}", f"{perr_cr[2]:.6g}", f"{perr_cr[3]:.6g}", f"{keff_err:.6g}"]})
                st.table(cr_table)
                all_tables.append((f"{aa}_carver_richards_params.txt", cr_table.to_csv(index=False)))
            except:
                st.markdown("_Fit failed._")
```

- **Functionality**: Displays parameter tables for each model.
- **Details**:
  - Organizes output into three columns.
  - Creates tables for No-Exchange (R₂), Luz–Meiboom (R₂, kₑₓ, φ), and Carver–Richards (R₂, k_AB, k_BA, Δδ, kₑₓ) parameters.
  - Saves tables as CSV files.
- **Purpose**: Presents quantitative results for scientific interpretation.
- **Image**: ![Sample Fit Curve](images/sample_fit_curve.jpg) _(Shows red Luz–Meiboom and blue Carver–Richards lines.)_

---

## 13. Code Analysis: Summarizing and Exporting Results

**Code Block: Model Comparison**

```python
        min_chi = min(chi_noex, chi_luz, chi_cr)
        if min_chi == chi_noex:
            better_model = "No-Exchange"
        elif min_chi == chi_luz:
            better_model = "Luz–Meiboom"
        else:
            better_model = "Carver–Richards"
        fit_summary.append({"Residue": aa, "Chi² (NoEx)": chi_noex, "Chi² (Luz)": chi_luz, "Chi² (CR)": chi_cr, "Better Fit": better_model})
    if fit_summary:
        st.markdown("<h4 style='font-family: Times New Roman; font-size: 18px;'>📊 Model Comparison Summary</h4>", unsafe_allow_html=True)
        summary_table = pd.DataFrame(fit_summary).sort_values(by="Residue").reset_index(drop=True)
        st.dataframe(summary_table.style.highlight_min(subset=["Chi² (NoEx)", "Chi² (Luz)", "Chi² (CR)"], axis=1, color="lightgreen"))
        all_tables.append(("model_comparison_summary.txt", summary_table.to_csv(index=False)))
```

- **Functionality**: Summarizes model fit quality.
- **Details**:
  - Identifies the model with the lowest chi-squared value.
  - Creates a table comparing chi-squared values across models, highlighting the best fit.
  - Saves the summary as a CSV.
- **Purpose**: Facilitates model selection based on fit quality.

**Code Block: Phi Values Visualization**

```python
    if phi_summary:
        summary_df = pd.DataFrame(phi_summary).sort_values(by='Residue').reset_index(drop=True)
        fig_phi, ax_phi = plt.subplots()
        ax_phi.errorbar(summary_df.index + 1, summary_df['phi'], yerr=summary_df['phi_err'], fmt='o', color='black', capsize=4, linewidth=2)
        ax_phi.plot(summary_df.index + 1, summary_df['phi'], color='red', linewidth=1.5)
        ax_phi.set_xlabel("# residue")
        ax_phi.set_ylabel("Phi value")
        ax_phi.set_title("Phi values (Luz–Meiboom)")
        ax_phi.grid(True)
        st.markdown("<h4 style='font-family: Times New Roman; font-size: 18px;'>📊 Summary of φ Values (Luz–Meiboom)</h4>", unsafe_allow_html=True)
        st.pyplot(fig_phi)
        buf_phi = io.BytesIO()
        fig_phi.savefig(buf_phi, format="png")
        buf_phi.seek(0)
        all_plots.append(("phi_values_plot.png", buf_phi.getvalue()))
        plt.close(fig_phi)
        all_tables.append(("phi_summary.txt", summary_df.to_csv(index=False)))
```

- **Functionality**: Plots phi values from the Luz–Meiboom model.
- **Details**:
  - Creates a table of phi values sorted by residue.
  - Generates a plot with error bars and a connecting line.
  - Saves the plot and table for export.
- **Purpose**: Visualizes trends in conformational exchange parameters.

**Code Block: Result Export**

```python
    if all_plots or all_tables:
        st.markdown("<h4 style='font-family: Times New Roman; font-size: 18px;'>⬇️ Download Results</h4>", unsafe_allow_html=True)
        for plot_name, plot_data in all_plots:
            st.download_button(label=f"Download {plot_name}", data=plot_data, file_name=plot_name, mime="image/png")
        for table_name, table_data in all_tables:
            st.download_button(label=f"Download {table_name}", data=table_data, file_name=table_name, mime="text/csv")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for table_name, table_data in all_tables:
                df_temp = pd.read_csv(io.StringIO(table_data))
                df_temp.to_excel(writer, sheet_name=table_name.replace('.txt', ''), index=False)
        output.seek(0)
        st.download_button(label="Download All Tables as Excel", data=output, file_name="all_tables.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
```

- **Functionality**: Provides download options for results.
- **Details**:
  - Offers buttons to download individual plots (PNG) and tables (CSV).
  - Combines all tables into a single Excel file with multiple sheets.
- **Purpose**: Enables users to save results for further analysis or publication.

---

## 14. Demonstration: Tool Operation

**Procedure**:

1. **Launch**: Run the app locally with `streamlit run app.py` or access a hosted version.
2. **Upload**: Load an Excel file with columns: `AminoAcid`, `Frequency`, `R2eff`, `R2eff_error`.
3. **Input B0**: Enter the magnetic field strength (e.g., 81 MHz).
4. **View Results**: Observe plots, parameter tables, and phi value summaries.
5. **Export**: Download results as PNG or Excel files.

**Sample Data** (`sample_data.xlsx`):

```
AminoAcid,Frequency,R2eff,R2eff_error
ALA,100,20.5,0.2
ALA,200,20.3,0.3
GLY,100,22.1,0.25
GLY,200,21.9,0.35
```

**Setup**:

- Install dependencies: `pip install streamlit pandas numpy matplotlib scipy xlsxwriter`.

**Key Feature**: The tool requires no coding, making it accessible to all researchers.

---

## 15. Benefits and Conclusion

**Benefits**:

- **Accessibility**: User-friendly interface eliminates coding barriers.
- **Efficiency**: Automates complex calculations, reducing analysis time.
- **Clarity**: Provides clear visualizations and tabular outputs for scientific reporting.
- **Impact**: Enhances understanding of protein dynamics, supporting advancements in biology and drug development.

**Python’s Contribution**: Python’s robust libraries and clear syntax enable the development of a powerful, automated tool for NMR data analysis.

**Conclusion**:
Developed during my internship at IIT Guwahati, this CPMG Fitting Tool demonstrates the synergy of Python programming and NMR spectroscopy. I express gratitude to my mentor Dr. Himanshu Singh and all the lab colleagues for their support. 

---
