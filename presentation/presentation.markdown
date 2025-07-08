# CPMG Fitting Tool Presentation: Python and Code Explained in Blocks

_Presented by [Your Name], Internship at IIT Guwahati, July 9, 2025_

Welcome! I’m excited to share a tool I built to simplify NMR data analysis using Python. This presentation is for everyone, even if you’ve never coded. I’ll explain Python basics, the science behind the tool, and how the code works in easy-to-follow blocks, like steps in a recipe. We’ll see how it helps study proteins and makes your research faster.

---

## 1. What is NMR and Why Use Python?

- **NMR (Nuclear Magnetic Resonance)**: A technique to study proteins by observing how their atoms behave in a magnetic field, like a camera capturing their movements.
- **CPMG Experiments**: Carr-Purcell-Meiboom-Gill (CPMG) measures how fast protein atoms “relax” (return to normal) after a magnetic pulse. This shows how proteins switch shapes, key for biology and drug design.
- **The Challenge**: CPMG data involves complex math, like fitting curves to numbers, which is slow and error-prone by hand.
- **CPMG Fitting Tool**: My tool automates this, letting you upload data, see graphs, and get results without coding.
- **Python**: A simple programming language, like writing clear instructions for a computer. It’s great for math, data, and building apps.
- **Why Python?**: It’s like a smart lab assistant that handles calculations, organizes data, and creates visuals, all in one.

---

## 2. Python Basics for Non-Coders

Python is like a recipe book for computers. You write instructions (code), and the computer follows them. Here’s what you need to know:

- **Variables**: Boxes to store numbers or words. Example: `box = 10` stores 10 in a box named `box`.
- **Functions**: Mini-recipes that do a job, like `add(2, 3)` giving 5.
- **Lists**: A way to store multiple items, like a shopping list: `[1, 2, 3]`.
- **Libraries**: Pre-made toolkits, like borrowing a calculator or graph maker.
- **Loops**: Repeat a task, like “for each item, do this.”
- **Conditionals**: Make decisions, like “if the data is ready, analyze it.”

Example:

```python
box = 5  # Store 5 in a box
print(box)  # Show 5
numbers = [1, 2, 3]  # A list
```

These are the building blocks of our tool’s code.

---

## 3. Libraries: The Toolkits We Use

Our tool uses Python **libraries**, like appliances in a kitchen. Each does a specific job:

- **Streamlit**: Builds a web app where you upload files and see results, like a user-friendly website.
- **Pandas**: Handles data tables, like Excel, to read and organize NMR data.
- **NumPy**: A super calculator for math with numbers and lists, used in our models.
- **Matplotlib**: Draws graphs, like plotting your data and model fits.
- **SciPy**: Fits math models to data, like finding the best curve.
- **XlsxWriter**: Saves results as Excel files for reports.
- **io and base64**: Help save graphs and files, like packaging data for download.

These libraries make our tool powerful and easy to use.

---

## 4. The CPMG Fitting Tool: What It Does

- **Purpose**: Analyzes CPMG data to understand protein movements by fitting three math models (No-Exchange, Luz–Meiboom, Carver–Richards).
- **How It Works**:
  - Upload an Excel file with columns: AminoAcid (e.g., ALA), Frequency (Hz), R2eff (relaxation rate), R2eff_error (uncertainty).
  - Enter the magnetic field strength (B0, e.g., 81 MHz).
  - See graphs comparing models, tables with parameters, and a phi value graph.
  - Download results for research papers.
- **Why It’s Great**: No coding needed—just upload and click!
- **Python’s Role**: Does the math, organizes data, and shows visuals, like an automated lab assistant.

---

## 5. The Science: Three Models and Their Terms

Our tool fits three models to describe how proteins move. Each model uses specific terms, explained below, and matches your data to a curve.

### No-Exchange Model

- **What It Does**: Assumes proteins don’t switch shapes, so the relaxation rate (R₂,eff) is constant, like a steady heartbeat.
- **Terms**:
  - **R₂,eff**: The effective relaxation rate (s⁻¹), how fast atoms return to normal.
  - **R₂**: The constant relaxation rate in this model.
  - **ν_CPMG**: The CPMG frequency (Hz), how often magnetic pulses are applied.
- **Equation**: R₂,eff(ν_CPMG) = R₂ (a flat line).

### Luz–Meiboom Model

- **What It Does**: Models fast switching between two protein shapes, like a dancer flipping between poses.
- **Terms**:
  - **R₂**: Base relaxation rate (s⁻¹).
  - **kₑₓ**: Exchange rate (s⁻¹), how fast the protein switches shapes.
  - **φ (phi)**: A parameter related to the magnetic field’s effect on the protein.
  - **B0**: Magnetic field strength (MHz), entered by the user.
  - **ν_CPMG**: CPMG frequency (Hz).
  - **tanh**: A math function (hyperbolic tangent) to model the switching effect.
    ![Luz–Meiboom Equation](images/luz_meiboom_model.jpg)

### Carver–Richards Model

- **What It Does**: Models complex switching with multiple rates, like a dancer with many moves.
- **Terms**:

  - **R₂**: Base relaxation rate (s⁻¹).
  - **k_AB, k_BA**: Forward and backward exchange rates (s⁻¹) between two protein states.
  - **Δδ**: Chemical shift difference (ppm), how much the magnetic environment differs.
  - **kₑₓ**: Total exchange rate (k_AB + k_BA).
  - **B0**: Magnetic field strength (MHz).
  - **ν_CPMG**: CPMG frequency (Hz).
  - **ψ, ζ, η, ξ, D_±, λ**: Intermediate values to compute the complex dynamics.

  ![Carver–Richards Equation](images/carver_richards_model.jpg)

- **Why These Models?**: They describe different protein behaviors, and Python fits them to find the best match for your data.

---

## 6. Code Walkthrough: Setting Up the App

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
rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 12, 'legend.fontsize': 12})
st.set_page_config(layout="centered")
```

- **What It Does**: Loads toolkits (libraries) and sets up the app’s appearance.
- **Details**:
  - Imports Streamlit for the web app, Pandas for data tables, NumPy for math, Matplotlib for graphs, SciPy’s `curve_fit` for model fitting, and `io`, `base64`, `xlsxwriter` for saving files.
  - Sets graph fonts to Times New Roman, size 12, for professional plots.
  - Centers the app’s layout in the browser.
- **Why**: Prepares the app, like gathering tools and setting up a workspace.
- **Analogy**: Like stocking a kitchen with appliances and setting the table.

---

## 7. Code Walkthrough: Creating the Sidebar and Inputs

**Code Block: Sidebar and File Upload**

```python
with st.sidebar:
    st.markdown("""
    <h3 style='font-family: Times New Roman; font-size: 18px;'>📄 Upload File Format</h3>
    <p style='font-family: Times New Roman; font-size: 14px;'>
        Upload an Excel file (.xlsx) with the following columns:
        <ul>
            <li><b>AminoAcid</b>: Name or identifier of the amino acid (e.g., ALA, GLY).</li>
            <li><b>Frequency</b>: CPMG frequency in Hz (numeric).</li>
            <li><b>R2eff</b>: Effective relaxation rate (s⁻¹, numeric).</li>
            <li><b>R2eff_error</b>: Error in R2eff (s⁻¹, numeric).</li>
        </ul>
        Ensure the file has a header row and no missing values in these columns.
    </p>
    """, unsafe_allow_html=True)
    st.number_input("Enter B0 value (MHz):", min_value=0.0, value=81.0, step=0.1, key="B0_input")
uploaded_file = st.file_uploader("\U0001F4C1 Upload Excel file with columns: AminoAcid, Frequency, R2eff, R2eff_error", type="xlsx")
```

- **What It Does**: Creates a sidebar with instructions and input fields for the user.
- **Details**:
  - Adds a sidebar with a formatted guide (using HTML) explaining the Excel file format.
  - Provides a box to enter B0 (default 81 MHz, decimals allowed).
  - Adds a button to upload an Excel file with specific columns.
- **Why**: Guides users to provide data and B0, like a user manual.
- **Analogy**: Like a restaurant menu explaining how to order.

---

## 8. Code Walkthrough: Displaying Titles and Equations

**Code Block: App Title and Model Equations**

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
        Luz–Meiboom Equation
    </h2>
""", unsafe_allow_html=True)
st.latex(r"""
R_{2,\mathrm{eff}}(\nu_\mathrm{CPMG}) = R_2 + \frac{\Phi}{k_\mathrm{ex}} \left(1 - \frac{4\nu_\mathrm{CPMG}}{k_\mathrm{ex}} \tanh\left(\frac{k_\mathrm{ex}}{4\nu_\mathrm{CPMG}}\right)\right)
""")
st.markdown("""
    <h2 style="font-family: 'Times New Roman'; font-size: 20px;">
        Carver–Richards Equation
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
    "<i>Carver–Richards fit uses full four-parameter expression including $k_{AB}$, $k_{BA}$, and $\Delta \delta$. No-Exchange model assumes constant $R_2$.</i>", unsafe_allow_html=True)
st.markdown("""
    <h3 style='font-family: Times New Roman; font-size: 16px;'>References</h3>
    <p style='font-family: Times New Roman; font-size: 14px;'>
        1. Luz, Z.; Meiboom, S. (1963) Nuclear Magnetic Resonance study of the protolysis of trimethylammonium ion in aqueous solution—order of the reaction with respect to solvent. <i>J. Chem. Phys.</i>, 39, 366–370.<br>
        2. Carver, J. P.; Richards, R. E. (1972) General 2-site solution for chemical exchange produced dependence of T2 upon Carr-Purcell pulse separation. <i>J. Magn. Reson.</i>, 6, 89-96.
    </p>
""", unsafe_allow_html=True)
```

- **What It Does**: Shows the app’s title, model equations, and references in the app.
- **Details**:
  - Displays “CPMG Fitting Tool” as a heading.
  - Shows the No-Exchange equation (R₂,eff = R₂).
  - Shows the Luz–Meiboom equation (see Section 5 image).
  - Shows the Carver–Richards equations (see Section 5 image).
  - Adds a note comparing models and cites original research.
- **Why**: Introduces the tool and explains the science to users.
- **Analogy**: Like a textbook chapter explaining the math.

---

## 9. Code Walkthrough: Defining Model Functions

**Code Block: Model Calculations**

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

- **What It Does**: Defines the math for the three models (No-Exchange, Luz–Meiboom, Carver–Richards).
- **Details**:
  - Stores the user’s B0 value in `B0`.
  - `no_exchange`: Returns a constant R₂ for all frequencies, modeling no shape switching.
  - `luz_meiboom`: Calculates R₂,eff for fast switching using R₂, kₑₓ, phi, and B0, with NumPy’s math tools (e.g., `np.tanh`).
  - `carver_richards`: Computes R₂,eff for complex dynamics using multiple parameters and intermediate calculations (ψ, ζ, etc.).
- **Why**: Sets up the equations to match data, like defining the rules for a game.
- **Analogy**: Like writing recipes for different dishes (simple to complex).
- **Images**:
  - ![Luz–Meiboom Equation](images/luz_meiboom_model.jpg)
  - ![Carver–Richards Equation](images/carver_richards_model.jpg)

---

## 10. Code Walkthrough: Loading and Processing Data

**Code Block: Reading Data and Checking Format**

```python
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    required_cols = {"AminoAcid", "Frequency", "R2eff", "R2eff_error"}
    if not required_cols.issubset(df.columns):
        st.error("❌ Excel file must contain: AminoAcid, Frequency, R2eff, R2eff_error")
        st.stop()
    amino_acids = df['AminoAcid'].unique()
    st.markdown(f"<div style='font-family: Times New Roman; font-size: 14px;'>✅ Loaded data for <b>{len(amino_acids)}</b> amino acid(s).</div>", unsafe_allow_html=True)
    fit_summary = []
    phi_summary = []
    all_plots = []
    all_tables = []
```

- **What It Does**: Loads the Excel file and prepares for analysis.
- **Details**:
  - Checks if a file was uploaded.
  - Reads the file into a table (`df`) using Pandas.
  - Ensures the table has required columns (AminoAcid, Frequency, R2eff, R2eff_error).
  - Shows an error and stops if columns are missing.
  - Lists unique amino acids and shows how many were found.
  - Creates empty lists for results (fits, phi values, graphs, tables).
- **Why**: Gets the NMR data ready, like checking ingredients before cooking.
- **Analogy**: Like opening a spreadsheet and making sure it has the right columns.

---

## 11. Code Walkthrough: Fitting Models and Plotting

**Code Block: Processing Each Amino Acid**

```python
    for aa in amino_acids:
        sub_df = df[df['AminoAcid'] == aa]
        x, y, yerr = sub_df['Frequency'].values, sub_df['R2eff'].values, sub_df['R2eff_error'].values
        st.markdown(f"<h4 style='font-family: Times New Roman; font-size: 18px;'>📈 Amino Acid: {aa}</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=yerr, fmt='o', color='black', capsize=4, label="Data")
```

- **What It Does**: Starts analyzing each amino acid and sets up a graph.
- **Details**:
  - Loops through each amino acid.
  - Filters the table for one amino acid’s data.
  - Gets frequencies (`x`), relaxation rates (`y`), and errors (`yerr`).
  - Shows a heading for the amino acid.
  - Creates a new graph and plots data points with error bars (black circles).
- **Why**: Prepares to fit models and show results for each amino acid.
- **Analogy**: Like focusing on one dish in a meal.

**Code Block: Fitting No-Exchange Model**

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

- **What It Does**: Fits the No-Exchange model and plots it.
- **Details**:
  - Initializes chi-squared (fit quality) and parameters as “bad” (infinity or not-a-number).
  - Tries fitting the No-Exchange model to data using `curve_fit`, guessing R₂ = 10.
  - Calculates error margins and fitted values.
  - Computes chi-squared to measure fit quality.
  - Plots the fit as a green line.
  - Shows an error if fitting fails.
- **Why**: Tests if the simple model matches the data.
- **Analogy**: Like trying a simple recipe and checking if it tastes right.
- **Image**: ![Sample Fit Curve](images/sample_fit_curve.jpg) _(Shows data points and model fits, including the green No-Exchange line.)_

**Code Block: Fitting Luz–Meiboom and Carver–Richards**

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

- **What It Does**: Fits the Luz–Meiboom and Carver–Richards models and plots them.
- **Details**:
  - Fits Luz–Meiboom with guesses for R₂, kₑₓ, phi, and plots a red line.
  - Saves phi values for later.
  - Fits Carver–Richards with guesses for four parameters, allowing more tries (`maxfev`), and plots a blue dashed line.
  - Calculates errors and chi-squared for each.
  - Shows errors if fitting fails.
- **Why**: Tests more complex models for better fits.
- **Analogy**: Like trying fancier recipes to match the taste.
- **Image**: ![Sample Fit Curve](images/sample_fit_curve.jpg) _(Shows red Luz–Meiboom and blue Carver–Richards lines.)_

---

## 12. Code Walkthrough: Displaying Graphs and Parameters

**Code Block: Finalizing and Showing Graphs**

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

- **What It Does**: Labels and shows the graph, then saves it.
- **Details**:
  - Labels x-axis (Frequency) and y-axis (R₂,eff).
  - Sets the title with the amino acid name.
  - Adds a legend and grid for clarity.
  - Displays the graph in the app.
  - Saves the graph as a PNG and stores it for downloading.
  - Closes the graph to save memory.
- **Why**: Shows the data and model fits visually.
- **Analogy**: Like hanging a finished painting.
- **Image**: ![Sample Fit Curve](images/sample_fit_curve.jpg)

**Code Block: Showing Parameter Tables**

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

- **What It Does**: Displays tables of fitted parameters for each model.
- **Details**:
  - Splits the app into three columns for tables.
  - Shows No-Exchange parameters (R₂), Luz–Meiboom parameters (R₂, kₑₓ, φ), and Carver–Richards parameters (R₂, k_AB, k_BA, Δδ, kₑₓ).
  - Calculates kₑₓ (k_AB + k_BA) for Carver–Richards.
  - Saves tables as CSV files.
  - Shows “Fit failed” if there’s an error.
- **Why**: Presents the model results clearly.
- **Analogy**: Like displaying ingredient lists for each dish.
- **Image**: ![Sample Fit Parameters](images/sample_fit_parameters.jpg) _(Shows tables with parameters like R₂, kₑₓ, φ.)_

---

## 13. Code Walkthrough: Summarizing and Downloading Results

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

- **What It Does**: Compares models and shows a summary table.
- **Details**:
  - Finds the model with the smallest chi-squared (best fit).
  - Labels it as “No-Exchange,” “Luz–Meiboom,” or “Carver–Richards.”
  - Adds results to a summary list.
  - Displays a table of chi-squared values, highlighting the best fit in green.
  - Saves the table as a CSV.
- **Why**: Shows which model best matches the data.
- **Analogy**: Like a scorecard picking the best dish.

**Code Block: Phi Values Plot**

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
        st.markdown("<h4 style='font-family: Times New Roman; font-size: 18px;'>📊 Summary of ϕ values (Luz–Meiboom)</h4>", unsafe_allow_html=True)
        st.pyplot(fig_phi)
        buf_phi = io.BytesIO()
        fig_phi.savefig(buf_phi, format="png")
        buf_phi.seek(0)
        all_plots.append(("phi_values_plot.png", buf_phi.getvalue()))
        plt.close(fig_phi)
        all_tables.append(("phi_summary.txt", summary_df.to_csv(index=False)))
```

- **What It Does**: Creates a graph of phi values from the Luz–Meiboom model.
- **Details**:
  - Checks if phi values exist.
  - Creates a table of phi values, sorted by amino acid.
  - Plots phi values with error bars (black circles) and a red line.
  - Labels axes and adds a title and grid.
  - Shows the graph and saves it as a PNG.
  - Saves the phi table as a CSV.
- **Why**: Shows trends in phi values across amino acids.
- **Analogy**: Like a chart showing ingredient amounts.

**Code Block: Downloading Results**

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

- **What It Does**: Adds buttons to download graphs and tables.
- **Details**:
  - Checks if there are results to download.
  - Adds buttons for each graph (PNG) and table (CSV).
  - Combines all tables into one Excel file with multiple sheets.
  - Adds a button to download the Excel file.
- **Why**: Lets users save results for reports.
- **Analogy**: Like a takeaway counter for dishes.

---

## 14. Demo: Seeing the Tool in Action

**Steps**:

1. **Run the App**: Use `streamlit run app.py` locally or a hosted version.
2. **Upload File**: Load a sample Excel file with columns:
   - AminoAcid (e.g., ALA, GLY)
   - Frequency (e.g., 100, 200 Hz)
   - R2eff (e.g., 20.5, 22.1 s⁻¹)
   - R2eff_error (e.g., 0.2, 0.3 s⁻¹)
3. **Enter B0**: Type 81 MHz in the sidebar.
4. **View Results**: Show graphs (like the sample fit curve), parameter tables (like the sample fit parameters), and the phi value graph.
5. **Download**: Click to download a graph or the Excel file.

**Key Point**: No coding needed—just upload and click!

**Preparation**:

- Create `sample_data.xlsx`:
  ```
  AminoAcid,Frequency,R2eff,R2eff_error
  ALA,100,20.5,0.2
  ALA,200,20.3,0.3
  GLY,100,22.1,0.25
  GLY,200,21.9,0.35
  ```
- Install libraries: `pip install streamlit pandas numpy matplotlib scipy xlsxwriter`.
- Test the app with the sample file.
- Keep the demo short (2–3 minutes).

**Images**:

- ![Sample Fit Curve](images/sample_fit_curve.jpg) _(Shows the graph output.)_
- ![Sample Fit Parameters](images/sample_fit_parameters.jpg) _(Shows the parameter tables.)_

---

## 15. Benefits and Conclusion

**Benefits**:

- **Easy**: No coding required—just upload and get results.
- **Fast**: Automates complex math, saving hours.
- **Clear**: Graphs and tables make results easy to use in research.
- **Impact**: Helps study protein dynamics for biology and medicine.

**Python’s Role**: Turns complex math into a simple app, like a lab assistant.

**Conclusion**:

- Built during my IIT Guwahati internship, this tool shows Python’s power in science.
- Thanks to my mentors, team, and you for listening!
- Questions? Want to see the demo again?

---
