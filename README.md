# Manas Data Analysis Tool

**An application for fitting CPMG relaxation dispersion data in protein NMR.**  
Developed during a research internship at **IIT Guwahati**.

---

## 🔬 Overview

This tool allows users to upload CPMG relaxation dispersion datasets and fit them to three different models:

- **No-Exchange Model** — Assumes a constant R₂ without chemical exchange.
- **Luz-Meiboom Model** — Designed for fast exchange regimes.
- **Carver-Richards Model** — Describes two-site chemical exchange (A ↔ B).

It outputs fitted parameters, visualizations, model comparisons, and allows data export.

---

## ⚙️ Features

- **📥 Data Input:** Upload an `.xlsx` file with columns for `AminoAcid`, `Frequency`, `R2eff`, and `R2eff_error`.

- **📈 Model Fitting:**

  - **No-Exchange:** Fits constant R₂.
  - **Luz-Meiboom:** Fits R₂, _kₑₓ_, and φ.
  - **Carver-Richards:** Fits R₂, _k_AB_, _k_BA_, and Δδ, and computes _kₑₓ = k_AB + k_BA_.

- **📊 Visualization:** Plots experimental data with error bars and fitted curves. Also includes a φ-value summary plot.

- **📋 Output:**

  - Tables of fitted parameters with ±1σ uncertainties.
  - χ² (chi-squared) values for comparing model fit quality.
  - Best-fitting model identified per residue.

- **💾 Export:**
  - Download individual plots (.png).
  - Export fitted parameter tables.
  - Export all results to a single Excel `.xlsx` file.

---

## 🚀 Installation

### 1. Clone the repository

```sh
git clone https://github.com/manasxzz/IITG.git
cd IITG
```

### 2. Create a virtual environment (recommended)

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Launch the app

```sh
streamlit run app.py
```

---

## 🧪 Usage

1. Upload a properly formatted `.xlsx` file (see below).
2. The app will:

   - Fit each amino acid to all three models.
   - Display fitted curves and error bars.
   - Show fitted parameters, φ plots (for Luz-Meiboom), and model comparisons.

3. Use the export options to download results.

---

## 📂 File Upload Format

The Excel file should include the following columns **with headers and no missing values**:

| AminoAcid | Frequency (Hz) | R2eff (s⁻¹) | R2eff_error (s⁻¹) |
| --------- | -------------- | ----------- | ----------------- |
| ALA       | 50             | 10.5        | 0.5               |
| ALA       | 100            | 11.2        | 0.4               |
| ALA       | 200            | 12.0        | 0.3               |
| GLY       | 50             | 9.8         | 0.6               |
| GLY       | 100            | 10.1        | 0.5               |
| GLY       | 200            | 10.5        | 0.4               |

---

## 📁 Repository Structure

```
IITG/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files (e.g., venv, __pycache__)
├── LICENSE                # MIT License
├── dev/                   # developmaental stages
```

---

## 📦 Dependencies

Listed in `requirements.txt`:

- Python ≥ 3.8
- streamlit ≥ 1.24.0
- pandas ≥ 1.5.0
- numpy ≥ 1.21.0
- scipy ≥ 1.9.0
- matplotlib ≥ 3.5.0
- xlsxwriter ≥ 3.0.0

---

## 📚 References

1. **Luz, Z., & Meiboom, S.** (1963). Nuclear magnetic resonance study of the protolysis of trimethylammonium ion in aqueous solution—order of the reaction with respect to solvent. _Journal of Chemical Physics, 39_(2), 366–370.

2. **Carver, J. P., & Richards, R. E.** (1972). A general two-site solution for the chemical exchange produced dependence of T2 upon the Carr-Purcell pulse separation. _Journal of Magnetic Resonance, 6_(1), 89–105.

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for more details.

---

## ✨ Acknowledgments

- Developed as part of a research internship at **IIT Guwahati**, Department of Biosciences and Bioengineering.
- Inspired by NMR relaxation studies in protein dynamics.
