import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Set font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
rcParams.update(
    {'axes.titlesize': 12, 'axes.labelsize': 12, 'legend.fontsize': 12})

st.set_page_config(layout="centered")

# Title and equations
st.markdown("""
    <h2 style="font-family: 'Times New Roman'; font-size: 20px;">
        CPMG Fitting Tool:
    </h2>
""", unsafe_allow_html=True)

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
    "<i>Carver–Richards fit uses full four-parameter expression including $k_{AB}$, $k_{BA}$, and $\Delta \delta$</i>", unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader(
    "\U0001F4C1 Upload CSV with columns: AminoAcid, Frequency, R2eff, R2eff_error", type="csv")

# Fit models


def luz_meiboom(v_cpmg, R2, k_ex, phi, B0=81.0):
    Phi = 4 * np.pi**2 * B0**2 * phi
    term = (4 * v_cpmg) / k_ex
    return R2 + (Phi / k_ex) * (1 - term * np.tanh(k_ex / (4 * v_cpmg)))


def carver_richards(v_cpmg, R2, k_AB, k_BA, delta_ppm, B0=81.0):
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


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = {"AminoAcid", "Frequency", "R2eff", "R2eff_error"}
    if not required_cols.issubset(df.columns):
        st.error("❌ CSV must contain: AminoAcid, Frequency, R2eff, R2eff_error")
        st.stop()

    amino_acids = df['AminoAcid'].unique()
    st.markdown(
        f"<div style='font-family: Times New Roman; font-size: 14px;'>✅ Loaded data for <b>{len(amino_acids)}</b> amino acid(s).</div>", unsafe_allow_html=True)

    phi_summary = []

    for aa in amino_acids:
        sub_df = df[df['AminoAcid'] == aa]
        x, y, yerr = sub_df['Frequency'].values, sub_df['R2eff'].values, sub_df['R2eff_error'].values

        st.markdown(
            f"<h4 style='font-family: Times New Roman; font-size: 18px;'>📈 Amino Acid: {aa}</h4>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=yerr, fmt='o',
                    color='black', capsize=4, label="Data")

        try:
            popt_luz, pcov_luz = curve_fit(luz_meiboom, x, y, sigma=yerr, absolute_sigma=True, p0=[
                                           10.0, 5000.0, 0.001], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            perr_luz = np.sqrt(np.diag(pcov_luz))
            fit_luz = luz_meiboom(np.linspace(min(x), max(x), 300), *popt_luz)
            ax.plot(np.linspace(min(x), max(x), 300),
                    fit_luz, 'r-', label="Luz–Meiboom Fit")
            phi_summary.append(
                {"Residue": aa, "phi": popt_luz[2], "phi_err": perr_luz[2]})
        except Exception as e:
            st.error(f"❌ Luz–Meiboom fit failed: {e}")
            popt_luz = perr_luz = [np.nan] * 3

        try:
            popt_cr, pcov_cr = curve_fit(carver_richards, x, y, sigma=yerr, p0=[
                                         20.0, 500.0, 500.0, 1.0], absolute_sigma=True, maxfev=10000)
            perr_cr = np.sqrt(np.diag(pcov_cr))
            fit_cr = carver_richards(
                np.linspace(min(x), max(x), 300), *popt_cr)
            ax.plot(np.linspace(min(x), max(x), 300), fit_cr,
                    'b--', label="Carver–Richards Fit")
        except Exception as e:
            st.error(f"❌ Carver–Richards fit failed: {e}")
            popt_cr = perr_cr = [np.nan] * 4

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)")
        ax.set_title(f"Model Fits for {aa}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🧾 Luz–Meiboom Parameters (± 1σ):**")
            try:
                table_luz = pd.DataFrame({
                    "Parameter": ["R₂", "kₑₓ", "φ"],
                    "Value": [f"{val:.6g}" for val in popt_luz],
                    "±1σ": [f"{err:.6g}" for err in perr_luz]
                })
                st.table(table_luz)
            except:
                st.markdown("_Fit failed._")

        with col2:
            st.markdown("**🧾 Carver–Richards Parameters (± 1σ):**")
            try:
                keff = popt_cr[1] + popt_cr[2]
                keff_err = np.sqrt(perr_cr[1]**2 + perr_cr[2]**2)
                table_cr = pd.DataFrame({
                    "Parameter": ["R₂", "k_AB", "k_BA", "Δδ", "kₑₓ"],
                    "Value": [f"{popt_cr[0]:.6g}", f"{popt_cr[1]:.6g}", f"{popt_cr[2]:.6g}", f"{popt_cr[3]:.6g}", f"{keff:.6g}"],
                    "±1σ": [f"{perr_cr[0]:.6g}", f"{perr_cr[1]:.6g}", f"{perr_cr[2]:.6g}", f"{perr_cr[3]:.6g}", f"{keff_err:.6g}"]
                })
                st.table(table_cr)
            except:
                st.markdown("_Fit failed._")

    if phi_summary:
        summary_df = pd.DataFrame(phi_summary).sort_values(
            by='Residue').reset_index(drop=True)
        fig_phi, ax_phi = plt.subplots()
        ax_phi.errorbar(summary_df.index + 1,
                        summary_df['phi'], yerr=summary_df['phi_err'], fmt='o', color='black', capsize=4, linewidth=2)
        ax_phi.plot(summary_df.index + 1,
                    summary_df['phi'], color='red', linewidth=1.5)
        ax_phi.set_xlabel("# residue")
        ax_phi.set_ylabel("Phi value")
        ax_phi.set_title("Phi values (Luz–Meiboom)")
        ax_phi.grid(True)
        st.markdown(
            "<h4 style='font-family: Times New Roman; font-size: 18px;'>📊 Summary of ϕ values (Luz–Meiboom)</h4>", unsafe_allow_html=True)
        st.pyplot(fig_phi)
