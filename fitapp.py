import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Set Times New Roman font and size 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 12, 'legend.fontsize': 12})

st.set_page_config(layout="centered")

# ---- Title ----
st.markdown("""
    <h2 style="font-family: 'Times New Roman'; font-size: 20px;">
        Luz–Meiboom CPMG Fitting Tool
    </h2>
""", unsafe_allow_html=True)

# ---- Luz–Meiboom Equation ----
st.latex(r"""
R_{2,\mathrm{eff}}(\nu_\mathrm{CPMG}) = R_2 + \frac{\Phi}{k_\mathrm{ex}} \left(1 - \frac{4\nu_\mathrm{CPMG}}{k_\mathrm{ex}} \tanh\left(\frac{k_\mathrm{ex}}{4\nu_\mathrm{CPMG}}\right)\right)
""")

# ---- Explanation of Variables ----
st.markdown("""
<div style="font-family: 'Times New Roman'; font-size: 14px;">
<b>Where:</b><br>
<ul>
  <li><b>R<sub>2</sub></b>: Baseline transverse relaxation rate (s⁻¹)</li>
  <li><b>k<sub>ex</sub></b>: Exchange rate between conformational states (s⁻¹)</li>
  <li><b>Φ</b>: Exchange contribution, defined as Φ = 4π² · B₀² · φ</li>
  <li><b>φ</b>: Population-weighted chemical shift difference (ppm²)</li>
  <li><b>ν<sub>CPMG</sub></b>: CPMG frequency (Hz)</li>
  <li><b>B₀</b>: Static magnetic field strength (fixed at 81 MHz)</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---- Upload File ----
uploaded_file = st.file_uploader("📁 Upload CSV with columns: AminoAcid, Frequency, R2eff, R2eff_error", type="csv")

# ---- Luz–Meiboom Model Function ----
def luz_meiboom(v_cpmg, R2, k_ex, phi, B0=81.0):
    Phi = 4 * np.pi**2 * B0**2 * phi
    term = (4 * v_cpmg) / k_ex
    return R2 + (Phi / k_ex) * (1 - term * np.tanh(k_ex / (4 * v_cpmg)))

# ---- Main Execution ----
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = {"AminoAcid", "Frequency", "R2eff", "R2eff_error"}

    if not required_cols.issubset(df.columns):
        st.error("❌ CSV must contain: AminoAcid, Frequency, R2eff, R2eff_error")
        st.stop()

    amino_acids = df['AminoAcid'].unique()
    st.markdown(f"<div style='font-family: Times New Roman; font-size: 14px;'>✅ Loaded data for <b>{len(amino_acids)}</b> amino acid(s).</div>", unsafe_allow_html=True)

    # Store phi values
    phi_summary = []

    for aa in amino_acids:
        sub_df = df[df['AminoAcid'] == aa]
        frequencies = sub_df['Frequency'].values
        r2eff = sub_df['R2eff'].values
        r2eff_error = sub_df['R2eff_error'].values

        try:
            p0 = [10.0, 5000.0, 0.001]
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(
                luz_meiboom,
                frequencies,
                r2eff,
                sigma=r2eff_error,
                absolute_sigma=True,
                p0=p0,
                bounds=bounds
            )
            perr = np.sqrt(np.diag(pcov))

            # Store phi for summary plot
            phi_summary.append({
                "Residue": aa,
                "phi": popt[2],
                "phi_err": perr[2]
            })

            # Plot per residue
            freq_smooth = np.linspace(min(frequencies), max(frequencies), 300)
            fit_vals = luz_meiboom(freq_smooth, *popt)

            st.markdown(f"<h4 style='font-family: Times New Roman; font-size: 18px;'>📈 Amino Acid: {aa}</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.errorbar(frequencies, r2eff, yerr=r2eff_error, fmt='o', color='black', capsize=4, label="Data")
            ax.plot(freq_smooth, fit_vals, color='red', linewidth=2, label="Luz–Meiboom Fit")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)")
            ax.set_title(f"Luz–Meiboom Fit for {aa}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Fitted parameters
            st.markdown("<b>🧾 Fitted Parameters (± 1σ):</b>", unsafe_allow_html=True)
            param_names = ['R₂', 'kₑₓ', 'φ']
            for name, val, err in zip(param_names, popt, perr):
                st.markdown(f"<span style='font-family: Times New Roman; font-size: 14px;'>{name} = {val:.6g} ± {err:.6g}</span>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Fit failed for {aa}: {e}")

    # ---- Final Summary Phi Plot ----
    if phi_summary:
        summary_df = pd.DataFrame(phi_summary)
        summary_df = summary_df.sort_values(by='Residue').reset_index(drop=True)

        fig_phi, ax_phi = plt.subplots()
        ax_phi.errorbar(
            summary_df.index + 1,
            summary_df['phi'],
            yerr=summary_df['phi_err'],
            fmt='o',
            color='black',
            capsize=4,
            linewidth=2
        )
        ax_phi.plot(summary_df.index + 1, summary_df['phi'], color='red', linewidth=1.5)
        ax_phi.set_xlabel("# residue")
        ax_phi.set_ylabel("Phi value")
        ax_phi.set_title("Phi values for residues")
        ax_phi.grid(True)

        st.markdown("<h4 style='font-family: Times New Roman; font-size: 18px;'>📊 Summary of ϕ values across residues</h4>", unsafe_allow_html=True)
        st.pyplot(fig_phi)
