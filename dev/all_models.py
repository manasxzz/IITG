import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import io
import base64
import xlsxwriter

# Set font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
rcParams.update(
    {'axes.titlesize': 12, 'axes.labelsize': 12, 'legend.fontsize': 12})

st.set_page_config(layout="centered")

# Sidebar for file format explanation
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
    st.number_input("Enter B0 value (MHz):", min_value=0.0,
                    value=81.0, step=0.1, key="B0_input")
# Title and equations
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

# References
st.markdown("""
    <h3 style='font-family: Times New Roman; font-size: 16px;'>References</h3>
    <p style='font-family: Times New Roman; font-size: 14px;'>
        1. Luz, Z.; Meiboom, S. (1963) Nuclear Magnetic Resonance study of the protolysis of trimethylammonium ion in aqueous solution—order of the reaction with respect to solvent. <i>J. Chem. Phys.</i>, 39, 366–370.<br>
        2. Carver, J. P.; Richards, R. E. (1972) General 2-site solution for chemical exchange produced dependence of T2 upon Carr-Purcell pulse separation. <i>J. Magn. Reson.</i>, 6, 89-96.
    </p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "\U0001F4C1 Upload Excel file with columns: AminoAcid, Frequency, R2eff, R2eff_error", type="xlsx")

# Model functions
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


if uploaded_file:
    df = pd.read_excel(uploaded_file)
    required_cols = {"AminoAcid", "Frequency", "R2eff", "R2eff_error"}
    if not required_cols.issubset(df.columns):
        st.error(
            "❌ Excel file must contain: AminoAcid, Frequency, R2eff, R2eff_error")
        st.stop()

    amino_acids = df['AminoAcid'].unique()
    st.markdown(
        f"<div style='font-family: Times New Roman; font-size: 14px;'>✅ Loaded data for <b>{len(amino_acids)}</b> amino acid(s).</div>", unsafe_allow_html=True)

    fit_summary = []
    phi_summary = []
    all_plots = []
    all_tables = []

    for aa in amino_acids:
        sub_df = df[df['AminoAcid'] == aa]
        x, y, yerr = sub_df['Frequency'].values, sub_df['R2eff'].values, sub_df['R2eff_error'].values

        st.markdown(
            f"<h4 style='font-family: Times New Roman; font-size: 18px;'>📈 Amino Acid: {aa}</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=yerr, fmt='o',
                    color='black', capsize=4, label="Data")

        chi_noex = chi_luz = chi_cr = np.inf
        popt_noex = perr_noex = [np.nan]
        popt_luz = perr_luz = [np.nan] * 3
        popt_cr = perr_cr = [np.nan] * 4

        # No-Exchange Fit
        try:
            popt_noex, pcov_noex = curve_fit(
                no_exchange, x, y, sigma=yerr, absolute_sigma=True, p0=[10.0], bounds=([0], [np.inf]))
            perr_noex = np.sqrt(np.diag(pcov_noex))
            yfit_noex = no_exchange(x, *popt_noex)
            chi_noex = np.sum(((y - yfit_noex) / yerr)**2)
            ax.plot(np.linspace(min(x), max(x), 300), no_exchange(np.linspace(
                min(x), max(x), 300), *popt_noex), 'g-', label="No-Exchange Fit")
        except Exception as e:
            st.error(f"❌ No-Exchange fit failed: {e}")

        # Luz-Meiboom Fit
        try:
            popt_luz, pcov_luz = curve_fit(luz_meiboom, x, y, sigma=yerr, absolute_sigma=True, p0=[
                                           10.0, 5000.0, 0.001], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            perr_luz = np.sqrt(np.diag(pcov_luz))
            yfit_luz = luz_meiboom(x, *popt_luz)
            chi_luz = np.sum(((y - yfit_luz) / yerr)**2)
            ax.plot(np.linspace(min(x), max(x), 300), luz_meiboom(np.linspace(
                min(x), max(x), 300), *popt_luz), 'r-', label="Luz–Meiboom Fit")
            phi_summary.append(
                {"Residue": aa, "phi": popt_luz[2], "phi_err": perr_luz[2]})
        except Exception as e:
            st.error(f"❌ Luz–Meiboom fit failed: {e}")

        # Carver-Richards Fit
        try:
            popt_cr, pcov_cr = curve_fit(carver_richards, x, y, sigma=yerr, p0=[
                                         20.0, 500.0, 500.0, 1.0], absolute_sigma=True, maxfev=10000)
            perr_cr = np.sqrt(np.diag(pcov_cr))
            yfit_cr = carver_richards(x, *popt_cr)
            chi_cr = np.sum(((y - yfit_cr) / yerr)**2)
            ax.plot(np.linspace(min(x), max(x), 300), carver_richards(np.linspace(
                min(x), max(x), 300), *popt_cr), 'b--', label="Carver–Richards Fit")
        except Exception as e:
            st.error(f"❌ Carver–Richards fit failed: {e}")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)")
        ax.set_title(f"Model Fits for {aa}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Save plot to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        all_plots.append((f"{aa}_fit_plot.png", buf.getvalue()))
        plt.close(fig)

        # Determine the best model
        min_chi = min(chi_noex, chi_luz, chi_cr)
        if min_chi == chi_noex:
            better_model = "No-Exchange"
        elif min_chi == chi_luz:
            better_model = "Luz–Meiboom"
        else:
            better_model = "Carver–Richards"
        fit_summary.append({"Residue": aa, "Chi² (NoEx)": chi_noex,
                           "Chi² (Luz)": chi_luz, "Chi² (CR)": chi_cr, "Better Fit": better_model})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🧾 No-Exchange Parameters (± 1σ):**")
            try:
                noex_table = pd.DataFrame({"Parameter": ["R₂"], "Value": [
                    f"{val:.6g}" for val in popt_noex], "±1σ": [f"{err:.6g}" for err in perr_noex]})
                st.table(noex_table)
                all_tables.append(
                    (f"{aa}_no_exchange_params.txt", noex_table.to_csv(index=False)))
            except:
                st.markdown("_Fit failed._")

        with col2:
            st.markdown("**🧾 Luz–Meiboom Parameters (± 1σ):**")
            try:
                luz_table = pd.DataFrame({"Parameter": ["R₂", "kₑₓ", "φ"], "Value": [
                    f"{val:.6g}" for val in popt_luz], "±1σ": [f"{err:.6g}" for err in perr_luz]})
                st.table(luz_table)
                all_tables.append(
                    (f"{aa}_luz_meiboom_params.txt", luz_table.to_csv(index=False)))
            except:
                st.markdown("_Fit failed._")

        with col3:
            st.markdown("**🧾 Carver–Richards Parameters (± 1σ):**")
            try:
                keff = popt_cr[1] + popt_cr[2]
                keff_err = np.sqrt(perr_cr[1]**2 + perr_cr[2]**2)
                cr_table = pd.DataFrame({"Parameter": ["R₂", "k_AB", "k_BA", "Δδ", "kₑₓ"], "Value": [f"{popt_cr[0]:.6g}", f"{popt_cr[1]:.6g}", f"{popt_cr[2]:.6g}", f"{popt_cr[3]:.6g}", f"{keff:.6g}"], "±1σ": [
                    f"{perr_cr[0]:.6g}", f"{perr_cr[1]:.6g}", f"{perr_cr[2]:.6g}", f"{perr_cr[3]:.6g}", f"{keff_err:.6g}"]})
                st.table(cr_table)
                all_tables.append(
                    (f"{aa}_carver_richards_params.txt", cr_table.to_csv(index=False)))
            except:
                st.markdown("_Fit failed._")

    if fit_summary:
        st.markdown(
            "<h4 style='font-family: Times New Roman; font-size: 18px;'>📊 Model Comparison Summary</h4>", unsafe_allow_html=True)
        summary_table = pd.DataFrame(fit_summary).sort_values(
            by="Residue").reset_index(drop=True)
        st.dataframe(summary_table.style.highlight_min(
            subset=["Chi² (NoEx)", "Chi² (Luz)", "Chi² (CR)"], axis=1, color="lightgreen"))
        all_tables.append(("model_comparison_summary.txt",
                          summary_table.to_csv(index=False)))

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
        buf_phi = io.BytesIO()
        fig_phi.savefig(buf_phi, format="png")
        buf_phi.seek(0)
        all_plots.append(("phi_values_plot.png", buf_phi.getvalue()))
        plt.close(fig_phi)
        all_tables.append(("phi_summary.txt", summary_df.to_csv(index=False)))

    # Download all plots and tables
    if all_plots or all_tables:
        st.markdown(
            "<h4 style='font-family: Times New Roman; font-size: 18px;'>⬇️ Download Results</h4>", unsafe_allow_html=True)

        # Download individual plots
        for plot_name, plot_data in all_plots:
            st.download_button(
                label=f"Download {plot_name}",
                data=plot_data,
                file_name=plot_name,
                mime="image/png"
            )

        # Download individual tables
        for table_name, table_data in all_tables:
            st.download_button(
                label=f"Download {table_name}",
                data=table_data,
                file_name=table_name,
                mime="text/csv"
            )

        # Download all as Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for table_name, table_data in all_tables:
                df_temp = pd.read_csv(io.StringIO(table_data))
                df_temp.to_excel(writer, sheet_name=table_name.replace(
                    '.txt', ''), index=False)
        output.seek(0)
        st.download_button(
            label="Download All Tables as Excel",
            data=output,
            file_name="all_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
