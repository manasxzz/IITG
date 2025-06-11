# app.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objs as go

# ----- Luz–Meiboom model -----
def luz_meiboom(v_cpmg, R2, k_ex, phi, B0=81.0):
    Phi = 4 * np.pi**2 * B0**2 * phi
    term = (4 * v_cpmg) / k_ex
    return R2 + (Phi / k_ex) * (1 - term * np.tanh(k_ex / (4 * v_cpmg)))

# ----- Streamlit App -----
st.set_page_config(page_title="Luz–Meiboom Fit", page_icon="📈", layout="centered")

st.title("📈 Luz–Meiboom Fit of CPMG Data")
st.write("Upload your **r2_results.csv** to fit the Luz–Meiboom model to your data.")

# File uploader
uploaded_file = st.file_uploader("Choose r2_results.csv", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Check required columns
    required_columns = {'Frequency', 'R2eff', 'R2eff_error'}
    if not required_columns.issubset(df.columns):
        st.error(f"CSV must contain columns: {required_columns}")
    else:
        # Extract data
        frequencies = df['Frequency'].values
        r2eff = df['R2eff'].values
        r2eff_error = df['R2eff_error'].values

        # ----- Fit data -----
        p0 = [10.0, 5000.0, 0.001]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

        try:
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

            # ----- Plot -----
            freq_smooth = np.linspace(np.min(frequencies), np.max(frequencies), 500)
            r2eff_fit = luz_meiboom(freq_smooth, *popt)

            fig = go.Figure()

            # Data points with error bars
            fig.add_trace(go.Scatter(
                x=frequencies,
                y=r2eff,
                mode='markers',
                name='Data',
                error_y=dict(
                    type='data',
                    array=r2eff_error,
                    visible=True
                ),
                marker=dict(color='black', size=8)
            ))

            # Fitted curve
            fig.add_trace(go.Scatter(
                x=freq_smooth,
                y=r2eff_fit,
                mode='lines',
                name='Luz–Meiboom Fit',
                line=dict(color='red', width=3)
            ))

            fig.update_layout(
                title='Luz–Meiboom Fit of CPMG Data',
                xaxis_title='Frequency (Hz)',
                yaxis_title='R₂eff (s⁻¹)',
                template='plotly_white',
                legend=dict(x=0.01, y=0.99),
                font=dict(size=16)
            )

            st.plotly_chart(fig, use_container_width=True)

            # ----- Show fit parameters -----
            st.subheader("📋 Fitted Parameters:")
            param_names = ['R2', 'k_ex', 'phi']

            param_table = pd.DataFrame({
                "Parameter": param_names,
                "Value": [f"{val:.6g}" for val in popt],
                "1σ Uncertainty": [f"{err:.6g}" for err in perr]
            })

            st.table(param_table)

        except Exception as e:
            st.error(f"An error occurred during fitting: {e}")

else:
    st.info("Please upload a CSV file to begin.")
