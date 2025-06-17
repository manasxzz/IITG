import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---- 1. Luz-Meiboom Equation ----
def luz_meiboom_model(nu_cpmg, R2, kex, phi, B0):
    # Φ = 4π² * B₀² * φ
    Phi = 4 * np.pi**2 * B0**2 * phi
    x = kex / (4 * nu_cpmg)
    return R2 + (Phi / kex) * (1 - (4 * nu_cpmg / kex) * np.tanh(x))

# Wrapper for fitting (fixing B0)
def model_wrapper(nu_cpmg, R2, kex, phi):
    B0 = 81  # MHz, change as needed
    return luz_meiboom_model(nu_cpmg, R2, kex, phi, B0)

# ---- 2. Load Excel Data ----
def load_data(file_path):
    df = pd.read_excel(file_path)
    freq = df.iloc[:, 0].values  # ν_CPMG in Hz
    r2_eff = df.iloc[:, 1].values  # R2_eff in 1/s
    return freq, r2_eff 

# ---- 3. Fit the Model ----
def fit_r2_data(freq, r2_eff):
    p0 = [10.0, 100.0, 0.01]  # R2, kex, phi
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(model_wrapper, freq, r2_eff, p0=p0, bounds=bounds)
    return popt, pcov

# ---- 4. Plotting ----
def plot_fit(freq, r2_eff, popt):
    freq_fit = np.linspace(min(freq), max(freq), 500)
    r2_fit = model_wrapper(freq_fit, *popt)

    plt.figure(figsize=(8, 5))
    plt.scatter(freq, r2_eff, label="Data", color="black")
    plt.plot(freq_fit, r2_fit, label="Luz–Meiboom Fit", color="red")
    plt.xlabel("ν_CPMG (Hz)")
    plt.ylabel("R2_eff (1/s)")
    plt.title("CPMG Relaxation Dispersion (Luz–Meiboom Model)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---- 5. Main ----
def main():
    file_path = "r2results.xlsx"  
    freq, r2_eff = load_data(file_path)
    popt, pcov = fit_r2_data(freq, r2_eff)
    print(f"Fitted parameters:\nR2 = {popt[0]:.2f}\nkex = {popt[1]:.2f}\nphi = {popt[2]:.5f}")
    plot_fit(freq, r2_eff, popt)

if __name__ == "__main__":
    main()
