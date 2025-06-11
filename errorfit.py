# luz_meiboom_fit.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----- 1. Read r2_results.csv -----
df = pd.read_csv("r2_results.csv")

# Extract data columns
frequencies = df['Frequency'].values
r2eff = df['R2eff'].values
r2eff_error = df['R2eff_error'].values

# ----- 2. Define Luz–Meiboom model -----
def luz_meiboom(v_cpmg, R2, k_ex, phi, B0=81.0):
    """
    Luz–Meiboom model for fitting R2eff vs frequency.
    
    Parameters:
    - v_cpmg : Frequency in Hz
    - R2     : Baseline relaxation rate
    - k_ex   : Exchange rate
    - phi    : Populations & chemical shift term
    - B0     : Static magnetic field strength in MHz (default 81 MHz)
    """
    Phi = 4 * np.pi**2 * B0**2 * phi
    term = (4 * v_cpmg) / k_ex
    return R2 + (Phi / k_ex) * (1 - term * np.tanh(k_ex / (4 * v_cpmg)))

# ----- 3. Fit data -----
# Initial guesses: R2 ≈ 10 s^-1, k_ex ≈ 5000 s^-1, phi ≈ 0.001
p0 = [10.0, 5000.0, 0.001]

# Bounds: all parameters must be positive
bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

# Perform curve fitting
popt, pcov = curve_fit(
    luz_meiboom,
    frequencies,
    r2eff,
    sigma=r2eff_error,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds
)

# Extract 1-σ uncertainties
perr = np.sqrt(np.diag(pcov))

# ----- 4. Plot -----
# Prepare smooth frequency range for plotting the fitted curve
freq_smooth = np.linspace(np.min(frequencies), np.max(frequencies), 500)
r2eff_fit = luz_meiboom(freq_smooth, *popt)

plt.figure(figsize=(8, 6))

# Plot experimental data with error bars
plt.errorbar(
    frequencies,
    r2eff,
    yerr=r2eff_error,
    fmt='o',
    label='Data',
    color='black',
    capsize=4
)

# Plot fitted model
plt.plot(freq_smooth, r2eff_fit, label='Luz–Meiboom Fit', color='red', linewidth=2)

# Labels, legend, title
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$R_{2,\mathrm{eff}}$ (s$^{-1}$)')
plt.title('Luz–Meiboom Fit of CPMG Data')
plt.legend()
plt.grid(True)

# Save plot
plt.savefig("r2_fit.png", dpi=300)
print("Plot saved as r2_fit.png")

# ----- 5. Print and save fit parameters -----
# Format parameters and errors
param_names = ['R2', 'k_ex', 'phi']
with open("fit_params.txt", "w", encoding="utf-8") as f:
    print("Fitted Parameters (with 1σ uncertainties):\n")
    f.write("Fitted Parameters (with 1σ uncertainties):\n\n")
    for name, val, err in zip(param_names, popt, perr):
        line = f"{name} = {val:.6g} ± {err:.6g}"
        print(line)
        f.write(line + "\n")

print("\nFit parameters saved to fit_params.txt")
