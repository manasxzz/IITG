import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Constants
pi = np.pi
B0 = 81  # MHz
gamma_H = 267.513e6  # rad T⁻¹ s⁻¹, for 1H

# Model function


def R2eff_model(v_cpmg, R2, k_AB, k_BA, delta_ppm):
    delta = 2 * pi * delta_ppm * B0  # Δ = 2πγB₀Δδ

    kex = k_AB + k_BA
    psi = (k_AB - k_BA) ** 2 - delta ** 2 + 4 * k_AB * k_BA
    zeta = 4 * pi * delta * (k_AB - k_BA)

    sqrt_term = np.sqrt(psi ** 2 + zeta ** 2)
    eta = np.sqrt(-psi + sqrt_term) / (2 * v_cpmg * np.sqrt(8))
    xi = np.sqrt(psi + sqrt_term) / (2 * v_cpmg * np.sqrt(8))

    D_plus = 0.5 * (1 + (psi + 2 * delta ** 2) / np.sqrt(psi ** 2 + zeta ** 2))
    D_minus = 0.5 * (-1 + (psi + 2 * delta ** 2) /
                     np.sqrt(psi ** 2 + zeta ** 2))

    lambda_val = np.sqrt(D_plus * np.cosh(xi) ** 2 - D_minus * np.cos(eta) ** 2) + \
        np.sqrt(D_plus * np.sinh(xi) ** 2 + D_minus * np.sin(eta) ** 2)

    R2eff = R2 + kex / 2 - 2 * v_cpmg * np.log(lambda_val)
    return R2eff


# Load Excel file
file_path = 'sample.xlsx'  # Change this to your actual file path
df = pd.read_excel(file_path)

v_cpmg = df['Frequency'].values
R2eff_exp = df['R2eff'].values
R2eff_err = df['R2eff_error'].values

# Initial guesses: [R2, k_AB, k_BA, delta_ppm]
p0 = [20.0, 500.0, 500.0, 1.0]

# Curve fitting
popt, pcov = curve_fit(
    R2eff_model,
    v_cpmg,
    R2eff_exp,
    sigma=R2eff_err,
    p0=p0,
    absolute_sigma=True,
    maxfev=10000
)

perr = np.sqrt(np.diag(pcov))
param_names = ['R2 (s⁻¹)', 'k_AB (Hz)', 'k_BA (Hz)', 'Δδ (ppm)']

# Calculate chi-squared
R2eff_calc = R2eff_model(v_cpmg, *popt)
chi2 = np.sum(((R2eff_exp - R2eff_calc) / R2eff_err) ** 2)

# Print results
print("Fitted Parameters:")
for name, val, err in zip(param_names, popt, perr):
    print(f"{name:10}: {val:.4f} ± {err:.4f}")
# === Compute k_ex and its uncertainty ===
k_ab, k_ba = popt[1], popt[2]
k_ab_err, k_ba_err = perr[1], perr[2]

k_ex = k_ab + k_ba
k_ex_err = np.sqrt(k_ab_err**2 + k_ba_err**2)

print(f"\nk_ex       = {k_ex:.2f} ± {k_ex_err:.2f} s⁻¹")

print(f"\nChi-squared (χ²): {chi2:.2f}")

# Plot
plt.errorbar(v_cpmg, R2eff_exp, yerr=R2eff_err,
             fmt='o', label='Experimental', capsize=4)
v_fit = np.linspace(min(v_cpmg), max(v_cpmg), 300)
R2eff_fit = R2eff_model(v_fit, *popt)
plt.plot(v_fit, R2eff_fit, '-', label='Fit')
plt.xlabel('CPMG Frequency (Hz)', fontsize=12)
plt.ylabel(r'$R_{2,\mathrm{eff}}$ (s$^{-1}$)', fontsize=12)
plt.title('CPMG Relaxation Dispersion Fit', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
