import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---- Constants ----
B0 = 81e6  # Hz
Tcp = 0.04  # s

# ---- 1. Load Data ----
df = pd.read_excel("r2_results.xlsx")  # Replace with your actual file name
nu_cpmg = df['Frequency'].values
r2eff_exp = df['R2eff'].values
r2eff_err = df['R2eff_error'].values

# ---- 2. Define Carver-Richards Model ----
def carver_richards_model(nu, R2, k_AB, k_BA, delta_ppm):
    delta_Hz = delta_ppm * B0  # Δδ in ppm to Hz
    k_ex = k_AB + k_BA
    p_A = k_BA / k_ex
    p_B = k_AB / k_ex
    psi = ((k_AB - k_BA)**2 +
           (2 * np.pi * delta_Hz)**2 -
           4 * np.pi * delta_Hz * k_AB)
    eta = np.sqrt(psi + k_ex**2) / (2 * nu)
    xi = np.sqrt(psi + k_ex**2) / (2 * nu)
    D = 0.5 * (1 + (psi + (2 * np.pi * delta_Hz)**2) / (psi + k_ex**2))
    lam = np.sqrt((D * np.cosh(xi) - D * np.cos(eta))**2 +
                  (D * np.sinh(xi) - D * np.sin(eta))**2)
    return R2 + 0.5 * k_ex - 2 * nu * np.log(lam)

# ---- 3. Fit using non-linear least squares ----
# Initial guesses: R2, k_AB, k_BA, Δδ (ppm)
p0 = [10.0, 200.0, 200.0, 1.0]
bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, 10])

popt, pcov = curve_fit(carver_richards_model, nu_cpmg, r2eff_exp, sigma=r2eff_err,
                       p0=p0, bounds=bounds, absolute_sigma=True)

# ---- 4. Extract and print parameters with uncertainties ----
param_names = ['R2 (s⁻¹)', 'k_AB (s⁻¹)', 'k_BA (s⁻¹)', 'Δδ (ppm)']
for name, val, err in zip(param_names, popt, np.sqrt(np.diag(pcov))):
    print(f"{name}: {val:.4f} ± {err:.4f}")

# ---- 5. Plot ----
plt.errorbar(nu_cpmg, r2eff_exp, yerr=r2eff_err, fmt='o', label='Data', capsize=3)
nu_fit = np.linspace(min(nu_cpmg), max(nu_cpmg), 500)
r2eff_fit = carver_richards_model(nu_fit, *popt)
plt.plot(nu_fit, r2eff_fit, label='Carver-Richards Fit', color='red')
plt.xlabel('ν_CPMG (Hz)')
plt.ylabel('R2,eff (s⁻¹)')
plt.title('CPMG Relaxation Dispersion Fit (Carver-Richards Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
