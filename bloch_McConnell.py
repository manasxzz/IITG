import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import expm

# Constants
B0 = 81e6  # Magnetic field strength in Hz
Tp = 0.040  # CPMG pulse train duration in seconds

# ---- 1. Read Data ----
file_path = 'r2_results.xlsx'  # Change to your file name
df = pd.read_excel(file_path)

frequencies = df['Frequency'].values  # Hz
r2eff_exp = df['R2eff'].values  # s^-1
r2eff_err = df['R2eff_error'].values  # s^-1

# ---- 2. Bloch-McConnell Model ----
def bloch_mcconnell_r2eff(nu_cpmg, R2, k_AB, k_BA, delta_ppm):
    delta = delta_ppm * B0  # convert ppm to Hz
    t = 1 / (4 * nu_cpmg)
    n = Tp * nu_cpmg

    r2eff_values = []

    for nu in nu_cpmg:
        t_cp = 1 / (4 * nu)
        n_cp = int(Tp * nu)

        R = np.array([
            [-R2 - k_AB,         k_BA],
            [k_AB, -R2 - k_BA + 2j * np.pi * delta]
        ])

        # Full echo unit block: U = exp(R t) exp(R† t) exp(R t) exp(R† t)
        U = expm(R * t_cp) @ expm(R.conj().T * t_cp)
        U = U @ expm(R * t_cp) @ expm(R.conj().T * t_cp)

        # Raise to power n
        M = np.linalg.matrix_power(U, n_cp)

        # Initial magnetization vector
        I0 = np.array([[1.0], [0.0]])
        I_t = M @ I0

        # Use magnitude of I(t)[0] as intensity
        I_intensity = np.abs(I_t[0, 0])
        I0_magnitude = np.abs(I0[0, 0])

        if I_intensity == 0:
            r2eff = 1e6  # penalize zero intensity
        else:
            r2eff = (1 / Tp) * np.log(I0_magnitude / I_intensity)

        r2eff_values.append(np.real(r2eff))

    return np.array(r2eff_values)

# ---- 3. Fit Function Wrapper ----
def fit_wrapper(nu_cpmg, R2, k_AB, k_BA, delta_ppm):
    return bloch_mcconnell_r2eff(nu_cpmg, R2, k_AB, k_BA, delta_ppm)

# Initial guesses
p0 = [10.0, 500.0, 500.0, 0.5]  # R2, kAB, kBA, delta_ppm

# Bounds to keep values physical
bounds = (
    [0, 0, 0, 0],    # Lower bounds
    [100, 1e5, 1e5, 10]  # Upper bounds
)

# ---- 4. Fit Data ----
popt, pcov = curve_fit(fit_wrapper, frequencies, r2eff_exp, sigma=r2eff_err, p0=p0, bounds=bounds, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))  # Standard deviation of parameters

# ---- 5. Plot Results ----
plt.errorbar(frequencies, r2eff_exp, yerr=r2eff_err, fmt='o', label='Experimental Data', capsize=3)

# Fitted curve
nu_fit = np.linspace(min(frequencies), max(frequencies), 200)
r2eff_fit = fit_wrapper(nu_fit, *popt)
plt.plot(nu_fit, r2eff_fit, '-', label='Fitted Model', color='red')

plt.xlabel('CPMG Frequency (Hz)')
plt.ylabel(r'$R_{2,eff}$ (s$^{-1}$)')
plt.title('Bloch-McConnell Fit to Relaxation Dispersion Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 6. Print Parameters ----
param_names = ['R2 (s^-1)', 'k_AB (s^-1)', 'k_BA (s^-1)', 'Δδ (ppm)']
for name, val, err in zip(param_names, popt, perr):
    print(f'{name}: {val:.4f} ± {err:.4f}')
