import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import expm

# --- Constants ---
B0 = 81  # MHz (81 MHz)
Tcp = 0.04  # s

# --- Read Excel ---
file_path = 'sample.xlsx'  # update this
df = pd.read_excel(file_path)

frequencies = df['Frequency'].values  # Hz
intensities = df['Intensity'].values

# --- Normalize using I0 = I(f=0) ---
I0 = intensities[frequencies == 0][0]
v_cpmg = frequencies[frequencies != 0]
I_v = intensities[frequencies != 0]

# Calculate R2eff from intensity
R2eff = (1 / Tcp) * np.log(I0 / I_v)

# --- Bloch-McConnell Model for R2eff ---
def bloch_mcconnell(v, R2, kex, dp, pA):
    pB = 1 - pA
    delta_omega = 2 * np.pi * dp * B0
    kAB = pB * kex
    kBA = pA * kex
    n = Tcp * v
    t = 1 / (4 * v)

    R = np.array([
        [-R2 - kAB,        kBA],
        [kAB, -R2 - kBA + 1j * delta_omega]
    ])

    M = expm(R * t) @ expm(R.conj().T * t) @ expm(R * t) @ expm(R.conj().T * t)
    M_n = np.linalg.matrix_power(M, int(n))
    I = np.real((M_n @ np.array([1.0, 0.0]))[0])
    return (1 / Tcp) * np.log(1 / I)  # since I0 / I = 1 / I (as I0 = 1)

# Vectorized model
def model(v_array, R2, kex, dp, pA):
    return np.array([bloch_mcconnell(v, R2, kex, dp, pA) for v in v_array])

# --- Fit ---
p0 = [10.0, 500.0, 0.5, 0.9]  # R2, kex, Δδ, pA
bounds = ([0.1, 1, 0.01, 0], [100, 1e4, 5, 1])

popt, pcov = curve_fit(model, v_cpmg, R2eff, p0=p0, bounds=bounds, maxfev=10000)
R2_fit, kex_fit, dp_fit, pA_fit = popt

# --- Plot ---
v_plot = np.linspace(min(v_cpmg), max(v_cpmg), 300)
R2_plot = model(v_plot, *popt)

plt.figure(figsize=(8, 6))
plt.scatter(v_cpmg, R2eff, label='Data', color='red')
plt.plot(v_plot, R2_plot, label='Fit', color='blue')
plt.xlabel('CPMG Frequency (Hz)')
plt.ylabel(r'$R_{2,eff}^{CPMG}$ (s⁻¹)')
plt.title('Bloch-McConnell Fit to R2eff')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --- Output ---
print(f"Fitted parameters:")
print(f"R2     = {R2_fit:.3f} s⁻¹")
print(f"kex    = {kex_fit:.3f} s⁻¹")
print(f"Δδ     = {dp_fit:.3f} ppm")
print(f"pA     = {pA_fit:.3f}")
