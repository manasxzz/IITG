import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- Luz-Meiboom Model ----------
def luz_meiboom_model(nu_cpmg, R2, kex, phi, B0=81):  # B0 in MHz
    Phi = 4 * np.pi**2 * B0**2 * phi
    x = kex / (4 * nu_cpmg)
    return R2 + (Phi / kex) * (1 - (4 * nu_cpmg / kex) * np.tanh(x))

def model_wrapper(nu_cpmg, R2, kex, phi):
    return luz_meiboom_model(nu_cpmg, R2, kex, phi)

# ---------- Load Data ----------
def load_multi_dataset(file_path):
    df = pd.read_excel(file_path)
    freq = df.iloc[:, 0].values
    datasets = {}

    for col in df.columns[1:]:
        r2 = df[col].values
        mask = (freq != 0) & ~np.isnan(r2)
        datasets[col] = (freq[mask], r2[mask])
    return datasets

# ---------- Fit Model ----------
def fit_r2_data(freq, r2_eff):
    p0 = [np.min(r2_eff), 100.0, 0.01]
    bounds = ([0, 1, 0], [np.inf, 1e5, 1])
    try:
        popt, _ = curve_fit(model_wrapper, freq, r2_eff, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan]
    return popt

# ---------- Plot + Table ----------
def plot_all_fits(datasets):
    plt.figure(figsize=(10, 6))
    xfit = np.linspace(10, 2000, 500)

    results = []

    for label, (freq, r2) in datasets.items():
        popt = fit_r2_data(freq, r2)
        if np.isnan(popt).any():
            print(f"⚠️ Skipping {label}: fit failed.")
            continue

        yfit = model_wrapper(xfit, *popt)
        plt.plot(xfit, yfit, label=f"{label} Fit", linewidth=2)
        plt.scatter(freq, r2, label=f"{label} Data", s=25)

        results.append({
            "Residue": label,
            "R2 (1/s)": round(popt[0], 3),
            "kex (1/s)": round(popt[1], 3),
            "phi": round(popt[2], 6)
        })

    plt.xlabel("ν_CPMG (Hz)")
    plt.ylabel("R₂_eff (1/s)")
    plt.title("CPMG Relaxation Dispersion Fits")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Output table
    print("\n📋 Fitted Parameters:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Optional: Save to CSV
    results_df.to_csv("fitted_parameters.csv", index=False)
    print("\n✅ Saved to fitted_parameters.csv")

# ---------- Main ----------
def main():
    file_path = "Sample2.xlsx"  # change to actual path
    datasets = load_multi_dataset(file_path)
    plot_all_fits(datasets)

if __name__ == "__main__":
    main()