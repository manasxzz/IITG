import argparse
import numpy as np
import pandas as pd

# --- Constants ---
T_CP = 0.040  # 40 ms in seconds
DEFAULT_TRIALS = 1000


def calculate_r2eff(I0: float, I_cpmg: float, t_cp: float = T_CP) -> float:
    """
    Compute the effective transverse relaxation rate R2eff.

    Parameters:
    - I0: Intensity at 0 Hz
    - I_cpmg: Intensity at given CPMG frequency
    - t_cp: Delay time (in seconds)
    """
    return (1.0 / t_cp) * np.log(I0 / I_cpmg)


def estimate_noise(intensities_50: np.ndarray, intensities_750: np.ndarray) -> float:
    """
    Estimate global measurement noise sigma from repeats at 50 Hz and 750 Hz.

    Parameters:
    - intensities_50: array of repeats at 50 Hz
    - intensities_750: array of repeats at 750 Hz

    Returns:
    - sigma: Mean of standard deviations
    """
    sigma_50 = np.std(intensities_50, ddof=1)
    sigma_750 = np.std(intensities_750, ddof=1)
    return np.mean([sigma_50, sigma_750])


def monte_carlo_error(I0: float, I_cpmg: float, sigma: float, trials: int = DEFAULT_TRIALS) -> float:
    """
    Perform Monte Carlo simulation to estimate R2eff error.

    Parameters:
    - I0: Intensity at 0 Hz
    - I_cpmg: Intensity at given frequency
    - sigma: Global noise estimate
    - trials: Number of Monte Carlo trials

    Returns:
    - std of simulated R2eff values
    """
    # Draw noisy samples
    I0_samples = np.random.normal(loc=I0, scale=sigma, size=trials)
    I_samples = np.random.normal(loc=I_cpmg, scale=sigma, size=trials)

    # Avoid negative or zero values
    I0_samples = np.clip(I0_samples, a_min=1e-12, a_max=None)
    I_samples = np.clip(I_samples, a_min=1e-12, a_max=None)

    # Compute R2eff samples
    r2_samples = (1.0 / T_CP) * np.log(I0_samples / I_samples)
    return np.std(r2_samples, ddof=1)


def process_data(input_file: str, output_file: str, trials: int) -> None:
    """
    Main processing function:
    - Read input Excel file
    - Compute R2eff and errors
    - Save results to CSV
    """
    # Read data
    df = pd.read_excel(input_file, engine='openpyxl')

    # Ensure required columns are present
    if not {'Frequency', 'Intensity'}.issubset(df.columns):
        raise ValueError("Input file must contain 'Frequency' and 'Intensity' columns.")

    # Extract I0 (intensity at 0 Hz): assume there's exactly one 0 Hz entry or use mean if multiple
    i0_values = df.loc[df['Frequency'] == 0, 'Intensity'].values
    if len(i0_values) == 0:
        raise ValueError("No intensity reading found at 0 Hz. Please include Frequency=0 row.")
    I0 = np.mean(i0_values)

    # Estimate noise sigma from repeats at 50 Hz and 750 Hz
    intens_50 = df.loc[df['Frequency'] == 50, 'Intensity'].values
    intens_750 = df.loc[df['Frequency'] == 750, 'Intensity'].values
    if len(intens_50) < 2 or len(intens_750) < 2:
        raise ValueError("Need at least two repeats at both 50 Hz and 750 Hz to estimate noise.")
    sigma = estimate_noise(intens_50, intens_750)

    # Prepare results
    results = []
    for freq, group in df.groupby('Frequency'):
        # Mean intensity at this frequency
        I_cpmg = group['Intensity'].mean()

        # Compute R2eff
        r2eff = calculate_r2eff(I0, I_cpmg)

        # Compute Monte Carlo error
        r2_err = monte_carlo_error(I0, I_cpmg, sigma, trials)

        results.append({'Frequency': freq,
                        'R2eff': r2eff,
                        'R2eff_error': r2_err})

    # Save to DataFrame and CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute R2eff and Monte Carlo errors from CPMG data.")
    parser.add_argument('-i', '--input', type=str, default='intensity.xlsx',
                        help='Path to input Excel file (default: intensity.xlsx)')
    parser.add_argument('-o', '--output', type=str, default='r2_results.csv',
                        help='Path to output CSV file (default: r2_results.csv)')
    parser.add_argument('-n', '--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'Number of Monte Carlo trials (default: {DEFAULT_TRIALS})')
    args = parser.parse_args()

    process_data(args.input, args.output, args.trials)
