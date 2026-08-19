"""
CBCSD Feature Extraction — all subjects.

Computes Coherence-Based Correntropy Spectral Density (CBCSD) features for
each subject's preprocessed inner-speech EEG data.

Input  (per subject): data/processed/X{sid}_InnerSpeech_Task.npy, shape (n_trials, 29, 641)
Output (per subject): features/CBCSD_Sub{sid}.npy, shape (n_trials, 3, n_channel_pairs)
                    3 = theta/alpha/beta bands, n_channel_pairs = 29*28/2 = 406   (symmetric connectivity matrix)
"""

import os
import numpy as np

SUBJECT_IDS = list(range(1, 11))
X_PATH_TEMPLATE = "data/processed/X{sid}_InnerSpeech_Task.npy"
FEATURES_DIR = "features"
os.makedirs(FEATURES_DIR, exist_ok=True)

FS = 256  # sampling rate after preprocessing
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


def calculate_optimal_sigma_auto(signal, coeff=0.9):
    """Silverman-type kernel bandwidth estimate from a single signal's own spread."""
    signal = np.asarray(signal)
    n = len(signal)
    std_val = np.std(signal, ddof=1)
    q75, q25 = np.percentile(signal, [75, 25])
    iqr_val = q75 - q25
    A = min(std_val, iqr_val / 1.34)
    if A == 0:
        A = 1
    return coeff * A * (n ** (-0.2))


def calculate_shared_sigma(signal1, signal2, coeff=0.9):
    """Kernel bandwidth shared between a channel pair (average of each channel's own sigma)."""
    sigma1 = calculate_optimal_sigma_auto(signal1, coeff)
    sigma2 = calculate_optimal_sigma_auto(signal2, coeff)
    return 0.5 * (sigma1 + sigma2)


def full_correntropy(signal1, signal2, sigma, coeff=0.9):
    """Cross-correntropy between two equal-length signals, evaluated at every lag."""
    signal1 = np.asarray(signal1, dtype=float)
    signal2 = np.asarray(signal2, dtype=float)

    if signal1.shape[0] != signal2.shape[0]:
        raise ValueError(
            "full_correntropy assumes equal-length signals; got "
            f"{signal1.shape[0]} and {signal2.shape[0]}."
        )

    N = len(signal1)
    M = len(signal2)  

    norm_factor = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    denom = 2 * (sigma ** 2)

    # Centering term: mean kernel value over the full N x M pairwise grid.
    diff2_full = (signal1[:, None] - signal2[None, :]) ** 2
    center_term = norm_factor * np.mean(np.exp(-diff2_full / denom))

    lags = np.arange(-(M - 1), N)
    correntropy_values = np.empty(len(lags))

    for idx, m in enumerate(lags):
        start1, end1 = max(0, m), min(N, M + m)
        start2, end2 = max(0, -m), min(M, N - m)

        s1_slice = signal1[start1:end1]
        s2_slice = signal2[start2:end2]

        if s1_slice.size == 0:
            correntropy_values[idx] = 0.0 - center_term
            continue

        diff2 = (s1_slice - s2_slice) ** 2
        k = norm_factor * np.exp(-diff2 / denom)
        correntropy_values[idx] = (np.sum(k) / N) - center_term 

    return list(lags), correntropy_values.tolist()


def correntropy_spectrum(V, fs=FS):
    """FFT of the correntropy function, restricted to non-negative frequencies."""
    V = np.asarray(V, dtype=float)
    V_shifted = np.fft.ifftshift(V)
    spectrum = np.fft.fft(V_shifted)
    freqs = np.fft.fftfreq(len(V), d=1 / fs)
    pos = freqs >= 0
    return freqs[pos], spectrum[pos]


def band_average(freqs, values, bands=BANDS):
    band_vals = []
    for _, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_vals.append(np.mean(values[idx]) if np.any(idx) else np.nan)
    return band_vals


def coherence_features(X, bands=BANDS, coeff=0.9, fs=FS, verbose=True):
    n_trials, n_channels, n_samples = X.shape
    n_bands = len(bands)
    pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]
    n_pairs = len(pairs)

    features = np.zeros((n_trials, n_bands, n_pairs))

    total_bins_checked = 0
    n_negative = 0
    n_above_one = 0

    for t in range(n_trials):
        if verbose and (t % 50 == 0):
            print(f"    trial {t + 1}/{n_trials}...")

        for p_idx, (i, j) in enumerate(pairs):
            xi, xj = X[t, i], X[t, j]
            sigma_pair = calculate_shared_sigma(xi, xj, coeff)

            _, Vxx = full_correntropy(xi, xi, sigma_pair, coeff)
            _, Vyy = full_correntropy(xj, xj, sigma_pair, coeff)
            _, Vxy = full_correntropy(xi, xj, sigma_pair, coeff)

            freqs, Pxx = correntropy_spectrum(Vxx, fs)
            _, Pyy = correntropy_spectrum(Vyy, fs)
            _, Pxy = correntropy_spectrum(Vxy, fs)

            cbcsd = (np.abs(Pxy) ** 2) / (np.real(Pxx) * np.real(Pyy) + 1e-12)

            total_bins_checked += cbcsd.size
            n_negative += np.sum(cbcsd < 0)
            n_above_one += np.sum(cbcsd > 1)

            features[t, :, p_idx] = band_average(freqs, cbcsd, bands)

    diagnostics = {
        "total_bins_checked": total_bins_checked,
        "n_negative": int(n_negative),
        "n_above_one": int(n_above_one),
        "pct_negative": 100 * n_negative / total_bins_checked if total_bins_checked else 0,
        "pct_above_one": 100 * n_above_one / total_bins_checked if total_bins_checked else 0,
    }
    return features, diagnostics


if __name__ == "__main__":
    for sid in SUBJECT_IDS:
        out_path = os.path.join(FEATURES_DIR, f"CBCSD_Sub{sid}.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] Subject {sid} already extracted -> {out_path}")
            continue

        print(f"\n=== Subject {sid} ===")
        X = np.load(X_PATH_TEMPLATE.format(sid=sid))
        features, diagnostics = coherence_features(X, verbose=True)

        print(f"  features shape: {features.shape}")
        print(f"  out-of-range CBCSD values: "
              f"{diagnostics['pct_negative']:.2f}% negative, "
              f"{diagnostics['pct_above_one']:.2f}% > 1")

        np.save(out_path, features)
        print(f"  saved -> {out_path}")
