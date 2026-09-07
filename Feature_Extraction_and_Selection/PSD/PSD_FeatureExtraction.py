"""
PSD Feature Extraction — all subjects.

Computes Power Spectral Density (Welch's method) band-power features for
each subject's preprocessed inner-speech EEG data (29 channels, 2.5 s
action window) and saves one feature array per subject.

Input  (per subject): data/processed/X{sid}_InnerSpeech_Task.npy, shape (n_trials, 29, n_samples)
Output (per subject): features/PSD_Sub{sid}.npy, shape (n_trials, 3, 29)
                       3 = theta/alpha/beta bands, 29 = channels
"""

import os
import numpy as np
from scipy import signal

SUBJECT_IDS = list(range(1, 11))
X_PATH_TEMPLATE = "data/processed/X{sid}_InnerSpeech_Task.npy"
FEATURES_DIR = "features"
os.makedirs(FEATURES_DIR, exist_ok=True)

FS = 256
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def extract_psd(X, fs=FS, bands=BANDS):
    """
    X: (n_trials, n_channels, n_samples) raw preprocessed EEG.
    Returns: (n_trials, n_bands, n_channels) band-power features.
    """
    n_trials, n_ch, n_samp = X.shape

    # Vectorized Welch over all trials/channels at once (axis=-1 = samples).
    freqs, psd = signal.welch(X, fs=fs, nperseg=min(256, n_samp), axis=-1)

    psd_features = np.zeros((n_trials, len(bands), n_ch))
    for bi, (name, (lo, hi)) in enumerate(bands.items()):
        mask = (freqs >= lo) & (freqs <= hi)
        psd_features[:, bi, :] = psd[:, :, mask].mean(axis=-1)
    return psd_features


if __name__ == "__main__":
    for sid in SUBJECT_IDS:
        out_path = os.path.join(FEATURES_DIR, f"PSD_Sub{sid}.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] Subject {sid} already extracted -> {out_path}")
            continue

        print(f"\n=== Subject {sid} ===")
        X = np.load(X_PATH_TEMPLATE.format(sid=sid))
        features = extract_psd(X)

        print(f"  features shape: {features.shape}")
        np.save(out_path, features)
        print(f"  saved -> {out_path}")
