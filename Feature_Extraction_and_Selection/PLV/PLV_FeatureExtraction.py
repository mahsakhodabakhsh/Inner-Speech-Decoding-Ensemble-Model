"""
PLV Feature Extraction — all subjects.

Computes Phase Locking Value (Lachaux et al., 1999) features for each
subject's preprocessed inner-speech EEG data (29 channels, 2.5 s action
window) and saves one feature array per subject.

Instantaneous phase (via the Hilbert transform) is only physically
meaningful for a narrowband signal, so band-pass filtering into
theta/alpha/beta before computing the phase is a mandatory step of this
method, not an optional design choice.

Input  (per subject): data/processed/X{sid}_InnerSpeech_Task.npy, shape (n_trials, 29, n_samples)
Output (per subject): features/PLV_Sub{sid}.npy, shape (n_trials, 3, n_channel_pairs)
                       3 = theta/alpha/beta bands, n_channel_pairs = 29*28/2 = 406
                       (off-diagonal pairs only — a channel's PLV with itself is always 1)
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


def extract_plv(X, fs=FS, bands=BANDS):
    """
    X: (n_trials, n_channels, n_samples) raw preprocessed EEG.
    Returns: (n_trials, n_bands, n_pairs) PLV features.
    """
    n_trials, n_ch, n_samp = X.shape
    tri_idx = np.triu_indices(n_ch, k=1)  # off-diagonal pairs only
    n_pairs = len(tri_idx[0])
    plv_features = np.empty((n_trials, len(bands), n_pairs))

    for bi, (name, (lo, hi)) in enumerate(bands.items()):
        b, a = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        Xb = signal.filtfilt(b, a, X, axis=2)
        analytic = signal.hilbert(Xb, axis=2)
        z = analytic / np.abs(analytic)  # unit phasors

        # Batched Gram matrix across trials: (n_trials, n_ch, n_ch).
        C = np.matmul(z, z.conj().transpose(0, 2, 1)) / n_samp
        plv_features[:, bi, :] = np.abs(C[:, tri_idx[0], tri_idx[1]])

    return plv_features


if __name__ == "__main__":
    for sid in SUBJECT_IDS:
        out_path = os.path.join(FEATURES_DIR, f"PLV_Sub{sid}.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] Subject {sid} already extracted -> {out_path}")
            continue

        print(f"\n=== Subject {sid} ===")
        X = np.load(X_PATH_TEMPLATE.format(sid=sid))
        features = extract_plv(X)

        assert not np.isnan(features).any(), "PLV contains NaN"
        assert not np.isinf(features).any(), "PLV contains Inf"

        print(f"  features shape: {features.shape}")
        np.save(out_path, features)
        print(f"  saved -> {out_path}")
