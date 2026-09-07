"""
CovRiem (Covariance / Riemannian Tangent Space) Feature Extraction and
Selection — all subjects.

Unlike CBCSD/PLV/PSD, extraction and selection CANNOT be split into two
separate scripts here: the Riemannian mean used to project each trial's
covariance matrix into tangent space is itself fit on data, so it must be
recomputed from the training fold only, inside every CV/LOO split.
Extracting tangent-space features once (from a mean fit on the full
dataset) and then selecting on top of them would leak test-fold
information into every feature. For this reason, per-trial covariance
matrices are computed once (a per-trial statistic, safe to reuse), but the
Riemannian mean and tangent projection are recomputed inside every fold.
No feature array is saved to disk for this modality.

Two-stage pipeline per subject:
  STAGE 1: 5-fold stratified CV (Riemannian mean computed once per training
           fold, features standardized with StandardScaler) over every
           (selector, classifier, k) combination -> keep the top 10 by
           mean CV accuracy.
  STAGE 2: for those top 10 configurations, run true Leave-One-Out CV (the
           Riemannian mean is rebuilt from the n-1 training trials on every
           LOO split) and compute resubstitution train accuracy. Rows are
           ranked by composite(LOO accuracy, train/test gap); all three
           accuracies (CV, train, LOO) are kept in the output for manual
           inspection when choosing the final per-subject configuration.

Outputs (per subject, in CSV_Results_CovRiem/):
  CovRiem_Sub{sid}_AllConfigurations.csv  -> every (selector, classifier, k) result
  CovRiem_Sub{sid}_Top10_LOO.csv          -> final top-10 configurations with LOO accuracy

Stage 2 is the slow part (LOO rebuilds the Riemannian mean n_trials times
per configuration), so it supports resuming from a checkpoint CSV if
interrupted.
"""

import os
import time
import traceback
import numpy as np
import pandas as pd
from scipy import linalg

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================================================
# CONFIG
# =========================================================
SUBJECT_IDS = list(range(1, 11))
X_PATH_TEMPLATE = "data/processed/X{sid}_InnerSpeech_Task.npy"
Y_PATH_TEMPLATE = "data/processed/y{sid}_InnerSpeech_Task.npy"

SAVE_DIR = "CSV_Results_CovRiem"
os.makedirs(SAVE_DIR, exist_ok=True)

FORCE_RERUN = False  # set True to ignore checkpoints and redo every subject

SHRINKAGE = 0.1  # covariance shrinkage toward identity, for numerical stability
RANDOM_STATE = 42
N_SPLITS = 5
CANDIDATE_K = list(range(5, 101))

LOO_WEIGHT = 0.9
GAP_WEIGHT = 0.1
TOP_N_STAGE1 = 10

# Feature scaling is fixed to StandardScaler throughout this study — it is
# not swept as a hyperparameter.

SELECTORS = {
    "mutual_info": lambda Xtr, ytr: mutual_info_classif(Xtr, ytr, random_state=RANDOM_STATE),
    "anova_f": lambda Xtr, ytr: f_classif(Xtr, ytr)[0],
}
CLASSIFIERS = {
    "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
    "svm_linear": lambda: SVC(kernel="linear", C=1.0),
    "lda": lambda: LinearDiscriminantAnalysis(),
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
}


# =========================================================
# Riemannian geometry helpers
# =========================================================
def _logm_sym(M):
    w, v = linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    return v @ np.diag(np.log(w)) @ v.T


def _sqrtm_sym(M):
    w, v = linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    return v @ np.diag(np.sqrt(w)) @ v.T


def _invsqrtm_sym(M):
    w, v = linalg.eigh(M)
    w = np.clip(w, 1e-10, None)
    return v @ np.diag(1 / np.sqrt(w)) @ v.T


def _expm_sym(M):
    w, v = linalg.eigh(M)
    return v @ np.diag(np.exp(w)) @ v.T


def riemannian_mean(covs, max_iter=20, tol=1e-6):
    """Affine-invariant (Karcher/Fréchet) mean of a stack of covariance matrices."""
    C = covs.mean(axis=0)
    for _ in range(max_iter):
        C_sqrt, C_invsqrt = _sqrtm_sym(C), _invsqrtm_sym(C)
        S = np.zeros_like(C)
        for i in range(covs.shape[0]):
            S += _logm_sym(C_invsqrt @ covs[i] @ C_invsqrt)
        S /= covs.shape[0]
        C_new = C_sqrt @ _expm_sym(S) @ C_sqrt
        if np.linalg.norm(C_new - C) < tol:
            C = C_new
            break
        C = C_new
    return C


def tangent_project(C_stack, mean_C, tri_idx):
    """Project each covariance matrix onto the tangent space at mean_C, vectorized (upper triangle incl. diagonal)."""
    C_invsqrt = _invsqrtm_sym(mean_C)
    feats = np.zeros((C_stack.shape[0], len(tri_idx[0])))
    for i in range(C_stack.shape[0]):
        S = _logm_sym(C_invsqrt @ C_stack[i] @ C_invsqrt)
        feats[i] = S[tri_idx]
    return feats


def compute_per_trial_covariances(X, shrinkage=SHRINKAGE):
    """Per-trial statistic only (depends solely on that trial's own raw signal) — safe to compute before any CV split."""
    n_trials, n_ch, n_samp = X.shape
    C = np.zeros((n_trials, n_ch, n_ch))
    for i in range(n_trials):
        c = np.cov(X[i])
        C[i] = (1 - shrinkage) * c + shrinkage * np.trace(c) / n_ch * np.eye(n_ch)
    return C


def _rank_order(scores):
    return np.argsort(np.nan_to_num(scores, nan=-np.inf), kind="mergesort")


def _minmax_norm(s):
    s = np.asarray(s, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


# =========================================================
# STAGE 1 — 5-fold sweep (Riemannian mean recomputed per fold)
# =========================================================
def run_stage1(C_all, y, subject_name, valid_k):
    tri_idx = np.triu_indices(C_all.shape[1])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(y)), y))

    # Riemannian mean + tangent projection computed once per fold (reused across all k/selector/classifier).
    fold_features = []
    for train_idx, test_idx in folds:
        mean_C_fold = riemannian_mean(C_all[train_idx])
        feat_train = tangent_project(C_all[train_idx], mean_C_fold, tri_idx)
        feat_test = tangent_project(C_all[test_idx], mean_C_fold, tri_idx)
        fold_features.append((feat_train, feat_test, y[train_idx], y[test_idx]))

    all_results = []
    for selector_name, selector_fn in SELECTORS.items():
        prepared = []
        for feat_train, feat_test, y_train, y_test in fold_features:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(feat_train)
            Xte = scaler.transform(feat_test)
            scores = selector_fn(Xtr, y_train)
            order = _rank_order(scores)
            prepared.append((Xtr, Xte, y_train, y_test, order))

        for clf_name, clf_factory in CLASSIFIERS.items():
            for k in valid_k:
                test_accs = []
                for Xtr, Xte, ytr, yte, order in prepared:
                    sel = order[-k:]
                    clf = clf_factory()
                    clf.fit(Xtr[:, sel], ytr)
                    test_accs.append(accuracy_score(yte, clf.predict(Xte[:, sel])) * 100.0)
                all_results.append({
                    "subject": subject_name,
                    "selector": selector_name,
                    "classifier": clf_name, "k_features": k,
                    "mean_accuracy": float(np.mean(test_accs)),
                    "std_accuracy": float(np.std(test_accs)),
                })

    df_all = pd.DataFrame(all_results).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    top10 = df_all.head(TOP_N_STAGE1).copy()
    return df_all, top10


# =========================================================
# STAGE 2 — LOO + train accuracy on the Stage-1 top 10
# =========================================================
def run_stage2(C_all, y, top10, tri_idx, verbose=True, checkpoint_path=None):
    """
    checkpoint_path: optional CSV path where each finished config's result
    row is appended as soon as it's done, so this stage can resume after an
    interruption instead of recomputing already-finished configurations
    (each True-LOO evaluation rebuilds the Riemannian mean n_trials times).
    """
    n_trials = len(y)
    rows = []

    existing = pd.read_csv(checkpoint_path) if (checkpoint_path and os.path.exists(checkpoint_path)) else None

    def _already_done(row):
        if existing is None:
            return False
        m = ((existing["selector"] == row["selector"]) &
             (existing["classifier"] == row["classifier"]) &
             (existing["k_features"] == row["k_features"]))
        return m.any()

    for _, row in top10.iterrows():
        if _already_done(row):
            if verbose:
                print(f"    [skip, checkpointed] {row['selector']}/{row['classifier']}/k={int(row['k_features'])}")
            continue

        selector_fn = SELECTORS[row["selector"]]
        clf_factory = CLASSIFIERS[row["classifier"]]
        k = int(row["k_features"])

        # Resubstitution train accuracy, using the full-dataset Riemannian mean.
        mean_C_full = riemannian_mean(C_all)
        feat_full = tangent_project(C_all, mean_C_full, tri_idx)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(feat_full)
        scores = selector_fn(Xs, y)
        sel = _rank_order(scores)[-k:]
        clf = clf_factory()
        clf.fit(Xs[:, sel], y)
        train_acc = float(accuracy_score(y, clf.predict(Xs[:, sel])) * 100.0)

        # True LOO: Riemannian mean rebuilt from the n-1 training trials on every split.
        correct = 0
        for i in range(n_trials):
            tr = np.array([j for j in range(n_trials) if j != i])
            te = np.array([i])
            mean_C_fold = riemannian_mean(C_all[tr])
            feat_tr = tangent_project(C_all[tr], mean_C_fold, tri_idx)
            feat_te = tangent_project(C_all[te], mean_C_fold, tri_idx)

            scaler_f = StandardScaler()
            Xtr = scaler_f.fit_transform(feat_tr)
            Xte = scaler_f.transform(feat_te)
            sel_f = _rank_order(selector_fn(Xtr, y[tr]))[-k:]

            clf_f = clf_factory()
            clf_f.fit(Xtr[:, sel_f], y[tr])
            correct += int(clf_f.predict(Xte[:, sel_f])[0] == y[i])

        loo_acc = 100.0 * correct / n_trials

        result_row = {
            **row.to_dict(),
            "mean_train_accuracy": train_acc,
            "train_test_gap": train_acc - row["mean_accuracy"],
            "loo_accuracy": loo_acc,
        }
        rows.append(result_row)
        if verbose:
            print(f"    {row['selector']}/{row['classifier']}/k={k}: "
                  f"CV={row['mean_accuracy']:.2f}% train={train_acc:.2f}% LOO={loo_acc:.2f}%", flush=True)

        if checkpoint_path is not None:
            df_row = pd.DataFrame([result_row])
            df_row.to_csv(checkpoint_path, mode="a", header=not os.path.exists(checkpoint_path), index=False)

    df_stage2 = pd.read_csv(checkpoint_path) if (checkpoint_path and os.path.exists(checkpoint_path)) else pd.DataFrame(rows)

    df_stage2["norm_loo_acc"] = _minmax_norm(df_stage2["loo_accuracy"])
    df_stage2["norm_gap"] = _minmax_norm(df_stage2["train_test_gap"])
    df_stage2["composite_score"] = (
        LOO_WEIGHT * df_stage2["norm_loo_acc"] + GAP_WEIGHT * (1 - df_stage2["norm_gap"])
    )
    return df_stage2.sort_values("composite_score", ascending=False).reset_index(drop=True)


# =========================================================
# Per-subject driver
# =========================================================
def run_subject(X_raw, y, subject_name, verbose=True):
    t0 = time.time()

    n_trials, n_ch, n_samp = X_raw.shape
    tri_idx = np.triu_indices(n_ch)
    n_feat_total = len(tri_idx[0])
    valid_k = [k for k in CANDIDATE_K if k <= n_feat_total]

    if verbose:
        print(f"[{subject_name}] computing per-trial covariances "
              f"({n_trials} trials, {n_ch} channels -> {n_feat_total} tangent-space features)...")
    C_all = compute_per_trial_covariances(X_raw)

    if verbose:
        print(f"[{subject_name}] stage 1: 5-fold sweep...")
    df_full, top10 = run_stage1(C_all, y, subject_name, valid_k)
    full_path = os.path.join(SAVE_DIR, f"CovRiem_{subject_name}_AllConfigurations.csv")
    df_full.to_csv(full_path, index=False)
    if verbose:
        print(f"    saved: {full_path}  ({len(df_full)} configs)")

    if verbose:
        print(f"[{subject_name}] stage 2: LOO + train accuracy on top {TOP_N_STAGE1}...")
    checkpoint_path = os.path.join(SAVE_DIR, f"_checkpoint_{subject_name}.csv")
    df_top10 = run_stage2(C_all, y, top10, tri_idx, verbose=verbose, checkpoint_path=checkpoint_path)
    top10_path = os.path.join(SAVE_DIR, f"CovRiem_{subject_name}_Top10_LOO.csv")
    df_top10.to_csv(top10_path, index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)  # stage 2 finished cleanly, checkpoint no longer needed
    if verbose:
        print(f"    saved: {top10_path}")

    best = df_top10.iloc[0]
    elapsed = time.time() - t0
    if verbose:
        print(f"[{subject_name}] done in {elapsed:.1f}s. "
              f"FINAL PICK: {best['selector']}/{best['classifier']}/k={int(best['k_features'])} "
              f"| CV={best['mean_accuracy']:.2f}% train={best['mean_train_accuracy']:.2f}% "
              f"LOO={best['loo_accuracy']:.2f}%")

    return {
        "subject": subject_name,
        "selector": best["selector"], "classifier": best["classifier"],
        "k_features": int(best["k_features"]),
        "cv_accuracy": best["mean_accuracy"], "cv_std": best["std_accuracy"],
        "mean_train_accuracy": best["mean_train_accuracy"],
        "train_test_gap": best["train_test_gap"],
        "loo_accuracy": best["loo_accuracy"],
        "elapsed_seconds": elapsed,
        "status": "ok",
    }


# =========================================================
# DRIVER — loop over all subjects, with checkpointing
# =========================================================
if __name__ == "__main__":
    summary_path = os.path.join(SAVE_DIR, "CovRiem_all_subjects_summary.csv")
    summary_rows = []
    if os.path.exists(summary_path) and not FORCE_RERUN:
        summary_rows = pd.read_csv(summary_path).to_dict("records")
    done_subjects = {r["subject"] for r in summary_rows if r.get("status") == "ok"}

    t_batch_start = time.time()

    for sid in SUBJECT_IDS:
        subject_name = f"Sub{sid}"
        if subject_name in done_subjects:
            print(f"[SKIP] {subject_name} already done (checkpoint hit).")
            continue

        print("\n" + "#" * 60)
        print(f"# SUBJECT {sid}")
        print("#" * 60)

        try:
            X = np.load(X_PATH_TEMPLATE.format(sid=sid))
            y = np.load(Y_PATH_TEMPLATE.format(sid=sid))
            assert X.ndim == 3, f"expected raw EEG (n_trials, n_ch, n_samples), got ndim={X.ndim}"
            assert X.shape[0] == y.shape[0], "trial count mismatch between X and y"

            result = run_subject(X, y, subject_name)
            summary_rows = [r for r in summary_rows if r["subject"] != subject_name]
            summary_rows.append(result)

        except Exception as e:
            print(f"[ERROR] {subject_name} failed: {e}")
            traceback.print_exc()
            summary_rows.append({"subject": subject_name, "status": f"FAILED: {e}"})

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nBatch finished in {(time.time() - t_batch_start) / 60:.1f} minutes.")
    print(f"Summary: {summary_path}")
