"""
Ensemble Evaluation — hard-voting ensemble across CBCSD, PLV, PSD, and
CovRiem, using each subject's final selected configuration from the
feature extraction/selection stage.

For each subject and each CV fold, all four modalities are fit and
predicted independently using their own pre-selected (selector,
classifier, k) configuration, then combined by majority (hard) vote across
the four predictions; ties go to the lower class label.

Produces four result files per run (all subjects), in CSV_Results_Ensemble/:
  Ensemble_KFold_AllSubjects.csv       -- 5-fold CV, test + train accuracy, per modality + ensemble
  Ensemble_LOO_AllSubjects.csv         -- Leave-One-Out CV, test + train accuracy, per modality + ensemble
  Ensemble_Permutation_AllSubjects.csv -- label-permutation test on the ensemble's 5-fold accuracy
  Ensemble_Binomial_AllSubjects.csv    -- one-sided binomial test on pooled 5-fold ensemble predictions vs. chance

Feature scaling is fixed to StandardScaler throughout; it is not swept as
a hyperparameter.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from scipy import linalg
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest


# ============================================================
# CONFIG -- edit paths/assumptions here
# ============================================================

SUBJECTS = [f"Sub{i}" for i in range(1, 11)]

SAVE_DIR = "CSV_Results_Ensemble"

CBCSD_DIR = Path("features")
PLV_DIR = Path("features")
PSD_DIR = Path("features")
RAW_DIR = Path("data/processed")

LABEL_PATH_TEMPLATE = "data/processed/y{n}_InnerSpeech_Task.npy"

COVRIEM_SHRINKAGE = 0.1

CBCSD_PATH_TEMPLATE = str(CBCSD_DIR / "CBCSD_Sub{n}.npy")
PLV_PATH_TEMPLATE = str(PLV_DIR / "PLV_Sub{n}.npy")
PSD_PATH_TEMPLATE = str(PSD_DIR / "PSD_Sub{n}.npy")
RAW_PATH_TEMPLATE = str(RAW_DIR / "X{n}_InnerSpeech_Task.npy")

CBCSD_BAND_INDEX = {"theta": 0, "alpha": 1, "beta": 2}
PLVPSD_BAND_INDEX = {"theta": 0, "alpha": 1, "beta": 2}

RANDOM_STATE = 42
KFOLD_SPLITS = 5
N_PERMUTATIONS = 200

MODALITIES = ["CBCSD", "PLV", "PSD", "CovRiem"]
ALL_COLS = MODALITIES + ["Ensemble"]
CHANCE_LEVEL = 0.25  # 4-class chance level, matches binomial-test convention

# --- per-subject best configs (unchanged from your sweep results) ---

CBCSD_BEST = {
    "Sub1":  dict(view="theta",          selector="mutual_info", classifier="logreg",     k=18),
    "Sub2":  dict(view="all_bands_flat", selector="anova_f",      classifier="lda",        k=75),
    "Sub3":  dict(view="theta",          selector="anova_f",      classifier="svm_rbf",    k=9),
    "Sub4":  dict(view="all_bands_flat", selector="mutual_info",  classifier="lda",        k=84),
    "Sub5":  dict(view="all_bands_flat", selector="anova_f",      classifier="svm_rbf",    k=15),
    "Sub6":  dict(view="beta",           selector="mutual_info",  classifier="svm_rbf",    k=21),
    "Sub7":  dict(view="all_bands_flat", selector="anova_f",      classifier="svm_rbf",    k=6),
    "Sub8":  dict(view="beta",           selector="anova_f",      classifier="svm_linear", k=22),
    "Sub9":  dict(view="beta",           selector="mutual_info",  classifier="svm_rbf",    k=20),
    "Sub10": dict(view="alpha",          selector="mutual_info",  classifier="svm_linear", k=26),
}

PLV_BEST = {
    "Sub1":  dict(view="beta",           selector="mutual_info", classifier="knn",        k=70),
    "Sub2":  dict(view="all_bands_flat", selector="anova_f",     classifier="lda",        k=14),
    "Sub3":  dict(view="theta",          selector="anova_f",     classifier="knn",        k=53),
    "Sub4":  dict(view="theta",          selector="anova_f",     classifier="logreg",     k=51),
    "Sub5":  dict(view="theta",          selector="mutual_info", classifier="knn",        k=43),
    "Sub6":  dict(view="alpha",          selector="mutual_info", classifier="svm_linear", k=36),
    "Sub7":  dict(view="all_bands_flat", selector="mutual_info", classifier="svm_linear", k=11),
    "Sub8":  dict(view="theta",          selector="mutual_info", classifier="lda",        k=95),
    "Sub9":  dict(view="all_bands_flat", selector="anova_f",     classifier="svm_rbf",    k=30),
    "Sub10": dict(view="alpha",          selector="anova_f",     classifier="svm_linear", k=36),
}

PSD_BEST = {
    "Sub1":  dict(view="theta",          selector="mutual_info", classifier="svm_rbf",    k=27),
    "Sub2":  dict(view="alpha",          selector="mutual_info", classifier="logreg",     k=20),
    "Sub3":  dict(view="beta",           selector="mutual_info", classifier="lda",        k=20),
    "Sub4":  dict(view="theta",          selector="mutual_info", classifier="svm_linear", k=12),
    "Sub5":  dict(view="alpha",          selector="mutual_info", classifier="knn",        k=14),
    "Sub6":  dict(view="theta",          selector="mutual_info", classifier="knn",        k=21),
    "Sub7":  dict(view="all_bands_flat", selector="mutual_info", classifier="logreg",     k=72),
    "Sub8":  dict(view="alpha",          selector="anova_f",     classifier="svm_rbf",    k=12),
    "Sub9":  dict(view="all_bands_flat", selector="mutual_info", classifier="svm_linear", k=32),
    "Sub10": dict(view="alpha",          selector="anova_f",     classifier="lda",        k=28),
}

COVRIEM_BEST = {
    "Sub1":  dict(selector="anova_f",     classifier="knn",        k=20),
    "Sub2":  dict(selector="anova_f",     classifier="lda",        k=5),
    "Sub3":  dict(selector="anova_f",     classifier="svm_linear", k=55),
    "Sub4":  dict(selector="anova_f",     classifier="lda",        k=85),
    "Sub5":  dict(selector="anova_f",     classifier="knn",        k=19),
    "Sub6":  dict(selector="anova_f",     classifier="lda",        k=8),
    "Sub7":  dict(selector="anova_f",     classifier="svm_linear", k=22),
    "Sub8":  dict(selector="mutual_info", classifier="knn",        k=52),
    "Sub9":  dict(selector="mutual_info", classifier="logreg",     k=30),
    "Sub10": dict(selector="anova_f",     classifier="logreg",     k=14),
}


# ============================================================
# Builders
# ============================================================

def build_classifier(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    if name == "svm_linear":
        return SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    raise ValueError(f"Unknown classifier: {name}")


def build_selector(name: str, k: int):
    if name == "anova_f":
        return SelectKBest(score_func=f_classif, k=k)
    if name == "mutual_info":
        return SelectKBest(score_func=partial(mutual_info_classif, random_state=RANDOM_STATE), k=k)
    raise ValueError(f"Unknown selector: {name}")


def build_flat_pipeline(selector_name: str, classifier_name: str, k: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", build_selector(selector_name, k)),
        ("clf", build_classifier(classifier_name)),
    ])


# ============================================================
# Riemannian geometry helpers (see CovRiem_FeatureExtractionSelection.py
# for the full explanation of why this must be recomputed per fold)
# ============================================================

def _logm_sym(M):
    w, v = linalg.eigh(M); w = np.clip(w, 1e-10, None)
    return v @ np.diag(np.log(w)) @ v.T

def _sqrtm_sym(M):
    w, v = linalg.eigh(M); w = np.clip(w, 1e-10, None)
    return v @ np.diag(np.sqrt(w)) @ v.T

def _invsqrtm_sym(M):
    w, v = linalg.eigh(M); w = np.clip(w, 1e-10, None)
    return v @ np.diag(1 / np.sqrt(w)) @ v.T

def _expm_sym(M):
    w, v = linalg.eigh(M)
    return v @ np.diag(np.exp(w)) @ v.T

def riemannian_mean(covs, max_iter=20, tol=1e-6):
    C = covs.mean(axis=0)
    for _ in range(max_iter):
        C_sqrt, C_invsqrt = _sqrtm_sym(C), _invsqrtm_sym(C)
        S = np.zeros_like(C)
        for i in range(covs.shape[0]):
            S += _logm_sym(C_invsqrt @ covs[i] @ C_invsqrt)
        S /= covs.shape[0]
        C_new = C_sqrt @ _expm_sym(S) @ C_sqrt
        if np.linalg.norm(C_new - C) < tol:
            C = C_new; break
        C = C_new
    return C

def tangent_project(C_stack, mean_C, tri_idx):
    C_invsqrt = _invsqrtm_sym(mean_C)
    feats = np.zeros((C_stack.shape[0], len(tri_idx[0])))
    for i in range(C_stack.shape[0]):
        S = _logm_sym(C_invsqrt @ C_stack[i] @ C_invsqrt)
        feats[i] = S[tri_idx]
    return feats

def compute_per_trial_covariances(X, shrinkage=COVRIEM_SHRINKAGE):
    n_trials, n_ch, n_samp = X.shape
    C = np.zeros((n_trials, n_ch, n_ch))
    for i in range(n_trials):
        c = np.cov(X[i])
        C[i] = (1 - shrinkage) * c + shrinkage * np.trace(c) / n_ch * np.eye(n_ch)
    return C

def covriem_selector_scores(selector_name: str, Xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    if selector_name == "anova_f":
        return f_classif(Xtr, ytr)[0]
    if selector_name == "mutual_info":
        return mutual_info_classif(Xtr, ytr, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown selector: {selector_name}")


# ============================================================
# Feature loading
# ============================================================

def load_labels(sub_n: int) -> np.ndarray:
    y = np.load(LABEL_PATH_TEMPLATE.format(n=sub_n))
    return np.asarray(y).ravel()

def load_flat_feature(path_template: str, sub_n: int, view: str, band_index: dict) -> np.ndarray:
    X = np.load(path_template.format(n=sub_n))
    if view in ("all_flat", "all_bands_flat"):
        return X.reshape(X.shape[0], -1)
    idx = band_index[view]
    return X[:, idx, :]

def load_raw_epochs(sub_n: int) -> np.ndarray:
    return np.load(RAW_PATH_TEMPLATE.format(n=sub_n))

def load_subject_data(sub_n: int) -> dict:
    """Load once per subject; reused across every fold / permutation.
    CovRiem's per-trial covariances depend only on each trial's own raw
    signal, so they are computed once here (not on every fold call)."""
    sub = f"Sub{sub_n}"
    X_raw = load_raw_epochs(sub_n)
    return {
        "CBCSD": load_flat_feature(CBCSD_PATH_TEMPLATE, sub_n, CBCSD_BEST[sub]["view"], CBCSD_BAND_INDEX),
        "PLV": load_flat_feature(PLV_PATH_TEMPLATE, sub_n, PLV_BEST[sub]["view"], PLVPSD_BAND_INDEX),
        "PSD": load_flat_feature(PSD_PATH_TEMPLATE, sub_n, PSD_BEST[sub]["view"], PLVPSD_BAND_INDEX),
        "raw": X_raw,
        "covriem_C_all": compute_per_trial_covariances(X_raw),
    }


# ============================================================
# Per-fold fit/predict -- unified across the 4 modalities
# ============================================================

def fit_predict_flat(cfg, X, y, train_idx, test_idx, need_train_pred):
    pipe = build_flat_pipeline(cfg["selector"], cfg["classifier"], cfg["k"])
    pipe.fit(X[train_idx], y[train_idx])
    test_pred = pipe.predict(X[test_idx])
    train_pred = pipe.predict(X[train_idx]) if need_train_pred else None
    return test_pred, train_pred


def fit_predict_covriem(cfg, C_all, y, train_idx, test_idx, need_train_pred):
    """C_all: precomputed per-trial covariances for this subject (see load_subject_data).
    Only the Riemannian mean and tangent projection are recomputed per fold."""
    n_ch = C_all.shape[1]
    tri_idx = np.triu_indices(n_ch)

    mean_C_fold = riemannian_mean(C_all[train_idx])
    feat_train = tangent_project(C_all[train_idx], mean_C_fold, tri_idx)
    feat_test = tangent_project(C_all[test_idx], mean_C_fold, tri_idx)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(feat_train)
    Xte = scaler.transform(feat_test)

    scores = covriem_selector_scores(cfg["selector"], Xtr, y[train_idx])
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
    sel = order[-cfg["k"]:]

    clf = build_classifier(cfg["classifier"])
    clf.fit(Xtr[:, sel], y[train_idx])
    test_pred = clf.predict(Xte[:, sel])
    train_pred = clf.predict(Xtr[:, sel]) if need_train_pred else None
    return test_pred, train_pred


def evaluate_fold_all_modalities(sub, X_dict, y, train_idx, test_idx, need_train_pred=False):
    """Fit+predict all 4 modalities for one CV fold.
    Returns dict name -> (test_pred, train_pred_or_None)."""
    out = {}
    cfg = CBCSD_BEST[sub]
    out["CBCSD"] = fit_predict_flat(cfg, X_dict["CBCSD"], y, train_idx, test_idx, need_train_pred)

    cfg = PLV_BEST[sub]
    out["PLV"] = fit_predict_flat(cfg, X_dict["PLV"], y, train_idx, test_idx, need_train_pred)

    cfg = PSD_BEST[sub]
    out["PSD"] = fit_predict_flat(cfg, X_dict["PSD"], y, train_idx, test_idx, need_train_pred)

    cfg = COVRIEM_BEST[sub]
    out["CovRiem"] = fit_predict_covriem(cfg, X_dict["covriem_C_all"], y, train_idx, test_idx, need_train_pred)

    return out


def hard_vote_from_preds(preds_by_modality: dict, key_index: int = 0):
    """key_index: 0 = test predictions, 1 = train predictions.
    Tie-break identical to the original ensemble script: np.unique returns
    sorted labels, np.argmax(counts) keeps the FIRST max -> ties go to the
    smaller class label."""
    names = list(preds_by_modality.keys())
    stack = np.stack([preds_by_modality[n][key_index] for n in names], axis=1)
    n = stack.shape[0]
    out = np.empty(n, dtype=stack.dtype)
    for i in range(n):
        vals, counts = np.unique(stack[i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


# ============================================================
# 5-fold CV: fold-level mean +/- std, per subject
# ============================================================

def evaluate_subject_kfold(sub_n, X_dict, y, n_splits=KFOLD_SPLITS, random_state=RANDOM_STATE):
    sub = f"Sub{sub_n}"
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accs = {name: [] for name in MODALITIES + ["Ensemble"]}

    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        preds = evaluate_fold_all_modalities(sub, X_dict, y, train_idx, test_idx, need_train_pred=False)
        y_test = y[test_idx]
        for name, (test_pred, _) in preds.items():
            fold_accs[name].append(accuracy_score(y_test, test_pred))
        ens_pred = hard_vote_from_preds(preds, key_index=0)
        fold_accs["Ensemble"].append(accuracy_score(y_test, ens_pred))

    # mean/std ACROSS THE 5 FOLDS, per subject
    return {name: (float(np.mean(v) * 100), float(np.std(v) * 100)) for name, v in fold_accs.items()}


def evaluate_subject_kfold_full(sub_n, X_dict, y, n_splits=KFOLD_SPLITS, random_state=RANDOM_STATE):
    """Single 5-fold pass that returns test AND train fold-level stats, plus
    the pooled ensemble OOF correct/total counts (reused by the binomial
    test so we don't run a second identical CV loop)."""
    sub = f"Sub{sub_n}"
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_fold_accs = {name: [] for name in ALL_COLS}
    train_fold_accs = {name: [] for name in ALL_COLS}
    total_correct = 0
    total_samples = 0

    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        preds = evaluate_fold_all_modalities(sub, X_dict, y, train_idx, test_idx, need_train_pred=True)
        y_test, y_train = y[test_idx], y[train_idx]

        for name, (test_pred, train_pred) in preds.items():
            test_fold_accs[name].append(accuracy_score(y_test, test_pred))
            train_fold_accs[name].append(accuracy_score(y_train, train_pred))

        ens_test_pred = hard_vote_from_preds(preds, key_index=0)
        ens_train_pred = hard_vote_from_preds(preds, key_index=1)
        test_fold_accs["Ensemble"].append(accuracy_score(y_test, ens_test_pred))
        train_fold_accs["Ensemble"].append(accuracy_score(y_train, ens_train_pred))

        total_correct += int(np.sum(ens_test_pred == y_test))
        total_samples += len(y_test)

    test_stats = {name: (float(np.mean(v) * 100), float(np.std(v) * 100)) for name, v in test_fold_accs.items()}
    train_stats = {name: (float(np.mean(v) * 100), float(np.std(v) * 100)) for name, v in train_fold_accs.items()}

    return test_stats, train_stats, total_correct, total_samples


# ============================================================
# LOO CV: pooled test accuracy (single number/subject) +
# training accuracy averaged across all LOO folds
# ============================================================

def evaluate_subject_loo(sub_n, X_dict, y):
    sub = f"Sub{sub_n}"
    n = len(y)
    loo = LeaveOneOut()

    test_preds = {name: np.empty(n, dtype=y.dtype) for name in MODALITIES}
    ensemble_test_pred = np.empty(n, dtype=y.dtype)
    train_acc_per_fold = {name: [] for name in MODALITIES + ["Ensemble"]}

    for train_idx, test_idx in loo.split(np.zeros(n)):
        preds = evaluate_fold_all_modalities(sub, X_dict, y, train_idx, test_idx, need_train_pred=True)
        for name, (test_pred, train_pred) in preds.items():
            test_preds[name][test_idx] = test_pred
            train_acc_per_fold[name].append(accuracy_score(y[train_idx], train_pred))

        ens_test = hard_vote_from_preds(preds, key_index=0)
        ensemble_test_pred[test_idx] = ens_test
        ens_train = hard_vote_from_preds(preds, key_index=1)
        train_acc_per_fold["Ensemble"].append(accuracy_score(y[train_idx], ens_train))

    test_acc = {name: float(accuracy_score(y, test_preds[name]) * 100) for name in test_preds}
    test_acc["Ensemble"] = float(accuracy_score(y, ensemble_test_pred) * 100)

    # avg train-fold accuracy across all n LOO folds, per requested definition
    train_acc = {name: float(np.mean(v) * 100) for name, v in train_acc_per_fold.items()}

    return test_acc, train_acc


# ============================================================
# Permutation test on ENSEMBLE accuracy (5-fold protocol)
#
# LOO permutation testing would require n_trials folds x n_permutations
# full refits of all 4 modalities per subject, which is computationally
# intractable at this dataset's trial count. The 5-fold protocol is used
# instead, consistent with standard practice.
# ============================================================

def _ensemble_kfold_acc(sub, X_dict, y_perm, n_splits, random_state):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in cv.split(np.zeros(len(y_perm)), y_perm):
        preds = evaluate_fold_all_modalities(sub, X_dict, y_perm, train_idx, test_idx, need_train_pred=False)
        ens_pred = hard_vote_from_preds(preds, key_index=0)
        accs.append(accuracy_score(y_perm[test_idx], ens_pred))
    return float(np.mean(accs))


def permutation_test_ensemble(sub_n, X_dict, y, n_permutations=N_PERMUTATIONS,
                               n_splits=KFOLD_SPLITS, random_state=RANDOM_STATE,
                               n_jobs=-1, verbose_interval=30):
    sub = f"Sub{sub_n}"
    rng = np.random.RandomState(random_state)

    observed = _ensemble_kfold_acc(sub, X_dict, y, n_splits, random_state)

    perm_ys = [rng.permutation(y) for _ in range(n_permutations)]

    t0 = time.time()
    null_scores = Parallel(n_jobs=n_jobs)(
        delayed(_ensemble_kfold_acc)(sub, X_dict, y_perm, n_splits, random_state)
        for y_perm in perm_ys
    )
    null_scores = np.array(null_scores)
    elapsed = time.time() - t0

    n_exceed = int(np.sum(null_scores >= observed))
    # same +1/+1 correction sklearn's permutation_test_score uses
    p_value = (n_exceed + 1) / (n_permutations + 1)

    return {
        "subject": sub,
        "observed_ensemble_accuracy_%": observed * 100,
        "null_mean_%": float(null_scores.mean() * 100),
        "null_std_%": float(null_scores.std() * 100),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_permutations_exceeding_observed": n_exceed,
        "significant_at_alpha_0.05": bool(p_value < 0.05),
        "elapsed_seconds": elapsed,
    }


# ============================================================
# Main
# ============================================================

def main():
    """Full run: 5-fold CV (test + train), LOO CV (test + train), binomial
    test, and permutation test, for all 10 subjects."""
    kfold_rows, loo_rows, perm_rows, binom_rows = [], [], [], []

    for sub_n in range(1, 11):
        sub = f"Sub{sub_n}"
        print(f"[{sub}] loading data ...")
        X_dict = load_subject_data(sub_n)
        y = load_labels(sub_n)

        print(f"[{sub}] 5-fold CV (test + train) + binomial test ...")
        test_stats, train_stats, total_correct, total_samples = evaluate_subject_kfold_full(sub_n, X_dict, y)
        row = {"subject": sub}
        for name in ALL_COLS:
            m, s = test_stats[name]
            row[f"{name}_mean_%"] = m
            row[f"{name}_std_%"] = s
        m, s = train_stats["Ensemble"]
        row["Ensemble_train_mean_%"] = m
        row["Ensemble_train_std_%"] = s
        kfold_rows.append(row)

        result = binomtest(k=total_correct, n=total_samples, p=CHANCE_LEVEL, alternative="greater")
        ci = result.proportion_ci(confidence_level=0.95)
        binom_rows.append({
            "subject": sub,
            "pooled_accuracy_%": 100.0 * total_correct / total_samples,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "binom_p_value": result.pvalue,
            "significant_at_alpha_0.05": bool(result.pvalue < 0.05),
            "ci_low_%": ci.low * 100,
            "ci_high_%": ci.high * 100,
        })

        print(f"[{sub}] LOO CV ...")
        test_acc, train_acc = evaluate_subject_loo(sub_n, X_dict, y)
        row = {"subject": sub}
        for name in ALL_COLS:
            row[f"{name}_test_%"] = test_acc[name]
            row[f"{name}_train_%"] = train_acc[name]
        loo_rows.append(row)

        print(f"[{sub}] permutation test (this is the slow part) ...")
        perm_result = permutation_test_ensemble(sub_n, X_dict, y)
        perm_rows.append(perm_result)
        print(f"    observed={perm_result['observed_ensemble_accuracy_%']:.2f}%  "
              f"null={perm_result['null_mean_%']:.2f}+/-{perm_result['null_std_%']:.2f}%  "
              f"p={perm_result['p_value']:.4f}  ({perm_result['elapsed_seconds']:.0f}s)")

    os.makedirs(SAVE_DIR, exist_ok=True)
    pd.DataFrame(kfold_rows).to_csv(os.path.join(SAVE_DIR, "Ensemble_KFold_AllSubjects.csv"), index=False)
    pd.DataFrame(loo_rows).to_csv(os.path.join(SAVE_DIR, "Ensemble_LOO_AllSubjects.csv"), index=False)
    pd.DataFrame(perm_rows).to_csv(os.path.join(SAVE_DIR, "Ensemble_Permutation_AllSubjects.csv"), index=False)
    pd.DataFrame(binom_rows).to_csv(os.path.join(SAVE_DIR, "Ensemble_Binomial_AllSubjects.csv"), index=False)

    print(f"\nSaved results to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
