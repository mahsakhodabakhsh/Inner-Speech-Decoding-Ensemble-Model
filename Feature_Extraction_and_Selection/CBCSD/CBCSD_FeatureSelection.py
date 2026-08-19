"""
CBCSD Feature Selection and Model Sweep — all subjects.

Two-stage pipeline per subject:
  STAGE 1: for every (view, selector, classifier, k) combination, run 5-fold stratified
           CV -> composite(test accuracy, train/test gap) -> keep the top
           10 configurations.
  STAGE 2: for those top 10 configurations, run Leave-One-Out CV. Rows are
           ranked automatically by composite(5-fold CV accuracy, LOO
           accuracy); mean_train_accuracy and train_test_gap are also kept
           in the output for manual inspection when choosing the final
           per-subject configuration (some subjects require trading a
           small amount of CV/LOO accuracy for a smaller train/test gap).

Views built from CBCSD's (n_trials, 3, 406) shape:
  "theta", "alpha", "beta"   -> each (n_trials, 406)
  "all_bands_flat"           -> (n_trials, 3*406), all bands concatenated

Feature selection (ANOVA F-test / mutual information) and scaling are fit
on the training fold only, inside every CV split, to avoid data leakage.

Outputs (per subject, in CSV_Results_CBCSD/):
  CBCSD_Sub{sid}_AllConfigurations.csv  -> every (view, selector, classifier, k) result
  CBCSD_Sub{sid}_Top10_LOO.csv          -> final top-10 configurations with LOO accuracy
"""

import os
import time
import traceback
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
SUBJECT_IDS = list(range(1, 11))
FEATURES_PATH_TEMPLATE = "features/CBCSD_Sub{sid}.npy"
Y_PATH_TEMPLATE = "data/processed/y{sid}_InnerSpeech_Task.npy"

BAND_NAMES = ["theta", "alpha", "beta"]  

SAVE_DIR = "CSV_Results_CBCSD"
os.makedirs(SAVE_DIR, exist_ok=True)

FORCE_RERUN = False  # set True to ignore checkpoints and redo every subject
RANDOM_STATE = 42
N_SPLITS = 5
KNN_N_NEIGHBORS = 5
CANDIDATE_K_FULL = list(range(5, 101))
ACC_WEIGHT_STAGE1 = 0.9
GAP_WEIGHT_STAGE1 = 0.1
CV_WEIGHT_STAGE2 = 0.4
LOO_WEIGHT_STAGE2 = 0.6
TOP_N_STAGE1 = 10

def _mi_scores(X_train, y_train):
    return mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)


def _anova_scores(X_train, y_train):
    f_vals, _ = f_classif(X_train, y_train)
    return f_vals


selectors = {"mutual_info": _mi_scores, "anova_f": _anova_scores}

classifiers = {
    "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
    "svm_linear": lambda: SVC(kernel="linear", C=1.0),
    "lda": lambda: LinearDiscriminantAnalysis(),
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "knn": lambda: KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS),
}


def topk_indices_from_order(order, k):
    return order[-k:]


def minmax_norm(s):
    s = np.asarray(s, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def build_views(CBCSD):
    """CBCSD: (n_trials, n_bands, n_pairs). Returns per-band 2D views plus one concatenated view."""
    n_trials, n_bands, n_pairs = CBCSD.shape
    views = {}
    for bi, name in enumerate(BAND_NAMES[:n_bands]):
        views[name] = CBCSD[:, bi, :]
    views["all_bands_flat"] = CBCSD.reshape(n_trials, -1)
    return views


def valid_k_for(X_view, candidate_k=CANDIDATE_K_FULL):
    return [k for k in candidate_k if k <= X_view.shape[1]]


# =========================================================
# STAGE 1 — full sweep, 5-fold CV
# =========================================================
def run_stage1(views, y, subject_name):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(y)), y))

    all_results = []
    for view_name, X_view in views.items():
        view_valid_k = valid_k_for(X_view)

        for selector_name, selector_fn in selectors.items():
            # Scale + score features once per fold (fit on train only), reused across all k and classifiers.
            prepared_folds = []
            for train_idx, test_idx in folds:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_view[train_idx])
                X_te = scaler.transform(X_view[test_idx])
                scores = selector_fn(X_tr, y[train_idx])
                order = np.argsort(np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
                prepared_folds.append((X_tr, X_te, y[train_idx], y[test_idx], order))

            for clf_name, clf_factory in classifiers.items():
                for k in view_valid_k:
                    test_accs, train_accs = [], []
                    for X_tr, X_te, y_tr, y_te, order in prepared_folds:
                        sel = topk_indices_from_order(order, k)
                        try:
                            clf = clf_factory()
                            clf.fit(X_tr[:, sel], y_tr)
                            test_accs.append(accuracy_score(y_te, clf.predict(X_te[:, sel])) * 100.0)
                            train_accs.append(accuracy_score(y_tr, clf.predict(X_tr[:, sel])) * 100.0)
                        except Exception:
                            test_accs.append(np.nan)
                            train_accs.append(np.nan)

                    mean_test = float(np.nanmean(test_accs))
                    mean_train = float(np.nanmean(train_accs))
                    all_results.append({
                        "subject": subject_name, "view": view_name,
                        "selector": selector_name,
                        "classifier": clf_name, "k_features": k,
                        "mean_test_accuracy": mean_test,
                        "std_test_accuracy": float(np.nanstd(test_accs)),
                        "mean_train_accuracy": mean_train,
                        "train_test_gap": mean_train - mean_test,
                    })

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(SAVE_DIR, f"CBCSD_{subject_name}_AllConfigurations.csv"), index=False)

    df_all["norm_acc"] = minmax_norm(df_all["mean_test_accuracy"])
    df_all["norm_gap"] = minmax_norm(df_all["train_test_gap"])
    df_all["composite_stage1"] = (
        ACC_WEIGHT_STAGE1 * df_all["norm_acc"] + GAP_WEIGHT_STAGE1 * (1 - df_all["norm_gap"])
    )
    df_ranked = df_all.sort_values("composite_stage1", ascending=False).reset_index(drop=True)

    return df_ranked.head(TOP_N_STAGE1).copy()


# =========================================================
# STAGE 2 — Leave-One-Out on the Stage-1 top 10
# =========================================================
def run_stage2(top10, views, y, subject_name):
    loo = LeaveOneOut()
    loo_splits = list(loo.split(np.zeros(len(y))))
    n_trials = len(y)

    stage2_rows = []
    for _, row in top10.iterrows():
        X_view = views[row["view"]]
        selector_fn = selectors[row["selector"]]
        clf_factory = classifiers[row["classifier"]]
        k = int(row["k_features"])

        correct = 0
        for train_idx, test_idx in loo_splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_view[train_idx])
            X_te = scaler.transform(X_view[test_idx])
            scores = selector_fn(X_tr, y[train_idx])
            order = np.argsort(np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
            sel = topk_indices_from_order(order, k)
            clf = clf_factory()
            clf.fit(X_tr[:, sel], y[train_idx])
            pred = clf.predict(X_te[:, sel])
            correct += int(pred[0] == y[test_idx][0])

        loo_acc = 100.0 * correct / n_trials
        stage2_rows.append({**row.to_dict(), "loo_accuracy": loo_acc})

    df_stage2 = pd.DataFrame(stage2_rows)
    df_stage2["norm_cv_acc"] = minmax_norm(df_stage2["mean_test_accuracy"])
    df_stage2["norm_loo_acc"] = minmax_norm(df_stage2["loo_accuracy"])
    df_stage2["composite_stage2"] = (
        CV_WEIGHT_STAGE2 * df_stage2["norm_cv_acc"] + LOO_WEIGHT_STAGE2 * df_stage2["norm_loo_acc"]
    )
    df_stage2 = df_stage2.sort_values("composite_stage2", ascending=False).reset_index(drop=True)
    df_stage2.to_csv(os.path.join(SAVE_DIR, f"CBCSD_{subject_name}_Top10_LOO.csv"), index=False)
    return df_stage2


# =========================================================
# DRIVER — loop over all subjects, with checkpointing
# =========================================================
def subject_already_done(subject_name):
    path = os.path.join(SAVE_DIR, f"CBCSD_{subject_name}_Top10_LOO.csv")
    return os.path.exists(path) and not FORCE_RERUN


if __name__ == "__main__":
    summary_path = os.path.join(SAVE_DIR, "CBCSD_all_subjects_summary.csv")
    summary_rows = []
    if os.path.exists(summary_path) and not FORCE_RERUN:
        summary_rows = pd.read_csv(summary_path).to_dict("records")

    t_batch_start = time.time()

    for sid in SUBJECT_IDS:
        subject_name = f"Sub{sid}"

        if subject_already_done(subject_name):
            print(f"[SKIP] {subject_name} already has Stage-2 output — checkpoint hit.")
            continue

        print("\n" + "#" * 60)
        print(f"# SUBJECT {sid}")
        print("#" * 60)

        try:
            t0 = time.time()
            CBCSD = np.load(FEATURES_PATH_TEMPLATE.format(sid=sid))
            y = np.load(Y_PATH_TEMPLATE.format(sid=sid))

            assert CBCSD.shape[0] == y.shape[0], "trial count mismatch between CBCSD and y"
            assert not np.isnan(CBCSD).any(), "CBCSD contains NaN"
            assert not np.isinf(CBCSD).any(), "CBCSD contains Inf"
            assert CBCSD.ndim == 3, f"expected 3D CBCSD features (n_trials, n_bands, n_pairs), got ndim={CBCSD.ndim}"

            views = build_views(CBCSD)
            print(f"  CBCSD shape={CBCSD.shape}, y shape={y.shape}, views={ {n: v.shape for n, v in views.items()} }")

            print("  -> stage 1 (coarse sweep, per-view)...", flush=True)
            top10 = run_stage1(views, y, subject_name)

            print("  -> stage 2 (LOO on top 10)...", flush=True)
            df_final = run_stage2(top10, views, y, subject_name)

            best = df_final.iloc[0]
            elapsed = time.time() - t0
            print(f"  {subject_name} done in {elapsed:.1f}s. "
                  f"Best: {best['view']}/{best['selector']}/{best['classifier']} "
                  f"k={int(best['k_features'])} loo_acc={best['loo_accuracy']:.2f}%")

            summary_rows.append({
                "subject": subject_name,
                "best_view": best["view"],
                "best_selector": best["selector"],
                "best_classifier": best["classifier"],
                "best_k": int(best["k_features"]),
                "cv_test_accuracy": best["mean_test_accuracy"],
                "mean_train_accuracy": best["mean_train_accuracy"],
                "train_test_gap": best["train_test_gap"],
                "loo_accuracy": best["loo_accuracy"],
                "elapsed_seconds": elapsed,
                "status": "ok",
            })

        except Exception as e:
            print(f"  [ERROR] {subject_name} failed: {e}")
            traceback.print_exc()
            summary_rows.append({"subject": subject_name, "status": f"FAILED: {e}"})

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nBatch finished in {(time.time() - t_batch_start) / 60:.1f} minutes.")
    print(f"Summary: {summary_path}")
