# Feature Extraction and Selection

This folder contains the feature extraction and feature selection code for
all four modalities used in this study: **CBCSD**, **PLV**, **PSD**, and
**CovRiem**.

## Why the four feature extraction methods are used together

Each of the four methods captures a different aspect of the EEG signal.
CBCSD is a connectivity measure that captures both linear and nonlinear
similarity between each pair of channels. PLV measures phase
synchronization between channel pairs, which is informative for
distinguishing inner-speech classes. PSD extracts power-spectrum
information from each individual channel, separately from any
inter-channel relationship. CovRiem captures the second-order statistical
relationship between every pair of channels through their covariance
structure, mapped to a Riemannian tangent space for better class
separability. Because these methods extract complementary properties of
the signal — spectral power, phase coupling, linear/nonlinear connectivity,
and covariance geometry — combining them through the ensemble voting
scheme allows the overall model to draw on more of the information present
in the EEG than any single feature extraction method alone.

## Folder structure

```
feature_extraction_and_selection/
├── cbcsd/
│   ├── CBCSD_FeatureExtraction.py
│   └── CBCSD_FeatureSelection.py
├── plv/
│   ├── PLV_FeatureExtraction.py
│   └── PLV_FeatureSelection.py
├── psd/
│   ├── PSD_FeatureExtraction.py
│   └── PSD_FeatureSelection.py
└── covriem/
    └── CovRiem_FeatureExtractionSelection.py
```

## Why CBCSD, PLV, and PSD each use two separate scripts, but CovRiem does not

For CBCSD, PLV, and PSD, feature extraction is a fixed, per-trial
transformation of the raw signal that does not depend on any other trial.
It is therefore computed once for the whole dataset and saved to disk
(`FeatureExtraction.py`); feature selection (ANOVA F-test / mutual
information) is then fit on the training fold only, inside each CV split,
using those precomputed features (`FeatureSelection.py`).

CovRiem cannot be split the same way. Its feature is not a fixed transform
of a single trial — it is each trial's covariance matrix **projected into
the tangent space at the training set's Riemannian mean**:

```
tangent_feature(trial) = log( mean_C^(-1/2) · cov(trial) · mean_C^(-1/2) )
```

`mean_C` (the Riemannian/Fréchet mean of the covariance matrices) is itself
estimated from data and must be computed from the **training trials only**.
If it were computed once from the entire dataset and reused across every CV
fold, every "feature" extracted for a supposedly held-out test trial would
already contain information from the test set baked into `mean_C` — data
leakage, even though no label information is involved. This means
extraction (the tangent-space projection) and selection cannot be staged as
two independent scripts; both must happen *inside* every CV/LOO split,
using only that split's training data to define `mean_C`.

What *is* safe to precompute is the **per-trial covariance matrix** itself
(before tangent-space projection), since it depends only on that one
trial's own raw signal. `CovRiem_FeatureExtractionSelection.py` computes
this once per subject and reuses it across every fold; only the Riemannian
mean estimation and tangent projection are repeated inside each fold. For
this reason, no fixed `CovRiem_Sub{id}.npy` feature file is produced —
unlike the other three modalities, there is no single, fold-independent
feature array that could be saved and reused.

## Shared conventions across all four modalities

- Feature scaling is fixed to `StandardScaler` throughout; it is not swept
  as a hyperparameter.
- Feature selection: ANOVA F-test and Mutual Information, with the number
  of selected features (`k`) swept from 5 to 100.
- Classifiers: linear and RBF SVM, KNN, Logistic Regression, and LDA.
- Model selection is a two-stage pipeline per subject: 5-fold stratified
  cross-validation (Stage 1) narrows every configuration down to the top
  10, which are then re-evaluated with Leave-One-Out cross-validation
  (Stage 2). The final configuration is chosen using 5-fold CV accuracy,
  LOO accuracy, and the train/test accuracy gap together, not by peak
  accuracy alone.
- Output naming: `{METHOD}_Sub{id}_AllConfigurations.csv` (every tested
  configuration) and `{METHOD}_Sub{id}_Top10_LOO.csv` (final top 10 with
  LOO accuracy), per subject.
