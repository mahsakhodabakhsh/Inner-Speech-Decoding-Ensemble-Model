# Inner-Speech-Decoding-Ensemble-Model

Ensemble EEG classification for inner speech Brain-Computer Interfaces
(BCI), combining four complementary feature extraction methods —
**CBCSD**, **PLV**, **PSD**, and **CovRiem** (covariance-based Riemannian
tangent space) — on the **Thinking Out Loud** dataset (Nieto et al., 2022).

## Overview

Decoding imagined (inner) speech directly from EEG is a promising but
difficult direction for BCI, since class-discriminative information can
show up in different aspects of the signal — spectral power, phase
coupling, connectivity, and signal geometry — depending on the subject.
This project extracts four complementary feature types from each subject's
EEG, selects the best-performing model configuration per subject through a
stability-based two-stage cross-validation procedure, and combines the
four modalities' predictions through hard-voting ensemble classification.

Each of the four methods captures a different property of the signal:
CBCSD is a connectivity measure that captures both linear and nonlinear
similarity between each pair of channels, PLV measures phase
synchronization between channel pairs, PSD extracts power-spectrum
information from each channel individually, and CovRiem captures the
second-order statistical relationship between channels through covariance
geometry mapped to a Riemannian tangent space. Combining these
complementary views through the ensemble allows the model to draw on more
of the information present in the EEG than any single feature extraction
method alone.

## Dataset

This project uses the **Thinking Out Loud** dataset:

> N. Nieto et al., *"Thinking out loud, an open-access EEG-based BCI
> dataset for inner speech recognition,"* Scientific Data, 2022.
> Dataset repository: https://github.com/N-Nieto/Inner_Speech_Dataset

10 healthy right-handed subjects performed inner speech imagination of four
Spanish command words (Arriba/Up, Abajo/Down, Izquierda/Left,
Derecha/Right). This project uses 29 of the original 128 EEG channels
(mapped to the standard 10-20 system) and the 2.5 s inner-speech action
interval of each trial.

## Repository structure

```
Inner-Speech-Decoding-Ensemble-Model/
├── preprocessing/
│   └── ChannelSelection_TimeIntervalSelection.ipynb # 29-channel mapping + 2.5s action-window cropping
├── feature_extraction_and_selection/
│   ├── README.md              
│   ├── cbcsd/
│   │   ├── CBCSD_FeatureExtraction.py
│   │   └── CBCSD_FeatureSelection.py
│   ├── plv/
│   │   ├── PLV_FeatureExtraction.py
│   │   └── PLV_FeatureSelection.py
│   ├── psd/
│   │   ├── PSD_FeatureExtraction.py
│   │   └── PSD_FeatureSelection.py
│   └── covriem/
│       └── CovRiem_FeatureExtractionSelection.py
└── ensemble/
    └── Ensemble_Evaluation.py
```

## Pipeline

1. **Preprocessing** (`preprocessing/`) — filtering, re-referencing, and
   epoching must first be run using Nieto et al.'s original preprocessing
   pipeline (see dataset repository linked above). The `ChannelSelection_TimeIntervalSelection.ipynb`
   notebook in this folder then takes that pipeline's output and performs
   29-channel selection and cropping to the 2.5 s inner-speech action
   window. Outputs `X{sid}_InnerSpeech_Task.npy` / `y{sid}_InnerSpeech_Task.npy`
   per subject.
2. **Feature extraction and selection** (`feature_extraction_and_selection/`)
   — one subfolder per modality. Each extracts its features from the
   preprocessed data, then sweeps feature-selection method (ANOVA F-test /
   mutual information), classifier (SVM-RBF, SVM-linear, LDA, Logistic
   Regression, KNN), and number of selected features (k = 5 to 100) for
   every subject. Model selection is a two-stage process: 5-fold stratified
   cross-validation narrows every configuration down to the top 10, which
   are then re-evaluated with Leave-One-Out cross-validation. The final
   per-subject configuration is chosen using 5-fold CV accuracy, LOO
   accuracy, and the train/test accuracy gap together, not by peak accuracy
   alone. See `feature_extraction_and_selection/README.md` for why CovRiem's
   extraction and selection cannot be split into two separate scripts like
   the other three modalities.
3. **Ensemble evaluation** (`ensemble/`) — for each subject, the four
   modalities' final selected configurations are combined via hard
   (majority) voting. Evaluated with 5-fold CV, LOO CV, a label-permutation
   test, and a one-sided binomial test against chance level.

## Requirements

```
numpy
pandas
scipy
scikit-learn
mne
joblib
```

## Feature scaling

`StandardScaler` is applied throughout this project and is not swept as a
hyperparameter.


## License

Released under the MIT License — see `LICENSE`.
