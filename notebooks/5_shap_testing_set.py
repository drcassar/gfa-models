#!/usr/bin/env python

import json
import os
import re
from collections import Counter
from pprint import pprint

# fmt:off
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             average_precision_score, balanced_accuracy_score,
                             brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
# fmt:on

RANDOM_SEED = 42

CLASSES_LABEL_BINARY = ["Crystal", "Glass"]

CHEM = True
FEATENG = False


def compute_shap_values(
    X_df,
    X_background,
    model,
    classes_label_binary,
    filename,
):

    explainer = shap.Explainer(
        model.predict_proba,
        masker=X_background,
        feature_names=X_df.columns.tolist(),
        output_names=classes_label_binary,
        seed=RANDOM_SEED,
    )

    explanation = explainer(X_df.values)

    os.makedirs("explanation_shap", exist_ok=True)
    joblib.dump(explanation, filename, compress=3)
    print("Explanation object saved successfully!")

    return explanation


###############################################################################
#                                  CHEM data                                  #
###############################################################################

if CHEM:
    df = pd.read_pickle(
        "./../data/processed/GFA_binary_CHEM.pkl.zip", compression="zip"
    )
    df = df.drop("Kod", axis=1)

    TARGET = ["GF"]
    FEATURES = df.columns.drop(TARGET)

    le = LabelEncoder()
    df["GF"] = le.fit_transform(df["GF"])

    train_indices = np.load("./../data/support/train_indices.npy")
    test_indices = np.load("./../data/support/test_indices.npy")
    test_data_leak_indices = np.load("./../data/support/test_data_leak_indices.npy")

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    test_df = test_df.loc[test_data_leak_indices]

    X_train = train_df.reindex(FEATURES, axis=1)
    y_train = train_df.reindex(TARGET, axis=1)

    X_test = test_df.reindex(FEATURES, axis=1)
    y_test = test_df.reindex(TARGET, axis=1)

    X = df.reindex(FEATURES, axis=1)
    y = df.reindex(TARGET, axis=1)

    str = f"./../src/hyperparameters_tunning/GFA_binary_CHEM/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

    model = joblib.load(str)

    filename = f"explanation_shap/explanation_shap_CHEM.joblib"

    try:
        explanation = joblib.load(filename)

    except FileNotFoundError:
        explanation = compute_shap_values(
            X_test,
            shap.kmeans(X_train, 100).data,
            model,
            CLASSES_LABEL_BINARY,
            filename,
        )


###############################################################################
#                                   FEATENG                                   #
###############################################################################

if FEATENG:
    df = pd.read_pickle(
        "./../data/processed/GFA_binary_FEATENG.pkl.zip", compression="zip"
    )
    df = df.drop("Kod", axis=1)

    TARGET = ["GF"]
    FEATURES = df.columns.drop(TARGET)

    le = LabelEncoder()
    df["GF"] = le.fit_transform(df["GF"])

    train_indices = np.load("./../data/support/train_indices.npy")
    test_indices = np.load("./../data/support/test_indices.npy")
    test_data_leak_indices = np.load("./../data/support/test_data_leak_indices.npy")

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    test_df = test_df.loc[test_data_leak_indices]

    X_train = train_df.reindex(FEATURES, axis=1)
    y_train = train_df.reindex(TARGET, axis=1)

    X_test = test_df.reindex(FEATURES, axis=1)
    y_test = test_df.reindex(TARGET, axis=1)

    X = df.reindex(FEATURES, axis=1)
    y = df.reindex(TARGET, axis=1)

    str = f"./../src/hyperparameters_tunning/GFA_binary_FEATENG/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

    str = f"./../src/hyperparameters_tunning/GFA_binary_FEATENG/model/calibrated_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

    model = joblib.load(str)

    filename = f"explanation_shap/explanation_shap_FEATENG.joblib"

    try:
        explanation = joblib.load(filename)

    except FileNotFoundError:
        explanation = compute_shap_values(
            X_test,
            shap.kmeans(X_train, 100).data,
            model,
            CLASSES_LABEL_BINARY,
            filename,
        )
