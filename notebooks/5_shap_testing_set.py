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

N_JOBS = 2
RANDOM_SEED = 42

CLASSES_LABEL_BINARY = ["Crystal", "Glass"]

CHEM = False
FEATENG = True


def compute_shap_values(
    X_df,
    X_background,
    model,
    classes_label_binary,
    filename,
    n_jobs=N_JOBS,
):

    explainer = shap.Explainer(
        model.predict_proba,
        masker=X_background,
        feature_names=X_df.columns.tolist(),
        output_names=classes_label_binary,
        seed=RANDOM_SEED,
    )

    n_chunks = n_jobs * 20
    chunks = np.array_split(X_df.values, n_chunks)

    print(f"Splitting data into {n_chunks} chunks to run on {n_jobs} cores...")

    def get_explanation(chunk):
        # check_additivity=False prevents common numerical precision errors
        # that can occur when parallelizing Random Forest calculations
        return explainer(chunk, check_additivity=False)

    # Run in parallel with TQDM
    print("Calculating SHAP values...")

    explanations_list = Parallel(n_jobs=n_jobs)(
        delayed(get_explanation)(chunk) for chunk in tqdm(chunks, desc="Progress")
    )

    # Concatenate results
    print("Concatenating results...")
    shap_values_all = np.concatenate([e.values for e in explanations_list], axis=0)
    base_values_all = np.concatenate([e.base_values for e in explanations_list], axis=0)
    data_all = np.concatenate([e.data for e in explanations_list], axis=0)

    # Reconstruct the final Explanation object
    explanation = shap.Explanation(
        values=shap_values_all,
        base_values=base_values_all,
        data=data_all,
        feature_names=X_df.columns.tolist(),
        output_names=classes_label_binary,
    )

    print("Done! Shape:", explanation.values.shape)

    # Save the explanation object to a file
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
            shap.kmeans(X_train, 100),
            model,
            CLASSES_LABEL_BINARY,
            filename,
            N_JOBS,
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
            X_test.sample(10),
            shap.kmeans(X_train, 100),
            model,
            CLASSES_LABEL_BINARY,
            filename,
            N_JOBS,
        )


###############################################################################
#                                     OLD                                     #
###############################################################################


# # ## Constants

# # In[2]:


# RANDOM_SEED = 0
# TEST_SIZE_SPLIT = 0.1


# # ## Functions

# # In[3]:


# # from sklearn.metrics import (
# #     accuracy_score,
# #     roc_auc_score,
# #     average_precision_score,
# #     f1_score,
# #     classification_report,
# #     confusion_matrix,
# #     ConfusionMatrixDisplay
# # )
# # import matplotlib.pyplot as plt


# def evaluate_model(model, X_test, y_test, labels_list=None, cm_normalize=None):
#     """
#     Evaluate a binary classification model with multiple metrics and plots.

#     Args:
#         model: Fitted scikit-learn model with predict() and predict_proba().
#         X_test (array-like): Test feature matrix.
#         y_test (array-like): True labels for the test set.
#         labels_list (LabelEncoder, optional): If provided, class names are them;
#             otherwise defaults to ["Class 0", "Class 1, ..."].
#         cm_normalize (str or None): Normalization method for confusion matrix.
#             Options: None, 'true', 'pred', 'all'.
#             Default: None (no normalization).

#     Prints:
#         - Accuracy, Balanced Accuracy
#         - ROC AUC (single value)
#         - PR AUC (class 0, class 1, macro, micro, weighted)
#         - F1 Score (binary, macro, micro, weighted, per class)
#         - Precision (macro, micro, weighted, per class)
#         - Recall (macro, micro, weighted, per class)
#         - Brier Score Loss
#         - Log Loss
#         - Classification report

#     Displays:
#         - Confusion matrix (normalized by predicted labels).
#     """

#     sns.reset_defaults()

#     if labels_list is None:
#         labels = ["Class " + str(i) for i in range(len(set(y_test)))]
#     else:
#         labels = labels_list

#     # Predictions
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)

#     # --- Base metrics ---
#     accuracy = accuracy_score(y_test, y_pred)
#     balanced_acc = balanced_accuracy_score(y_test, y_pred)

#     # --- ROC AUC (single value) ---
#     roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

#     # --- PR AUC variations ---
#     pr_auc_class1 = average_precision_score(y_test, y_pred_proba[:, 1])
#     pr_auc_class0 = average_precision_score(1 - y_test, y_pred_proba[:, 0])
#     pr_auc_macro = np.mean([pr_auc_class0, pr_auc_class1])
#     pr_auc_micro = average_precision_score(y_test, y_pred_proba[:, 1], average="micro")
#     pr_auc_weighted = pr_auc_class0 * np.mean(y_test == 0) + pr_auc_class1 * np.mean(
#         y_test == 1
#     )

#     # --- F1 scores ---
#     f1_binary = f1_score(y_test, y_pred)
#     f1_macro = f1_score(y_test, y_pred, average="macro")
#     f1_micro = f1_score(y_test, y_pred, average="micro")
#     f1_weighted = f1_score(y_test, y_pred, average="weighted")
#     f1_per_class = f1_score(y_test, y_pred, average=None)

#     # --- Precision ---
#     precision_macro = precision_score(y_test, y_pred, average="macro")
#     precision_micro = precision_score(y_test, y_pred, average="micro")
#     precision_weighted = precision_score(y_test, y_pred, average="weighted")
#     precision_per_class = precision_score(y_test, y_pred, average=None)

#     # --- Recall ---
#     recall_macro = recall_score(y_test, y_pred, average="macro")
#     recall_micro = recall_score(y_test, y_pred, average="micro")
#     recall_weighted = recall_score(y_test, y_pred, average="weighted")
#     recall_per_class = recall_score(y_test, y_pred, average=None)

#     # --- Calibration & loss metrics ---
#     brier = brier_score_loss(y_test, y_pred_proba[:, 1])
#     logloss = log_loss(y_test, y_pred_proba)

#     # --- Print metrics ---
#     print("===== Evaluation Metrics =====")
#     print(f"Accuracy:              {accuracy:.4f}")
#     print(f"Balanced Accuracy:     {balanced_acc:.4f}")
#     print(f"ROC AUC:               {roc_auc:.4f}")
#     print("\n--- PR AUC ---")
#     print(f"PR AUC (micro):        {pr_auc_micro:.4f}")
#     print(f"PR AUC (macro):        {pr_auc_macro:.4f}")
#     print(f"PR AUC (weighted):     {pr_auc_weighted:.4f}")
#     print(f"PR AUC (class 0):      {pr_auc_class0:.4f}")
#     print(f"PR AUC (class 1):      {pr_auc_class1:.4f}")
#     print("\n--- F1-Scores ---")
#     print(f"F1-Score (micro):            {f1_micro:.4f}")
#     print(f"F1-Score (macro):            {f1_macro:.4f}")
#     print(f"F1-Score (weighted):         {f1_weighted:.4f}")
#     print(f"F1-Score (class 0):          {f1_per_class[0]:.4f}")
#     print(f"F1-Score (class 1):          {f1_per_class[1]:.4f}")
#     print("\n--- Precision ---")
#     print(f"Precision (micro):     {precision_micro:.4f}")
#     print(f"Precision (macro):     {precision_macro:.4f}")
#     print(f"Precision (weighted):  {precision_weighted:.4f}")
#     print(f"Precision (class 0):   {precision_per_class[0]:.4f}")
#     print(f"Precision (class 1):   {precision_per_class[1]:.4f}")
#     print("\n--- Recall ---")
#     print(f"Recall (micro):        {recall_micro:.4f}")
#     print(f"Recall (macro):        {recall_macro:.4f}")
#     print(f"Recall (weighted):     {recall_weighted:.4f}")
#     print(f"Recall (class 0):      {recall_per_class[0]:.4f}")
#     print(f"Recall (class 1):      {recall_per_class[1]:.4f}")
#     print("\n--- Loss Metrics ---")
#     print(f"Brier Score Loss:      {brier:.4f}")
#     print(f"Log Loss:              {logloss:.4f}\n")

#     # --- Classification report ---
#     print("===== Classification Report =====")
#     print(classification_report(y_test, y_pred, target_names=labels))

#     # --- Confusion Matrix ---
#     fig, ax = plt.subplots(figsize=(10, 8))
#     cm = confusion_matrix(y_test, y_pred, normalize=cm_normalize)

#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)

#     im = ax.images[0]
#     im.set_clim(0, 1 if cm_normalize else cm.max())
#     fig.colorbar(im, ax=ax)

#     plt.title(f"Confusion Matrix")
#     plt.xticks(rotation=45)
#     plt.show()


# # ## Configurations

# # In[4]:


# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.serif": ["Times New Roman", "DejaVu Serif"],
#         "font.size": 12,
#         "axes.labelsize": 14,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "legend.fontsize": 10,
#         "mathtext.fontset": "cm",
#     }
# )


# # ## Machine Learning Models
# #
# # ### Loading Indices

# # In[5]:


# # train_indices = np.load("../data/support/train_indices.npy")
# # test_indices = np.load("../data/support/test_indices.npy")

# df = pd.read_pickle("../data/processed/GFA_binary_processed.pkl.zip", compression="zip")
# df = df.drop("sample_id", axis=1)

# TARGET = ["target"]
# FEATURES = df.columns.drop(TARGET)

# le = LabelEncoder()
# df["target"] = le.fit_transform(df["target"])

# indices = df.index
# train_indices, test_indices = train_test_split(
#     indices, test_size=TEST_SIZE_SPLIT, random_state=RANDOM_SEED, stratify=df[TARGET]
# )


# # ## CHEM

# # In[6]:


# df_chem = pd.read_pickle("../data/processed/GFA_binary_CHEM.pkl.zip", compression="zip")
# df_chem


# # In[7]:


# df_chem = pd.read_pickle("../data/processed/GFA_binary_CHEM.pkl.zip", compression="zip")
# df_chem = df_chem.drop("Kod", axis=1)
# display(df_chem)
# print(df_chem.columns)
# print(Counter(df_chem["GF"]))

# TARGET_name = "GF"
# TARGET = [TARGET_name]
# FEATURES = df_chem.columns.drop(TARGET)

# le = LabelEncoder()
# df_chem[TARGET_name] = le.fit_transform(df_chem[TARGET_name])


# # In[8]:


# train_df_chem = df_chem.loc[train_indices]
# test_df_chem = df_chem.loc[test_indices]

# X_train_chem = train_df_chem.reindex(FEATURES, axis=1).values
# y_train_chem = train_df_chem.reindex(TARGET, axis=1).values.ravel()

# X_test_chem = test_df_chem.reindex(FEATURES, axis=1).values
# y_test_chem = test_df_chem.reindex(TARGET, axis=1).values.ravel()

# X_chem = df_chem.reindex(FEATURES, axis=1).values
# y_chem = df_chem.reindex(TARGET, axis=1).values.ravel()

# optimized_model_path = "../src/hyperparameters_tunning/GFA_binary_CHEM/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"
# optimized_model = joblib.load(optimized_model_path)
# all_hyperparams = optimized_model.get_params()

# print(f"\nFound {len(all_hyperparams)} optimized hyperparameters.")
# print("Listing all parameters:")
# pprint(all_hyperparams)

# # with open('../data/support/optimized_hyperparams_CHEM.json', 'w') as f:
# #     json.dump(all_hyperparams, f, indent=2, default=str)


# # In[9]:


# # RF = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
# RF = RandomForestClassifier(**all_hyperparams)

# RF.fit(X_train_chem, y_train_chem)

# y_pred_chem = RF.predict(X_test_chem)
# y_pred_proba_chem = RF.predict_proba(X_test_chem)

# classes_label_binary = ["Crystal", "Glass"]

# evaluate_model(
#     RF, X_test_chem, y_test_chem, labels_list=le.classes_, cm_normalize="pred"
# )


# # In[10]:


# # ### FEATENG

# # In[11]:


# df_feateng = pd.read_pickle(
#     "../data/processed/GFA_binary_FEATENG.pkl.zip", compression="zip"
# )
# df_feateng = df_feateng.drop("Kod", axis=1)
# display(df_feateng)
# print(df_feateng.columns)
# print(Counter(df_feateng["GF"]))

# TARGET_name = "GF"
# TARGET = [TARGET_name]
# FEATURES = df_feateng.columns.drop(TARGET)

# le = LabelEncoder()
# df_feateng[TARGET_name] = le.fit_transform(df_feateng[TARGET_name])


# # In[12]:


# train_df_feateng = df_feateng.loc[train_indices]
# test_df_feateng = df_feateng.loc[test_indices]

# X_train_feateng = train_df_feateng.reindex(FEATURES, axis=1).values
# y_train_feateng = train_df_feateng.reindex(TARGET, axis=1).values.ravel()

# X_test_feateng = test_df_feateng.reindex(FEATURES, axis=1).values
# y_test_feateng = test_df_feateng.reindex(TARGET, axis=1).values.ravel()

# X_feateng = df_feateng.reindex(FEATURES, axis=1).values
# y_feateng = df_feateng.reindex(TARGET, axis=1).values.ravel()

# optimized_model_path = "../src/hyperparameters_tunning/GFA_binary_FEATENG/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"
# optimized_model = joblib.load(optimized_model_path)
# all_hyperparams = optimized_model.get_params()

# print(f"\nFound {len(all_hyperparams)} optimized hyperparameters.")
# print("Listing all parameters:")
# pprint(all_hyperparams)

# # with open('../data/support/optimized_hyperparams_FEATENG.json', 'w') as f:
# #     json.dump(all_hyperparams, f, indent=2, default=str)


# # In[13]:


# # RF = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
# RF = RandomForestClassifier(**all_hyperparams)

# RF.fit(X_train_feateng, y_train_feateng)

# y_pred_feateng = RF.predict(X_test_feateng)
# y_pred_proba_feateng = RF.predict_proba(X_test_feateng)

# classes_label_binary = ["Crystal", "Glass"]

# evaluate_model(
#     RF, X_test_feateng, y_test_feateng, labels_list=le.classes_, cm_normalize="pred"
# )


# # In[14]:


# # import numpy as np
# # import shap
# # from joblib import Parallel, delayed
# # from tqdm import tqdm

# # n_jobs = 14
# # feat_names = df_feateng[FEATURES].columns.to_list()
# # explainer = shap.Explainer(RF, feature_names=feat_names, output_names=classes_label_binary, seed=RANDOM_SEED)

# # # Split X_train into chunks (if one chunk finishes early, the core immediately picks up the next one).
# # n_chunks = n_jobs * 20
# # chunks = np.array_split(X_train_feateng, n_chunks)

# # print(f"Splitting data into {n_chunks} chunks to run on {n_jobs} cores...")

# # # Helper function
# # def get_explanation(chunk):
# #     # check_additivity=False prevents common numerical precision errors
# #     # that can occur when parallelizing Random Forest calculations
# #     return explainer(chunk, check_additivity=False)

# # # Run in parallel with TQDM
# # print("Calculating SHAP values...")

# # explanations_list = Parallel(n_jobs=n_jobs)(
# #     delayed(get_explanation)(chunk) for chunk in tqdm(chunks, desc="Progress")
# # )

# # # Concatenate results
# # print("Concatenating results...")
# # shap_values_all = np.concatenate([e.values for e in explanations_list], axis=0)
# # base_values_all = np.concatenate([e.base_values for e in explanations_list], axis=0)
# # data_all = np.concatenate([e.data for e in explanations_list], axis=0)

# # # Reconstruct the final Explanation object
# # explanation = shap.Explanation(
# #     values=shap_values_all,
# #     base_values=base_values_all,
# #     data=data_all,
# #     feature_names=feat_names,
# #     output_names=classes_label_binary
# # )

# # print("Done! Shape:", explanation.values.shape)

# # # Save the explanation object to a file
# # joblib.dump(explanation, '../data/support/explanation_shap/explanation_shap_FEATENG.joblib')
# # print("Explanation object saved successfully!")


# # ### GS

# # In[15]:


# df_gs = pd.read_pickle("../data/processed/GFA_binary_GS.pkl.zip", compression="zip")
# df_gs = df_gs.drop("Kod", axis=1)
# display(df_gs)
# print(df_gs.columns)
# print(Counter(df_gs["GF"]))

# TARGET_name = "GF"
# TARGET = [TARGET_name]
# FEATURES = df_gs.columns.drop(TARGET)

# le = LabelEncoder()
# df_gs[TARGET_name] = le.fit_transform(df_gs[TARGET_name])


# # In[16]:


# train_df_gs = df_gs.loc[train_indices]
# test_df_gs = df_gs.loc[test_indices]

# X_train_gs = train_df_gs.reindex(FEATURES, axis=1).values
# y_train_gs = train_df_gs.reindex(TARGET, axis=1).values.ravel()

# X_test_gs = test_df_gs.reindex(FEATURES, axis=1).values
# y_test_gs = test_df_gs.reindex(TARGET, axis=1).values.ravel()

# X_gs = df_gs.reindex(FEATURES, axis=1).values
# y_gs = df_gs.reindex(TARGET, axis=1).values.ravel()

# optimized_model_path = "../src/hyperparameters_tunning/GFA_binary_GS/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"
# optimized_model = joblib.load(optimized_model_path)
# all_hyperparams = optimized_model.get_params()

# print(f"\nFound {len(all_hyperparams)} optimized hyperparameters.")
# print("Listing all parameters:")
# pprint(all_hyperparams)

# # with open('../data/support/optimized_hyperparams_GS.json', 'w') as f:
# #     json.dump(all_hyperparams, f, indent=2, default=str)


# # In[17]:


# missing_rows = test_df_gs[test_df_gs.isnull().any(axis=1)]
# display(missing_rows)


# # In[18]:


# # RF = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
# RF = RandomForestClassifier(**all_hyperparams)

# RF.fit(X_train_gs, y_train_gs)

# y_pred_gs = RF.predict(X_test_gs)
# y_pred_proba_gs = RF.predict_proba(X_test_gs)

# classes_label_binary = ["Crystal", "Glass"]

# evaluate_model(RF, X_test_gs, y_test_gs, labels_list=le.classes_, cm_normalize="pred")


# # In[19]:


# # import numpy as np
# # import shap
# # from joblib import Parallel, delayed
# # from tqdm import tqdm

# # n_jobs = 14
# # feat_names = df_gs[FEATURES].columns.to_list()
# # explainer = shap.Explainer(RF, feature_names=feat_names, output_names=classes_label_binary, seed=RANDOM_SEED)

# # # Split X_train into chunks (if one chunk finishes early, the core immediately picks up the next one).
# # n_chunks = n_jobs * 20
# # chunks = np.array_split(X_train_gs, n_chunks)

# # print(f"Splitting data into {n_chunks} chunks to run on {n_jobs} cores...")

# # # Helper function
# # def get_explanation(chunk):
# #     # check_additivity=False prevents common numerical precision errors
# #     # that can occur when parallelizing Random Forest calculations
# #     return explainer(chunk, check_additivity=False)

# # # Run in parallel with TQDM
# # print("Calculating SHAP values...")

# # explanations_list = Parallel(n_jobs=n_jobs)(
# #     delayed(get_explanation)(chunk) for chunk in tqdm(chunks, desc="Progress")
# # )

# # # Concatenate results
# # print("Concatenating results...")
# # shap_values_all = np.concatenate([e.values for e in explanations_list], axis=0)
# # base_values_all = np.concatenate([e.base_values for e in explanations_list], axis=0)
# # data_all = np.concatenate([e.data for e in explanations_list], axis=0)

# # # Reconstruct the final Explanation object
# # explanation = shap.Explanation(
# #     values=shap_values_all,
# #     base_values=base_values_all,
# #     data=data_all,
# #     feature_names=feat_names,
# #     output_names=classes_label_binary
# # )

# # print("Done! Shape:", explanation.values.shape)

# # # Save the explanation object to a file
# # joblib.dump(explanation, '../data/support/explanation_shap/explanation_shap_GS.joblib')
# # print("Explanation object saved successfully!")


# # ### FEATING AND GS

# # In[20]:


# df_feateng_gs = pd.read_pickle(
#     "../data/processed/GFA_binary_FEATENG_GS.pkl.zip", compression="zip"
# )
# df_feateng_gs = df_feateng_gs.drop("Kod", axis=1)
# display(df_feateng_gs)
# print(df_feateng_gs.columns)
# print(Counter(df_feateng_gs["GF"]))

# TARGET_name = "GF"
# TARGET = [TARGET_name]
# FEATURES = df_feateng_gs.columns.drop(TARGET)

# le = LabelEncoder()
# df_feateng_gs[TARGET_name] = le.fit_transform(df_feateng_gs[TARGET_name])


# # In[21]:


# train_df_feateng_gs = df_feateng_gs.loc[train_indices]
# test_df_feateng_gs = df_feateng_gs.loc[test_indices]

# X_train_feateng_gs = train_df_feateng_gs.reindex(FEATURES, axis=1).values
# y_train_feateng_gs = train_df_feateng_gs.reindex(TARGET, axis=1).values.ravel()

# X_test_feateng_gs = test_df_feateng_gs.reindex(FEATURES, axis=1).values
# y_test_feateng_gs = test_df_feateng_gs.reindex(TARGET, axis=1).values.ravel()

# X_feateng_gs = df_feateng_gs.reindex(FEATURES, axis=1).values
# y_feateng_gs = df_feateng_gs.reindex(TARGET, axis=1).values.ravel()

# optimized_model_path = "../src/hyperparameters_tunning/GFA_binary_FEATENG_GS/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"
# optimized_model = joblib.load(optimized_model_path)
# all_hyperparams = optimized_model.get_params()

# print(f"\nFound {len(all_hyperparams)} optimized hyperparameters.")
# print("Listing all parameters:")
# pprint(all_hyperparams)

# # with open('../data/support/optimized_hyperparams_FEATENG_GS.json', 'w') as f:
# #     json.dump(all_hyperparams, f, indent=2, default=str)


# # In[22]:


# # RF = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
# RF = RandomForestClassifier(**all_hyperparams)

# RF.fit(X_train_feateng_gs, y_train_feateng_gs)

# y_pred_feateng_gs = RF.predict(X_test_feateng_gs)
# y_pred_proba_feateng_gs = RF.predict_proba(X_test_feateng_gs)

# classes_label_binary = ["Crystal", "Glass"]

# evaluate_model(
#     RF,
#     X_test_feateng_gs,
#     y_test_feateng_gs,
#     labels_list=le.classes_,
#     cm_normalize="pred",
# )


# # In[23]:


# # import numpy as np
# # import shap
# # from joblib import Parallel, delayed
# # from tqdm import tqdm

# # n_jobs = 14
# # feat_names = df_feateng_gs[FEATURES].columns.to_list()
# # explainer = shap.Explainer(RF, feature_names=feat_names, output_names=classes_label_binary, seed=RANDOM_SEED)

# # # Split X_train into chunks (if one chunk finishes early, the core immediately picks up the next one).
# # n_chunks = n_jobs * 20
# # chunks = np.array_split(X_train_feateng_gs, n_chunks)

# # print(f"Splitting data into {n_chunks} chunks to run on {n_jobs} cores...")

# # # Helper function
# # def get_explanation(chunk):
# #     # check_additivity=False prevents common numerical precision errors
# #     # that can occur when parallelizing Random Forest calculations
# #     return explainer(chunk, check_additivity=False)

# # # Run in parallel with TQDM
# # print("Calculating SHAP values...")

# # explanations_list = Parallel(n_jobs=n_jobs)(
# #     delayed(get_explanation)(chunk) for chunk in tqdm(chunks, desc="Progress")
# # )

# # # Concatenate results
# # print("Concatenating results...")
# # shap_values_all = np.concatenate([e.values for e in explanations_list], axis=0)
# # base_values_all = np.concatenate([e.base_values for e in explanations_list], axis=0)
# # data_all = np.concatenate([e.data for e in explanations_list], axis=0)

# # # Reconstruct the final Explanation object
# # explanation = shap.Explanation(
# #     values=shap_values_all,
# #     base_values=base_values_all,
# #     data=data_all,
# #     feature_names=feat_names,
# #     output_names=classes_label_binary
# # )

# # print("Done! Shape:", explanation.values.shape)

# # # Save the explanation object to a file
# # joblib.dump(explanation, '../data/support/explanation_shap/explanation_shap_FEATENG_GS.joblib')
# # print("Explanation object saved successfully!")


# # ## Curve Validation

# # In[24]:


# y_test = y_test_gs  # Same test set for all models

# # Data Dictionary
# models_data = {
#     "CHEM": y_pred_proba_chem,
#     "FEATENG": y_pred_proba_feateng,
#     "GS": y_pred_proba_gs,
#     "FEATENG + GS": y_pred_proba_feateng_gs,
# }

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
# linestyles = ["-", "--", "-.", ":"]
# # linestyles = ['-', '-', '-', '-']


# # Plot 1: ROC Curve
# for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):

#     if y_proba.ndim == 2 and y_proba.shape[1] == 2:
#         y_score = y_proba[:, 1]
#     else:
#         y_score = y_proba.ravel()

#     fpr, tpr, _ = roc_curve(y_test, y_score)
#     roc_auc = auc(fpr, tpr)
#     ax1.plot(
#         fpr,
#         tpr,
#         color=color,
#         linestyle=ls,
#         lw=2,
#         label=f"{name} (ROC-AUC = {roc_auc:.3f})",
#     )

# # Random guess line
# ax1.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", alpha=0.8)

# ax1.set_xlim([-0.02, 1.02])
# ax1.set_ylim([0.0, 1.05])
# ax1.set_xlabel("False Positive Rate")
# ax1.set_ylabel("True Positive Rate")
# ax1.legend(loc="lower right", frameon=False)
# ax1.grid(alpha=0.2)
# ax1.spines["top"].set_visible(False)
# ax1.spines["right"].set_visible(False)

# ax1.text(
#     -0.1,
#     1.05,
#     "(a)",
#     transform=ax1.transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="top",
#     ha="right",
# )

# # Plot 2: Precision-Recall Curve
# for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):

#     if y_proba.ndim == 2 and y_proba.shape[1] == 2:
#         y_score = y_proba[:, 1]
#     else:
#         y_score = y_proba.ravel()

#     precision, recall, _ = precision_recall_curve(y_test, y_score)
#     ap = average_precision_score(y_test, y_score)
#     ax2.plot(
#         recall,
#         precision,
#         color=color,
#         linestyle=ls,
#         lw=2,
#         label=f"{name} (PR-AUC = {ap:.3f})",
#     )

# ax2.set_xlabel("Recall")
# ax2.set_ylabel("Precision")
# ax2.legend(loc="lower left", frameon=False)
# ax2.grid(alpha=0.2)
# ax2.spines["top"].set_visible(False)
# ax2.spines["right"].set_visible(False)

# ax2.text(
#     -0.1,
#     1.05,
#     "(b)",
#     transform=ax2.transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="top",
#     ha="right",
# )

# plt.tight_layout()
# plt.savefig(
#     "../reports/figures/model_comparison_curves.png", dpi=300, bbox_inches="tight"
# )
# plt.show()


# # In[25]:


# fig, ax = plt.subplots(figsize=(8, 6))

# # Perfect calibration line
# # ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", linewidth=1.5)
# ax.plot(
#     [0, 1],
#     [0, 1],
#     color="gray",
#     label="Perfectly Calibrated",
#     lw=1.2,
#     linestyle="--",
#     alpha=0.8,
# )

# # Colors and markers
# colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
# markers = ["s", "o", "^", "v"]
# linestyles = ["-", "--", "-.", ":"]
# # linestyles = ['-', '-', '-', '-']

# # Plot Calibration Curves
# for (name, y_proba), color, marker, ls in zip(
#     models_data.items(), colors, markers, linestyles
# ):

#     if y_proba.ndim == 2 and y_proba.shape[1] == 2:
#         y_score = y_proba[:, 1]
#     else:
#         y_score = y_proba.ravel()

#     # Calculate calibration curve
#     # n_bins=10 means we check deciles (0-10%, 10-20%, etc.)
#     prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=10)

#     # Calculate Brier Score using the 1D score (Lower is better)
#     bs = brier_score_loss(y_test, y_score)

#     ax.plot(
#         prob_pred,
#         prob_true,
#         marker=marker,
#         markersize=5,
#         color=color,
#         linewidth=1.8,
#         linestyle=ls,
#         label=f"{name} (BS = {bs:.3f})",
#     )

# ax.set_ylabel("Fraction of Positives")
# ax.set_xlabel("Mean Predicted Probability")
# ax.set_ylim([-0.05, 1.05])
# ax.set_xlim([-0.05, 1.05])

# ax.legend(loc="upper left", frameon=False)
# ax.grid(alpha=0.3)

# # Remove top and right spines
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# plt.tight_layout()
# plt.savefig("../reports/figures/calibration_curve.png", dpi=300, bbox_inches="tight")
# plt.show()


# # ## Classification Metrics Comparison

# # In[29]:


# def extract_metrics_from_notebook(file_path):
#     """Reads the .ipynb file and extracts evaluation metrics from the outputs."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         nb = json.load(f)

#     metrics_blocks = []
#     # Loop through all cells looking for text outputs containing the metrics
#     for cell in nb.get("cells", []):
#         if cell.get("cell_type") == "code":
#             for out in cell.get("outputs", []):
#                 if out.get("output_type") == "stream":
#                     text = "".join(out.get("text", []))
#                     if "===== Evaluation Metrics =====" in text:
#                         metrics_blocks.append(text)

#     # Dictionary with the target metrics and their respective Regex patterns
#     patterns = {
#         r"Accuracy (\uparrow)": r"Accuracy:\s+([0-9.]+)",
#         r"Weighted Accuracy (\uparrow)": r"Balanced Accuracy:\s+([0-9.]+)",
#         r"ROC-AUC (\uparrow)": r"ROC AUC:\s+([0-9.]+)",
#         r"PR-AUC (\uparrow)": r"PR AUC \(class 1\):\s+([0-9.]+)",
#         r"Macro PR-AUC (\uparrow)": r"PR AUC \(macro\):\s+([0-9.]+)",
#         r"F1-Score (\uparrow)": r"F1-Score \(class 1\):\s+([0-9.]+)",
#         r"Macro F1-Score (\uparrow)": r"F1-Score \(macro\):\s+([0-9.]+)",
#         r"Brier Score (\downarrow)": r"Brier Score Loss:\s+([0-9.]+)",
#         r"Log Loss (\downarrow)": r"Log Loss:\s+([0-9.]+)",
#     }

#     model_names = ["CHEM", "FEATENG", "GS", "FEATENG+GS"]
#     data = {"Dataset": list(patterns.keys())}
#     for model in model_names:
#         data[model] = []

#     # Extract values block by block
#     for i, block in enumerate(metrics_blocks):
#         if i >= len(model_names):
#             break

#         model = model_names[i]
#         for metric, pattern in patterns.items():
#             match = re.search(pattern, block)
#             val = round(float(match.group(1)), 3) if match else np.nan
#             data[model].append(val)

#     return data


# def generate_and_save_outputs(
#     data,
#     path_txt="../data/support/analysis_table/model_metrics_comparison.txt",
#     path_xlsx="../data/support/analysis_table/model_metrics_comparison.xlsx",
# ):
#     """Transforms the data into Markdown, saves as TXT, and exports to XLSX."""
#     df = pd.DataFrame(data)

#     # Convert to object type to allow string assignments (*0.824*)
#     formatted_df = df.astype(object).copy()

#     # Logic to find the best value (maximum or minimum)
#     for i, row in df.iterrows():
#         metric = row["Dataset"]
#         values = row[1:].astype(float).values

#         # If the metric has a down arrow (\downarrow), the LOWEST value is best
#         if "downarrow" in metric:
#             best_val = np.nanmin(values)
#         else:
#             # Otherwise, the HIGHEST value is best
#             best_val = np.nanmax(values)

#         # Apply asterisks (*) to the best values and format with 3 decimal places
#         for col in df.columns[1:]:
#             val = row[col]
#             if val == best_val:
#                 formatted_df.at[i, col] = f"*{val:.3f}*"
#             else:
#                 formatted_df.at[i, col] = f"{val:.3f}"

#     # Generate Markdown String
#     markdown_table = (
#         "| Dataset                      |   CHEM | FEATENG |     GS | FEATENG+GS |\n"
#     )
#     markdown_table += (
#         "| <l>                          |    <r> |     <r> |    <r> |        <r> |\n"
#     )
#     markdown_table += (
#         "|------------------------------+--------+---------+--------+------------|\n"
#     )
#     for i, row in formatted_df.iterrows():
#         col1 = f"| {row['Dataset']:<28} "
#         col2 = f"| {row['CHEM']:>6} "
#         col3 = f"| {row['FEATENG']:>7} "
#         col4 = f"| {row['GS']:>6} "
#         col5 = f"| {row['FEATENG+GS']:>10} |\n"
#         markdown_table += col1 + col2 + col3 + col4 + col5
#     markdown_table += (
#         "|------------------------------+--------+---------+--------+------------|"
#     )

#     # Save as TXT file
#     with open(path_txt, "w", encoding="utf-8") as f:
#         f.write(markdown_table)
#     print("-> Successfully saved", path_txt)

#     # Save as XLSX (Excel) file
#     # Requires the 'openpyxl' library installed (pip install openpyxl)
#     with pd.ExcelWriter(path_xlsx) as writer:
#         df.to_excel(writer, sheet_name="Raw Numbers", index=False)
#         formatted_df.to_excel(
#             writer, sheet_name="Formatted with Asterisks", index=False
#         )
#     print("-> Successfully saved", path_xlsx)

#     return markdown_table


# # --- Main Execution ---
# file_path = "2_machine_learning_analysis.ipynb"

# # Extract data
# extracted_data = extract_metrics_from_notebook(file_path)

# # Generate strings and save output files
# final_table = generate_and_save_outputs(extracted_data)

# # Print out the final table
# print("\n" + final_table)


# # ## Save Parameters Model

# # In[27]:


# import ast
# import json
# import os


# def extract_and_save_hyperparameters(
#     file_path, output_dir="../data/support/model_parameters"
# ):
#     """Extracts the hyperparameters for each model in the notebook and saves them in a directory."""

#     os.makedirs(output_dir, exist_ok=True)

#     with open(file_path, "r", encoding="utf-8") as f:
#         nb = json.load(f)

#     param_blocks = []

#     # Iterate through the notebook cells looking for configurations
#     for cell in nb.get("cells", []):
#         if cell.get("cell_type") == "code":
#             for out in cell.get("outputs", []):
#                 if out.get("output_type") == "stream":
#                     text = "".join(out.get("text", []))
#                     if "Listing all parameters:" in text:
#                         # Extract only the dictionary part (after "Listing all parameters:")
#                         dict_str = text.split("Listing all parameters:\n")[1].strip()
#                         param_blocks.append(dict_str)

#     # Model names in the order they appear
#     model_names = ["CHEM", "FEATENG", "GS", "FEATENG_GS"]
#     saved_files = []

#     # Convert strings to dictionaries and save
#     for i, dict_str in enumerate(param_blocks):
#         if i >= len(model_names):
#             break

#         model_name = model_names[i]

#         try:
#             # Convert the text string to an actual Python dictionary
#             params_dict = ast.literal_eval(dict_str)

#             # File path (e.g., support/CHEM_params.json)
#             file_name = f"{model_name}_params.json"
#             save_path = os.path.join(output_dir, file_name)

#             # Save the dictionary as a properly formatted JSON file
#             with open(save_path, "w", encoding="utf-8") as f:
#                 json.dump(params_dict, f, indent=4)

#             saved_files.append(save_path)
#             print(f"-> Successfully saved: {save_path}")

#         except Exception as e:
#             print(f"Error processing parameters for model {model_name}: {e}")

#     return saved_files


# # --- Execution ---
# file_path = "2_machine_learning_analysis.ipynb"
# extract_and_save_hyperparameters(file_path)


# # In[28]:


# import ast
# import json

# import numpy as np
# import pandas as pd


# def extract_hyperparameters_from_notebook(file_path):
#     """Reads the .ipynb file and extracts hyperparameter dictionaries from the outputs."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         nb = json.load(f)

#     param_blocks = []
#     # Loop through all cells looking for text outputs containing the hyperparameters
#     for cell in nb.get("cells", []):
#         if cell.get("cell_type") == "code":
#             for out in cell.get("outputs", []):
#                 if out.get("output_type") == "stream":
#                     text = "".join(out.get("text", []))
#                     if "Listing all parameters:" in text:
#                         # Extract the dictionary string
#                         dict_str = text.split("Listing all parameters:\n")[1].strip()
#                         param_blocks.append(dict_str)

#     model_names = ["CHEM", "FEATENG", "GS", "FEATENG+GS"]
#     data = {model: {} for model in model_names}

#     # Extract values block by block
#     for i, block in enumerate(param_blocks):
#         if i >= len(model_names):
#             break

#         model = model_names[i]
#         try:
#             # Safely evaluate the string as a Python dictionary
#             data[model] = ast.literal_eval(block)
#         except Exception as e:
#             print(f"Error parsing hyperparameters for {model}: {e}")

#     return data


# def format_hyperparam_value(v):
#     """Formats the values to match the table standard (None, strings, integers, or \\num notation)."""
#     if v is None:
#         return "None"
#     elif isinstance(v, bool):
#         return str(v)
#     elif isinstance(v, str):
#         return v
#     elif isinstance(v, int):
#         return str(v)
#     elif isinstance(v, float):
#         if v == 0.0:
#             return "0"
#         else:
#             # Format floats to LaTeX \num{X.XXe-Y}
#             sci_str = f"{v:.2e}"
#             base, exp = sci_str.split("e")
#             exp = int(exp)  # Convert to int to remove leading zeros (e.g., e-05 -> e-5)
#             return rf"\num{{{base}e{exp}}}"
#     return str(v)


# def generate_and_save_hyperparams(
#     data,
#     path_txt="../data/support/analysis_table/model_hyperparameters_comparison.txt",
#     path_xlsx="../data/support/analysis_table/model_hyperparameters_comparison.xlsx",
# ):
#     """Transforms the hyperparameter data into Markdown, saves as TXT, and exports to XLSX."""

#     # Define the search space mappings and order
#     search_spaces = {
#         "n_estimators": r"{10, 11, ..., 1000}",
#         "criterion": r"{gini, entropy, log_loss}",
#         "min_samples_split": r"{2, 3, ..., 20}",
#         "min_samples_leaf": r"{1, 2, ..., 20}",
#         "max_depth": r"{None} $\cup$ {1, 2, ..., 100}",
#         "max_features": r"{sqrt, log2, None} $\cup$ [\num{e-5}, 1]",
#         "max_leaf_nodes": r"{None} $\cup$ {10, 11, ..., 1000}",
#         "min_impurity_decrease": r"{0} $\cup$ [\num{e-5}, 1]",
#         "class_weight": r"{None, balanced, balanced_subsample}",
#         "ccp_alpha": r"{0} $\cup$ [\num{e-5}, 1]",
#         "max_samples": r"{None} $\cup$ [\num{e-4}, 1]",
#     }

#     model_names = ["CHEM", "FEATENG", "GS", "FEATENG+GS"]

#     # Build dictionaries for DataFrames
#     raw_dict = {
#         "Hyperparameter": list(search_spaces.keys()),
#         "Search space": list(search_spaces.values()),
#     }
#     fmt_dict = {
#         "Hyperparameter": list(search_spaces.keys()),
#         "Search space": list(search_spaces.values()),
#     }

#     for model in model_names:
#         raw_dict[model] = []
#         fmt_dict[model] = []
#         for hp in search_spaces.keys():
#             val = data[model].get(hp, None)
#             raw_dict[model].append(val)
#             fmt_dict[model].append(format_hyperparam_value(val))

#     df_raw = pd.DataFrame(raw_dict)
#     df_fmt = pd.DataFrame(fmt_dict)

#     # Generate Markdown String
#     markdown_table = "| Hyperparameter        | Search space                             |               CHEM |  FEATENG |                 GS |    FEATENG+GS |\n"
#     markdown_table += "| <l>                   | <l>                                      |                <r> |      <r> |                <r> |           <r> |\n"
#     markdown_table += "|-----------------------+------------------------------------------+--------------------+----------+--------------------+---------------|\n"

#     for i, row in df_fmt.iterrows():
#         hp = row["Hyperparameter"]
#         space = row["Search space"]
#         chem = row["CHEM"]
#         feateng = row["FEATENG"]
#         gs = row["GS"]
#         feateng_gs = row["FEATENG+GS"]

#         markdown_table += f"| {hp:<21} | {space:<40} | {chem:>18} | {feateng:>8} | {gs:>18} | {feateng_gs:>13} |\n"

#     markdown_table += "|-----------------------+------------------------------------------+--------------------+----------+--------------------+---------------|"

#     # Save as TXT file
#     with open(path_txt, "w", encoding="utf-8") as f:
#         f.write(markdown_table)
#     print("-> Successfully saved", path_txt)

#     # Save as XLSX (Excel) file
#     with pd.ExcelWriter(path_xlsx) as writer:
#         df_raw.to_excel(writer, sheet_name="Raw Numbers", index=False)
#         df_fmt.to_excel(writer, sheet_name="Formatted Table", index=False)
#     print("-> Successfully saved", path_xlsx)

#     return markdown_table


# # --- Main Execution ---
# file_path = "2_machine_learning_analysis.ipynb"

# # Extract data
# extracted_hyperparams = extract_hyperparameters_from_notebook(file_path)

# # Generate strings and save output files
# final_table_hp = generate_and_save_hyperparams(extracted_hyperparams)

# # Print out the final table
# print("\n" + final_table_hp)
