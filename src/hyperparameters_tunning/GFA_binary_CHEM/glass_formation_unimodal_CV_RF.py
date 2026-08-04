###################################################################################
#           Importing Libraries and Resources                                    #
###################################################################################

import os
import sys
import time
from pprint import pprint

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optuna import Trial, create_study
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


def name_print(name):
    """Displays a three line message with the given name"""

    name_size = len(name)
    space_size = 107 - name_size

    if space_size % 2 == 0:
        space_size_left, space_size_right = int(space_size / 2), int(space_size / 2)
    else:
        space_size_left, space_size_right = int(space_size // 2 + 1), int(
            space_size // 2
        )

    print(
        f"#####################################################################\n"
        f"#{' ' * space_size_left}{name}{' ' * space_size_right}#\n"
        f"#####################################################################\n"
    )


def evaluate_binary_model(model, X_test, y_test, classe=1):
    """
    Evaluate a binary classifier and return a dictionary of metrics.
    'classe' defines which class is treated as the positive class.
    """

    y_pred = model.predict(X_test)
    y_pred_proba_all = model.predict_proba(X_test)
    y_pred_proba = y_pred_proba_all[:, classe]

    # ROC-AUC: no pos_label param in sklearn, so flip y_test when classe=0
    # so that the chosen class is always the positive one in the AUC computation.
    # Result should be ~0.875 for both classes (ROC-AUC is symmetric).
    y_test_roc = y_test if classe == 1 else (1 - y_test)

    metrics = {
        # --- Ranking metrics ---
        "roc_auc": roc_auc_score(y_test_roc, y_pred_proba),
        "pr_auc": average_precision_score(y_test, y_pred_proba, pos_label=classe),
        "pr_auc_macro": average_precision_score(
            y_test, y_pred_proba, average="macro", pos_label=classe
        ),
        "pr_auc_micro": average_precision_score(
            y_test, y_pred_proba, average="micro", pos_label=classe
        ),
        # --- Threshold-based metrics ---
        "f1": f1_score(y_test, y_pred, pos_label=classe),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "precision": precision_score(y_test, y_pred, pos_label=classe, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=classe),
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        # --- Calibration metrics ---
        "log_loss": log_loss(y_test, y_pred_proba_all),
        "brier_score": brier_score_loss(y_test, y_pred_proba, pos_label=classe),
    }

    return metrics


def plot_reliability_diagram_and_ece(
    model, X_test, y_test, n_bins=10, model_name="Model"
):
    """
    Plots the reliability diagram (calibration curve) and computes ECE.
    """
    y_prob = model.predict_proba(X_test)[:, 1]  # P(glass)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_test, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error (ECE)
    # Weighted average of |accuracy - confidence| per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_confidence = y_prob[in_bin].mean()
            bin_accuracy = y_test[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_prob)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "s-", label=f"{model_name} (ECE={ece:.4f})")
    ax.set_xlabel("Mean predicted probability (confidence)")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title(f"Reliability Diagram — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"reliability_{model_name}.png", dpi=150)
    plt.show()

    print(f"ECE ({model_name}): {ece:.4f}")
    print(f"Brier Score ({model_name}): {brier_score_loss(y_test, y_prob):.4f}")

    return ece


###############################################################################
#       Constants                                  #
###############################################################################

SEARCH = False
STANDARD_MODEL = False

RANDOM_SEED = 0
TEST_SPLIT_SIZE = 0.1
VAL_SPLIT_SIZE = (
    (1 - TEST_SPLIT_SIZE) * TEST_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE)
)  # test size = val sisze
NUM_FOLD = 5
N_TRIALS = 2000
N_TRIALS_CROSS_VALIDATION = (
    int(N_TRIALS * 0.05) if N_TRIALS >= 20 else 1
)  # 5% of the best trials

DATASET_TYPE = "Glass X Crystal Elements CVs Feature Engineering with Unimodal Method F1-Score Macro"
STUDY_NAME = "glass_formation_cv_FE_RF"

METRIC_NAME = "f1_score_macro"
MAIN_METRIC_SCORE = f1_score
# MAIN_METRIC_CV = 'f1_macro'
OPTIMIZER = "TPESampler"  # 'RandomSampler' # 'TPESampler' # 'CmaEsSampler' # 'NSGAIISampler' # 'GridSampler' # 'RandomSampler' # 'TPESampler'

# TIME_LIMIT_TRIAL = 10 * 60 # 10 min


###################################################################################
#    Loading Data and Selecting Features and Target                              #
###################################################################################

df = pd.read_pickle(
    "./../../../data/processed/GFA_binary_CHEM.pkl.zip", compression="zip"
)
df = df.drop("Kod", axis=1)

TARGET = ["GF"]
FEATURES = df.columns.drop(TARGET)

le = LabelEncoder()
df["GF"] = le.fit_transform(df["GF"])

###################################################################################
#        Splitting Data into Train and Test Sets                                 #
###################################################################################

# indices = df.index
# train_indices, test_indices = train_test_split(
#     indices, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=df[TARGET]
# )

train_indices = np.load("./../../../data/support/train_indices.npy")
test_indices = np.load("./../../../data/support/test_indices.npy")
test_data_leak_indices = np.load("./../../../data/support/test_data_leak_indices.npy")

train_df = df.loc[train_indices]
test_df = df.loc[test_indices]

# para evitar data leakage
test_df = test_df.loc[test_data_leak_indices]

X_train = train_df.reindex(FEATURES, axis=1).values
y_train = train_df.reindex(TARGET, axis=1).values.ravel()

X_test = test_df.reindex(FEATURES, axis=1).values
y_test = test_df.reindex(TARGET, axis=1).values.ravel()

X = df.reindex(FEATURES, axis=1).values
y = df.reindex(TARGET, axis=1).values.ravel()

# # Splitting Train Data into Final Train and Validations Sets
# X_final_train, X_val, y_final_train, y_val = train_test_split(
#     X_train, y_train, test_size=VAL_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=y_train
# )

###################################################################################
#            Define the Search Space                                             #
###################################################################################


def create_rf_model(trial):
    """Creates an instance of RandomForestClassifier (RF) with Optuna-tuned parameters."""

    parametros = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000, log=True),
        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": RANDOM_SEED,
    }

    has_max_depth = trial.suggest_categorical("has_max_depth", [True, False])
    if has_max_depth:
        parametros["max_depth"] = trial.suggest_int("max_depth", 1, 100, log=True)
    else:
        parametros["max_depth"] = None

    max_features_is_float = trial.suggest_categorical(
        "max_features_is_float", [True, False]
    )
    if max_features_is_float:
        parametros["max_features"] = trial.suggest_float(
            "max_features_float", 1e-5, 1, log=True
        )
    else:
        parametros["max_features"] = trial.suggest_categorical(
            "max_features_categorical", ["sqrt", "log2", None]
        )

    has_max_leaf_nodes = trial.suggest_categorical("has_max_leaf_nodes", [True, False])
    if has_max_leaf_nodes:
        parametros["max_leaf_nodes"] = trial.suggest_int(
            "max_leaf_nodes", 10, 1000, log=True
        )
    else:
        parametros["max_leaf_nodes"] = None

    min_impurity_decrease_is_float = trial.suggest_categorical(
        "min_impurity_decrease_is_float", [True, False]
    )
    if min_impurity_decrease_is_float:
        parametros["min_impurity_decrease"] = trial.suggest_float(
            "min_impurity_decrease", 1e-5, 1, log=True
        )
    else:
        parametros["min_impurity_decrease"] = 0

    has_class_weight = trial.suggest_categorical("has_class_weight", [True, False])
    if has_class_weight:
        parametros["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"]
        )
    else:
        parametros["class_weight"] = None

    has_ccp_alpha = trial.suggest_categorical("has_ccp_alpha", [True, False])
    if has_ccp_alpha:
        parametros["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-5, 1, log=True)
    else:
        parametros["ccp_alpha"] = 0

    max_samples_is_float = trial.suggest_categorical(
        "max_samples_is_float", [True, False]
    )
    if max_samples_is_float:
        parametros["max_samples"] = trial.suggest_float(
            "max_samples", 1e-4, 1, log=True
        )
    else:
        parametros["max_samples"] = None

    model = RandomForestClassifier(**parametros)

    return model


###################################################################################
#         Objective Function for Optuna CV                                     #
###################################################################################


def objective_function(trial, X, y):
    """Objective function using 10-fold CV and Optuna hyperparameter tuning."""

    model = create_rf_model(trial)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=NUM_FOLD,
        scoring=make_scorer(MAIN_METRIC_SCORE, average="macro"),
        n_jobs=-1,
    )

    mean_score = np.mean(cv_scores)
    median_score = np.median(cv_scores)
    std_score = np.std(cv_scores)
    final_score = np.min([mean_score, median_score])

    trial.set_user_attr("cv_scores", cv_scores.tolist())
    trial.set_user_attr("mean_score", float(mean_score))
    trial.set_user_attr("median_score", float(median_score))
    trial.set_user_attr("std_score", float(std_score))
    trial.set_user_attr("final_score", float(final_score))

    return final_score


def partial_objective_function(trial):
    return objective_function(trial, X_train, y_train)


###################################################################################
#        Create and Optimize the Study Objective                                 #
###################################################################################

study_object = create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    storage=f"sqlite:///model/{STUDY_NAME}_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}.db",
    load_if_exists=True,
)

if SEARCH:

    initial_guess = {
        "n_estimators": 100,
        "criterion": "gini",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "has_max_depth": False,
        "max_features_is_float": False,
        "max_features_categorical": "sqrt",
        "has_max_leaf_nodes": False,
        "min_impurity_decrease_is_float": False,
        "has_class_weight": False,
        "has_ccp_alpha": False,
        "max_samples_is_float": False,
        # CONSTANTS
        # "random_state": RANDOM_SEED,
        # "bootstrap": True,
        # "n_jobs": -1,
    }

    study_object.enqueue_trial(initial_guess)

    trials_time_list = []

    for _ in range(N_TRIALS):

        start_time = time.time()

        study_object.optimize(partial_objective_function, n_trials=1)
        # study_object.optimize(partial_objective_function, n_trials=1, timeout=TIME_LIMIT_TRIAL)
        # study_object.trials_dataframe().to_csv(f'model/trials_RF_{METRIC_NAME}_{N_TRIALS}.csv')

        end_time = time.time()

        # Calculate the execution time per trial
        execution_time = (end_time - start_time) / 60
        trials_time_list.append(execution_time)

    # Calculate the mean and the standard deviation with trials time list
    trial_time_mean = np.mean(trials_time_list)

    trial_time_std = np.std(trials_time_list)


###################################################################################
#              Save Trials to CSV                                                #
###################################################################################

results = []
for trial in study_object.trials:
    if trial.value is None:
        continue
    results.append(
        {
            "trial_number": trial.number,
            "params": trial.params,
            "folds_scores": trial.user_attrs.get("cv_scores"),
            "score_used": trial.user_attrs.get("final_score"),
            "mean": trial.user_attrs.get("mean_score"),
            "median": trial.user_attrs.get("median_score"),
            "std": trial.user_attrs.get("std_score"),
        }
    )

results_df = pd.DataFrame(results)

# results_df.to_csv(
#     f"model/trials_RF_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}.csv", index=False
# )


###################################################################################
#              Display Basic Informations                                        #
###################################################################################

best_trial_model = study_object.best_trial

name_print(METRIC_NAME)
print(f"Dataset type: {DATASET_TYPE}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Test split size: {TEST_SPLIT_SIZE}")
print(
    f"Total number of trials: {N_TRIALS}, "
    f"with cross-validation applied to the {N_TRIALS_CROSS_VALIDATION} best trials"
)
# print(f"Mean time per trial: {trial_time_mean:.3f} ± {trial_time_std:.3f} minutes")
# print(f"Time limit per trial: {(TIME_LIMIT_TRIAL / 60):.2f} minutes")
print(f"Best trial number: {best_trial_model.number}")
print(f"Best trial parameters: {best_trial_model.params}")
print(
    f"Folds scores of {METRIC_NAME} for the best trial by cross validation: {best_trial_model.user_attrs['cv_scores']}"
)
print(
    f"Value of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'score_used']:.4f}"
)
print(
    f"Mean of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'mean']:.4f}"
)
print(
    f"Median of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'median']:.4f}"
)
print(
    f"Standard Deviation of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'std']:.4f} \n"
)


###############################################################################
#                                 Calibration                                 #
###############################################################################

str = f"calibrated_model_binary_unimodal_RF_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}"

try:
    calibrated_model = joblib.load(f"model/{str}.pkl")

except FileNotFoundError:

    base_estimator = create_rf_model(best_trial_model)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_estimator,
        method="isotonic",
        cv=5,
        n_jobs=-1,
    )

    calibrated_model.fit(X_train, y_train)

    joblib.dump(calibrated_model, f"model/{str}.pkl")

print(f"## Best RF Model (calibrated) - Related to majority class (glass)\n")
metrics = evaluate_binary_model(calibrated_model, X_test, y_test, classe=1)
pprint(metrics)
ap_glass = metrics["pr_auc"]
print()

print(f"## Best RF Model (calibrated) - Related to minority class (crystal)\n")
metrics = evaluate_binary_model(calibrated_model, X_test, y_test, classe=0)
pprint(metrics)
ap_crystal = metrics["pr_auc"]
print()

pr_auc_macro_true = (ap_glass + ap_crystal) / 2
print("True PR-macro", pr_auc_macro_true)
print()
print()


###################################################################################
#          Display the Optimized Model Metrics                                   #
###################################################################################

print(f"#### Best RF Model\n")

str = f"best_model_binary_unimodal_RF_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}"

try:
    best_model = joblib.load(f"model/{str}.pkl")

except FileNotFoundError:
    best_model = create_rf_model(best_trial_model)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f"model/{str}.pkl")

print(f"## Best RF Model (uncalibrated) - Related to majority class (glass)\n")
metrics = evaluate_binary_model(best_model, X_test, y_test, classe=1)
pprint(metrics)
ap_glass = metrics["pr_auc"]
print()

print(f"## Best RF Model (uncalibrated) - Related to minority class (crystal)\n")
metrics = evaluate_binary_model(best_model, X_test, y_test, classe=0)
pprint(metrics)
ap_crystal = metrics["pr_auc"]
print()

pr_auc_macro_true = (ap_glass + ap_crystal) / 2
print("True PR-macro", pr_auc_macro_true)
print()
print()


ece_before = plot_reliability_diagram_and_ece(
    best_model, X_test, y_test, model_name="RF_uncalibrated"
)
ece_after = plot_reliability_diagram_and_ece(
    calibrated_model, X_test, y_test, model_name="RF_calibrated"
)

print()
print()


###############################################################################
#                                   Baseline                                  #
###############################################################################

print(f"#### Dummy Model\n")

dummy_model = DummyClassifier()
dummy_model.fit(X_train, y_train)

print(f"## Dummy Model - Related to majority class (glass)\n")
metrics = evaluate_binary_model(dummy_model, X_test, y_test, classe=1)
pprint(metrics)
ap_glass = metrics["pr_auc"]
print()

print(f"## Dummy Model - Related to minority class (crystal)\n")
metrics = evaluate_binary_model(dummy_model, X_test, y_test, classe=0)
pprint(metrics)
ap_crystal = metrics["pr_auc"]
print()

pr_auc_macro_true = (ap_glass + ap_crystal) / 2
print("True PR-macro", pr_auc_macro_true)
print()
print()

###################################################################################
#           Display the Standard Model Metrics                                   #
###################################################################################

if STANDARD_MODEL:
    print(f"#### Standard RF Model\n")

    # The best model is trained with all the data and tested
    RF_model = RandomForestClassifier()

    RF_model.fit(X_train, y_train)

    metrics = evaluate_binary_model(RF_model, X_test, y_test)
    pprint(metrics)
