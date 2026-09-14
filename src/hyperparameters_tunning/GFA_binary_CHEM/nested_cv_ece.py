###################################################################################
#           Importing Libraries and Resources                                    #
###################################################################################

import numpy as np
import pandas as pd
from optuna import create_study
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict

N_OUTER_FOLDS = 5  # outer CV folds for ECE estimation
N_INNER_FOLDS = 5  # inner CV folds used by CalibratedClassifierCV
RANDOM_STATE = 42

RANDOM_SEED = 0
TEST_SPLIT_SIZE = 0.1
NUM_FOLD = 5
N_TRIALS = 2000

DATASET_TYPE = "Glass X Crystal Elements CVs Feature Engineering with Unimodal Method F1-Score Macro"
STUDY_NAME = "glass_formation_cv_FE_RF"

METRIC_NAME = "f1_score_macro"
MAIN_METRIC_SCORE = f1_score
OPTIMIZER = "TPESampler"


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) from true labels and
    predicted probabilities for the positive class.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels.
    y_prob : array-like of shape (n,)
        Predicted probability of the positive class (class 1).
    n_bins : int
        Number of equal-width bins over [0, 1].

    Returns
    -------
    ece : float
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_confidence = y_prob[in_bin].mean()
            bin_accuracy = y_true[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_prob)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


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

study_object = create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    storage=f"sqlite:///model/{STUDY_NAME}_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}.db",
    load_if_exists=True,
)

best_trial_model = study_object.best_trial


###############################################################################
#                                 Calibration                                 #
###############################################################################

outer_cv = StratifiedKFold(
    n_splits=N_OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

### Main Model

main_estimator = create_rf_model(best_trial_model)

oof_proba_main = cross_val_predict(
    main_estimator,
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=-1,
)

ece_main = compute_ece(y_train, oof_proba_main[:, 1], n_bins=10)
print(f"ECE Main Model (out-of-fold CV on training data): {ece_main:.4f}")


### Calibrated model

base_estimator = create_rf_model(best_trial_model)

calibrated_estimator = CalibratedClassifierCV(
    estimator=base_estimator, method="isotonic", cv=N_INNER_FOLDS, n_jobs=-1
)

oof_proba_calibrated = cross_val_predict(
    calibrated_estimator,
    X_train,
    y_train,
    cv=outer_cv,
    method="predict_proba",
    n_jobs=-1,
)

ece_calibrated = compute_ece(y_train, oof_proba_calibrated[:, 1], n_bins=10)
print(f"ECE Calibrated Model (out-of-fold CV on training data): {ece_calibrated:.4f}")


### Decision

ECE_THRESHOLD = 10

ece_reduction = (1 - ece_calibrated / ece_main) * 100

print(f"\nECE reduction: {ece_reduction:.4f} %")

if ece_reduction < ECE_THRESHOLD:
    print("Reduction below threshold: retaining Main Model.")
else:
    print("Reduction above threshold: selecting Calibrated Model.")
