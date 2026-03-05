#############################################################################################################
#                                      Importing Libraries and Resources                                    #
#############################################################################################################

import time
import pandas as pd
import numpy as np
import joblib
import sys
import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../functions'))
#from auxiliary_functions import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, average_precision_score, f1_score, balanced_accuracy_score, precision_score, recall_score, log_loss, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from optuna import create_study, Trial

def name_print(name):
    '''Displays a three line message with the given name'''
    
    name_size = len(name)
    space_size = 107 - name_size
    
    if space_size % 2 == 0:
        space_size_left, space_size_right = int(space_size/2), int(space_size/2)
    else: 
        space_size_left, space_size_right = int(space_size//2 + 1), int(space_size//2)

    print(f"#############################################################################################################\n"
          f"#{' ' * space_size_left}{                           name                           }{' ' * space_size_right}#\n"
          f"#############################################################################################################\n")


#############################################################################################################
#                                                    Constants                                              #
#############################################################################################################

RANDOM_SEED = 0
TEST_SPLIT_SIZE = 0.1
VAL_SPLIT_SIZE = (1 - TEST_SPLIT_SIZE) * TEST_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE) # test size = val sisze
NUM_FOLD = 10
N_TRIALS = 2000
N_TRIALS_CROSS_VALIDATION = int(N_TRIALS * 0.05) if N_TRIALS >= 20 else 1 # 5% of the best trials

DATASET_TYPE = "Glass X Crystal GFA and Top Features CVs Feature Engineering with Unimodal Method F1-Score Macro"
STUDY_NAME = 'glass_formation_GFAand_top_features_CV_RF_F1_macro'

METRIC_NAME = 'f1_score_macro'
MAIN_METRIC_SCORE = f1_score
#MAIN_METRIC_CV = 'f1_macro'
OPTIMIZER = 'TPESampler' # 'RandomSampler' # 'TPESampler' # 'CmaEsSampler' # 'NSGAIISampler' # 'GridSampler' # 'RandomSampler' # 'TPESampler'

#TIME_LIMIT_TRIAL = 10 * 60 # 10 min


#############################################################################################################
#                               Loading Data and Selecting Features and Target                              #
#############################################################################################################

df = pd.read_pickle("./../../../data/processed/GFA_binary_FEATENG_GS.pkl.zip", compression='zip')
df = df.drop('Kod', axis=1)

TARGET = ['GF']
FEATURES = df.columns.drop(TARGET)

le = LabelEncoder()
df['GF'] = le.fit_transform(df['GF'])

#############################################################################################################
#                                   Splitting Data into Train and Test Sets                                 #
#############################################################################################################

# indices = df.index
# train_indices, test_indices = train_test_split(
#     indices, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=df[TARGET]
# )

train_indices = np.load("./../../../data/support/train_indices.npy")
test_indices = np.load("./../../../data/support/test_indices.npy")

train_df = df.loc[train_indices]
test_df = df.loc[test_indices]

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

#############################################################################################################
#                                       Define the Search Space                                             #
#############################################################################################################

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

    has_max_depth = trial.suggest_categorical(
        "has_max_depth", [True, False]
    )
    if has_max_depth:
        parametros["max_depth"] = trial.suggest_int(
            "max_depth", 1, 100, log=True
        )
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

    has_max_leaf_nodes = trial.suggest_categorical(
        "has_max_leaf_nodes", [True, False]
    )
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

    has_class_weight = trial.suggest_categorical(
        "has_class_weight", [True, False]
    )
    if has_class_weight:
        parametros["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"]
        )
    else:
        parametros["class_weight"] = None

    has_ccp_alpha = trial.suggest_categorical(
        "has_ccp_alpha", [True, False]
    )
    if has_ccp_alpha:
        parametros["ccp_alpha"] = trial.suggest_float("ccp_alpha", 1e-5, 1, log=True)
    else:
        parametros["ccp_alpha"] = 0


    max_samples_is_float = trial.suggest_categorical(
        "max_samples_is_float", [True, False]
    )
    if max_samples_is_float:
        parametros["max_samples"] = trial.suggest_float("max_samples", 1e-4, 1, log=True)
    else:
        parametros["max_samples"] = None

    model = RandomForestClassifier(**parametros)

    return model

#############################################################################################################
#                                    Objective Function for Optuna CV                                     #
#############################################################################################################

def objective_function(trial, X, y):
    """Objective function using 10-fold CV and Optuna hyperparameter tuning."""
    
    model = create_rf_model(trial)
    
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=NUM_FOLD,
        scoring=make_scorer(MAIN_METRIC_SCORE, average='macro'),
        n_jobs=-1
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


#############################################################################################################
#                                   Create and Optimize the Study Objective                                 #
#############################################################################################################

study_object = create_study(
    direction="maximize",
    study_name=STUDY_NAME,
    storage=f"sqlite:///model/{STUDY_NAME}_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}.db",
    load_if_exists=True,
)

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
    "max_samples_is_float": False
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
    #study_object.optimize(partial_objective_function, n_trials=1, timeout=TIME_LIMIT_TRIAL)
    #study_object.trials_dataframe().to_csv(f'model/trials_RF_{METRIC_NAME}_{N_TRIALS}.csv')

    end_time = time.time()

    # Calculate the execution time per trial
    execution_time = (end_time - start_time) / 60
    trials_time_list.append(execution_time)

# Calculate the mean and the standard deviation with trials time list
trial_time_mean = np.mean(trials_time_list)

trial_time_std = np.std(trials_time_list)


#############################################################################################################
#                                         Save Trials to CSV                                                #
#############################################################################################################

results = []
for trial in study_object.trials:
    if trial.value is None:
        continue
    results.append({
        'trial_number': trial.number,
        'params': trial.params,
        'folds_scores': trial.user_attrs.get("cv_scores"),
        'score_used': trial.user_attrs.get("final_score"),
        'mean': trial.user_attrs.get("mean_score"),
        'median': trial.user_attrs.get("median_score"),
        'std': trial.user_attrs.get("std_score"),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(f'model/trials_RF_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}.csv', index=False)

#############################################################################################################
#                                         Display Basic Informations                                        #
#############################################################################################################

best_trial_model = study_object.best_trial

name_print(METRIC_NAME)
print(f"Dataset type: {DATASET_TYPE}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Test split size: {TEST_SPLIT_SIZE}")
print(f"Total number of trials: {N_TRIALS}, " 
      f"with cross-validation applied to the {N_TRIALS_CROSS_VALIDATION} best trials")
print(f"Mean time per trial: {trial_time_mean:.3f} ± {trial_time_std:.3f} minutes")
# print(f"Time limit per trial: {(TIME_LIMIT_TRIAL / 60):.2f} minutes")
print(f"Best trial number: {best_trial_model.number}")
print(f"Best trial parameters: {best_trial_model.params}")
print(f"Folds scores of {METRIC_NAME} for the best trial by cross validation: {best_trial_model.user_attrs['cv_scores']}")
print(f"Value of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'score_used']:.4f}")
print(f"Mean of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'mean']:.4f}")
print(f"Median of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'median']:.4f}")
print(f"Standard Deviation of {METRIC_NAME} for the best trial by cross validation: {results_df.loc[best_trial_model.number, 'std']:.4f} \n")
print(f'#############################################################################################################\n')


#############################################################################################################
#                                     Display the Optimized Model Metrics                                   #
#############################################################################################################

# The best model is trained with all the data and tested
best_RF_model_total = create_rf_model(best_trial_model)

best_RF_model_total.fit(X_train, y_train)
y_pred_RF = best_RF_model_total.predict(X_test)
y_pred_proba = best_RF_model_total.predict_proba(X_test)[:,1]

roc_auc_RF = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC metric for optimized model: {roc_auc_RF:.4f} \n')

pr_auc_RF = average_precision_score(y_test, y_pred_proba)
print(f'PR AUC metric for optimized model: {pr_auc_RF:.4f}')

pr_auc_macro_RF = average_precision_score(y_test, y_pred_proba, average="macro")
print(f'PR AUC Macro metric for optimized model: {pr_auc_macro_RF:.4f}')

pr_auc_micro_RF = average_precision_score(y_test, y_pred_proba, average="micro")
print(f'PR AUC Micro metric for optimized model: {pr_auc_micro_RF:.4f} \n')

f1_score_RF = f1_score(y_test, y_pred_RF)
print(f'F1-Score Binary (BCC) metric for optimized model: {f1_score_RF:.4f}')

f1_score_macro_RF = f1_score(y_test, y_pred_RF, average="macro")
print(f'F1-Score Macro metric for optimized model: {f1_score_macro_RF:.4f}')

f1_score_micro_RF = f1_score(y_test, y_pred_RF, average="micro")
print(f'F1-Score Micro metric for optimized model: {f1_score_micro_RF:.4f} \n')

precision_RF = precision_score(y_test, y_pred_RF)
print(f'Precision Binary (BCC) metric for optimized model: {precision_RF:.4f}')

recall_RF = recall_score(y_test, y_pred_RF)
print(f'Recall Binary (BCC) metric for optimized model: {recall_RF:.4f} \n')

accuracy_RF = accuracy_score(y_test, y_pred_RF)
print(f'Accuracy metric for optimized model: {accuracy_RF:.4f}')

balanced_accuracy_RF = balanced_accuracy_score(y_test, y_pred_RF)
print(f'Balanced Accuracy metric for optimized model: {balanced_accuracy_RF:.4f} \n')

log_loss_RF = log_loss(y_test, y_pred_proba)
print(f'Log Loss metric for optimized model: {log_loss_RF:.4f}') 

brier_score_RF = brier_score_loss(y_test, y_pred_proba)
print(f'Brier Score metric for optimized model: {brier_score_RF:.4f} \n') 

joblib.dump(best_RF_model_total, f"model/best_model_binary_unimodal_RF_{METRIC_NAME}_{N_TRIALS}_{OPTIMIZER}.pkl")


#############################################################################################################
#                                      Display the Standard Model Metrics                                   #
#############################################################################################################

print(f'#############################################################################################################\n')

# The best model is trained with all the data and tested
RF_model = RandomForestClassifier()

RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)
y_pred_proba = RF_model.predict_proba(X_test)[:,1]

roc_auc_RF = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC metric for standard model: {roc_auc_RF:.4f} \n')

pr_auc_RF = average_precision_score(y_test, y_pred_proba)
print(f'PR AUC metric for standard model: {pr_auc_RF:.4f}')

pr_auc_macro_RF = average_precision_score(y_test, y_pred_proba, average="macro")
print(f'PR AUC Macro metric for standard model: {pr_auc_macro_RF:.4f}')

pr_auc_micro_RF = average_precision_score(y_test, y_pred_proba, average="micro")
print(f'PR AUC Micro metric for standard model: {pr_auc_micro_RF:.4f} \n')

f1_score_RF = f1_score(y_test, y_pred_RF)
print(f'F1-Score Binary (BCC) metric for standard model: {f1_score_RF:.4f}')

f1_score_macro_RF = f1_score(y_test, y_pred_RF, average="macro")
print(f'F1-Score Macro metric for standard model: {f1_score_macro_RF:.4f}')

f1_score_micro_RF = f1_score(y_test, y_pred_RF, average="micro")
print(f'F1-Score Micro metric for standard model: {f1_score_micro_RF:.4f} \n')

precision_RF = precision_score(y_test, y_pred_RF)
print(f'Precision Binary (BCC) metric for standard model: {precision_RF:.4f}')

recall_RF = recall_score(y_test, y_pred_RF)
print(f'Recall Binary (BCC) metric for standard model: {recall_RF:.4f} \n')

accuracy_RF = accuracy_score(y_test, y_pred_RF)
print(f'Accuracy metric for standard model: {accuracy_RF:.4f}')

balanced_accuracy_RF = balanced_accuracy_score(y_test, y_pred_RF)
print(f'Balanced Accuracy metric for standard model: {balanced_accuracy_RF:.4f} \n')

log_loss_RF_std = log_loss(y_test, y_pred_proba)
print(f'Log Loss metric for standard model: {log_loss_RF_std:.4f}') 

brier_score_RF_std = brier_score_loss(y_test, y_pred_proba)
print(f'Brier Score metric for standard model: {brier_score_RF_std:.4f} \n') 
