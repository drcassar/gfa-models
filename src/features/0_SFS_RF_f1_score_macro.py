# Sequential Forward Selection (SFS) for Random Forest with F1-Score Macro Evaluation
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    classification_report,
    make_scorer,
)

# --- Configuration & Constants ---
INPUT_FILE = "../../data/interim/GF_binary_feature_engineering.pkl.zip"
SPLIT_DATA_DIR = "../../data/support/"
OUTPUT_DIR = "log"

RANDOM_STATE_NUM = 0
TEST_DATA_SIZE = 0.1
MAX_FEATURES = 100  # Number of features to select
NUM_FOLDS = 10


def evaluate_model(model, X_test, y_test, label_encoder=None):
    """
    Evaluate a binary classification model with multiple metrics.
    """
    if label_encoder is not None:
        labels = label_encoder.classes_
    else:
        labels = ["Class 0", "Class 1"]

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # --- Base metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    # --- PR AUC variations ---
    pr_auc_class1 = average_precision_score(y_test, y_pred_proba[:, 1])
    pr_auc_class0 = average_precision_score(1 - y_test, y_pred_proba[:, 0])
    pr_auc_macro = np.mean([pr_auc_class0, pr_auc_class1])
    pr_auc_micro = average_precision_score(y_test, y_pred_proba[:, 1], average="micro")
    pr_auc_weighted = pr_auc_class0 * np.mean(y_test == 0) + pr_auc_class1 * np.mean(
        y_test == 1
    )

    # --- F1 scores ---
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_per_class = f1_score(y_test, y_pred, average=None)

    # --- Precision & Recall ---
    precision_macro = precision_score(y_test, y_pred, average="macro")
    precision_micro = precision_score(y_test, y_pred, average="micro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    precision_per_class = precision_score(y_test, y_pred, average=None)

    recall_macro = recall_score(y_test, y_pred, average="macro")
    recall_micro = recall_score(y_test, y_pred, average="micro")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    recall_per_class = recall_score(y_test, y_pred, average=None)

    # --- Calibration & loss metrics ---
    brier = brier_score_loss(y_test, y_pred_proba[:, 1])
    logloss = log_loss(y_test, y_pred_proba)

    # --- Print metrics ---
    print("===== Evaluation Metrics =====")
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Balanced Accuracy:     {balanced_acc:.4f}")
    print(f"ROC AUC:               {roc_auc:.4f}")
    print("\n--- PR AUC ---")
    print(f"PR AUC (micro):        {pr_auc_micro:.4f}")
    print(f"PR AUC (macro):        {pr_auc_macro:.4f}")
    print(f"PR AUC (weighted):     {pr_auc_weighted:.4f}")
    print(f"PR AUC (class 0):      {pr_auc_class0:.4f}")
    print(f"PR AUC (class 1):      {pr_auc_class1:.4f}")
    print("\n--- F1-Scores ---")
    print(f"F1-Score (micro):      {f1_micro:.4f}")
    print(f"F1-Score (macro):      {f1_macro:.4f}")
    print(f"F1-Score (weighted):   {f1_weighted:.4f}")
    print(f"F1-Score (class 0):    {f1_per_class[0]:.4f}")
    print(f"F1-Score (class 1):    {f1_per_class[1]:.4f}")
    print("\n--- Precision ---")
    print(f"Precision (micro):     {precision_micro:.4f}")
    print(f"Precision (macro):     {precision_macro:.4f}")
    print(f"Precision (weighted):  {precision_weighted:.4f}")
    print(f"Precision (class 0):   {precision_per_class[0]:.4f}")
    print(f"Precision (class 1):   {precision_per_class[1]:.4f}")
    print("\n--- Recall ---")
    print(f"Recall (micro):        {recall_micro:.4f}")
    print(f"Recall (macro):        {recall_macro:.4f}")
    print(f"Recall (weighted):     {recall_weighted:.4f}")
    print(f"Recall (class 0):      {recall_per_class[0]:.4f}")
    print(f"Recall (class 1):      {recall_per_class[1]:.4f}")
    print("\n--- Loss Metrics ---")
    print(f"Brier Score Loss:      {brier:.4f}")
    print(f"Log Loss:              {logloss:.4f}\n")

    print("===== Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=labels))


def run_feature_selection():
    start_time_total = time.time()

    # --- Load and Prepare Data ---
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_pickle(INPUT_FILE, compression="zip")

    X_elements = df["elements"]
    X_abs = df["absolute_properties"]
    X_weighted = df["weighted_properties"]
    X_meta = df["metadata"]
    y = df["target"]

    df_selection = pd.concat([X_elements, X_abs, X_weighted, X_meta, y], axis=1)

    TARGET = ["GF"]
    FEATURES = df_selection.columns.drop(TARGET)

    le = LabelEncoder()
    df_selection["GF"] = le.fit_transform(df_selection["GF"])

    # train_indices, test_indices = train_test_split(
    #     df_selection.index,
    #     test_size=TEST_DATA_SIZE,
    #     random_state=RANDOM_STATE_NUM,
    #     stratify=df_selection[TARGET]
    # )

    train_indices = np.load(SPLIT_DATA_DIR + "train_indices.npy")
    test_indices = np.load(SPLIT_DATA_DIR + "test_indices.npy")

    train_df_selection = df_selection.loc[train_indices]
    test_df_selection = df_selection.loc[test_indices]

    X_train = train_df_selection.reindex(FEATURES, axis=1).values
    y_train = train_df_selection.reindex(TARGET, axis=1).values.ravel()

    X_test = test_df_selection.reindex(FEATURES, axis=1).values
    y_test = test_df_selection.reindex(TARGET, axis=1).values.ravel()

    # --- Setup Estimator ---
    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE_NUM, class_weight="balanced", n_jobs=-1
    )
    scorer = make_scorer(f1_score, average="macro")
    cv = StratifiedKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE_NUM
    )

    # --- Initial Evaluation (All Features) ---
    print("\n=== Initial Model Evaluation on Full Feature Set ===")
    scores = cross_val_score(
        estimator, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1
    )

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    criterion_score = min(mean_score, median_score)

    print(f"Number of features: {X_train.shape[1]}")
    print(f"Mean F1 Score (macro): {mean_score:.4f}")
    print(f"Median F1 Score (macro): {median_score:.4f}")
    print(f"Std F1 Score (macro): {std_score:.4f}")
    print(f"Criterion Score (min of mean/median): {criterion_score:.4f}")

    print("\nTraining baseline model on full training set...")

    estimator.fit(X_train, y_train)
    evaluate_model(estimator, X_test, y_test, label_encoder=le)

    # --- Setup Output Directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Sequential Forward Selection (SFS) ---
    selected_features = []
    remaining_features = FEATURES.to_list().copy()

    results = []
    all_combinations_results = []
    iteration_times = []

    for iteration in range(MAX_FEATURES):
        iteration_start_time = time.time()
        iteration_results = []

        print(
            f"\n=== Iteration {iteration + 1}: Selecting feature {iteration + 1} of {MAX_FEATURES} ==="
        )
        print(f"Currently selected: {selected_features}")

        for feat in tqdm(
            remaining_features, desc=f"Testing features (n={len(selected_features)})"
        ):
            candidate_features = selected_features + [feat]

            # Subsetting the features
            X_candidate = pd.DataFrame(X_train, columns=FEATURES)[
                candidate_features
            ].values

            scores = cross_val_score(
                estimator, X_candidate, y_train, cv=cv, scoring=scorer, n_jobs=-1
            )

            iter_mean = np.mean(scores)
            iter_median = np.median(scores)

            iteration_result = {
                "iteration": iteration + 1,
                "added_feature": feat,
                "selected_features": candidate_features,
                "num_features": len(candidate_features),
                "scores": scores,
                "median_score": iter_median,
                "mean_score": iter_mean,
                "std_score": np.std(scores),
                "criterion_score": min(iter_mean, iter_median),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
            }

            iteration_results.append(iteration_result)
            all_combinations_results.append(iteration_result)

        # Pick the best candidate
        best_candidate = max(iteration_results, key=lambda x: x["criterion_score"])
        selected_features = best_candidate["selected_features"]
        remaining_features = [
            f for f in remaining_features if f != best_candidate["added_feature"]
        ]
        results.append(best_candidate)

        # Track timing
        iteration_time = time.time() - iteration_start_time
        iteration_times.append(iteration_time)

        print(f"Selected feature {iteration + 1}: {best_candidate['added_feature']}")
        print(f"Criterion score: {best_candidate['criterion_score']:.4f}")
        print(f"Iteration time: {iteration_time:.2f}s")
        print("-" * 80)

    # --- Final Comprehensive Saves ---
    print("\n=== Saving Final Results ===")
    df_results_final = pd.DataFrame(results)

    # 1. Save Final SFS Results
    final_res_file = os.path.join(OUTPUT_DIR, "SFS_final_results.xlsx")

    with pd.ExcelWriter(final_res_file, engine="openpyxl") as writer:
        df_results_final.to_excel(
            writer, sheet_name="Selected_Features_Summary", index=False
        )

        perf_summary = df_results_final[
            [
                "added_feature",
                "median_score",
                "mean_score",
                "std_score",
                "criterion_score",
            ]
        ].copy()
        perf_summary["cumulative_features"] = perf_summary.index + 1
        perf_summary.to_excel(writer, sheet_name="Performance_Summary", index=False)

        pd.DataFrame({"final_selected_features": selected_features}).to_excel(
            writer, sheet_name="Final_Selection", index=False
        )

        pd.DataFrame(
            {
                "total_combinations_tested": [len(all_combinations_results)],
                "total_iterations": [MAX_FEATURES],
                "average_combinations_per_iteration": [
                    len(all_combinations_results) / MAX_FEATURES
                ],
            }
        ).to_excel(writer, sheet_name="Combinations_Count", index=False)

    print(f"Saved: {final_res_file}")

    # --- Train Final Model ---
    print("\n=== Final Selected Features ===")
    print(
        df_results_final[
            [
                "added_feature",
                "median_score",
                "mean_score",
                "std_score",
                "criterion_score",
            ]
        ]
    )
    print(f"\nTraining final model on {len(selected_features)} selected features...")

    X_train_top = pd.DataFrame(X_train, columns=FEATURES)[selected_features]
    X_test_top = pd.DataFrame(X_test, columns=FEATURES)[selected_features]

    final_estimator = RandomForestClassifier(
        random_state=RANDOM_STATE_NUM, class_weight="balanced", n_jobs=-1
    )
    final_estimator.fit(X_train_top, y_train)

    evaluate_model(final_estimator, X_test_top, y_test, label_encoder=le)

    total_time = time.time() - start_time_total

    # --- Print Summary ---
    print(f"\n=== EXECUTION SUMMARY ===")
    print(f"Total iterations: {MAX_FEATURES}")
    print(f"Total combinations tested: {len(all_combinations_results)}")
    print(
        f"Average combinations per iteration: {len(all_combinations_results) / MAX_FEATURES:.1f}"
    )
    print(f"Final selected features: {selected_features}")

    print(f"\n=== TIMING INFORMATION ===")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Average time per iteration: {np.mean(iteration_times):.2f}s")

    # 2. Save Timing Information
    timing_file = os.path.join(OUTPUT_DIR, "execution_timing.xlsx")
    pd.DataFrame(
        {
            "total_execution_time_seconds": [total_time],
            "total_execution_time_minutes": [total_time / 60],
            "average_iteration_time_seconds": [np.mean(iteration_times)],
            "total_iterations_time_seconds": [np.sum(iteration_times)],
            "number_of_iterations": [MAX_FEATURES],
            "total_combinations_tested": [len(all_combinations_results)],
        }
    ).to_excel(timing_file, index=False)

    print(f"Saved: {timing_file}")


if __name__ == "__main__":
    run_feature_selection()
