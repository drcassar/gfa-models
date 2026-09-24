import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


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


###############################################################################
#                                     CHEM                                    #
###############################################################################

# str = f"./../src/hyperparameters_tunning/GFA_binary_CHEM/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

model_path_chem = f"./../src/hyperparameters_tunning/GFA_binary_CHEM/model/calibrated_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"
if not os.path.exists(model_path_chem):
    model_path_chem = f"./../src/hyperparameters_tunning/GFA_binary_CHEM/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

df = pd.read_pickle("./../data/processed/GFA_binary_CHEM.pkl.zip", compression="zip")
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

model = joblib.load(model_path_chem)

y_pred_proba_chem = model.predict_proba(X_test)

ece = compute_ece(y_test, y_pred_proba_chem[:, 1], n_bins=10)
print(f"ECE CHEM test: {ece}")


###############################################################################
#                                   FEATENG                                   #
###############################################################################

model_path_feateng = f"./../src/hyperparameters_tunning/GFA_binary_FEATENG/model/calibrated_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"
if not os.path.exists(model_path_feateng):
    model_path_feateng = f"./../src/hyperparameters_tunning/GFA_binary_FEATENG/model/best_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

df = pd.read_pickle("./../data/processed/GFA_binary_FEATENG.pkl.zip", compression="zip")
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

model = joblib.load(model_path_feateng)

y_pred_proba_feateng = model.predict_proba(X_test)

ece = compute_ece(y_test, y_pred_proba_feateng[:, 1], n_bins=10)
print(f"ECE FEATENG test: {ece}")


###############################################################################
#                                    DUMMY                                    #
###############################################################################

model = DummyClassifier()
model.fit(X_train, y_train)

y_pred_proba_dummy = model.predict_proba(X_test)

ece = compute_ece(y_test, y_pred_proba_dummy[:, 1], n_bins=10)
print(f"ECE DUMMY test: {ece}")

###############################################################################
#                                     Plot                                    #
###############################################################################

models_data = {
    "CHEM": y_pred_proba_chem,
    "FEATENG": y_pred_proba_feateng,
}

y_test_array = y_test.values.ravel() if hasattr(y_test, "values") else np.asarray(y_test).ravel()
baseline_pos = float(np.mean(y_test_array == 1))
baseline_neg = float(np.mean(y_test_array == 0))
baseline_macro = 0.5

print(f"Test samples: {len(y_test_array)}")
print(f"Baseline Positive (Glass): {baseline_pos:.4f}")
print(f"Baseline Negative (Crystal): {baseline_neg:.4f}")
print(f"Baseline Macro-average: {baseline_macro:.4f}")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8.5))

colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
linestyles = ["-", "--", "-.", ":"]

# Plot 1: ROC-AUC
for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba.ravel()

    fpr, tpr, _ = roc_curve(y_test_array, y_score)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=color, linestyle=ls, lw=2, label=f"{name}")

ax1.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", alpha=0.8, label="DUMMY")
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="lower right", frameon=False)
ax1.grid(alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.text(
    -0.1,
    1.05,
    "(a)",
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="right",
)

# Plot 2: PR Curve - Glass (Positive Class)
for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score_pos = y_proba[:, 1]
    else:
        y_score_pos = y_proba.ravel()

    precision_pos, recall_pos, _ = precision_recall_curve(y_test_array, y_score_pos, pos_label=1)
    ap_pos = average_precision_score(y_test_array, y_score_pos, pos_label=1)
    ax2.plot(recall_pos, precision_pos, color=color, linestyle=ls, lw=2, label=f"{name}")

ax2.plot([0, 1], [baseline_pos, baseline_pos], color="gray", linestyle="--", lw=1.2, alpha=0.8, label="DUMMY")
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc="lower left", frameon=False)
ax2.grid(alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.text(
    -0.1,
    1.05,
    "(b)",
    transform=ax2.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="right",
)

# Plot 3: PR Curve - Crystal (Negative Class)
for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score_neg = y_proba[:, 0]
    else:
        y_score_neg = 1.0 - y_proba.ravel()

    precision_neg, recall_neg, _ = precision_recall_curve(y_test_array == 0, y_score_neg)
    ap_neg = average_precision_score(y_test_array == 0, y_score_neg)
    ax3.plot(recall_neg, precision_neg, color=color, linestyle=ls, lw=2, label=f"{name}")

ax3.plot([0, 1], [baseline_neg, baseline_neg], color="gray", linestyle="--", lw=1.2, alpha=0.8, label="DUMMY")
ax3.set_xlim([-0.02, 1.02])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.legend(loc="lower left", frameon=False)
ax3.grid(alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.text(
    -0.1,
    1.05,
    "(c)",
    transform=ax3.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="right",
)

# Plot 4: PR Curve - Macro-average
recall_grid = np.linspace(0, 1, 1000)
for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score_pos = y_proba[:, 1]
        y_score_neg = y_proba[:, 0]
    else:
        y_score_pos = y_proba.ravel()
        y_score_neg = 1.0 - y_proba.ravel()

    precision_pos, recall_pos, _ = precision_recall_curve(y_test_array, y_score_pos, pos_label=1)
    precision_neg, recall_neg, _ = precision_recall_curve(y_test_array == 0, y_score_neg)

    ap_pos = average_precision_score(y_test_array, y_score_pos, pos_label=1)
    ap_neg = average_precision_score(y_test_array == 0, y_score_neg)
    macro_ap = (ap_pos + ap_neg) / 2.0

    interp_pos = np.interp(recall_grid, recall_pos[::-1], precision_pos[::-1])
    interp_neg = np.interp(recall_grid, recall_neg[::-1], precision_neg[::-1])
    precision_macro = (interp_pos + interp_neg) / 2.0

    ax4.plot(recall_grid, precision_macro, color=color, linestyle=ls, lw=2, label=f"{name}")

ax4.plot([0, 1], [baseline_macro, baseline_macro], color="gray", linestyle="--", lw=1.2, alpha=0.8, label="DUMMY")
ax4.set_xlim([-0.02, 1.02])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel("Recall")
ax4.set_ylabel("Precision")
ax4.legend(loc="lower left", frameon=False)
ax4.grid(alpha=0.2)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.text(
    -0.1,
    1.05,
    "(d)",
    transform=ax4.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="right",
)

plt.tight_layout()
plt.savefig("model_comparison_curves.png", dpi=300, bbox_inches="tight")
if os.path.exists("./../reports/figures"):
    plt.savefig("./../reports/figures/model_comparison_curves.png", dpi=300, bbox_inches="tight")
plt.close()


###############################################################################
#                               PLOT calibration                              #
###############################################################################

fig, ax = plt.subplots(figsize=(6, 4.5))

# Perfect calibration line
ax.plot(
    [0, 1],
    [0, 1],
    color="gray",
    label="Perfectly Calibrated",
    lw=1.2,
    linestyle="--",
    alpha=0.8,
)

# Colors and markers
colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
markers = ["s", "o", "^", "v"]
linestyles = ["-", "--", "-.", ":"]

# Plot Calibration Curves
for (name, y_proba), color, marker, ls in zip(
    models_data.items(), colors, markers, linestyles
):

    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba.ravel()

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_test_array, y_score, n_bins=10)

    # Calculate Brier Score using the 1D score (Lower is better)
    bs = brier_score_loss(y_test_array, y_score)

    ax.plot(
        prob_pred,
        prob_true,
        marker=marker,
        markersize=5,
        color=color,
        linewidth=1.8,
        linestyle=ls,
        label=f"{name} (BS = {bs:.3f})",
    )

ax.set_ylabel("Fraction of Positives")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylim([-0.05, 1.05])
ax.set_xlim([-0.05, 1.05])

ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.3)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=300, bbox_inches="tight")
if os.path.exists("./../reports/figures"):
    plt.savefig("./../reports/figures/calibration_curve.png", dpi=300, bbox_inches="tight")
plt.close()
