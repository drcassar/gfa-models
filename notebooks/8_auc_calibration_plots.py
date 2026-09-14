import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

str = f"./../src/hyperparameters_tunning/GFA_binary_CHEM/model/calibrated_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

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

model = joblib.load(str)

y_pred_proba_chem = model.predict_proba(X_test)

ece = compute_ece(y_test, y_pred_proba_chem[:, 1], n_bins=10)
print(f"ECE CHEM test: {ece}")


###############################################################################
#                                   FEATENG                                   #
###############################################################################

str = f"./../src/hyperparameters_tunning/GFA_binary_FEATENG/model/calibrated_model_binary_unimodal_RF_f1_score_macro_2000_TPESampler.pkl"

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

model = joblib.load(str)

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
    "DUMMY": y_pred_proba_dummy,
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
linestyles = ["-", "--", "-.", ":"]


# Plot 1: ROC-AUC
for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):

    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba.ravel()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=color, linestyle=ls, lw=2, label=f"{name}")

# Random guess line
# ax1.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--', alpha=0.8)

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
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

# Plot 2: Precision-Recall Curve
for (name, y_proba), color, ls in zip(models_data.items(), colors, linestyles):

    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba.ravel()

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    ax2.plot(recall, precision, color=color, linestyle=ls, lw=2, label=f"{name}")

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
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

plt.tight_layout()
plt.savefig("model_comparison_curves.png", dpi=300, bbox_inches="tight")
plt.close()


###############################################################################
#                               PLOT calibration                              #
###############################################################################

del models_data["DUMMY"]

fig, ax = plt.subplots(figsize=(6, 4.5))

# Perfect calibration line
# ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated", linewidth=1.5)
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
# linestyles = ['-', '-', '-', '-']

# Plot Calibration Curves
for (name, y_proba), color, marker, ls in zip(
    models_data.items(), colors, markers, linestyles
):

    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba.ravel()

    # Calculate calibration curve
    # n_bins=10 means we check deciles (0-10%, 10-20%, etc.)
    prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=10)

    # Calculate Brier Score using the 1D score (Lower is better)
    bs = brier_score_loss(y_test, y_score)

    ax.plot(
        prob_pred,
        prob_true,
        marker=marker,
        markersize=5,
        color=color,
        linewidth=1.8,
        linestyle=ls,
        label=f"{name}",
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
plt.close()
