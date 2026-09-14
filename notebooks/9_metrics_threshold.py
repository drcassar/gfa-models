import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


###############################################################################
#                                    DUMMY                                    #
###############################################################################

model = DummyClassifier()
model.fit(X_train, y_train)

y_pred_proba_dummy = model.predict_proba(X_test)


###############################################################################
#                                     CODE                                    #
###############################################################################


def compute_metrics_across_thresholds(
    y_true, y_prob_glass, thresholds=None, n_points=300
):
    """
    Computes class-specific and aggregate metrics across a range of
    decision thresholds.

    Parameters
    ----------
    y_true        : array-like, true binary labels (1=glass, 0=crystal)
    y_prob_glass  : array-like, predicted P(glass)
    thresholds    : optional array of thresholds; if None, uses n_points
                    evenly spaced values between min and max predicted prob
    n_points      : number of threshold values to evaluate

    Returns
    -------
    df : pd.DataFrame with one row per threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob_glass)

    if thresholds is None:
        thresholds = np.linspace(y_prob.min(), y_prob.max(), n_points)

    records = []
    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)

        # Skip degenerate thresholds where only one class is predicted
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue

        records.append(
            {
                "threshold": tau,
                "precision_glass": precision_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                ),
                "recall_glass": recall_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                ),
                "f1_glass": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                "precision_crystal": precision_score(
                    y_true, y_pred, pos_label=0, zero_division=0
                ),
                "recall_crystal": recall_score(
                    y_true, y_pred, pos_label=0, zero_division=0
                ),
                "f1_crystal": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
        )

    return pd.DataFrame(records)


def find_operating_points(df):
    """
    Identifies notable operating points on the threshold curve.
    Returns a dict of {label: threshold_value}.
    """
    ops = {}

    # Default threshold
    ops["Default (τ=0.50)"] = 0.50

    # Maximum balanced accuracy
    idx = df["balanced_accuracy"].idxmax()
    ops["Max balanced accuracy"] = df.loc[idx, "threshold"]

    # Maximum macro F1
    idx = df["f1_macro"].idxmax()
    ops["Max macro F1"] = df.loc[idx, "threshold"]

    # Maximum crystal recall (discovery screening: reject non-glass-formers)
    idx = df["recall_crystal"].idxmax()
    # This trivially gives tau → 0; find the highest tau where recall_crystal > 0.80
    # high_recall_crystal = df[df["recall_crystal"] >= 0.80]
    # if not high_recall_crystal.empty:
    #     ops["Crystal recall ≥ 0.80"] = high_recall_crystal["threshold"].max()

    # Maximum glass recall (miss as few glass formers as possible)
    # high_recall_glass = df[df["recall_glass"] >= 0.95]
    # if not high_recall_glass.empty:
    #     ops["Glass recall ≥ 0.95"] = high_recall_glass["threshold"].min()

    return ops


def plot_threshold_analysis(df, operating_points, model_name="CHEM", save_path=None):
    """
    Plots metric curves as a function of decision threshold with
    operating point annotations.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8 * 0.8, 9 * 0.8), sharex=True)

    tau = df["threshold"]

    # ── Top panel: class-specific precision and recall ────────────────────────
    ax = axes[0]
    ax.plot(tau, df["precision_glass"], color="#4C72B0", lw=2, label="Precision, glass")
    ax.plot(
        tau, df["recall_glass"], color="#4C72B0", lw=2, ls="--", label="Recall, glass"
    )
    ax.plot(
        tau, df["precision_crystal"], color="#C44E52", lw=2, label="Precision, crystal"
    )
    ax.plot(
        tau,
        df["recall_crystal"],
        color="#C44E52",
        lw=2,
        ls="--",
        label="Recall, crystal",
    )
    ax.set_ylabel("Score", fontsize=11)
    # ax.set_title(
    #     f"Threshold Analysis — {model_name} model", fontsize=12, fontweight="bold"
    # )
    ax.legend(fontsize=9, loc="center left")
    # ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

    # ── Bottom panel: aggregate metrics ──────────────────────────────────────
    ax = axes[1]
    ax.plot(
        tau, df["balanced_accuracy"], color="#55A868", lw=2, label="Balanced accuracy"
    )
    ax.plot(tau, df["f1_macro"], color="#8172B2", lw=2, label="Macro F1")
    ax.plot(tau, df["f1_glass"], color="#4C72B0", lw=2, ls=":", label="F1, glass")
    ax.plot(tau, df["f1_crystal"], color="#C44E52", lw=2, ls=":", label="F1, crystal")
    ax.set_xlabel("Decision threshold τ", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=9, loc="lower center")
    # ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1)

    # ── Annotate operating points on both panels ──────────────────────────────
    colors_ops = ["black", "#55A868", "#8172B2", "#C44E52", "#4C72B0"]
    for (label, tau_val), color in zip(operating_points.items(), colors_ops):
        for ax in axes:
            ax.axvline(tau_val, color=color, lw=1.2, ls=":", alpha=0.8)
        # Annotate only on top panel to avoid clutter
        axes[0].annotate(
            label,
            xy=(tau_val, 0.05),
            xytext=(tau_val + 0.01, 0.15),
            fontsize=7.5,
            color=color,
            rotation=90,
            va="bottom",
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def build_operating_points_table(df, operating_points):
    """
    Builds a compact table of metrics at each operating point.
    Suitable for inclusion in the paper as a supplementary table.
    """
    cols = [
        "threshold",
        "precision_glass",
        "recall_glass",
        "f1_glass",
        "precision_crystal",
        "recall_crystal",
        "f1_crystal",
        "f1_macro",
        "balanced_accuracy",
    ]
    col_labels = [
        "τ",
        "Prec. glass",
        "Rec. glass",
        "F1 glass",
        "Prec. crystal",
        "Rec. crystal",
        "F1 crystal",
        "Macro F1",
        "Bal. acc.",
    ]

    rows = []
    for label, tau_val in operating_points.items():
        # Find the row in df closest to the requested threshold
        idx = (df["threshold"] - tau_val).abs().idxmin()
        row = df.loc[idx, cols].tolist()
        rows.append([label] + [f"{v:.3f}" for v in row])

    result = pd.DataFrame(rows, columns=["Operating point"] + col_labels)
    return result


if __name__ == "__main__":

    y_proba_glass = y_pred_proba_chem[:, 1]

    df_thresholds = compute_metrics_across_thresholds(y_test, y_proba_glass)
    operating_points = find_operating_points(df_thresholds)

    plot_threshold_analysis(
        df_thresholds,
        operating_points,
        model_name="CHEM",
        save_path="threshold_analysis_CHEM.png",
    )

    table = build_operating_points_table(df_thresholds, operating_points)
    print(table.to_string(index=False))

    y_proba_glass = y_pred_proba_feateng[:, 1]

    df_thresholds = compute_metrics_across_thresholds(y_test, y_proba_glass)
    operating_points = find_operating_points(df_thresholds)

    plot_threshold_analysis(
        df_thresholds,
        operating_points,
        model_name="FEATENG",
        save_path="threshold_analysis_FEATENG.png",
    )

    table = build_operating_points_table(df_thresholds, operating_points)
    print(table.to_string(index=False))
