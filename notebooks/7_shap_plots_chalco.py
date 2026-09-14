import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from joblib import load
from sklearn.preprocessing import LabelEncoder

from translation import AtMol_translation, agg, gs, physchem

DPI = 300
EXT = ".png"
# EXT = ".pdf"

TARGET = "GF"
MAIN = ["Ge", "Te", "As", "Se"]

LOWESS_FRAC = 1 / 5


def plot_scatter(
    shap_values,
    feat_cols,
    figname,
    x_feature,
    color_feature=None,
    prop_name="P(glass)",
    unit="",
    base_value=0.0,
    figsize=(6, 5),
    lowess=LOWESS_FRAC,
):
    """
    Plots a SHAP scatter plot.

    Parameters
    ----------
    shap_values : np.ndarray
        2D array of SHAP values, shape (n_samples, n_features).
    feat_cols : list[str]
        Feature names corresponding to columns in shap_values.
    figname : str
        Output file path for the saved figure.
    x_feature : str
        Feature name to use as the x-axis (feature values).
    color_feature : str
        Feature name to use for the color scale.
    prop_name : str
        Label for the y-axis (SHAP value axis).
    unit : str
        Optional unit string appended to the title.
    base_value : float
        Base (expected) value shown in the plot title.
    figsize : tuple
        Figure size as (width, height) in inches.
    """

    fig, ax = plt.subplots(figsize=figsize)

    feat_cols = list(feat_cols)

    x_idx = feat_cols.index(x_feature)

    # Extract arrays
    x_vals = shap_values.data[:, x_idx]  # feature values for x-axis
    y_vals = shap_values.values[:, x_idx]  # SHAP values for x_feature (y-axis)

    if color_feature:
        color_idx = feat_cols.index(color_feature)
        color_vals = shap_values.data[:, color_idx]

        sc = ax.scatter(
            x_vals,
            y_vals,
            c=color_vals,
            cmap="coolwarm",
            alpha=0.6,
            edgecolors="none",
            s=16,
        )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_feature)

    else:

        sc = ax.scatter(
            x_vals,
            y_vals,
            alpha=0.6,
            edgecolors="none",
            s=16,
        )

    if lowess:

        lw = sm.nonparametric.lowess
        w = lw(y_vals, x_vals, frac=lowess)
        (line,) = ax.plot(w[:, 0], w[:, 1], label="LOWESS", color="green")

        line.set_path_effects(
            [pe.Stroke(linewidth=3, foreground="white"), pe.Normal()]  # contour/outline
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel(x_feature)
    ax.set_ylabel(f"SHAP value for {x_feature}\n({prop_name})")
    ax.set_title(f"Base value = {base_value:.2f}" + (f" {unit}" if unit else ""))

    fig.savefig(figname, dpi=DPI, bbox_inches="tight", pad_inches=2e-2)
    plt.close(fig)


def pretty_feature(txt):

    if txt in AtMol_translation.values():
        return txt

    if "|" in txt:
        things = txt.split("|")
        fun = agg[things[-1]]
        feat = physchem[things[1]]

        if txt.startswith("A"):
            return rf"${fun}(\lceil {feat} \rceil)$"
        else:
            return rf"${fun}({feat})$"

    else:
        return f"${gs[txt]}$"


def getdata():

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

    return X_test


###############################################################################
#                                    Config                                   #
###############################################################################

plt.rcParams.update({"mathtext.fontset": "dejavuserif"})

###############################################################################
#                                     Main                                    #
###############################################################################

PLOT_ALL_FILTERED = True
PLOT_COLORED = True

if __name__ == "__main__":

    # Shap values
    file_ = "explanation_shap/explanation_shap_CHEM_calibrado.joblib"
    explanation_CHEM = load(file_)
    explanation_glass_CHEM = explanation_CHEM[:, :, 1]
    base_value = explanation_glass_CHEM.base_values[0]
    feature_names_ = explanation_CHEM.feature_names
    feature_names = [pretty_feature(f) for f in feature_names_]
    shap_values = explanation_glass_CHEM

    # data
    X_test = getdata()

    for main in MAIN:

        logic = X_test[main] > 0
        shap_focus = shap_values[logic.values]

        plot_scatter(
            shap_focus,
            feature_names,
            rf"SHAP_CHEM_SCATTER_{main}" + EXT,
            x_feature=main,
            base_value=base_value,
        )

        if PLOT_COLORED:

            for col in X_test.columns:

                if main != col:

                    plot_scatter(
                        shap_focus,
                        feature_names,
                        rf"corrs_CHEM_COLORED/SHAP_CHEM_SCATTER_{main}_{col}" + EXT,
                        x_feature=main,
                        color_feature=col,
                        base_value=base_value,
                    )

        if PLOT_ALL_FILTERED:

            for col in X_test.columns:

                if main != col:

                    df = X_test.query(f"{main} > 0 & {col} > 0")

                    if len(df) > 0:

                        mask = X_test.index.isin(df.index)
                        shap_focus = shap_values[mask]

                        plot_scatter(
                            shap_focus,
                            feature_names,
                            rf"corrs_CHEM/SHAP_CHEM_SCATTER_{main}_{col}" + EXT,
                            x_feature=main,
                            color_feature=col,
                            base_value=base_value,
                        )
