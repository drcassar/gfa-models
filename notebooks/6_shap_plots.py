import matplotlib.pyplot as plt
import numpy as np
import shap
from joblib import load

from translation import AtMol_translation, agg, gs, physchem

DPI = 300
EXT = ".png"
# EXT = ".pdf"


def signed_log(x):
    """Signed log1p: preserves sign, handles zeros, compresses large values."""
    return np.sign(x) * np.log1p(np.abs(x))


def plot(
    shap_values,
    feat_cols,
    figname,
    colorbar=False,
    prop_name="P(glass)",
    unit="",
    max_display=10,
    base_value=None,
):

    n_features = min(max_display, len(feat_cols))
    width = 4
    height = max(4, n_features * 0.4)  # ~0.4 inches per feature row, min 3 inches

    shap.summary_plot(
        shap_values,
        feature_names=feat_cols,
        show=False,
        # plot_type="violin",
        plot_type="dot",
        color_bar=colorbar,
        max_display=max_display,
        plot_size=(width, height),
    )

    plt.xlabel(f"SHAP value, {prop_name}")
    plt.title(f"Base value = {base_value:.2f}" + (f" {unit}" if unit else ""))
    fig = plt.gcf()
    fig.savefig(figname, dpi=DPI, bbox_inches="tight", pad_inches=2e-2)
    plt.close(fig)


def plot_scatter(
    shap_values,
    feat_cols,
    figname,
    x_feature,
    color_feature,
    prop_name="P(glass)",
    unit="",
    base_value=0.0,
    figsize=(6, 5),
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

    feat_cols = list(feat_cols)

    x_idx = feat_cols.index(x_feature)
    color_idx = feat_cols.index(color_feature)

    # Extract arrays
    x_vals = shap_values.data[:, x_idx]  # feature values for x-axis
    y_vals = shap_values.values[:, x_idx]  # SHAP values for x_feature (y-axis)
    color_vals = shap_values.data[:, color_idx]  # feature values for color scale

    fig, ax = plt.subplots(figsize=figsize)

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


###############################################################################
#                                    Config                                   #
###############################################################################

plt.rcParams.update({"mathtext.fontset": "dejavuserif"})

###############################################################################
#                                     Main                                    #
###############################################################################

CHEM = False
CHEM = True
FEATENG = True
FEATENG = False
OTHER = False

if CHEM:
    file_ = "explanation_shap/explanation_shap_CHEM_calibrado.joblib"
    explanation_CHEM = load(file_)
    explanation_glass_CHEM = explanation_CHEM[:, :, 1]
    base_value = explanation_glass_CHEM.base_values[0]
    feature_names_ = explanation_CHEM.feature_names
    feature_names = [pretty_feature(f) for f in feature_names_]
    shap_values = explanation_glass_CHEM

    plot(
        shap_values,
        feature_names,
        "shap_CHEM" + EXT,
        False,
        base_value=base_value,
    )
    plot(
        shap_values,
        feature_names,
        "shap_CHEM_cbar" + EXT,
        True,
        base_value=base_value,
    )
    plot(
        shap_values,
        feature_names,
        "SHAP_CHEM_all" + EXT,
        False,
        max_display=len(feature_names),
        base_value=base_value,
    )

if FEATENG:

    file_ = "explanation_shap/explanation_shap_FEATENG_calibrado.joblib"
    explanation_FEATENG = load(file_)
    explanation_glass_FEATENG = explanation_FEATENG[:, :, 1]
    base_value = explanation_glass_FEATENG.base_values[0]
    feature_names_ = explanation_FEATENG.feature_names
    feature_names = [pretty_feature(f) for f in feature_names_]
    shap_values = explanation_glass_FEATENG

    plot(
        shap_values,
        feature_names,
        "shap_FEATENG" + EXT,
        False,
        base_value=base_value,
    )
    plot(
        shap_values,
        feature_names,
        "shap_FEATENG_cbar" + EXT,
        True,
        base_value=base_value,
    )
    plot(
        shap_values,
        feature_names,
        "SHAP_FEATENG_all" + EXT,
        False,
        max_display=len(feature_names),
        base_value=base_value,
    )

    for x_name, save_name1 in zip(feature_names, feature_names_):
        for color_name, save_name2 in zip(feature_names, feature_names_):

            plot_scatter(
                shap_values,
                feature_names,
                rf"corrs/SHAP_FEATENG_SCATTER_{save_name1}_{save_name2}" + EXT,
                x_feature=x_name,
                color_feature=color_name,
                base_value=base_value,
            )


if OTHER:
    # GS dataset
    explanation_GS = load("2_explanation_shap_GS.joblib")
    explanation_glass_GS = explanation_GS[:, :, 1]
    base_value = explanation_glass_GS.base_values[0]
    feature_names = explanation_GS.feature_names
    feature_names = [pretty_feature(f) for f in feature_names]
    shap_values = explanation_glass_GS

    plot(shap_values, feature_names, "shap_GS" + EXT, False)
    plot(shap_values, feature_names, "shap_GS_cbar" + EXT, True)

    # GS dataset
    explanation_GS = load("2_explanation_shap_GS.joblib")
    explanation_glass_GS = explanation_GS[:, :, 1]
    base_value = explanation_glass_GS.base_values[0]
    feature_names = explanation_GS.feature_names
    feature_names = [pretty_feature(f) for f in feature_names]
    shap_values = explanation_glass_GS
    jezica_idx = list(shap_values.feature_names).index("jezica")
    shap_values.data[:, jezica_idx] = np.log10(shap_values.data[:, jezica_idx])
    feature_names[jezica_idx] = r"$\log$(Jezica)"

    plot(shap_values, feature_names, "shap_GS" + EXT, False)
    plot(shap_values, feature_names, "shap_GS_cbar" + EXT, True)

    # Jezica log
    explanation_FEATENG_GS = load("3_explanation_shap_FEATENG_GS.joblib")
    explanation_glass_FEATENG_GS = explanation_FEATENG_GS[:, :, 1]
    base_value = explanation_glass_FEATENG_GS.base_values[0]
    feature_names = explanation_FEATENG_GS.feature_names
    feature_names = [pretty_feature(f) for f in feature_names]
    shap_values = explanation_glass_FEATENG_GS
    jezica_idx = list(shap_values.feature_names).index("jezica")
    shap_values.data[:, jezica_idx] = np.log10(shap_values.data[:, jezica_idx])
    feature_names[jezica_idx] = r"$\log$(Jezica)"

    plot(shap_values, feature_names, "shap_FEATENG_GS" + EXT, False)
    plot(shap_values, feature_names, "shap_FEATENG_GS_cbar" + EXT, True)
    plot(
        shap_values,
        feature_names,
        "SHAP_FEATENG_GS_all" + EXT,
        False,
        max_display=len(feature_names),
    )
