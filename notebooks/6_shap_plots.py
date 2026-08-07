import matplotlib.pyplot as plt
import numpy as np
import shap
from joblib import load

from translation import AtMol_translation, agg, gs, physchem

PLOT = False
PLOT = True

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

# CHEM dataset
explanation_CHEM = load("0_explanation_shap_CHEM.joblib")
explanation_glass_CHEM = explanation_CHEM[:, :, 1]
base_value = explanation_glass_CHEM.base_values[0]
feature_names = explanation_CHEM.feature_names
feature_names = [pretty_feature(f) for f in feature_names]
shap_values = explanation_glass_CHEM
if PLOT:
    plot(shap_values, feature_names, "shap_CHEM" + EXT, False)
    plot(shap_values, feature_names, "shap_CHEM_cbar" + EXT, True)
    plot(
        shap_values,
        feature_names,
        "SHAP_CHEM_all" + EXT,
        False,
        max_display=len(feature_names),
    )

# FEATENG dataset
explanation_FEATENG = load("1_explanation_shap_FEATENG.joblib")
explanation_glass_FEATENG = explanation_FEATENG[:, :, 1]
base_value = explanation_glass_FEATENG.base_values[0]
feature_names = explanation_FEATENG.feature_names
feature_names = [pretty_feature(f) for f in feature_names]
shap_values = explanation_glass_FEATENG
if PLOT:
    plot(shap_values, feature_names, "shap_FEATENG" + EXT, False)
    plot(shap_values, feature_names, "shap_FEATENG_cbar" + EXT, True)
    plot(
        shap_values,
        feature_names,
        "SHAP_FEATENG_all" + EXT,
        False,
        max_display=len(feature_names),
    )

# GS dataset
explanation_GS = load("2_explanation_shap_GS.joblib")
explanation_glass_GS = explanation_GS[:, :, 1]
base_value = explanation_glass_GS.base_values[0]
feature_names = explanation_GS.feature_names
feature_names = [pretty_feature(f) for f in feature_names]
shap_values = explanation_glass_GS
if PLOT:
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
if PLOT:
    plot(shap_values, feature_names, "shap_GS" + EXT, False)
    plot(shap_values, feature_names, "shap_GS_cbar" + EXT, True)

# FEATENG_GS dataset
explanation_FEATENG_GS = load("3_explanation_shap_FEATENG_GS.joblib")
explanation_glass_FEATENG_GS = explanation_FEATENG_GS[:, :, 1]
base_value = explanation_glass_FEATENG_GS.base_values[0]
feature_names = explanation_FEATENG_GS.feature_names
feature_names = [pretty_feature(f) for f in feature_names]
shap_values = explanation_glass_FEATENG_GS
jezica_idx = list(shap_values.feature_names).index("jezica")
shap_values.data[:, jezica_idx] = np.log10(shap_values.data[:, jezica_idx])
feature_names[jezica_idx] = r"$\log$(Jezica)"
if PLOT:
    plot(shap_values, feature_names, "shap_FEATENG_GS" + EXT, False)
    plot(shap_values, feature_names, "shap_FEATENG_GS_cbar" + EXT, True)
    plot(
        shap_values,
        feature_names,
        "SHAP_FEATENG_GS_all" + EXT,
        False,
        max_display=len(feature_names),
    )
