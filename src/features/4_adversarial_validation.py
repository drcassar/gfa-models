#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import describe, kstest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "../../data/interim/GF_binary_feature_engineering_32bit.pkl.zip"
SPLIT_DATA_DIR = "../../data/support/"

N_REPEATS = 30
N_CV = 5


def get_data():

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_pickle(INPUT_FILE, compression="zip")

    df_selection = df["elements"]

    train_indices = np.load(SPLIT_DATA_DIR + "train_indices.npy")
    test_indices = np.load(SPLIT_DATA_DIR + "test_indices.npy")

    train_df = df_selection.loc[train_indices]
    test_df = df_selection.loc[test_indices]

    return train_df, test_df


def histplot(data, name):
    axe = sns.histplot(data, discrete=False, stat="percent")
    fig = axe.get_figure()
    figext = ".png"
    path = Path(rf"plots/{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path.with_suffix(figext),
        dpi=150,
        bbox_inches="tight",
        pad_inches=2e-2,
    )
    plt.close(fig)


if __name__ == "__main__":

    if "x_train" not in globals():
        x_train, x_test = get_data()

    n_test = len(x_test)
    aucs = []

    for seed in range(N_REPEATS):
        print(f"Starting seed {seed}")
        x_train_sub = x_train.sample(n=n_test, random_state=seed)

        X = pd.concat([x_train_sub, x_test])
        y = np.array([0] * n_test + [1] * n_test)

        clf = GradientBoostingClassifier(n_estimators=100, random_state=seed)

        auc = cross_val_score(clf, X, y, cv=N_CV, scoring="roc_auc").mean()
        aucs.append(auc)

    print(f"AUC média: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
