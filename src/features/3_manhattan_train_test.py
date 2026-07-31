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

# from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "../../data/interim/GF_binary_feature_engineering_32bit.pkl.zip"
SPLIT_DATA_DIR = "../../data/support/"


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

    x_train, x_test = get_data()

    ###########################################################################
    #                                Manhattan                                #
    ###########################################################################

    LIMIT = 0.01

    min_dist = cdist(x_train, x_test, "cityblock").min(axis=0)

    mask = min_dist > LIMIT
    plot_data = min_dist[mask]

    print(len(plot_data) / len(min_dist) * 100)

    histplot(plot_data, "manhattan histogram")

    ###########################################################################
    #                Comparando treino-treino com treino-teste                #
    ###########################################################################

    treino_treino = cdist(x_train, x_train, "cityblock")
    mask = treino_treino != 0
    treino_treino = treino_treino[mask].min(axis=0)

    treino_teste = cdist(x_train, x_test, "cityblock").min(axis=0)

    histplot(treino_treino, "manhattan treino_treino")
    histplot(treino_teste, "manhattan treino_teste")

    nbrs = NearestNeighbors(n_neighbors=1, metric="manhattan", n_jobs=-1)
    nbrs.fit(x_train)

    treino_treino, indices = nbrs.kneighbors()  # (n_samples, 1) distances
    treino_treino = treino_treino.ravel()

    df = pd.concat(
        [
            pd.DataFrame({"value": treino_treino, "Source": "Train x Train"}),
            pd.DataFrame({"value": treino_teste, "Source": "Train x Test"}),
        ],
        ignore_index=True,
    )

    ###########################################################################
    #                                   Plot                                  #
    ###########################################################################

    bins = np.histogram_bin_edges(df["value"], bins=150)

    axe = sns.histplot(
        data=df,
        x="value",
        hue="Source",
        bins=bins,
        alpha=0.5,
        stat="percent",
        common_norm=False,
    )

    axe.axvline(LIMIT, ls="--", c="red")

    fig = axe.get_figure()
    axe.set(xlim=[0, 0.25])

    figext = ".png"
    path = Path(rf"plots/check distribution")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path.with_suffix(figext),
        dpi=150,
        bbox_inches="tight",
        pad_inches=2e-2,
    )
    plt.close(fig)

    ###########################################################################
    #                         Kolmogorov-Smirnov test                         #
    ###########################################################################

    print(len(treino_treino), len(treino_teste))

    result = kstest(treino_treino, treino_teste)
    print(result)
