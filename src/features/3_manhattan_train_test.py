#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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


if __name__ == "__main__":
    train_df, test_df = get_data()

    dist = cdist(train_df, test_df, "cityblock").flatten()

    print(f"Min: {dist.min()}")
    print(f"Max: {dist.max()}")
