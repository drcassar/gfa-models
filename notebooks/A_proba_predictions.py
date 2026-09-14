import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from glasspy.predict import VITRIFY

comps = [
    "Li2O(B2O3)2",
    "Li2O(SiO2)2",
    "Na2O(B2O3)2",
    "Na2O(SiO2)2",
    "CaOAl2O3(SiO2)2",
]

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


model_ = VITRIFY("CHEM")

feats = []

for comp in comps:
    feat = model_._featurizer(comp)
    feats.append(feat)

df = pd.concat(feats, axis=0)

proba = model.predict_proba(df)
print(proba)


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

model_ = VITRIFY("FEATENG")

feats = []

for comp in comps:
    feat = model_._featurizer(comp)
    feats.append(feat)

df = pd.concat(feats, axis=0)

proba = model.predict_proba(df)
print(proba)
