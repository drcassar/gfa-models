# This script splits the dataset into training and testing sets, ensuring that the target variable is stratified.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
TEST_SIZE_SPLIT = 0.1
RANDOM_SEED = 0

# File paths
INPUT_FILE = "../data/processed/GF_binary_processed.pkl.zip"
OUTPUT_DIR = "../data/support/"

df = pd.read_pickle(INPUT_FILE, compression="zip")

TARGET_name = 'target'
TARGET = [TARGET_name]
FEATURES = df.columns.drop(TARGET)

le = LabelEncoder()
df[TARGET_name] = le.fit_transform(df[TARGET_name])

indices = df.index
train_indices, test_indices = train_test_split(
    indices, test_size=TEST_SIZE_SPLIT, random_state=RANDOM_SEED, stratify=df[TARGET]
)

# Save as .npy files
np.save(OUTPUT_DIR + 'train_indices.npy', train_indices)
np.save(OUTPUT_DIR + 'test_indices.npy', test_indices)