# This script calculates Glass Stability (GS) parameters for a dataset of glass compositions.
import pandas as pd
import numpy as np
from glasspy.predict import GlassNet

# Define file paths and constants at the top for easy configuration
INPUT_FILE = "../../data/interim/GF_binary_feature_engineering.pkl.zip"
OUTPUT_FILE = "../../data/processed/GFA_binary_processed.pkl.zip"
OUTPUT_FILE_32BIT = "../../data/processed/GFA_binary_processed_32bit.pkl.zip"
BAD_ELEMENTS = ["Th", "U"]

def calculate_glass_stability():
    """
    Loads intermediate glass data, removes unwanted elements, predicts 
    thermal properties using GlassNet, calculates Glass Stability (GS) 
    parameters, and saves both standard (64-bit) and memory-optimized (32-bit) datasets.
    """
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_pickle(INPUT_FILE, compression="zip")

    print(f"Filtering out unwanted elements: {BAD_ELEMENTS}...")
    for e in BAD_ELEMENTS:
        # Assuming a MultiIndex dataframe setup based on ("elements", e)
        logic = df[("elements", e)] > 0
        df = df.loc[~logic]
        df = df.drop(("elements", e), axis=1)

    print("Initializing GlassNet model...")
    model = GlassNet()

    print("Predicting thermal properties (Tg, Tc, Tx, Tm)...")
    el = df["elements"]
    preds = model.predict(el)

    Tg = preds["Tg"]
    Tc = preds["CrystallizationPeak"]
    Tx = preds["CrystallizationOnset"]
    Tm = preds["Tliquidus"]
    
    print("Predicting viscosity at Tm...")
    eta_Tm = model.predict_viscosity(Tm, el)

    print("Calculating Glass Stability (GS) parameters...")
    Kw_Tx = (Tx - Tg) / Tm
    Kw_Tc = (Tc - Tg) / Tm
    Kh_Tc = (Tc - Tg) / (Tm - Tc)
    H_Tx = (Tx - Tg) / Tg
    gamma_Tc = Tc / (Tg + Tm)
    jezica = eta_Tm / (Tm**2)

    print("Appending new GS features to the dataframe...")
    df[("GS_parameters", "Kw_Tx")] = Kw_Tx.values
    df[("GS_parameters", "Kw_Tc")] = Kw_Tc.values
    df[("GS_parameters", "Kh_Tc")] = Kh_Tc.values
    df[("GS_parameters", "H_Tx")] = H_Tx.values
    df[("GS_parameters", "gamma_Tc")] = gamma_Tc.values
    df[("GS_parameters", "jezica")] = jezica.values

    # Reorder columns to ensure GS features are grouped together and maintain a consistent structure
    new_order = ['compounds','elements','absolute_properties','weighted_properties','GS_parameters','metadata','sample_id','target']
    cols = [c for name in new_order for c in df.columns[df.columns.get_level_values(0)==name]]
    df = df.loc[:, cols]

    # 1. Save the original 64-bit version
    print(f"Saving standard 64-bit processed data to {OUTPUT_FILE}...")
    df.to_pickle(OUTPUT_FILE, compression="zip")
    
    # 2. Create the 32-bit version
    print("Converting data to 32-bit format to optimize memory usage...")
    df_32bit = df.copy()
    
    # Downcast float64 to float32
    float_cols = df_32bit.select_dtypes(include=['float64']).columns
    df_32bit[float_cols] = df_32bit[float_cols].astype(np.float32)
    
    # Downcast int64 to int32
    int_cols = df_32bit.select_dtypes(include=['int64']).columns
    df_32bit[int_cols] = df_32bit[int_cols].astype(np.int32)

    # Save the 32-bit version
    print(f"Saving optimized 32-bit processed data to {OUTPUT_FILE_32BIT}...")
    df_32bit.to_pickle(OUTPUT_FILE_32BIT, compression="zip")
    
    print("Process completed successfully!")

if __name__ == "__main__":
    calculate_glass_stability()