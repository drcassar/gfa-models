import os
import pandas as pd

# --- File Paths ---
DATASET_PATH = "../../data/processed/GFA_binary_processed.pkl.zip"
SFS_RESULTS_PATH = "log/SFS_final_results.xlsx"
OUTPUT_DIR = "../../data/processed/"

def build_final_datasets():
    """
    Builds the final ML datasets by merging specific blocks of the 
    processed dataset with the features selected by SFS.
    """
    print(f"Loading processed dataset from {DATASET_PATH}...")
    df = pd.read_pickle(DATASET_PATH, compression="zip")

    print(f"Loading SFS results from {SFS_RESULTS_PATH}...")
    sfs_results = pd.read_excel(SFS_RESULTS_PATH, sheet_name='Selected_Features_Summary')

    # Extract the list of selected features from the SFS results
    selected_features = sfs_results['added_feature'].tolist()
    selected_features = selected_features[:20]  # Limit to top 20 features

    print(f"Found {len(selected_features)} selected features. Filtering...")
    
    # Extract base blocks common to all datasets
    sample_id_df = df.get('sample_id', pd.DataFrame(index=df.index))
    target_df = df['target']
    
    # Group all feature categories evaluated during SFS
    df_all_features = pd.concat([
        df.get('elements', pd.DataFrame()),
        df.get('absolute_properties', pd.DataFrame()),
        df.get('weighted_properties', pd.DataFrame()),
        df.get('metadata', pd.DataFrame())
    ], axis=1)

    # Filter for the selected features present in the dataset
    available_features = [feat for feat in selected_features if feat in df_all_features.columns]
    
    if not available_features:
        print("Error: No selected features found in the dataset. Please check the feature names.")
        return

    df_selected_features = df_all_features[available_features]
    
    # --- Build Final Datasets ---
    
    # 1. CHEM Dataset: Compounds + Sample ID + Target
    print("Building GFA_binary_CHEM...")
    df_chem = pd.concat([df['compounds'], sample_id_df, target_df], axis=1)
    df_chem.to_pickle(os.path.join(OUTPUT_DIR, "GFA_binary_CHEM.pkl.zip"), compression="zip")

    # 2. FEATENG Dataset: Selected Features + Sample ID + Target
    print("Building GFA_binary_FEATENG...")
    df_feateng = pd.concat([df_selected_features, sample_id_df, target_df], axis=1)
    df_feateng.to_pickle(os.path.join(OUTPUT_DIR, "GFA_binary_FEATENG.pkl.zip"), compression="zip")

    # 3. GS Dataset: GS_parameters + Sample ID + Target
    print("Building GFA_binary_GS...")
    if 'GS_parameters' in df:
        df_gs = pd.concat([df['GS_parameters'], sample_id_df, target_df], axis=1)
        df_gs.to_pickle(os.path.join(OUTPUT_DIR, "GFA_binary_GS.pkl.zip"), compression="zip")
    else:
        print("Warning: 'GS_parameters' not found in the dataset. Skipping GFA_binary_GS.")

    # 4. FEATENG_GS Dataset: Selected Features + GS_parameters + Sample ID + Target
    print("Building GFA_binary_FEATENG_GS...")
    if 'GS_parameters' in df:
        df_feateng_gs = pd.concat([df_selected_features, df['GS_parameters'], sample_id_df, target_df], axis=1)
        df_feateng_gs.to_pickle(os.path.join(OUTPUT_DIR, "GFA_binary_FEATENG_GS.pkl.zip"), compression="zip")
    
    print("\nAll final datasets have been successfully saved!")

if __name__ == "__main__":
    build_final_datasets()