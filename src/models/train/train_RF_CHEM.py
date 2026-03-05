import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_full_model():
    """
    Trains a Random Forest classifier on the entire CHEM dataset using pre-optimized hyperparameters.
    """
    print("--- Training CHEM Model on ALL DATA ---")
    
    # Load data
    data_path = "../../../data/processed/GFA_binary_CHEM.pkl.zip"
    df = pd.read_pickle(data_path, compression='zip')
    
    # Pre-process
    df = df.drop('Kod', axis=1)
    target_name = 'GF'
    features = df.columns.drop([target_name])
    
    le = LabelEncoder()
    df[target_name] = le.fit_transform(df[target_name])
    
    X_full = df[features].values
    y_full = df[target_name].values.ravel()
    
    # Load Hyperparameters
    params_path = "../../../data/support/model_parameters/CHEM_params.json"
    with open(params_path, 'r') as f:
        hyperparams = json.load(f)
        
    # Initialize and Train Random Forest
    print(f"Loaded {len(hyperparams)} parameters. Training...")
    rf_model = RandomForestClassifier(**hyperparams)
    rf_model.fit(X_full, y_full)
    
    # Save the trained model
    output_model_path = "../../../models/RF_FULL_CHEM.pkl"
    joblib.dump(rf_model, output_model_path, compress=('zlib', 3))
    print(f"Model successfully saved to: {output_model_path}\n")

if __name__ == "__main__":
    train_full_model()