import pandas as pd
import joblib
import os

def predict_new_data(input_csv_path, output_csv_path):
    """
    Loads the trained FEATENG+GS Random Forest model and predicts on new data.
    """
    print("--- Running Inference: FEATENG+GS Model ---")
    
    # Load the compressed model
    model_path = "../../../models/RF_FULL_FEATENG_GS.pkl.z"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
    print(f"Loading model from: {model_path}")
    rf_model = joblib.load(model_path)
    
    # Load the new user data
    print(f"Loading new data from: {input_csv_path}")
    df_new = pd.read_csv(input_csv_path)
    
    df_output = df_new.copy()
    
    # Pre-process the input (Drop 'Kod' or 'GF' if the user included them)
    columns_to_drop = ['Kod', 'GF']
    for col in columns_to_drop:
        if col in df_new.columns:
            df_new = df_new.drop(col, axis=1)
            
    # Ensure the columns in df_new exactly match the features used during training
    
    # Predict
    print("Making predictions...")
    predictions_numeric = rf_model.predict(df_new.values)
    probabilities = rf_model.predict_proba(df_new.values)
    
    # Map numeric predictions back to original labels (Assuming 0: Crystal, 1: Glass)
    class_mapping = {0: "Crystal", 1: "Glass"}
    predictions_labels = [class_mapping[pred] for pred in predictions_numeric]
    
    # Get the probability of the predicted class (max probability)
    confidence_scores = probabilities.max(axis=1)
    
    # Save results
    df_output['Predicted_Class'] = predictions_labels
    df_output['Confidence_Score'] = confidence_scores
    
    df_output.to_csv(output_csv_path, index=False)
    print(f"Predictions successfully saved to: {output_csv_path}\n")

if __name__ == "__main__":
    # Change these paths to point to the actual files
    INPUT_DATA = "path/to/user/new_compounds.csv"
    OUTPUT_DATA = "path/to/user/predictions_FEATENG_GS.csv"
    
    predict_new_data(INPUT_DATA, OUTPUT_DATA)