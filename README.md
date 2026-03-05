# Will it form a glass?

Machine Learning pipeline to predict the **Glass Forming Ability (GFA)** of inorganic liquid composition, classifying them as either **Crystal** or **Glass**. 

This repository contains the end-to-end workflow: from data preprocessing and exploratory data analysis to feature engineering, Optuna-based hyperparameter tuning, model training (using Random Forests), and model interpretability (using SHAP). 

The project evaluates four different datasets to predict GFA:
* **CHEM**: Chemical composition features only.
* **FEATENG**: Engineered physical-chemical features.
* **GS**: Glass Stability parameters.
* **FEATENG_GS**: Combination of engineered features and GS parameters.

## 🛠️ Requirements & Environment Setup

This project requires **Python 3.11.14**. 

To reproduce the environment and install all dependencies, we highly recommend creating a **Conda** environment:

```bash
# Create a Conda environment named 'gfa_env' with the specific Python version
conda create -n gfa_env python=3.11.14

# Activate the environment
conda activate gfa_env

# Install the required packages
pip install -r requirements.txt
```

## 🚀 How to Use

If you have new compounds and want to predict their Glass Forming Ability without retraining the models:

1. Prepare a `.csv` file with your compounds. Ensure the dataset contains the exact same feature columns as the training set used for the specific model.
2. Navigate to the `src/models/predict/` directory.
3. Open the prediction script corresponding to the features you have (e.g., `predict_RF_CHEM.py`).
4. Update the `INPUT_DATA` and `OUTPUT_DATA` paths at the bottom of the script.
5. Run the script:
```bash
python src/models/predict/predict_RF_CHEM.py
```


The script will output a new `.csv` containing the `Predicted_Class` (Crystal or Glass) and the `Confidence_Score` (which represents the model's estimated probability for the predicted class, ranging from 0.50 to 1.00, where higher values indicate greater certainty).

## 🧠 How to Retrain Models

If you update the dataset or wish to regenerate the models using 100% of the data:

1. (Optional) Rerun the Optuna scripts inside `src/hyperparameters_tunning/` to find new optimal parameters. The current optimal parameters are already saved in `data/support/model_parameters/`.
2. Run the desired training script inside `src/models/train/`:
```bash
python src/models/train/train_RF_CHEM.py
```

3. The newly trained model will be serialized and saved to the `models/` directory as a `.pkl` file, ready for inference.

---

## 📂 Project Organization

```
├── LICENSE
├── README.md
├── requirements.txt               <- Dependencies to reproduce the environment.
│
├── data/
│   ├── interim/                   <- Intermediate transformed data.
│   ├── processed/                 <- Final canonical datasets for modeling (CHEM, FEATENG, etc.).
│   ├── raw/                       <- The original, immutable data dump.
│   └── support/                   <- Auxiliary files (train/test split indices, translations).
│       ├── analysis_table/        <- Output tables (metrics comparisons, feature statistics).
│       ├── explanation_shap/      <- Serialized SHAP explainers (joblib files).
│       └── model_parameters/      <- JSON files containing the optimized hyperparameters.
│
├── models/                        <- Final serialized models trained on 100% of data.
│   ├── RF_FULL_CHEM.pkl
│   ├── RF_FULL_FEATENG.pkl
│   ├── RF_FULL_FEATENG_GS.pkl
│   └── RF_FULL_GS.pkl
│
├── notebooks/                     <- Jupyter notebooks for exploration and analysis.
│   ├── 0_data_processing.ipynb    <- Initial data cleaning and preprocessing.
│   ├── 1_data_analysis.ipynb      <- Exploratory Data Analysis (EDA).
│   ├── 2_machine_learning_analysis.ipynb <- Model evaluation and metrics generation.
│   └── 3_shap_analysis.ipynb      <- Model interpretability using SHAP.
│
├── reports/
│   └── figures/                   <- Generated figures
│
└── src/                           <- Source code for the pipeline.
    ├── features/                  <- Sequential Feature Selection (SFS), Add GS parameters and create final datasets.
    │
    ├── hyperparameters_tunning/   <- Optuna scripts and databases to find best model parameters.
    │   ├── GFA_binary_CHEM/
    │   ├── GFA_binary_FEATENG/
    │   ├── GFA_binary_FEATENG_GS/
    │   └── GFA_binary_GS/
    │
    ├── models/
    │   ├── predict/               <- Inference scripts for end-users on new data.
    │   │   └── predict_RF_*.py
    │   └── train/                 <- Scripts to train full models using optimized params.
    │       └── train_RF_*.py
    │
    ├── split_data.py              <- Script to separate data indices in train/test splits.
    └── create_requirements.py     <- Script to handle dependency tracking.

```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>