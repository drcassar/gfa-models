import pandas as pd

def main():
    # Loading the data
    print("Loading data...")
    df_raw = pd.read_csv("../data/raw/GF_data_compound.zip", sep="\t")
    df_chem = pd.read_pickle("../data/processed/GFA_binary_CHEM.pkl.zip", compression='zip')
    df_feateng = pd.read_pickle("../data/processed/GFA_binary_FEATENG.pkl.zip", compression='zip')
    df_gs = pd.read_pickle("../data/processed/GFA_binary_GS.pkl.zip", compression='zip')
    df_feateng_gs = pd.read_pickle("../data/processed/GFA_binary_FEATENG_GS.pkl.zip", compression='zip')

    # Saving the data as Excel files in the same directories
    print("Saving files to Excel...")
    df_raw.to_excel("../data/raw/GF_data_compound.xlsx", index=False)
    df_chem.to_excel("../data/processed/GFA_binary_CHEM.xlsx", index=False)
    df_feateng.to_excel("../data/processed/GFA_binary_FEATENG.xlsx", index=False)
    df_gs.to_excel("../data/processed/GFA_binary_GS.xlsx", index=False)
    df_feateng_gs.to_excel("../data/processed/GFA_binary_FEATENG_GS.xlsx", index=False)

    print("Conversion to Excel completed successfully!")

if __name__ == "__main__":
    main()