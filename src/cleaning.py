import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Les colonnes où 0 est impossible biologiquement
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Remplacer les 0 par NaN
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    
    # Remplir les NaN par la médiane de chaque colonne
    for col in cols_with_zeros:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print("=== Doublons ===")
    print(f"Nombre de doublons : {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print("Doublons supprimés !")
    else:
        print("Aucun doublon trouvé ✅")
    return df

def remove_outliers(df):
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    print("\n=== Outliers détectés (méthode IQR) ===")
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        print(f"{col} : {outliers} outliers → clippés")
        df[col] = df[col].clip(lower, upper)
    return df

def get_info(df):
    print("=== Aperçu des données ===")
    print(df.head())
    print("\n=== Infos générales ===")
    print(df.info())
    print("\n=== Statistiques ===")
    print(df.describe())