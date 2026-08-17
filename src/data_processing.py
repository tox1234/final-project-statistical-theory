import pandas as pd
import os

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the environmental dataset and standardizes column names
    to prevent parsing errors in statsmodels R-like formulas.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place 'pollution_dataset.csv' in the data directory.")
        
    df = pd.read_csv(file_path)
    
    # Standardizing columns for OLS and GLRT formulas
    rename_mapping = {
        'PM2.5': 'PM2_5',
        'Population Density': 'Population_Density',
        'Proximity to Industrial Areas': 'Proximity_to_Industrial_Areas',
        'Air Quality': 'Air_Quality'
    }
    df.rename(columns=rename_mapping, inplace=True)
    
    return df
