import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print("Loaded data successfully")
        return df
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return None