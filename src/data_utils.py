import pandas as pd
from src.config import DATA_PATH, TARGET_COLUMN

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df

def basic_data_checks(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df.columns else {}
    }