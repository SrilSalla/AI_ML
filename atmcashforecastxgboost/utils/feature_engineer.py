import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def engineer_features(df):
    """Create time-based and categorical features"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Time features
    df["day_of_week"] = df["date"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["quarter"] = df["date"].dt.quarter
    
    return df

def get_preprocessor():
    """Return configured ColumnTransformer"""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), ["atm_id", "location_type"])
        ],
        remainder="passthrough"
    )