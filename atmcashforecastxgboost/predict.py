import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
from config import MODEL_DIR, DATA_DIR, FORECAST_DIR
from utils.feature_engineer import engineer_features
import os

def predict_next_30_days():
    # Load artifacts
    model = joblib.load(MODEL_DIR / "xgboost_atm_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    
    # Get last date from training data
    training_data = pd.read_csv(DATA_DIR / "synthetic_atm_data.csv")
    last_date = pd.to_datetime(training_data['date']).max()
    
    # Generate future dates
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30,
        freq="D"
    )
    
    # Get unique categories from training data
    atm_ids = training_data['atm_id'].unique()
    location_types = training_data['location_type'].unique()
    
    # Create future data
    future_data = []
    for date in future_dates:
        for atm_id in atm_ids:
            for loc_type in location_types:
                future_data.append({
                    "date": date,
                    "atm_id": atm_id,
                    "location_type": loc_type,
                })
    
    # Process features
    future_df = engineer_features(pd.DataFrame(future_data))
    X_future = future_df.drop(columns=["date"])
    X_future_processed = preprocessor.transform(X_future)
    
    # Predict
    dfuture = xgb.DMatrix(X_future_processed)
    future_df["predicted_cash"] = model.predict(dfuture)
    
    # Save results
    os.makedirs(FORECAST_DIR, exist_ok=True)
    future_df.to_csv(FORECAST_DIR / "next_30_days.csv", index=False)
    print(f"\nPredictions saved to {FORECAST_DIR / 'next_30_days.csv'}")

if __name__ == "__main__":
    predict_next_30_days()