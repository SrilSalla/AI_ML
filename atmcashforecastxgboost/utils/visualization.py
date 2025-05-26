import matplotlib.pyplot as plt
from config import FORECAST_DIR
import pandas as pd
import xgboost as xgb
from config import MODEL_DIR, DATA_DIR
from utils.feature_engineer import engineer_features
import joblib



def plot_forecast(atm_id="ATM_001"):
    """Plot 30-day forecast for a specific ATM"""
    df = pd.read_csv(FORECAST_DIR / "next_30_days.csv")
    subset = df[df["atm_id"] == atm_id]
    
    plt.figure(figsize=(12, 6))
    plt.plot(subset["date"], subset["predicted_cash"], marker="o")
    plt.title(f"30-Day Cash Forecast for {atm_id}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Cash Withdrawal ($)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted():
    """Compare model predictions vs actual values"""
    # Load model and preprocessor
    model = joblib.load(MODEL_DIR / "xgboost_atm_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    
    # Load and prepare data
    df = pd.read_csv(DATA_DIR / "synthetic_atm_data.csv")
    df = engineer_features(df)
    
    # Preprocess features
    X = df.drop(columns=["cash_withdrawn", "date"])
    y = df["cash_withdrawn"]
    X_processed = preprocessor.transform(X)
    
    # Make predictions
    dmatrix = xgb.DMatrix(X_processed)
    df["predicted"] = model.predict(dmatrix)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, df["predicted"], alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Cash Withdrawn ($)")
    plt.ylabel("Predicted Cash Withdrawn ($)")
    plt.title("Actual vs Predicted Values")
    plt.grid(True)
    plt.show()

def plot_forecast(atm_id="ATM_001"):
    """Plot 30-day forecast for a specific ATM"""
    df = pd.read_csv(DATA_DIR / "forecasts" / "next_30_days.csv")
    subset = df[df["atm_id"] == atm_id]
    
    plt.figure(figsize=(12, 6))
    plt.plot(subset["date"], subset["predicted_cash"], marker="o")
    plt.title(f"30-Day Forecast for {atm_id}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Cash Withdrawn ($)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()