import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from config import MODEL_DIR, DATA_DIR, XGB_PARAMS
from utils.data_generator import generate_synthetic_data
from utils.feature_engineer import engineer_features, get_preprocessor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_and_save_model():
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    df = engineer_features(df)
    
    # Train/validation split
    X = df.drop(columns=["cash_withdrawn", "date"])
    y = df["cash_withdrawn"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Preprocessing
    print("Preprocessing data...")
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    dval = xgb.DMatrix(X_val_processed, label=y_val)
    
    # Train model
    print("Training model...")
    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    # Save artifacts
    joblib.dump(model, MODEL_DIR / "xgboost_atm_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    
    # Evaluate
    val_pred = model.predict(dval)
    print(f"\nValidation MAE: ${mean_absolute_error(y_val, val_pred):.2f}")
    print(f"Model saved to {MODEL_DIR}")

    # On validation set
    val_pred = model.predict(dval)
    print(f"MAE: ${mean_absolute_error(y_val, val_pred):.2f}")
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae = mean_absolute_error(y_val, val_pred)
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE as % of Avg Demand: {(mae / y_val.mean()) * 100:.1f}%")
    

if __name__ == "__main__":
    train_and_save_model()