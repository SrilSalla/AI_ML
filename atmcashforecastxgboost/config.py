import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FORECAST_DIR = DATA_DIR / "forecasts"

# Data generation
NUM_ATMS = 20
START_DATE = "2023-01-01"
END_DATE = "2025-05-13"

# Model hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": 42
}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)