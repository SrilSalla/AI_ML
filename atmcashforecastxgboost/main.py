from train_model import train_and_save_model
from predict import predict_next_30_days
from config import MODEL_DIR
import os

def main():
    # Check if model exists, otherwise train first
    if not os.path.exists(MODEL_DIR / "xgboost_atm_model.pkl"):
        print("Training model first...")
        train_and_save_model()
    
    print("Generating predictions...")
    predict_next_30_days()
    print("Process completed successfully!")

if __name__ == "__main__":
    main()