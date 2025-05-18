import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def get_next_holiday(date):
    """Calculate the next holiday date dynamically."""
    holiday_date = pd.Timestamp(year=date.year, month=12, day=25)
    if date > holiday_date:
        holiday_date = pd.Timestamp(year=date.year + 1, month=12, day=25)
    return holiday_date

def generate_synthetic_data():
    np.random.seed(42)
    num_atms = 10
    # Dynamically set the end date as the current date
    current_date = pd.Timestamp("2025-05-11")  # Set the current date
    dates = pd.date_range(start="2023-01-01", end=current_date, freq='D')  # Use current_date as the end date
    atm_ids = [f"ATM_{i:03d}" for i in range(1, num_atms + 1)]
    location_types = np.random.choice(["Urban", "Suburban", "Rural"], num_atms)
    
    data = []
    for date in dates:
        for atm_id, loc_type in zip(atm_ids, location_types):
            cash_withdrawn = np.random.randint(1000, 20000)
            if loc_type == "Urban":
                cash_withdrawn += np.random.randint(2000, 5000)
            elif loc_type == "Suburban":
                cash_withdrawn += np.random.randint(1000, 3000)
            if date.weekday() >= 5:
                cash_withdrawn += np.random.randint(1000, 4000)
            cash_withdrawn += np.random.randint(-500, 500)
            data.append({
                "date": date,
                "atm_id": atm_id,
                "location_type": loc_type,
                "foot_traffic": np.random.randint(1, 10),
                "cash_withdrawn": cash_withdrawn,
            })
    df = pd.DataFrame(data)
    df.to_csv("atm_data.csv", index=False)
    return df

def preprocess_data(df):
    current_date = pd.Timestamp("2025-05-11")  # Set a fixed current date
    df['date'] = pd.to_datetime(df['date'])
    df['days_until_holiday'] = df['date'].apply(lambda x: (get_next_holiday(x) - x).days)
    return df

def train_model(X_train, y_train):
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['atm_id', 'location_type'])],
        remainder='passthrough'
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_processed, y_train)
    return model, preprocessor

def predict_future(model, preprocessor, last_date, atm_ids, location_types):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq='D')
    future_data = []
    
    # Ensure location_types matches the length of atm_ids
    if len(location_types) != len(atm_ids):
        location_types = np.random.choice(location_types, len(atm_ids), replace=True)
    
    for date in future_dates:
        for atm_id, loc_type in zip(atm_ids, location_types):
            future_data.append({
                "date": date,
                "atm_id": atm_id,
                "location_type": loc_type,
                "foot_traffic": np.random.randint(1, 10),
                "day_of_week": date.weekday(),
                "is_weekend": (date.weekday() >= 5),
                "month": date.month,
                "is_month_end": date.is_month_end,
                "days_until_holiday": (get_next_holiday(date) - date).days,
            })
    future_df = pd.DataFrame(future_data)
    future_X = future_df.drop(columns=['date'])
    future_X_processed = preprocessor.transform(future_X)
    future_df['predicted_cash'] = model.predict(future_X_processed)
    return future_df

if __name__ == "__main__":
    # Step 1: Generate and preprocess data
    print("1.Generating synthetic data...")
    df = generate_synthetic_data()
    df = preprocess_data(df)
    
    # Step 2: Train-test split
    print("2.Splitting data into train and test sets...")
    df['date'] = pd.to_datetime(df['date'])
    test_mask = df['date'] >= (df['date'].max() - pd.Timedelta(days=14))
    X_train, X_test = df[~test_mask].drop(columns=['cash_withdrawn', 'date']), df[test_mask].drop(columns=['cash_withdrawn', 'date'])
    y_train, y_test = df[~test_mask]['cash_withdrawn'], df[test_mask]['cash_withdrawn']
    
    # Step 3: Train model
    print("3.Training model...")
    model, preprocessor = train_model(X_train, y_train)
    
    # Step 4: Evaluate
    print("4.Evaluating model...")
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    
    # Step 5: Predict future
    print("5.Predicting future cash withdrawals...")
    atm_ids = df['atm_id'].unique()
    location_types = df['location_type'].unique()
    future_df = predict_future(model, preprocessor, df['date'].max(), atm_ids, location_types)
    print(future_df.head(4))
    
    # Step 6: Save model and predictions
    print("6.Saving model and predictions...")
    joblib.dump(model, "cash_forecasting_model.pkl")

    # Drop 'days_until_holiday' before saving to CSV
    future_df.drop(columns=['days_until_holiday'], inplace=True)
    future_df.to_csv("future_predictions.csv", index=False)