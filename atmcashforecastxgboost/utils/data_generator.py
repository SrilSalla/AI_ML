import numpy as np
import pandas as pd
from config import NUM_ATMS, START_DATE, END_DATE, DATA_DIR

def generate_synthetic_data(save_to_disk=True):
    """Generate synthetic ATM transaction data with location features"""
    print("Generating synthetic ATM transaction data...")
    """Generate synthetic ATM transaction data with location features"""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    atm_ids = [f'ATM_{i:03d}' for i in range(1, NUM_ATMS + 1)]
    location_types = np.random.choice(['Urban', 'Suburban', 'Rural'], NUM_ATMS)
    
    data = []
    for date in dates:
        for atm_id, loc_type in zip(atm_ids, location_types):
            # Base demand + location effects
            cash = _generate_daily_cash(date, loc_type)
            data.append({
                "date": date,
                "atm_id": atm_id,
                "location_type": loc_type,
                "cash_withdrawn": cash,
            })
    
    df = pd.DataFrame(data)
    if save_to_disk:
        df.to_csv(DATA_DIR / "synthetic_atm_data.csv", index=False)
    return df

def _generate_daily_cash(date, location_type):
    """Helper: Generate realistic cash withdrawals"""
    base = np.random.randint(3000, 8000)
    
    # Location modifiers
    if location_type == "Urban":
        base += np.random.randint(2000, 5000)
    elif location_type == "Suburban":
        base += np.random.randint(1000, 3000)
    
    # Time effects
    if date.weekday() >= 5:  # Weekend
        base *= 1.2
    if date.is_month_end:  # Month-end
        base *= 1.3
    
    return max(1000, base + np.random.normal(0, 500))