from utils.visualization import plot_forecast

# Plot for a specific ATM
plot_forecast(atm_id="ATM_001")  

# To plot all ATMs (example for 3 ATMs)
for atm_id in ["ATM_001", "ATM_002", "ATM_003"]:
    plot_forecast(atm_id=atm_id)