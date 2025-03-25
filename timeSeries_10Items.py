import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("data/Annual 2023 Profits.csv")  # Ensure correct file path

# Clean column names and drop unnamed columns
df.columns = df.columns.str.strip()
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

# Define month mapping
month_mapping = {
    "January": "2023-01-01", "February": "2023-02-01", "March": "2023-03-01",
    "April": "2023-04-01", "May": "2023-05-01", "June": "2023-06-01",
    "July": "2023-07-01", "August": "2023-08-01", "September": "2023-09-01",
    "October": "2023-10-01", "November": "2023-11-01", "December": "2023-12-01"
}

# Convert 'Month' column to datetime
df['ds'] = df['Month'].map(month_mapping)
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# Define top 10 most profitable items
top_10_items = [
    "Obstetrics and Gynecology", "Obstetrics and Gynecology Consultation",
    "Pediatrics Consultation", "Medicine", "Flu Vaccine", "Rotateq Vaccine",
    'PCV "Pneumoccoccal" Vaccine', "Medical Packages",
    '6 IN 1 "Hexaxim" Vaccine', "Laser Circumcision"
]

# Filter and group data for the top 10 items
df_filtered = df[df['Item'].isin(top_10_items)]
df_grouped = df_filtered.groupby(['ds', 'Item'], as_index=False).agg({'Total Profit': 'sum'})

# Store forecast results
forecast_results = {}

# Forecasting loop for each item
for i, item in enumerate(top_10_items):
    item_df = df_grouped[df_grouped["Item"] == item].sort_values(by="ds")
    
    if len(item_df) < 2:
        print(f"Skipping {item}: Not enough data")
        continue

    item_df = item_df.rename(columns={'Total Profit': 'y'})

    # Fit Prophet model with multiplicative seasonality
    model = Prophet(seasonality_mode='multiplicative')
    model.fit(item_df[['ds', 'y']])

    # Create future dataframe (1-year prediction)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # Store forecast results
    forecast_results[item] = forecast[['ds', 'yhat']].rename(columns={'yhat': 'Predicted Profit'})
    forecast_results[item]['Item'] = item  # Add item name for table

    # Calculate evaluation metrics
    actual_values = item_df['y'].values
    predicted_values = forecast['yhat'][:len(item_df)].values

    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = mse ** 0.5
    mape = (abs(actual_values - predicted_values) / actual_values).mean() * 100

    # Print evaluation results
    print(f"\n=== Evaluation Metrics for {item} ===")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Plot actual and predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(item_df['ds'], item_df['y'], label=f"Actual {item}", marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], linestyle='dashed', label=f"Predicted {item}")
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    plt.title(f"Profit Forecast for {item}")
    plt.xlabel("Date")
    plt.ylabel("Total Profit")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 📊 Combine forecast results into a table (ensure all items appear)
forecast_table = pd.concat(forecast_results.values(), ignore_index=True)
forecast_table = forecast_table[['Item', 'ds', 'Predicted Profit']]

# Filter for forecasted values from Jan 1, 2024 to Jan 1, 2025 (next 12 months)
forecast_table = forecast_table[forecast_table['ds'] >= '2024-01-01']

# Format the Predicted Profit as currency
forecast_table['Predicted Profit'] = forecast_table['Predicted Profit'].apply(lambda x: f"${x:,.2f}")

# Export the forecast table to CSV
# forecast_table.to_csv("forecasted_profits_2024_2025.csv", index=False)                 ---- remove comment to actuaally store it

print("\n📊 Forecasted Profits Table (Jan 2024 - Jan 2025) exported to 'forecasted_profits_2024_2025.csv'.")
